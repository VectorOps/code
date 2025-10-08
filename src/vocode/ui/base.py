import asyncio
from typing import Optional, TYPE_CHECKING, List, Union, Any, Dict
import contextlib
from vocode.settings import build_model_from_settings

from vocode.logger import logger
from vocode.runner.runner import Runner
from vocode.runner.models import (
    RunEvent,
    RunInput,
    RespPacket,
    RespMessage,
    RespApproval,
    PACKET_TOKEN_USAGE,
    PACKET_FINAL_MESSAGE,
    PACKET_STATUS_CHANGE,
)
from vocode.state import Assignment, Message, RunnerStatus, RunStatus
from vocode.models import Graph, Workflow
from .proto import (
    UIPacket,
    UIPacketEnvelope,
    UIPacketRunEvent,
    UIPacketUIReset,
    UIPacketStatus,
    UIPacketRunInput,
    UIPacketCustomCommands,
    UIPacketCompletionRequest,
    UIPacketCompletionResult,
    UIPacketCommandResult,
    UIPacketRunCommand,
    UICommand,
    PACKET_RUN_INPUT,
    PACKET_RUN_COMMAND,
    PACKET_COMPLETION_REQUEST,
    PACKET_COMPLETION_RESULT,
)
from .rpc import RpcHelper

if TYPE_CHECKING:
    from vocode.project import Project


from .autocomplete import AutoCompletionManager
from . import autocomplete_providers as acp


class UIState:
    """
    Holds UI-related state and provides a high-level API around Runner.
    Exposes an in-memory async bi-directional protocol to connect a concrete UI.
    """

    def __init__(self, project: "Project") -> None:
        self.project = project
        self.runner: Optional[Runner] = None
        self.workflow: Optional[Workflow] = None
        self.assignment: Optional[Assignment] = None
        self._initial_message: Optional[Message] = None

        self._outgoing: "asyncio.Queue[UIPacketEnvelope]" = asyncio.Queue()
        self._incoming: "asyncio.Queue[UIPacketEnvelope]" = asyncio.Queue()
        self._rpc = RpcHelper(
            self._outgoing.put, "UIState", id_generator=self._next_msg_id
        )
        self.autocomplete = AutoCompletionManager(self)

        self.autocomplete.register(acp.PROVIDER_WORKFLOW_LIST, acp.ac_workflow_list)
        self.autocomplete.register(acp.PROVIDER_FILELIST, acp.ac_filelist)
        # Dedicated task for forwarding command events and a router for incoming packets
        self._drive_task: Optional[asyncio.Task] = None
        self._stop_watcher_task: Optional[asyncio.Task] = None
        self._cmd_events_task: Optional[asyncio.Task] = None
        self._incoming_router_task: Optional[asyncio.Task] = None
        self._msg_counter: int = 0
        self._client_msg_counter: int = 0
        self._last_status: Optional[RunnerStatus] = None
        self._selected_workflow_name: Optional[str] = None
        self._current_node_name: Optional[str] = None
        self._lock = asyncio.Lock()
        self._stop_signal: asyncio.Event = asyncio.Event()
        # LLM usage totals are stored on Project.llm_usage and proxied via properties.
        # Start background tasks for command deltas and incoming packet routing
        # LLM usage totals are stored on Project.llm_usage and proxied via properties.
        # Start background tasks for command deltas and incoming packet routing
        self._cmd_events_task = asyncio.create_task(self._forward_command_events())
        self._incoming_router_task = asyncio.create_task(self._route_incoming_packets())

    def _next_msg_id(self) -> int:
        self._msg_counter += 1
        return self._msg_counter

    def next_client_msg_id(self) -> int:
        """Generate a new message ID for a client-initiated message."""
        self._client_msg_counter += 1
        return self._client_msg_counter

    async def _watch_stop_signal(self) -> None:
        try:
            await self._stop_signal.wait()
            if self._drive_task and not self._drive_task.done():
                self._drive_task.cancel("Stop signal received")
        except asyncio.CancelledError:
            pass

    # ------------------------
    # Public protocol endpoints
    # ------------------------

    async def recv(self) -> UIPacketEnvelope:
        """
        Await next request from UIState (to be handled by UI client).
        """
        return await self._outgoing.get()

    async def send(self, envelope: UIPacketEnvelope) -> None:
        """
        Send a response from UI client back to UIState.
        """
        if envelope.source_msg_id is not None and self._rpc.handle_response(envelope):
            return

        # All incoming packets (including run_command and completion_request) are placed on the single incoming queue.
        await self._incoming.put(envelope)

    # ------------------------
    # Runner lifecycle control
    # ------------------------

    async def start(
        self,
        workflow: Workflow,
        *,
        initial_message: Optional[Message] = None,
        assignment: Optional[Assignment] = None,
    ) -> None:
        """
        Start a new runner for the given workflow. If a runner is already active, raises RuntimeError.
        """
        async with self._lock:
            if self._drive_task and not self._drive_task.done():
                raise RuntimeError("Runner is already active")

            self.workflow = workflow
            if workflow.name:
                self._selected_workflow_name = workflow.name

            if assignment and assignment.status == RunStatus.finished:
                self.assignment = Assignment()
            else:
                self.assignment = assignment or Assignment()

            self._initial_message = initial_message
            self.runner = Runner(
                workflow=workflow, project=self.project, initial_message=initial_message
            )
            self._msg_counter = 0
            self._last_status = None
            self._stop_signal.clear()
            # Always instruct UI to reset when a runner starts.
            await self._send_ui_reset()
            self._drive_task = asyncio.create_task(self._drive_runner())
            self._stop_watcher_task = asyncio.create_task(self._watch_stop_signal())

    async def stop(self) -> None:
        """
        Politely stop the runner (allowing it to be resumed later).
        """
        async with self._lock:
            if self.runner is None:
                return
            self._stop_signal.set()
            self.runner.stop()

    async def cancel(self) -> None:
        """
        Cancel the current in-flight executor work (non-resumable).
        """
        async with self._lock:
            if self.runner is None:
                return
            self._stop_signal.set()
            self.runner.cancel()

    async def restart(self) -> None:
        """
        Restart execution. If current runner is stopped, it will resume with existing Assignment.
        If it has finished or been canceled, a fresh Runner is created with the last workflow.
        """
        async with self._lock:
            prev_status = self.status
            if self.workflow is None:
                raise RuntimeError("No workflow available to restart")

            # If there's no runner, or it finished/canceled, create a new one
            if self.runner is None or self.runner.status in (
                RunnerStatus.finished,
                RunnerStatus.canceled,
            ):
                self.runner = Runner(
                    workflow=self.workflow,
                    project=self.project,
                    initial_message=self._initial_message,
                )

            # Re-drive the same assignment (resume if stopped)
            if self.assignment is None or self.assignment.status == RunStatus.finished:
                self.assignment = Assignment()

            # Prevent multiple drivers
            if self._drive_task and not self._drive_task.done():
                raise RuntimeError("Runner is already active")

            self._msg_counter = 0
            self._last_status = None
            self._stop_signal.clear()
            # Always instruct UI to reset when a runner restarts.
            await self._send_ui_reset()
            self._drive_task = asyncio.create_task(self._drive_runner())

    # ------------------------
    # Workflow helpers and accessors
    # ------------------------
    def list_workflows(self) -> List[str]:
        if not self.project.settings or not self.project.settings.workflows:
            return []
        return list(self.project.settings.workflows.keys())

    async def start_by_name(
        self, name: str, *, initial_message: Optional[Message] = None
    ) -> None:
        # Clear project-level state when switching to a different workflow
        if (
            self._selected_workflow_name is not None
            and name != self._selected_workflow_name
        ):
            self.project.project_state.clear()
            with contextlib.suppress(Exception):
                self.project.commands.clear()
        if not self.project.settings or name not in (
            self.project.settings.workflows or {}
        ):
            raise KeyError(f"Unknown workflow: {name}")
        wf_cfg = self.project.settings.workflows[name]

        graph = Graph(nodes=wf_cfg.nodes, edges=wf_cfg.edges)
        wf = Workflow(name=name, graph=graph)

        self._selected_workflow_name = name
        await self.start(wf, initial_message=initial_message)

    async def _shutdown_current_runner(self, *, cancel: bool) -> None:
        if not self._drive_task:
            return
        if cancel:
            await self.cancel()
        else:
            await self.stop()
        try:
            await asyncio.wait_for(self._drive_task, timeout=5.0)
        except asyncio.TimeoutError:
            self._drive_task.cancel()
            with contextlib.suppress(Exception):
                await self._drive_task
        except asyncio.CancelledError:
            # Task was cancelled but should have handled cleanup; ignore
            pass
        finally:
            self._drive_task = None
            self.runner = None
            self._last_status = None
            self._current_node_name = None
            # Drain any leftover outbound requests and inbound responses from the previous run
            with contextlib.suppress(asyncio.QueueEmpty):
                while True:
                    self._outgoing.get_nowait()
            with contextlib.suppress(asyncio.QueueEmpty):
                while True:
                    self._incoming.get_nowait()

    async def use(
        self, name: str, *, initial_message: Optional[Message] = None
    ) -> None:
        await self._shutdown_current_runner(cancel=False)
        await self.start_by_name(name, initial_message=initial_message)

    async def reset(self) -> None:
        if not self._selected_workflow_name:
            raise RuntimeError("No workflow selected to reset")
        await self._shutdown_current_runner(cancel=False)
        # Clear project-level state on reset
        self.project.project_state.clear()
        with contextlib.suppress(Exception):
            self.project.commands.clear()
        await self.start_by_name(
            self._selected_workflow_name, initial_message=self._initial_message
        )

    def is_active(self) -> bool:
        if self.status in (
            RunnerStatus.finished,
            RunnerStatus.stopped,
            RunnerStatus.canceled,
        ):
            return False
        return self._drive_task is not None and not self._drive_task.done()

    @property
    def status(self) -> RunnerStatus:
        return self.runner.status if self.runner else RunnerStatus.idle

    @property
    def current_node_name(self) -> Optional[str]:
        return self._current_node_name

    @property
    def selected_workflow_name(self) -> Optional[str]:
        return self._selected_workflow_name

    @property
    def acc_prompt_tokens(self) -> int:
        return self.project.llm_usage.prompt_tokens

    @property
    def acc_completion_tokens(self) -> int:
        return self.project.llm_usage.completion_tokens

    @property
    def acc_cost_dollars(self) -> float:
        return self.project.llm_usage.cost_dollars

    async def rewind(self, n: int = 1) -> None:
        """
        Rewind the last n steps. Allowed only when the runner is not running or waiting for input.
        """
        async with self._lock:
            if self.runner is None or self.assignment is None:
                raise RuntimeError("No runner/assignment to rewind")
            if self.runner.status in (RunnerStatus.running, RunnerStatus.waiting_input):
                raise RuntimeError(
                    "Cannot rewind while runner is running or waiting for input"
                )
            await self.runner.rewind(self.assignment, n)

    async def replace_user_input(
        self, resp: Union[RespMessage, RespApproval], n: Optional[int] = 1
    ) -> None:
        """
        Replace a user input (final prompt/confirm or prior message_request) and prepare resume.
        By default, it targets the last user input. A specific step can be targeted with step_index.
        """
        async with self._lock:
            if self.runner is None or self.assignment is None:
                raise RuntimeError("No runner/assignment available")
            if self.runner.status in (RunnerStatus.running, RunnerStatus.waiting_input):
                raise RuntimeError(
                    "Cannot replace input while runner is running or waiting for input"
                )
            self.runner.replace_user_input(self.assignment, resp, n=n)

    # ------------------------
    # Internal driver
    # ------------------------
    async def _send_ui_reset(self) -> None:
        await self._outgoing.put(
            UIPacketEnvelope(
                msg_id=self._next_msg_id(),
                payload=UIPacketUIReset(),
            )
        )

    async def _emit_status_if_changed(self) -> None:
        if self.runner is None:
            return
        curr = self.runner.status
        if curr != self._last_status:
            await self._outgoing.put(
                UIPacketEnvelope(
                    msg_id=self._next_msg_id(),
                    payload=UIPacketStatus(prev=self._last_status, curr=curr),
                )
            )
            self._last_status = curr
        # (Ticker logic removed: UIState emits status changes only.)

    async def _drive_runner(self) -> None:
        """
        Drive the Runner async generator, forwarding events to the UI and
        returning responses back to the Runner.
        """
        if self.runner is None or self.assignment is None:
            return

        agen = self.runner.run(self.assignment)
        to_send: Optional[RunInput] = None

        try:
            while True:
                try:
                    # Start/continue the runner
                    req: RunEvent = await agen.asend(to_send)
                    to_send = None
                except StopAsyncIteration:
                    # Final status notification
                    await self._emit_status_if_changed()
                    break

                # Notify status transition, if any
                await self._emit_status_if_changed()

                # Convert runner status-change packets to UIPacketStatus
                if req.event.kind == PACKET_STATUS_CHANGE:
                    sc = req.event  # type: ignore[attr-defined]
                    await self._outgoing.put(
                        UIPacketEnvelope(
                            msg_id=self._next_msg_id(),
                            payload=UIPacketStatus(
                                prev=sc.old_status,
                                curr=sc.new_status,
                                prev_node=sc.old_node,
                                curr_node=sc.new_node,
                            ),
                        )
                    )
                    # Track current node for UI convenience
                    self._current_node_name = sc.new_node
                    to_send = None
                    continue

                # Token usage packets are emitted but Project totals are updated by LLMExecutor directly.

                # Decide if this event should be forwarded to the UI client.
                # Suppress node finals when hide_final_output is True and no input is requested.
                suppress_event = False
                if req.event.kind == PACKET_FINAL_MESSAGE and not req.input_requested:
                    if self.runner is not None:
                        rn = self.runner.runtime_graph.get_runtime_node_by_name(
                            req.node
                        )
                        if rn is not None and rn.model.hide_final_output:
                            suppress_event = True

                # Await UI response only if required
                if req.input_requested and not suppress_event:
                    self._current_node_name = req.node
                    # Disable timeout while waiting for user input: block indefinitely
                    response_payload = await self._rpc.call(
                        UIPacketRunEvent(event=req), timeout=None
                    )

                    if response_payload and response_payload.kind == PACKET_RUN_INPUT:
                        to_send = response_payload.input
                    else:
                        if response_payload:
                            logger.warning(
                                "UIState: Ignored mismatched RPC response",
                                response=response_payload,
                            )
                        to_send = None
                else:
                    to_send = None
                    if not suppress_event:
                        # Forward the run event to the UI client with a correlation id
                        self._current_node_name = req.node
                        await self._outgoing.put(
                            UIPacketEnvelope(
                                msg_id=self._next_msg_id(),
                                payload=UIPacketRunEvent(event=req),
                            )
                        )

        except asyncio.CancelledError:
            # Runner was cancelled/stopped while waiting (e.g., for UI input).
            # Close the generator and emit final status, then exit cleanly.
            with contextlib.suppress(Exception):
                await agen.aclose()
            await self._emit_status_if_changed()
        except Exception as e:
            logger.exception("UIState: runner driver failed: %s", e)
        finally:
            # Cancel watcher task if it's still running
            if self._stop_watcher_task:
                self._stop_watcher_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._stop_watcher_task
            self._stop_watcher_task = None
            # Final status emission (in case it changed just before exit)
            await self._emit_status_if_changed()
            # Clear stop signal to leave UIState in a clean state for future starts.
            with contextlib.suppress(Exception):
                self._stop_signal.clear()
            self._current_node_name = None

    # ------------------------
    # Custom commands plumbing
    # ------------------------
    async def _forward_command_events(self) -> None:
        try:
            q = self.project.commands.subscribe()
            while True:
                delta = await q.get()
                added = [
                    UICommand(
                        name=c.name,
                        help=c.help,
                        usage=c.usage,
                        autocompleter=c.autocompleter,
                    )
                    for c in delta.added
                ]
                await self._outgoing.put(
                    UIPacketEnvelope(
                        msg_id=self._next_msg_id(),
                        payload=UIPacketCustomCommands(
                            added=added, removed=delta.removed
                        ),
                    )
                )
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception("UIState: command events forwarder failed: %s", e)

    async def _route_incoming_packets(self) -> None:
        try:
            while True:
                envelope = await self._incoming.get()
                kind = envelope.payload.kind
                if kind == PACKET_RUN_COMMAND:
                    await self._handle_run_command(envelope)
                elif kind == PACKET_COMPLETION_REQUEST:
                    await self._handle_completion_request(envelope)
                else:
                    # Non-command/completion requests should be handled elsewhere (e.g., RPC responses).
                    # Ignore unknown requests.
                    pass
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception("UIState: incoming packet router failed: %s", e)

    async def _handle_run_command(self, envelope: UIPacketEnvelope) -> None:
        from vocode.commands import CommandContext as ExecCommandContext

        rc = envelope.payload
        if rc.kind != PACKET_RUN_COMMAND:
            return

        name = rc.name
        args: List[str] = list(rc.input or [])
        ctx = ExecCommandContext(project=self.project, ui=self)  # type: ignore[arg-type]
        ok = True
        output: Optional[str] = None
        error: Optional[str] = None

        try:
            output = await self.project.commands.execute(name, ctx, args)
        except Exception as ex:
            ok = False
            error = str(ex)

        await self._outgoing.put(
            UIPacketEnvelope(
                msg_id=self._next_msg_id(),
                source_msg_id=envelope.msg_id,
                payload=UIPacketCommandResult(
                    name=name, ok=ok, output=output, error=error
                ),
            )
        )

    async def _handle_completion_request(self, envelope: UIPacketEnvelope) -> None:
        req = envelope.payload
        if req.kind != PACKET_COMPLETION_REQUEST:
            return

        handler = self.autocomplete.get(req.name)
        if not handler:
            resp = UIPacketCompletionResult(
                ok=False, suggestions=[], error=f"Unknown autocompleter '{req.name}'"
            )
        else:
            try:
                suggestions = await handler(self, req.params or {})
                resp = UIPacketCompletionResult(ok=True, suggestions=suggestions)
            except Exception as ex:
                resp = UIPacketCompletionResult(ok=False, suggestions=[], error=str(ex))

        await self._outgoing.put(
            UIPacketEnvelope(
                msg_id=self._next_msg_id(),
                source_msg_id=envelope.msg_id,
                payload=resp,
            )
        )
