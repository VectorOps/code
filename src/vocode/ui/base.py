import asyncio
from typing import Optional, TYPE_CHECKING, List, Union
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
)
from vocode.state import Assignment, Message, RunnerStatus
from vocode.models import Graph, Workflow
from .proto import (
    UIRequest,
    UIReqRunEvent,
    UIReqStatus,
    UIResponse,
    UIRespRunInput,
    UI_PACKET_RUN_INPUT,
    UIReqCustomCommands,
    UICommand,
    UIRespRunCommand,
    UIReqCommandResult,
    UI_PACKET_RUN_COMMAND,
)

if TYPE_CHECKING:
    from vocode.project import Project


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

        self._outgoing: "asyncio.Queue[UIRequest]" = asyncio.Queue()
        self._incoming: "asyncio.Queue[UIResponse]" = asyncio.Queue()
        # Dedicated queue for custom run_command invocations
        self._incoming_cmds: "asyncio.Queue[UIRespRunCommand]" = asyncio.Queue()
        self._drive_task: Optional[asyncio.Task] = None
        self._cmd_events_task: Optional[asyncio.Task] = None
        self._cmd_calls_task: Optional[asyncio.Task] = None
        self._req_counter: int = 0
        self._last_status: Optional[RunnerStatus] = None
        self._selected_workflow_name: Optional[str] = None
        self._current_node_name: Optional[str] = None
        self._lock = asyncio.Lock()
        self._stop_signal: asyncio.Event = asyncio.Event()
        # LLM usage totals are stored on Project.llm_usage and proxied via properties.
        # Start background tasks for command deltas and command invocations
        self._cmd_events_task = asyncio.create_task(self._forward_command_events())
        self._cmd_calls_task = asyncio.create_task(self._process_command_calls())

    # ------------------------
    # Public protocol endpoints
    # ------------------------

    async def recv(self) -> UIRequest:
        """
        Await next request from UIState (to be handled by UI client).
        """
        return await self._outgoing.get()

    async def send(self, resp: UIResponse) -> None:
        """
        Send a response from UI client back to UIState.
        """
        # Route run_command packets to a dedicated queue to avoid interfering with runner flow.
        if getattr(resp, "kind", None) == UI_PACKET_RUN_COMMAND:
            await self._incoming_cmds.put(resp)  # type: ignore[arg-type]
        else:
            await self._incoming.put(resp)

    # Convenience helpers to build and send responses
    async def respond_packet(self, req_id: int, packet: Optional[RespPacket]) -> None:
        inp = (
            RunInput(response=packet) if packet is not None else RunInput(response=None)
        )
        await self.send(UIRespRunInput(req_id=req_id, input=inp))

    async def respond_message(self, req_id: int, message: Message) -> None:
        await self.respond_packet(req_id, RespMessage(message=message))

    async def respond_approval(self, req_id: int, approved: bool) -> None:
        await self.respond_packet(req_id, RespApproval(approved=approved))

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
            self._selected_workflow_name = getattr(
                workflow, "name", self._selected_workflow_name
            )
            self.assignment = assignment or Assignment()
            self._initial_message = initial_message
            # New workflow: clear workflow-scoped custom commands
            with contextlib.suppress(Exception):
                self.project.commands.clear()
            self.runner = Runner(
                workflow=workflow, project=self.project, initial_message=initial_message
            )
            self._req_counter = 0
            self._last_status = None
            self._stop_signal.clear()
            self._drive_task = asyncio.create_task(self._drive_runner())

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
            if self.assignment is None:
                self.assignment = Assignment()

            # Prevent multiple drivers
            if self._drive_task and not self._drive_task.done():
                raise RuntimeError("Runner is already active")
            self._req_counter = 0
            self._last_status = None
            self._stop_signal.clear()
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
            with contextlib.suppress(asyncio.QueueEmpty):
                while True:
                    self._incoming_cmds.get_nowait()

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

    async def replace_last_user_input(
        self, resp: Union[RespMessage, RespApproval]
    ) -> None:
        """
        Replace the last user input (final prompt/confirm or prior message_request) and prepare resume.
        """
        async with self._lock:
            if self.runner is None or self.assignment is None:
                raise RuntimeError("No runner/assignment available")
            if self.runner.status in (RunnerStatus.running, RunnerStatus.waiting_input):
                raise RuntimeError(
                    "Cannot replace input while runner is running or waiting for input"
                )
            self.runner.replace_last_user_input(self.assignment, resp)

    # ------------------------
    # Internal driver
    # ------------------------

    async def _emit_status_if_changed(self) -> None:
        if self.runner is None:
            return
        curr = self.runner.status
        if curr != self._last_status:
            await self._outgoing.put(UIReqStatus(prev=self._last_status, curr=curr))
            self._last_status = curr

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
                except asyncio.CancelledError:
                    # Propagate cancellation state out; runner.cancel/stop should set status accordingly
                    await self._emit_status_if_changed()
                    break

                # Notify status transition, if any
                await self._emit_status_if_changed()

                # Token usage packets are emitted but Project totals are updated by LLMExecutor directly.

                # Decide if this event should be forwarded to the UI client.
                # Suppress node finals when hide_final_output is True and no input is requested.
                suppress_event = False
                if (
                    req.event.kind == PACKET_FINAL_MESSAGE
                    and not req.input_requested
                    and self.workflow is not None
                ):
                    try:
                        graph = getattr(self.workflow, "graph", None)
                        if graph is not None:
                            rn = graph.get_runtime_node_by_name(req.node)
                            if rn is not None and getattr(
                                rn.model, "hide_final_output", False
                            ):
                                suppress_event = True
                    except Exception:
                        # Be conservative: if we cannot resolve the node, do not suppress.
                        suppress_event = False

                if not suppress_event:
                    # Forward the run event to the UI client with a correlation id
                    self._req_counter += 1
                    req_id = self._req_counter
                    self._current_node_name = req.node
                    await self._outgoing.put(UIReqRunEvent(req_id=req_id, event=req))
                else:
                    # Do not forward or wait for input. Leave to_send as None.
                    req_id = None

                # Await UI response only if required
                if req.input_requested and not suppress_event:
                    # Wait for a matching response or a stop signal.
                    while True:
                        resp_task = asyncio.create_task(self._incoming.get())
                        stop_task = asyncio.create_task(self._stop_signal.wait())
                        done, pending = await asyncio.wait(
                            {resp_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
                        )
                        # If stop signal fired, close the runner generator and exit promptly.
                        if stop_task in done:
                            for t in pending:
                                t.cancel()
                            with contextlib.suppress(Exception):
                                await agen.aclose()
                            await self._emit_status_if_changed()
                            return
                        # Otherwise we have a response task completed.
                        ui_resp = resp_task.result()
                        for t in pending:
                            t.cancel()
                        if (
                            ui_resp.kind == UI_PACKET_RUN_INPUT
                            and ui_resp.req_id == req_id
                        ):
                            to_send = ui_resp.input
                            break
                        else:
                            logger.warning(
                                "UIState: Ignored mismatched response", expected=req_id
                            )

                else:
                    to_send = None

        except Exception as e:
            logger.exception("UIState: runner driver failed: %s", e)
        finally:
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
                    UICommand(name=c.name, help=c.help, usage=c.usage) for c in delta.added
                ]
                await self._outgoing.put(
                    UIReqCustomCommands(added=added, removed=delta.removed)
                )
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception("UIState: command events forwarder failed: %s", e)

    async def _process_command_calls(self) -> None:
        # Handle UIRespRunCommand packets arriving from the UI client.
        from vocode.commands import CommandContext as ExecCommandContext
        try:
            while True:
                rc = await self._incoming_cmds.get()
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
                    UIReqCommandResult(name=name, ok=ok, output=output, error=error)
                )
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception("UIState: command call processor failed: %s", e)
