import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, List, Union, Any, Dict
import contextlib
from vocode.settings_loader import build_model_from_settings

from vocode.logger import logger
from vocode.runner.runner import Runner
from vocode.runner.models import (
    RunEvent,
    RunInput,
    ReqLogMessage,
    RespPacket,
    RespMessage,
    RespApproval,
    PACKET_TOKEN_USAGE,
    PACKET_FINAL_MESSAGE,
    PACKET_STATUS_CHANGE,
    PACKET_START_WORKFLOW,
)
from vocode.state import (
    Assignment,
    Message,
    RunnerStatus,
    RunStatus,
    Activity,
    ActivityType,
    LogLevel,
)
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
    UIPacketAck,
    UICommand,
    PACKET_RUN_INPUT,
    PACKET_RUN_COMMAND,
    PACKET_COMPLETION_REQUEST,
    PACKET_COMPLETION_RESULT,
    PACKET_UI_RELOAD,
    PACKET_PROJECT_OP_START,
    PACKET_PROJECT_OP_PROGRESS,
    PACKET_PROJECT_OP_FINISH,
    UIPacketProjectOpStart,
    UIPacketProjectOpProgress,
    UIPacketProjectOpFinish,
    UIPacketLog,
)
from .rpc import RpcHelper
from .logging_interceptor import UILoggingHandler

if TYPE_CHECKING:
    from vocode.project import Project


from .autocomplete import AutoCompletionManager
from . import autocomplete_providers as acp


@dataclass
class RunnerFrame:
    runner: Runner
    workflow: Workflow
    assignment: Assignment
    agen: Any  # Async generator: AsyncIterator[RunEvent]
    to_send: Optional[RunInput] = None
    last_final: Optional[Message] = None


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
        self.runner_stack: List[RunnerFrame] = []
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
        # UI logging forwarder state (installed when a run starts)
        self._log_queue: Optional["asyncio.Queue[dict]"] = None
        self._log_forwarder_task: Optional[asyncio.Task] = None
        self._log_handler: Optional[logging.Handler] = None
        # LLM usage totals are stored on Project.llm_usage and proxied via properties.
        # Start background tasks for command deltas and incoming packet routing
        # LLM usage totals are stored on Project.llm_usage and proxied via properties.
        # Start background tasks for command deltas and incoming packet routing
        self._cmd_events_task = asyncio.create_task(self._forward_command_events())
        self._incoming_router_task = asyncio.create_task(self._route_incoming_packets())
        # Subscribe to project messages and forward to UI
        self._project_messages_task = asyncio.create_task(
            self._forward_project_messages()
        )

    def _top_frame(self) -> Optional[RunnerFrame]:
        return self.runner_stack[-1] if self.runner_stack else None

    def _create_runner_frame(
        self,
        workflow: Workflow,
        *,
        initial_message: Optional[Message] = None,
        assignment: Optional[Assignment] = None,
    ) -> RunnerFrame:
        assign = assignment or Assignment()
        runner = Runner(
            workflow=workflow, project=self.project, initial_message=initial_message
        )
        agen = runner.run(assign)
        return RunnerFrame(
            runner=runner,
            workflow=workflow,
            assignment=assign,
            agen=agen,
            to_send=None,
            last_final=None,
        )

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
                assignment = Assignment()

            self._initial_message = initial_message
            top_frame = self._create_runner_frame(
                workflow, initial_message=initial_message, assignment=assignment
            )
            self.runner_stack = [top_frame]
            self.runner = top_frame.runner
            self.workflow = top_frame.workflow
            self.assignment = top_frame.assignment
            self._msg_counter = 0
            self._last_status = None
            self._stop_signal.clear()
            # Install stdlib logging interceptor for the duration of the run
            self._install_logging_interceptor()
            # Always instruct UI to reset when a runner starts.
            await self._send_ui_reset()
            self._drive_task = asyncio.create_task(self._drive_runner())
            self._stop_watcher_task = asyncio.create_task(self._watch_stop_signal())

    async def stop(self, wait: bool = True) -> None:
        """
        Politely stop the top-level runner (allowing it to be resumed later).
        """
        async with self._lock:
            top = self._top_frame()
            if top is None:
                return
            # Request runner to stop (sets status and cancels any in-flight executor work)
            top.runner.stop()
            # Proactively break the driver out of any waits (e.g., pending UI RPC)
            if self._drive_task and not self._drive_task.done():
                # Inform watcher (if any) and directly cancel to ensure prompt exit
                self._stop_signal.set()
                self._drive_task.cancel()
            if wait and self._drive_task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await self._drive_task
            # Ensure the async generator is not reused after being closed by the driver
            # so a later restart() will create a fresh generator tied to the current runner/assignment.
            if top.agen is not None:
                top.agen = None

    async def cancel(self) -> None:
        """
        Cancel all in-flight executor work (non-resumable).
        """
        async with self._lock:
            if not self.runner_stack:
                return
            for f in self.runner_stack:
                with contextlib.suppress(Exception):
                    f.runner.cancel()
            if self._drive_task and not self._drive_task.done():
                self._drive_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._drive_task
            self.runner_stack.clear()
            self.runner = None
            self.workflow = None
            self.assignment = None
            self._current_node_name = None
            self._last_status = None

    async def restart(self) -> None:
        """
        Restart execution. If current runner is stopped, it will resume with existing Assignment.
        If it has finished or been canceled, a fresh Runner is created with the last workflow.
        """
        async with self._lock:
            # Restart only the top-of-stack runner
            top = self._top_frame()
            if top is None:
                raise RuntimeError("No workflow available to restart")

            # If there's no runner, or it finished/canceled, create a new one
            if top.runner.status in (RunnerStatus.finished, RunnerStatus.canceled):
                replacement = self._create_runner_frame(
                    top.workflow, initial_message=self._initial_message
                )
                # Preserve/refresh assignment if needed
                if (
                    top.assignment is None
                    or top.assignment.status == RunStatus.finished
                ):
                    top.assignment = replacement.assignment
                top.runner = replacement.runner
                top.agen = replacement.agen

            if top.assignment is None or top.assignment.status == RunStatus.finished:
                top.assignment = Assignment()
            self.assignment = top.assignment
            self.runner = top.runner
            self.workflow = top.workflow

            # Prevent multiple drivers
            if self._drive_task and not self._drive_task.done():
                raise RuntimeError("Runner is already active")

            self._msg_counter = 0
            self._last_status = None
            self._stop_signal.clear()
            # Reinstall logging interceptor on restart
            self._install_logging_interceptor()
            # Always instruct UI to reset when a runner restarts.
            await self._send_ui_reset()
            self._drive_task = asyncio.create_task(self._drive_runner())
            self._stop_watcher_task = asyncio.create_task(self._watch_stop_signal())

    # ------------------------
    # Workflow helpers and accessors
    # ------------------------
    def list_workflows(self) -> List[str]:
        if not self.project.settings or not self.project.settings.workflows:
            return []
        return list(self.project.settings.workflows.keys())

    async def start_by_name(
        self, name: str, initial_message: Optional[Message] = None
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
            RunnerStatus.idle,
            RunnerStatus.finished,
            RunnerStatus.stopped,
            RunnerStatus.canceled,
        ):
            return False
        return self._drive_task is not None and not self._drive_task.done()

    @property
    def status(self) -> RunnerStatus:
        top = self._top_frame()
        return top.runner.status if top else RunnerStatus.idle

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
            top = self._top_frame()
            if top is None:
                raise RuntimeError("No runner/assignment to rewind")
            if top.runner.status in (RunnerStatus.running, RunnerStatus.waiting_input):
                raise RuntimeError(
                    "Cannot rewind while runner is running or waiting for input"
                )
            await top.runner.rewind(top.assignment, n)

    async def replace_user_input(
        self, resp: Union[RespMessage, RespApproval], n: Optional[int] = 1
    ) -> None:
        """
        Replace a user input (final prompt/confirm or prior message_request) and prepare resume.
        By default, it targets the last user input. A specific step can be targeted with step_index.
        """
        async with self._lock:
            top = self._top_frame()
            if top is None:
                raise RuntimeError("No runner/assignment available")
            if top.runner.status in (RunnerStatus.running, RunnerStatus.waiting_input):
                raise RuntimeError(
                    "Cannot replace input while runner is running or waiting for input"
                )
            top.runner.replace_user_input(top.assignment, resp, n=n)

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

    async def _emit_status(self, runner):
        curr = runner.status
        if curr != self._last_status:
            await self._outgoing.put(
                UIPacketEnvelope(
                    msg_id=self._next_msg_id(),
                    payload=UIPacketStatus(prev=self._last_status, curr=curr),
                )
            )
            self._last_status = curr

    async def _emit_status_if_changed(self) -> None:
        top = self._top_frame()
        if top is None:
            return

        await self._emit_status(top.runner)

    async def _drive_runner(self) -> None:
        """
        Drive the Runner async generator, forwarding events to the UI and
        returning responses back to the Runner.
        """
        if not self.runner_stack:
            return

        # Ensure top frame agen is initialized
        top = self._top_frame()
        if top and top.agen is None:
            top.agen = top.runner.run(top.assignment)

        try:
            while self.runner_stack:
                frame = self._top_frame()
                if frame is None:
                    break
                runner = frame.runner
                workflow = frame.workflow
                agen = frame.agen
                to_send = frame.to_send
                frame.to_send = None
                try:
                    req: RunEvent = await agen.asend(to_send)
                except StopAsyncIteration:
                    # Done: bubble final message to previous frame (if any)
                    finished = self.runner_stack.pop()
                    parent = self._top_frame()
                    if parent is not None and finished.last_final is not None:
                        parent.to_send = RunInput(
                            response=RespMessage(message=finished.last_final)
                        )
                        # Continue loop to drive parent
                        continue
                    else:
                        # No parent or no final to bubble; emit final status and stop
                        await self._emit_status(finished.runner)
                        break

                # Notify status transition, if any
                await self._emit_status_if_changed()

                # Convert runner status-change packets to UIPacketStatus
                if req.event.kind == PACKET_STATUS_CHANGE:
                    sc = req.event
                    node_description = None
                    if sc.new_node and workflow:
                        node = workflow.graph.node_by_name.get(sc.new_node)
                        if node:
                            node_description = node.description

                    await self._outgoing.put(
                        UIPacketEnvelope(
                            msg_id=self._next_msg_id(),
                            payload=UIPacketStatus(
                                prev=sc.old_status,
                                curr=sc.new_status,
                                prev_node=sc.old_node,
                                curr_node=sc.new_node,
                                curr_node_description=node_description,
                            ),
                        )
                    )
                    # Track current node for UI convenience
                    self._current_node_name = sc.new_node
                    to_send = None
                    continue

                # Decide if this event should be forwarded to the UI client.
                # Suppress node finals when hide_final_output is True and no input is requested.
                suppress_event = False
                if req.event.kind == PACKET_FINAL_MESSAGE and not req.input_requested:
                    rn = runner.runtime_graph.get_runtime_node_by_name(req.node)
                    if rn is not None and rn.model.hide_final_output:
                        suppress_event = True

                # Intercept start_workflow: push a child frame and continue
                if req.event.kind == PACKET_START_WORKFLOW:
                    sw = req.event
                    try:
                        if not self.project.settings or sw.workflow not in (
                            self.project.settings.workflows or {}
                        ):
                            raise KeyError(f"Unknown workflow: {sw.workflow}")
                        wf_cfg = self.project.settings.workflows[sw.workflow]
                        child_graph = Graph(nodes=wf_cfg.nodes, edges=wf_cfg.edges)
                        child_wf = Workflow(name=sw.workflow, graph=child_graph)
                        child_frame = self._create_runner_frame(
                            child_wf, initial_message=sw.initial_message
                        )
                        self.runner_stack.append(child_frame)
                        # Update convenience refs to child
                        self.runner = child_frame.runner
                        self.workflow = child_frame.workflow
                        self.assignment = child_frame.assignment
                        # Continue to drive the child
                        continue
                    except Exception as e:
                        # Child could not be started (e.g., unknown workflow or executor).
                        # Emit a log event to the UI with the reason so the user sees details.
                        error_text = (
                            f"Failed to start workflow '{sw.workflow}': "
                            f"{e.__class__.__name__}: {e}"
                        )
                        log_event = RunEvent(
                            node=req.node,
                            execution=(
                                req.execution
                                if req.execution is not None
                                else Activity(type=ActivityType.executor)
                            ),
                            event=ReqLogMessage(text=error_text, level=LogLevel.error),
                            input_requested=False,
                        )
                        await self._outgoing.put(
                            UIPacketEnvelope(
                                msg_id=self._next_msg_id(),
                                payload=UIPacketRunEvent(event=log_event),
                            )
                        )

                        # Resume parent with a generic error response so the workflow can continue/finalize.
                        if frame is not None:
                            frame.to_send = RunInput(
                                response=RespMessage(
                                    message=Message(
                                        role="agent",
                                        text="[error: child workflow failed to start]",
                                    )
                                )
                            )
                        continue

                # Await UI response only if required
                if req.input_requested and not suppress_event:
                    self._current_node_name = req.node
                    # Disable timeout while waiting for user input: block indefinitely
                    response_payload = await self._rpc.call(
                        UIPacketRunEvent(event=req), timeout=None
                    )

                    if response_payload and response_payload.kind == PACKET_RUN_INPUT:
                        frame.to_send = response_payload.input
                    else:
                        if response_payload:
                            logger.warning(
                                "UIState: Ignored mismatched RPC response",
                                response=response_payload,
                            )
                        frame.to_send = None
                else:
                    if not suppress_event:
                        # Forward the run event to the UI client with a correlation id
                        self._current_node_name = req.node
                        await self._outgoing.put(
                            UIPacketEnvelope(
                                msg_id=self._next_msg_id(),
                                payload=UIPacketRunEvent(event=req),
                            )
                        )

                # Record frame final messages for bubbling when generator exits
                if req.event.kind == PACKET_FINAL_MESSAGE:
                    frame.last_final = req.event.message
                    if runner is not None:
                        rn = runner.runtime_graph.get_runtime_node_by_name(req.node)
                        if rn is not None and rn.model.hide_final_output:
                            suppress_event = True

        except asyncio.CancelledError:
            # Runner was cancelled/stopped while waiting (e.g., for UI input).
            # Close the top generator if available, then emit final status.
            top = self._top_frame()
            if top and top.agen:
                with contextlib.suppress(Exception):
                    await top.agen.aclose()
                # Mark agen as cleared so a future restart can recreate it
                top.agen = None
            await self._emit_status_if_changed()
        except Exception as e:
            logger.exception("UIState: runner driver failed: %s", e)
        finally:
            # Remove logging interceptor when the driver exits
            with contextlib.suppress(Exception):
                self._uninstall_logging_interceptor()
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
                elif kind == PACKET_UI_RELOAD:
                    await self._handle_ui_reload(envelope)
                else:
                    # Non-command/completion requests should be handled elsewhere (e.g., RPC responses).
                    # Ignore unknown requests.
                    pass
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception("UIState: incoming packet router failed: %s", e)

    async def _forward_project_messages(self) -> None:
        try:
            async for p in self.project.message_generator():
                try:
                    if p.kind == PACKET_PROJECT_OP_START:
                        await self._outgoing.put(
                            UIPacketEnvelope(
                                msg_id=self._next_msg_id(),
                                payload=UIPacketProjectOpStart(message=p.message),
                            )
                        )
                    elif p.kind == PACKET_PROJECT_OP_PROGRESS:
                        await self._outgoing.put(
                            UIPacketEnvelope(
                                msg_id=self._next_msg_id(),
                                payload=UIPacketProjectOpProgress(
                                    progress=p.progress,
                                    total=p.total,
                                ),
                            )
                        )
                    elif p.kind == PACKET_PROJECT_OP_FINISH:
                        await self._outgoing.put(
                            UIPacketEnvelope(
                                msg_id=self._next_msg_id(),
                                payload=UIPacketProjectOpFinish(),
                            )
                        )
                except AttributeError:
                    logger.warning("UIState: ignoring malformed project message: %r", p)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception("UIState: project message forwarder failed: %s", e)

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

    async def _handle_ui_reload(self, envelope: UIPacketEnvelope) -> None:
        """
        Force reload:
        - Cancel any current runner
        - Stop command forwarding
        - Shutdown current Project
        - Create and start a new Project from the same base_path
        - Restart command forwarding and send UI reset
        - ACK the request
        """
        try:
            # Force cancel any active runner
            await self._shutdown_current_runner(cancel=True)

            # Stop forwarding command events
            if self._cmd_events_task and not self._cmd_events_task.done():
                self._cmd_events_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._cmd_events_task
            self._cmd_events_task = None

            # Shutdown current project
            with contextlib.suppress(Exception):
                await self.project.shutdown()

            # Recreate a fresh project and start it
            from vocode.project import Project as VocodeProject

            base = self.project.base_path
            new_project = VocodeProject.from_base_path(base)
            await new_project.start()

            # Swap the project
            self.project = new_project

            # Reset UIState high-level selections
            self.workflow = None
            self.assignment = None
            self._initial_message = None
            self._selected_workflow_name = None
            self._current_node_name = None
            self._last_status = None

            # Restart command forwarder with the new project
            self._cmd_events_task = asyncio.create_task(self._forward_command_events())

            # Ask UI to clear any state
            await self._send_ui_reset()
        except Exception as e:
            logger.exception("UIState: project reload failed: %s", e)
        finally:
            # Always ACK so the caller can proceed
            await self._outgoing.put(
                UIPacketEnvelope(
                    msg_id=self._next_msg_id(),
                    source_msg_id=envelope.msg_id,
                    payload=UIPacketAck(),
                )
            )

    # ------------------------
    # Stdlib logging integration
    # ------------------------
    def _install_logging_interceptor(self) -> None:
        if self._log_handler is not None:
            return
        loop = asyncio.get_running_loop()
        self._log_queue = asyncio.Queue()
        self._log_forwarder_task = asyncio.create_task(self._forward_logs())
        self._log_handler = UILoggingHandler(loop, self._log_queue)  # type: ignore[arg-type]
        logging.getLogger().addHandler(self._log_handler)

    def _uninstall_logging_interceptor(self) -> None:
        # Remove handler from root logger
        if self._log_handler is not None:
            try:
                logging.getLogger().removeHandler(self._log_handler)
            finally:
                self._log_handler = None
        # Stop forwarder task
        if self._log_forwarder_task is not None:
            self._log_forwarder_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                asyncio.get_running_loop().run_until_complete(self._log_forwarder_task)  # type: ignore[misc]
            self._log_forwarder_task = None
        # Drop queue reference
        self._log_queue = None

    async def _forward_logs(self) -> None:
        try:
            while True:
                assert self._log_queue is not None
                item = await self._log_queue.get()
                try:
                    payload = UIPacketLog(
                        level=item.get("level"),
                        message=item.get("message", ""),
                        logger=item.get("logger"),
                        pathname=item.get("pathname"),
                        lineno=item.get("lineno"),
                        exc_text=item.get("exc_text"),
                    )
                    await self._outgoing.put(
                        UIPacketEnvelope(msg_id=self._next_msg_id(), payload=payload)
                    )
                except Exception as e:
                    logger.warning("UIState: failed to forward log item: %s", e)
        except asyncio.CancelledError:
            return
