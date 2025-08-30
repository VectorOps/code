import asyncio
from typing import Optional, TYPE_CHECKING, List
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
)
from vocode.state import Assignment, Message, RunnerStatus
from vocode.graph import Graph, Workflow
from .proto import (
    UIRequest,
    UIReqRunEvent,
    UIReqStatus,
    UIResponse,
    UIRespRunInput,
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
        self._drive_task: Optional[asyncio.Task] = None
        self._req_counter: int = 0
        self._last_status: Optional[RunnerStatus] = None
        self._selected_workflow_name: Optional[str] = None
        self._current_node_name: Optional[str] = None
        self._lock = asyncio.Lock()

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
        await self._incoming.put(resp)

    # Convenience helpers to build and send responses

    async def respond_packet(self, req_id: int, packet: Optional[RespPacket]) -> None:
        inp = RunInput(response=packet) if packet is not None else RunInput(response=None)
        await self.send(UIRespRunInput(req_id=req_id, input=inp))

    async def respond_message(self, req_id: int, message: Message) -> None:
        await self.respond_packet(req_id, RespMessage(message=message))

    async def respond_approval(self, req_id: int, approved: bool) -> None:
        await self.respond_packet(req_id, RespApproval(approved=approved))

    # ------------------------
    # Runner lifecycle control
    # ------------------------

    async def start(self, workflow: Workflow, *, initial_message: Optional[Message] = None, assignment: Optional[Assignment] = None) -> None:
        """
        Start a new runner for the given workflow. If a runner is already active, raises RuntimeError.
        """
        async with self._lock:
            if self._drive_task and not self._drive_task.done():
                raise RuntimeError("Runner is already active")

            self.workflow = workflow
            self._selected_workflow_name = getattr(workflow, "name", self._selected_workflow_name)
            self.assignment = assignment or Assignment()
            self._initial_message = initial_message
            self.runner = Runner(workflow=workflow, project=self.project, initial_message=initial_message)
            self._req_counter = 0
            self._last_status = None
            self._drive_task = asyncio.create_task(self._drive_runner())

    async def stop(self) -> None:
        """
        Politely stop the runner (allowing it to be resumed later).
        """
        async with self._lock:
            if self.runner is None:
                return
            self.runner.stop()

    async def cancel(self) -> None:
        """
        Cancel the current in-flight executor work (non-resumable).
        """
        async with self._lock:
            if self.runner is None:
                return
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
            if self.runner is None or self.runner.status in (RunnerStatus.finished, RunnerStatus.canceled):
                self.runner = Runner(workflow=self.workflow, project=self.project, initial_message=self._initial_message)

            # Re-drive the same assignment (resume if stopped)
            if self.assignment is None:
                self.assignment = Assignment()

            # Prevent multiple drivers
            if self._drive_task and not self._drive_task.done():
                raise RuntimeError("Runner is already active")
            self._req_counter = 0
            self._last_status = None
            self._drive_task = asyncio.create_task(self._drive_runner())

    # ------------------------
    # Workflow helpers and accessors
    # ------------------------

    def list_workflows(self) -> List[str]:
        if not self.project.settings or not self.project.settings.workflows:
            return []
        return list(self.project.settings.workflows.keys())

    async def start_by_name(self, name: str, *, initial_message: Optional[Message] = None) -> None:
        if not self.project.settings or name not in (self.project.settings.workflows or {}):
            raise KeyError(f"Unknown workflow: {name}")
        wf_cfg = self.project.settings.workflows[name]

        graph = Graph.build(nodes=wf_cfg.nodes, edges=wf_cfg.edges)
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

    async def use(self, name: str, *, initial_message: Optional[Message] = None) -> None:
        await self._shutdown_current_runner(cancel=False)
        await self.start_by_name(name, initial_message=initial_message)

    async def reset(self) -> None:
        if not self._selected_workflow_name:
            raise RuntimeError("No workflow selected to reset")
        await self._shutdown_current_runner(cancel=False)
        await self.start_by_name(self._selected_workflow_name, initial_message=self._initial_message)

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

                # Forward the run event to the UI client with a correlation id
                self._req_counter += 1
                req_id = self._req_counter
                self._current_node_name = req.node
                await self._outgoing.put(UIReqRunEvent(req_id=req_id, event=req))

                # Await UI response only if required
                if req.input_requested:
                    # Wait for a matching response
                    while True:
                        ui_resp = await self._incoming.get()
                        if isinstance(ui_resp, UIRespRunInput) and ui_resp.req_id == req_id:
                            to_send = ui_resp.input
                            break
                        else:
                            logger.warning("UIState: Ignored mismatched response", expected=req_id)

                else:
                    to_send = None

        except Exception as e:
            logger.exception("UIState: runner driver failed: %s", e)
        finally:
            # Final status emission (in case it changed just before exit)
            await self._emit_status_if_changed()
            self._current_node_name = None
