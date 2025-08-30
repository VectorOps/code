
from typing import AsyncIterator, ClassVar, Dict, Optional, Type, List, TYPE_CHECKING
from enum import Enum

import asyncio
if TYPE_CHECKING:
    from vocode.project import Project
from vocode.graph import Workflow, Node
from vocode.state import Message, RunnerStatus, Assignment, ToolCallStatus, Step, Activity, StepStatus, ActivityType
from vocode.runner.models import (
    ReqPacket,
    RespPacket,
    RunEvent,
    RunInput,
    RespToolCall,
    ReqMessageRequest,
    ReqToolCall,
    ReqInterimMessage,
    ReqFinalMessage,
    RespMessage,
    RespApproval,
    PACKET_MESSAGE_REQUEST,
    PACKET_TOOL_CALL,
    PACKET_MESSAGE,
    PACKET_FINAL_MESSAGE,
    PACKET_APPROVAL,
)

class ResumeMode(str, Enum):
    PROMPT = "prompt"
    RERUN = "rerun"


class Executor:
    # Subclasses must set 'type' to the Node.type they handle
    type: ClassVar[Optional[str]] = None
    _registry: ClassVar[Dict[str, Type["Executor"]]] = {}

    def __init_subclass__(cls, **kwargs):
        """Register Executor subclasses by their 'type' into the registry."""
        super().__init_subclass__(**kwargs)
        t = getattr(cls, "type", None)
        if isinstance(t, str) and t:
            Executor._registry[t] = cls

    def __init__(self, config: Node, project: "Project"):
        """Initialize an executor instance with its corresponding Node config and Project."""
        # Configuration object of the corresponding Node (may be a Node subclass)
        self.config = config
        self.project = project

    @classmethod
    def register(cls, type_name: str, exec_cls: Type["Executor"]) -> None:
        """Manually register an Executor class under a node type name."""
        cls._registry[type_name] = exec_cls

    @classmethod
    def create_for_node(cls, node: Node, project: "Project") -> "Executor":
        """Create an Executor instance for the given Node using the registry."""
        sub = cls._registry.get(node.type)
        if sub is None:
            raise ValueError(f"No executor registered for node type '{node.type}'")
        return sub(config=node, project=project)

    async def run(self, messages: List[Message]) -> AsyncIterator[ReqPacket]:
        """
        Async generator from Executor to Runner using a yield-and-reply protocol:
        - yield ExecMessage(kind='message', message=...)
        - yield ExecToolCall(kind='tool_call', tool_calls=[ToolCall(...)]) and expect the runner to resume
          this generator with a reply via `agen.asend(ExecutorPacket(...))`. The yielded
          expression inside the generator will evaluate to that ExecutorPacket instance.
        - yield ExecFinalMessage(kind='final', message=..., outcome_name=...) once to finish.
        Runner drives the generator with `anext()` and `asend(...)`.
        """
        raise NotImplementedError("Executor subclasses must implement 'run' as an async generator")


class Runner:
    def __init__(self, workflow, project: "Project", initial_message: Optional[Message] = None):
        """Prepare the runner with a graph, initial message, status flags, and per-node executors."""
        self.workflow = workflow
        self.project = project
        self.initial_message: Optional[Message] = initial_message
        self.status: RunnerStatus = RunnerStatus.idle
        self._current_exec_task: Optional[asyncio.Task] = None
        self._executors: Dict[str, Executor] = {
            n.name: Executor.create_for_node(n, project=self.project) for n in self.workflow.graph.nodes
        }
        self._stop_requested: bool = False

    def cancel(self) -> None:
        """Cancel the currently running executor step, if any."""
        if self._current_exec_task and not self._current_exec_task.done():
            self._current_exec_task.cancel()

    def stop(self) -> None:
        """Stop the runner: set status to 'stopped' and cancel the current executor, if any."""
        self._stop_requested = True
        self.status = RunnerStatus.stopped
        if self._current_exec_task and not self._current_exec_task.done():
            self._current_exec_task.cancel()

    def _compute_node_transition(self, current_runtime_node, exec_activity: Activity, step: Step):
        """
        Decide the next runtime node and the input messages for it based on the executed node's result.
        Returns (next_runtime_node, next_input_messages) or (None, None) if the flow should finish.
        """
        node_model = current_runtime_node.model

        # If the node has no outcomes, we're done
        if not node_model.outcomes:
            return None, None

        # Choose next outcome
        outcome_name = exec_activity.outcome_name
        if outcome_name:
            next_runtime_node = current_runtime_node.get_child_by_outcome(outcome_name)
            if next_runtime_node is None:
                raise ValueError(f"No edge defined from node '{node_model.name}' via outcome '{outcome_name}'")
        else:
            # No outcome provided: if there is exactly one, follow it; otherwise error
            if len(node_model.outcomes) == 1:
                next_runtime_node = current_runtime_node.get_child_by_outcome(node_model.outcomes[0].name)
                if next_runtime_node is None:
                    raise ValueError(f"No edge defined from node '{node_model.name}' via its only outcome")
            else:
                raise ValueError(
                    f"Node '{node_model.name}' did not provide an outcome and has {len(node_model.outcomes)} outcomes"
                )

        # Determine input messages for the next node
        if node_model.pass_all_messages:
            msgs = [a.message for a in step.executions if a.message is not None]
        else:
            last_msg = next((a.message for a in reversed(step.executions) if a.message is not None), None)
            msgs = [last_msg] if last_msg is not None else []

        return next_runtime_node, msgs

    def _find_runtime_node_by_name(self, name: str):
        """Locate the RuntimeNode by name using Graph's runtime map."""
        return self.workflow.graph.get_runtime_node_by_name(name)

    def _prepare_resume(self, task: Assignment):
        """
        Returns (runtime_node, step, last_activity, mode) or None.
        mode ∈ {ResumeMode.PROMPT, ResumeMode.RERUN}:
          - ResumeMode.PROMPT: last activity was executor-provided final; synthesize a final_message event.
          - ResumeMode.RERUN: last activity was user-provided; immediately rerun the node.
        """
        if not task.steps:
            return None
        for step in reversed(task.steps):
            if step.status == StepStatus.finished and step.executions:
                last_act = step.executions[-1]
                if not last_act.is_complete and last_act.type == ActivityType.executor:
                    # Prefer resuming only from a complete executor final; otherwise skip
                    continue
                rn = self._find_runtime_node_by_name(step.node)
                if rn is None:
                    return None
                mode = ResumeMode.PROMPT if last_act.type == ActivityType.executor else ResumeMode.RERUN
                return rn, step, last_act, mode
        return None

    async def run(
        self,
        task: Assignment,
    ) -> AsyncIterator[RunEvent]:
        """
        Async generator that executes the graph from current task state; yields RunEvent and expects
        optional RunInput after each yield.
        """
        # Only allow starting when idle or previously stopped
        if self.status not in (RunnerStatus.idle, RunnerStatus.stopped):
            raise RuntimeError(
                f"run() not allowed when runner status is '{self.status}'. Allowed: 'idle', 'stopped'"
            )
        prev_status = self.status
        self.status = RunnerStatus.running
        graph = self.workflow.graph

        # Runner resuming
        resume_plan = None
        if prev_status == RunnerStatus.stopped:
            resume_plan = self._prepare_resume(task)

        reuse_step: Optional[Step] = None
        pending_input_messages: List[Message] = []

        if resume_plan is not None:
            rn, last_step, last_act, mode = resume_plan
            if mode is ResumeMode.PROMPT:
                req = ReqFinalMessage(message=last_act.message, outcome_name=last_act.outcome_name)
                self.status = RunnerStatus.waiting_input
                resume_event = RunEvent(
                    node=rn.name,
                    execution=last_act,
                    event=req,
                    input_requested=True,
                )
                run_input: Optional[RunInput] = (yield resume_event)
                resp = run_input.response if run_input is not None else None
                if resp is not None and resp.kind == PACKET_MESSAGE:
                    last_step.status = StepStatus.running
                    last_step.executions.append(Activity(type=ActivityType.user, message=resp.message))
                    current_runtime_node = rn
                    reuse_step = last_step
                    pending_input_messages = []
                else:
                    next_runtime_node, next_input_messages = self._compute_node_transition(rn, last_act, last_step)
                    if next_runtime_node is None:
                        self.status = RunnerStatus.finished
                        return
                    current_runtime_node = next_runtime_node
                    pending_input_messages = list(next_input_messages)
            else:  # ResumeMode.RERUN
                current_runtime_node = rn
                reuse_step = last_step
                pending_input_messages = []
        else:
            current_runtime_node = graph.root
            pending_input_messages = [self.initial_message] if self.initial_message is not None else []

        # Main loop
        while True:
            step: Optional[Step] = reuse_step
            reuse_step = None

            if step is None:
                step = Step(node=current_runtime_node.name, status=StepStatus.running)
                task.steps.append(step)
                # Add any pending input messages (including initial_message or carry-over messages)
                for m in pending_input_messages:
                    step.executions.append(Activity(type=ActivityType.user, message=m))
                pending_input_messages = []

            # Node loop
            rerun_same_node = True
            while rerun_same_node:
                rerun_same_node = False

                executor = self._executors[current_runtime_node.name]
                # Prepare messages for executor from step executions
                messages_for_exec = [a.message for a in step.executions if a.message is not None]
                agen = executor.run(messages_for_exec)
                current_activity: Optional[Activity] = None
                user_message_for_rerun: Optional[Message] = None
                to_send: Optional[RespPacket] = None

                while True:
                    try:
                        self._current_exec_task = asyncio.create_task(agen.asend(to_send))
                        req: ReqPacket = await self._current_exec_task
                        to_send = None
                    except StopAsyncIteration:
                        # Executor ended without explicit final; mark as completed
                        if current_activity is not None:
                            current_activity.is_complete = True
                        break
                    except asyncio.CancelledError:
                        # Runner.cancel() interrupted the in-flight executor
                        if current_activity is not None:
                            current_activity.is_canceled = True
                        self.status = RunnerStatus.stopped if self._stop_requested else RunnerStatus.canceled
                        try:
                            await agen.aclose()
                        except Exception:
                            pass
                        finally:
                            # Reset stop flag after handling cancellation
                            self._stop_requested = False
                        # NEW: mark current step if one exists
                        if step is not None:
                            step.status = StepStatus.stopped if self.status == RunnerStatus.stopped else StepStatus.canceled
                        return
                    finally:
                        self._current_exec_task = None

                    # Update execution with any interim output for visibility
                    if req.kind == PACKET_MESSAGE:
                        current_activity = Activity(type=ActivityType.executor, message=req.message)
                        step.executions.append(current_activity)

                    input_requested = req.kind in (
                        PACKET_MESSAGE_REQUEST,
                        PACKET_TOOL_CALL,
                        PACKET_FINAL_MESSAGE,
                    )
                    self.status = RunnerStatus.waiting_input if input_requested else RunnerStatus.running

                    # Emit event and await optional input
                    run_event = RunEvent(
                        node=current_runtime_node.name,
                        execution=current_activity or Activity(type=ActivityType.executor),
                        event=req,
                        input_requested=input_requested,
                    )
                    run_input: Optional[RunInput] = (yield run_event)
                    resp = run_input.response if run_input is not None else None

                    # Default: no response back to executor unless specified below
                    to_send = None

                    # Handle executor requests
                    if req.kind == PACKET_FINAL_MESSAGE:
                        current_activity = Activity(
                            type=ActivityType.executor,
                            message=req.message,
                            outcome_name=req.outcome_name,
                            is_complete=True,
                        )
                        step.executions.append(current_activity)

                        # Approval handling: None or approved=True => success => proceed to next node
                        if resp is None or (resp.kind == PACKET_APPROVAL and resp.approved):
                            # No response expected back to executor
                            break

                        # If a user message was provided instead, re-run same node with that message
                        if resp is not None and resp.kind == PACKET_MESSAGE:
                            step.executions.append(Activity(type=ActivityType.user, message=resp.message))
                            user_message_for_rerun = resp.message
                            break

                        # Otherwise, treat as success and proceed
                        break

                    elif req.kind == PACKET_TOOL_CALL:
                        # Determine approval; default to rejected unless explicitly approved
                        approved = (resp is not None and resp.kind == PACKET_APPROVAL and resp.approved)

                        # Update request objects for event visibility, and build a deep-copied response payload
                        updated_tool_calls = []
                        for tc in req.tool_calls:
                            tc.status = ToolCallStatus.completed if approved else ToolCallStatus.rejected
                            if approved:
                                if tc.result is None:
                                    tc.result = "{}"
                            else:
                                # On rejection ensure result is cleared
                                tc.result = None
                            updated_tool_calls.append(tc.model_copy(deep=True))

                        # Reply to executor with updated tool calls
                        to_send = RespToolCall(tool_calls=updated_tool_calls)
                        continue

                    elif req.kind == PACKET_MESSAGE_REQUEST:
                        # Require a message; if not provided, keep prompting until we get one
                        if resp is None or resp.kind != PACKET_MESSAGE:
                            while True:
                                self.status = RunnerStatus.waiting_input
                                run_input = (yield run_event)
                                resp = run_input.response if run_input is not None else None
                                if resp is not None and resp.kind == PACKET_MESSAGE:
                                    break
                        step.executions.append(Activity(type=ActivityType.user, message=resp.message))
                        to_send = resp  # guaranteed RespMessage here
                        continue

                    else:
                        # For other request kinds (e.g., interim messages), send nothing
                        to_send = None
                        continue

                # If we got a user-provided message to re-run the same node, do so
                if user_message_for_rerun is not None:
                    rerun_same_node = True
                    continue

                # Mark this step as finished (no rerun for this node)
                step.status = StepStatus.finished

                last_exec_activity = next((a for a in reversed(step.executions) if a.type == ActivityType.executor and a.is_complete), None)
                if last_exec_activity is None:
                    raise RuntimeError("No completed executor activity found to compute transition")
                next_runtime_node, next_input_messages = self._compute_node_transition(
                    current_runtime_node, last_exec_activity, step
                )
                if next_runtime_node is None:
                    self.status = RunnerStatus.finished
                    return

                current_runtime_node = next_runtime_node
                pending_input_messages = list(next_input_messages)

                break
