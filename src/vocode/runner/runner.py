
from typing import AsyncIterator, ClassVar, Dict, Optional, Type, List, TYPE_CHECKING
from enum import Enum

import asyncio
import contextlib
from pydantic import BaseModel
if TYPE_CHECKING:
    from vocode.project import Project
from vocode.graph import Workflow, Node
from vocode.graph.models import Confirmation, ResetPolicy, MessageMode
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
        # Map of node_name -> async generator (executor.run(...)) to support long-lived executors
        self._agen_map: Dict[str, AsyncIterator[ReqPacket]] = {}
        self._stop_requested: bool = False
        self._internal_cancel_requested: bool = False

    def cancel(self) -> None:
        """Cancel the currently running executor step, if any."""
        if self._current_exec_task and not self._current_exec_task.done():
            self._internal_cancel_requested = True
            self._current_exec_task.cancel()

    def stop(self) -> None:
        """Stop the runner: set status to 'stopped' and cancel the current executor, if any."""
        self._stop_requested = True
        self.status = RunnerStatus.stopped
        if self._current_exec_task and not self._current_exec_task.done():
            self._internal_cancel_requested = True
            self._current_exec_task.cancel()

    def _compute_node_transition(self, current_runtime_node, exec_activity: Activity, step: Step):
        """
        Decide the next runtime node and the input messages for it based on the executed node's result.
        Returns (next_runtime_node, next_input_messages, edge_reset_policy) or (None, None, None) if the flow should finish.
        """
        node_model = current_runtime_node.model

        if not node_model.outcomes:
            return None, None, None

        # Choose outcome and next node
        outcome_name = exec_activity.outcome_name
        if outcome_name:
            next_runtime_node = current_runtime_node.get_child_by_outcome(outcome_name)
            if next_runtime_node is None:
                raise ValueError(
                    f"No edge defined from node '{node_model.name}' via outcome '{outcome_name}'"
                )
            selected_outcome = outcome_name
        else:
            if len(node_model.outcomes) == 1:
                selected_outcome = node_model.outcomes[0].name
                next_runtime_node = current_runtime_node.get_child_by_outcome(selected_outcome)
                if next_runtime_node is None:
                    raise ValueError(
                        f"No edge defined from node '{node_model.name}' via its only outcome"
                    )
            else:
                raise ValueError(
                    f"Node '{node_model.name}' did not provide an outcome and has {len(node_model.outcomes)} outcomes"
                )

        # Determine input messages for the next node based on message_mode
        mode = node_model.message_mode
        if mode == MessageMode.all_messages:
            msgs = [a.message for a in step.executions if a.message is not None]
        elif mode == MessageMode.final_response:
            final_msg = exec_activity.message or next(
                (a.message for a in reversed(step.executions) if a.message is not None),
                None,
            )
            msgs = [final_msg] if final_msg is not None else []
        elif mode == MessageMode.concatenate_final:
            input_msgs = [
                a.message
                for a in step.executions
                if a.message is not None and a.type == ActivityType.user
            ]
            final_msg = exec_activity.message
            parts = [m.text for m in input_msgs if m.text is not None]
            if final_msg is not None and final_msg.text is not None:
                parts.append(final_msg.text)
            combined_text = "\n".join(parts)
            if combined_text:
                role = (final_msg.role if final_msg is not None else (input_msgs[-1].role if input_msgs else "agent"))
                msgs = [Message(role=role, text=combined_text)]
            else:
                msgs = []
        else:
            # Fallback to final_response semantics
            final_msg = exec_activity.message
            msgs = [final_msg] if final_msg is not None else []

        # Look up the edge to obtain the reset_policy override (if any)
        edge_reset_policy = None
        try:
            graph = self.workflow.graph
            edge_reset_policy = next(
                (
                    e.reset_policy
                    for e in graph.edges
                    if e.source_node == node_model.name
                    and e.source_outcome == selected_outcome
                    and e.target_node == next_runtime_node.name
                ),
                None,
            )
        except Exception:
            edge_reset_policy = None

        return next_runtime_node, msgs, edge_reset_policy

    def _find_runtime_node_by_name(self, name: str):
        """Locate the RuntimeNode by name using Graph's runtime map."""
        return self.workflow.graph.get_runtime_node_by_name(name)

    def _get_previous_node_messages(
        self,
        task: Assignment,
        node_name: str,
        exclude_step: Optional[Step] = None,
    ) -> List[Message]:
        """
        Collect all prior messages for the given node across earlier steps, excluding interim executor chunks.
        Includes:
          - user messages
          - executor final messages (is_complete=True)
        Excludes:
          - interim executor messages (is_complete=False)
          - messages from 'exclude_step' (the current in-flight step)
        """
        msgs: List[Message] = []
        for s in task.steps:
            if s is exclude_step:
                continue
            if s.node != node_name:
                continue
            for a in s.executions:
                if a.message is None:
                    continue
                if a.type == ActivityType.executor and not a.is_complete:
                    continue
                msgs.append(a.message)
        return msgs

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

    async def _close_all_generators(self) -> None:
        gens = list(self._agen_map.values())
        self._agen_map.clear()
        for g in gens:
            try:
                await g.aclose()
            except Exception:
                pass

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
                    incoming_policy_override: Optional[ResetPolicy] = None
                else:
                    next_runtime_node, next_input_messages, next_edge_policy = self._compute_node_transition(
                        rn, last_act, last_step
                    )
                    if next_runtime_node is None:
                        self.status = RunnerStatus.finished
                        await self._close_all_generators()
                        return
                    current_runtime_node = next_runtime_node
                    pending_input_messages = list(next_input_messages)
                    incoming_policy_override = next_edge_policy
            else:  # ResumeMode.RERUN
                current_runtime_node = rn
                reuse_step = last_step
                pending_input_messages = []
                incoming_policy_override: Optional[ResetPolicy] = None
        else:
            current_runtime_node = graph.root
            pending_input_messages = [self.initial_message] if self.initial_message is not None else []
            incoming_policy_override: Optional[ResetPolicy] = None

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

            # Node selection
            node_model = current_runtime_node.model
            node_name = current_runtime_node.name
            # Determine reset policy for this run (edge override wins once)
            policy = incoming_policy_override or node_model.reset_policy
            incoming_policy_override = None


            # Collect initial inputs (added when the step was created)
            initial_user_inputs: List[Message] = [
                a.message for a in step.executions
                if a.message is not None and a.type == ActivityType.user
            ]

            # Policy: executor instance and generator lifecycle
            # Close any existing generator if policy requires fresh run
            if policy in (ResetPolicy.always_reset, ResetPolicy.keep_results):
                old_gen = self._agen_map.pop(node_name, None)
                if old_gen is not None:
                    try:
                        await old_gen.aclose()
                    except Exception:
                        pass

            # For always_reset, also replace the executor instance to reset internal state
            if policy == ResetPolicy.always_reset:
                # Replace executor instance
                self._executors[node_name] = Executor.create_for_node(node_model, project=self.project)

            executor = self._executors[node_name]

            # Obtain or create the async generator
            agen = self._agen_map.get(node_name)
            to_send: Optional[RespPacket] = None

            if agen is None:
                # Build initial messages depending on policy
                start_messages: List[Message] = []
                if policy == ResetPolicy.keep_results:
                    prev_msgs = self._get_previous_node_messages(task, node_name, exclude_step=step)
                    if prev_msgs:
                        start_messages.extend(prev_msgs)
                # For always_reset and keep_state (first run), add current initial inputs
                start_messages.extend(initial_user_inputs)
                agen = executor.run(start_messages)
                self._agen_map[node_name] = agen
            else:
                # keep_state: resume existing generator with a new input message (if any)
                if initial_user_inputs:
                    last_input = initial_user_inputs[-1]
                    to_send = RespMessage(message=last_input)

            current_activity: Optional[Activity] = None

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
                    if not self._internal_cancel_requested:
                        raise

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
                        self._internal_cancel_requested = False
                    # NEW: mark current step if one exists
                    if step is not None:
                        step.status = StepStatus.stopped if self.status == RunnerStatus.stopped else StepStatus.canceled
                    # Ensure current node generator is removed; avoid double-closing if it's the same object
                    gen_in_map = self._agen_map.pop(current_runtime_node.name, None)
                    if gen_in_map is not None and gen_in_map is not agen:
                        try:
                            await gen_in_map.aclose()
                        except Exception:
                            pass
                    return
                finally:
                    self._current_exec_task = None

                # Update execution with any interim output for visibility
                if req.kind == PACKET_MESSAGE:
                    current_activity = Activity(type=ActivityType.executor, message=req.message)
                    step.executions.append(current_activity)

                if req.kind in (PACKET_MESSAGE_REQUEST, PACKET_TOOL_CALL):
                    input_requested = True
                elif req.kind == PACKET_FINAL_MESSAGE:
                    # Request input for 'prompt' and 'confirm' modes
                    node_conf = current_runtime_node.model.confirmation
                    input_requested = node_conf in (Confirmation.prompt, Confirmation.confirm)
                else:
                    input_requested = False
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

                    node_conf = current_runtime_node.model.confirmation

                    if node_conf == Confirmation.confirm:
                        # Force explicit Y/N approval; re-prompt until we get an approval packet.
                        while True:
                            if resp is not None and resp.kind == PACKET_APPROVAL:
                                if resp.approved:
                                    # Proceed to next node
                                    break
                                # Rejected: stop executor and runner politely.
                                self.status = RunnerStatus.stopped
                                step.status = StepStatus.stopped
                                with contextlib.suppress(Exception):
                                    await agen.aclose()
                                gen_in_map = self._agen_map.pop(current_runtime_node.name, None)
                                if gen_in_map is not None and gen_in_map is not agen:
                                    with contextlib.suppress(Exception):
                                        await gen_in_map.aclose()
                                return
                            # Not an approval response: ask again.
                            self.status = RunnerStatus.waiting_input
                            run_event = RunEvent(
                                node=current_runtime_node.name,
                                execution=current_activity,
                                event=req,
                                input_requested=True,
                            )
                            run_input = (yield run_event)
                            resp = run_input.response if run_input is not None else None
                        # Reaching here means approved => proceed to next node.
                        break

                    # prompt mode: allow additional user message sent back to executor
                    if resp is not None and resp.kind == PACKET_MESSAGE:
                        to_send = resp  # Send directly back into the executor
                        continue

                    # Approval handling: None or approved=True => success => proceed to next node
                    if resp is None or (resp.kind == PACKET_APPROVAL and resp.approved):
                        break

                    # Any other response: treat as success and proceed
                    break

                elif req.kind == PACKET_TOOL_CALL:
                    # Default to approved unless explicitly rejected by a response
                    approved = True
                    if resp is not None and resp.kind == PACKET_APPROVAL:
                        approved = resp.approved

                    updated_tool_calls = []
                    for tc in req.tool_calls:
                        # Respect explicit rejection
                        if not approved:
                            tc.status = ToolCallStatus.rejected
                            tc.result = {"error": "Tool call rejected by user"}
                            updated_tool_calls.append(tc.model_copy(deep=True))
                            continue

                        # Resolve the tool from the project registry
                        tool = self.project.tools.get(tc.name)
                        if tool is None:
                            tc.status = ToolCallStatus.rejected
                            tc.result = {"error": f"Unknown tool '{tc.name}'"}
                            updated_tool_calls.append(tc.model_copy(deep=True))
                            continue

                        # Validate/construct the tool's input model
                        try:
                            args_model = tool.input_model.model_validate(tc.arguments)
                        except Exception as e:
                            tc.status = ToolCallStatus.rejected
                            tc.result = {"error": f"Invalid arguments for tool '{tc.name}': {str(e)}"}
                            updated_tool_calls.append(tc.model_copy(deep=True))
                            continue

                        # Execute the tool
                        try:
                            result_obj = await tool.run(self.project, args_model)
                            if isinstance(result_obj, BaseModel):
                                result_payload = result_obj.model_dump()
                            elif isinstance(result_obj, list):
                                result_payload = [
                                    (item.model_dump() if isinstance(item, BaseModel) else item)
                                    for item in result_obj
                                ]
                            else:
                                result_payload = result_obj
                            tc.status = ToolCallStatus.completed
                            tc.result = result_payload
                        except Exception as e:
                            tc.status = ToolCallStatus.rejected
                            tc.result = {"error": f"Tool '{tc.name}' failed: {str(e)}"}

                        updated_tool_calls.append(tc.model_copy(deep=True))

                    # Reply to executor with executed tool call results
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


            # Mark this step as finished (no rerun for this node)
            step.status = StepStatus.finished

            last_exec_activity = next((a for a in reversed(step.executions) if a.type == ActivityType.executor and a.is_complete), None)
            if last_exec_activity is None:
                raise RuntimeError("No completed executor activity found to compute transition")
            next_runtime_node, next_input_messages, next_edge_policy = self._compute_node_transition(
                current_runtime_node, last_exec_activity, step
            )
            if next_runtime_node is None:
                self.status = RunnerStatus.finished
                await self._close_all_generators()
                return

            # If this node's executor should not be reused, close and drop its generator now.
            if policy in (ResetPolicy.always_reset, ResetPolicy.keep_results):
                gen_to_close = self._agen_map.pop(node_name, None)
                if gen_to_close is not None:
                    try:
                        await gen_to_close.aclose()
                    except Exception:
                        pass

            current_runtime_node = next_runtime_node
            pending_input_messages = list(next_input_messages)
            incoming_policy_override = next_edge_policy
