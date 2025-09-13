
from typing import AsyncIterator, ClassVar, Dict, Optional, Type, List, TYPE_CHECKING, Any

import asyncio
from pydantic import BaseModel
if TYPE_CHECKING:
    from vocode.project import Project
from vocode.graph import Node
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
    ExecRunInput,
    PACKET_MESSAGE_REQUEST,
    PACKET_TOOL_CALL,
    PACKET_MESSAGE,
    PACKET_LOG,
    PACKET_FINAL_MESSAGE,
    PACKET_APPROVAL,
)




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

    async def run(self, inp: ExecRunInput) -> AsyncIterator[tuple[ReqPacket, Optional[Any]]]:
        """
        Async generator from Executor to Runner. Executors yield (packet, state) tuples:
        - may yield zero or more (PACKET_MESSAGE, state) interim messages,
        - must end a cycle by yielding a non-PACKET_MESSAGE packet as (packet, state).
        They DO NOT expect responses via .asend(); runner re-invokes run() with ExecRunInput(response=...) and state.
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

    def _get_previous_node_state(
        self,
        task: Assignment,
        node_name: str,
        exclude_step: Optional[Step] = None,
    ) -> Optional[object]:
        """
        Returns the most recent executor state for a node from prior steps.
        Prefers the last execution with a non-None state, skipping exclude_step.
        """
        for s in reversed(task.steps):
            if s is exclude_step:
                continue
            if s.node != node_name:
                continue
            for a in reversed(s.executions):
                if a.type == ActivityType.executor and a.state is not None:
                    return a.state
        return None

    def _prepare_resume(self, task: Assignment):
        """
        Find the last successfully finished step, compute the transition to the next node,
        and return (next_runtime_node, input_messages_for_next, incoming_edge_reset_policy).
        If no finished steps are found, return None (start from the graph root).
        If the last finished node has no outgoing edges, return (None, None, None) to indicate completion.
        """
        if not task.steps:
            return None
        for step in reversed(task.steps):
            if step.status == StepStatus.finished:
                last_exec = next(
                    (a for a in reversed(step.executions) if a.type == ActivityType.executor and a.is_complete),
                    None,
                )
                if last_exec is None:
                    continue
                cur_rn = self._find_runtime_node_by_name(step.node)
                if cur_rn is None:
                    return None
                next_rn, msgs, edge_policy = self._compute_node_transition(cur_rn, last_exec, step)
                return next_rn, msgs, edge_policy
        return None


    async def rewind(self, task: Assignment, n: int = 1) -> None:
        """
        Rewind history by removing the last n steps from the assignment
        and resetting executors so the rewound nodes are as if they never executed.

        Allowed when runner is idle, stopped, or finished.
        """
        if n <= 0:
            raise ValueError("rewind 'n' must be >= 1")

        if self.status in (RunnerStatus.running, RunnerStatus.waiting_input):
            raise RuntimeError(f"Cannot rewind while runner status is '{self.status}'")

        # Reset executors so the rewound nodes are as if they never executed.

        # Recreate fresh executor instances for all nodes to forget any prior state.
        self._executors = {
            node.name: Executor.create_for_node(node, project=self.project)
            for node in self.workflow.graph.nodes
        }

        # Remove the last n steps (if fewer, remove all).
        to_remove = min(n, len(task.steps))
        for _ in range(to_remove):
            task.steps.pop()

        # Reset internal flags and set a resumable state.
        self._stop_requested = False
        self._internal_cancel_requested = False
        self._current_exec_task = None
        self.status = RunnerStatus.stopped

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
        incoming_policy_override: Optional[ResetPolicy] = None

        # Runner resuming
        resume_info = None
        if prev_status == RunnerStatus.stopped:
            resume_info = self._prepare_resume(task)

        pending_input_messages: List[Message] = []

        if resume_info is not None:
            next_rn, next_msgs, incoming_policy_override = resume_info
            if next_rn is None:
                self.status = RunnerStatus.finished
                return
            current_runtime_node = next_rn
            pending_input_messages = list(next_msgs or [])
        else:
            current_runtime_node = graph.root
            pending_input_messages = [self.initial_message] if self.initial_message is not None else []
            incoming_policy_override: Optional[ResetPolicy] = None

        # Main loop
        while True:
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

            # Build base input messages for this node execution (first cycle only)
            base_input_messages: List[Message] = []
            if policy == ResetPolicy.keep_results:
                base_input_messages.extend(
                    self._get_previous_node_messages(task, node_name, exclude_step=step)
                )
            base_input_messages.extend(initial_user_inputs)

            # Compute initial state based on reset policy
            if policy == ResetPolicy.keep_state:
                current_state = self._get_previous_node_state(task, node_name, exclude_step=step)
            else:
                current_state = None

            # For always_reset, also replace the executor instance to reset internal state
            if policy == ResetPolicy.always_reset:
                self._executors[node_name] = Executor.create_for_node(node_model, project=self.project)
            executor = self._executors[node_name]

            # Runner cycles: repeatedly invoke executor until it yields a final or request we handle externally
            resp: Optional[RespPacket] = None
            last_exec_activity: Optional[Activity] = None
            placeholder_exec = Activity(type=ActivityType.executor)

            while True:
                # Invoke executor for one cycle
                agen = executor.run(ExecRunInput(messages=base_input_messages, state=current_state, response=resp))
                resp = None  # response is one-shot for each cycle

                # Stream interim messages; stop at first non-message packet
                req: Optional[ReqPacket] = None
                try:
                    while True:
                        try:
                            self._current_exec_task = asyncio.create_task(anext(agen))
                            pkt, yielded_state = await self._current_exec_task
                        finally:
                            self._current_exec_task = None

                        if pkt.kind in (PACKET_MESSAGE, PACKET_LOG):
                            # Interim message/log: do not generate an Activity or populate message/state.
                            self.status = RunnerStatus.running
                            run_event = RunEvent(
                                node=current_runtime_node.name,
                                execution=placeholder_exec,
                                event=pkt,
                                input_requested=False,
                            )
                            _ = (yield run_event)
                            continue

                        # Non-message => end of this executor cycle
                        req = pkt

                        # Capture updated state if provided
                        if yielded_state is not None:
                            current_state = yielded_state

                        break
                except StopAsyncIteration:
                    # Executor ended without emitting a completion packet; treat as done for this cycle
                    req = None

                # If nothing returned, mark step finished and transition
                if req is None:
                    step.status = StepStatus.finished

                    # Find last complete executor activity for transition (may be from previous cycles)
                    last_exec_activity = next(
                        (a for a in reversed(step.executions) if a.type == ActivityType.executor and a.is_complete),
                        None,
                    )
                    if last_exec_activity is None:
                        # No explicit final; allow transition with last interim if any (message_mode may handle)
                        last_interim = next(
                            (a for a in reversed(step.executions) if a.type == ActivityType.executor and a.message is not None),
                            None,
                        )
                        if last_interim is None:
                            raise RuntimeError("Executor finished without emitting any messages")
                        last_exec_activity = last_interim.clone(is_complete=True)
                        step.executions.append(last_exec_activity)

                    # Compute next node/inputs
                    next_runtime_node, next_input_messages, next_edge_policy = self._compute_node_transition(
                        current_runtime_node, last_exec_activity, step
                    )
                    if next_runtime_node is None:
                        self.status = RunnerStatus.finished
                        return
                    current_runtime_node = next_runtime_node
                    pending_input_messages = list(next_input_messages)
                    incoming_policy_override = next_edge_policy
                    break

                # Handle completion packet kinds
                # Figure out if input is requested from UI
                if req.kind in (PACKET_MESSAGE_REQUEST, PACKET_TOOL_CALL):
                    input_requested = True
                elif req.kind == PACKET_FINAL_MESSAGE:
                    node_conf = current_runtime_node.model.confirmation
                    input_requested = node_conf in (Confirmation.prompt, Confirmation.confirm)
                else:
                    input_requested = False

                # Prepare default activity placeholder for UI emission
                run_event = RunEvent(
                    node=current_runtime_node.name,
                    execution=placeholder_exec,
                    event=req,
                    input_requested=input_requested,
                )
                run_input: Optional[RunInput] = (yield run_event)
                resp_packet: Optional[RespPacket] = run_input.response if run_input is not None else None

                # PACKET_FINAL_MESSAGE: finalize (with confirm/prompt)
                if req.kind == PACKET_FINAL_MESSAGE:
                    final_act = Activity(
                        type=ActivityType.executor,
                        message=req.message,
                        outcome_name=req.outcome_name,
                        is_complete=True,
                        state=current_state,
                    )

                    node_conf = current_runtime_node.model.confirmation

                    # confirm: require explicit approval
                    if node_conf == Confirmation.confirm:
                        # Re-prompt until approval or stop on reject
                        while True:
                            if resp_packet is not None and resp_packet.kind == PACKET_APPROVAL:
                                if resp_packet.approved:
                                    step.executions.append(final_act)
                                    break
                                # Rejected: record final and stop runner
                                step.executions.append(final_act)
                                self.status = RunnerStatus.stopped
                                step.status = StepStatus.stopped
                                return
                            # Ask again
                            self.status = RunnerStatus.waiting_input
                            run_event = RunEvent(
                                node=current_runtime_node.name,
                                execution=final_act,
                                event=req,
                                input_requested=True,
                            )
                            run_input = (yield run_event)
                            resp_packet = run_input.response if run_input is not None else None

                    # prompt: allow one more user message; if provided, run another cycle
                    elif node_conf == Confirmation.prompt:
                        if resp_packet is not None and resp_packet.kind == PACKET_MESSAGE:
                            # Record this final, but continue same node with provided user message (not recorded)
                            step.executions.append(final_act)
                            resp = resp_packet
                            continue
                        # No extra message provided; accept final
                        step.executions.append(final_act)

                    else:
                        # No confirmation/prompt: accept final
                        step.executions.append(final_act)

                    # Final accepted: finish step and transition
                    step.status = StepStatus.finished
                    last_exec_activity = final_act
                    next_runtime_node, next_input_messages, next_edge_policy = self._compute_node_transition(
                        current_runtime_node, last_exec_activity, step
                    )
                    if next_runtime_node is None:
                        self.status = RunnerStatus.finished
                        return
                    current_runtime_node = next_runtime_node
                    pending_input_messages = list(next_input_messages)
                    incoming_policy_override = next_edge_policy
                    break  # move to next node

                # PACKET_TOOL_CALL: execute tools and re-run executor with results
                if req.kind == PACKET_TOOL_CALL:
                    # Default to approved unless explicitly rejected by a response
                    approved = True
                    if resp_packet is not None and resp_packet.kind == PACKET_APPROVAL:
                        approved = resp_packet.approved

                    updated_tool_calls = []
                    for tc in req.tool_calls:
                        if not approved:
                            tc.status = ToolCallStatus.rejected
                            tc.result = {"error": "Tool call rejected by user"}
                            updated_tool_calls.append(tc.model_copy(deep=True))
                            continue

                        tool = self.project.tools.get(tc.name)
                        if tool is None:
                            tc.status = ToolCallStatus.rejected
                            tc.result = {"error": f"Unknown tool '{tc.name}'"}
                            updated_tool_calls.append(tc.model_copy(deep=True))
                            continue

                        try:
                            args_model = tool.input_model.model_validate(tc.arguments)
                        except Exception as e:
                            tc.status = ToolCallStatus.rejected
                            tc.result = {"error": f"Invalid arguments for tool '{tc.name}': {str(e)}"}
                            updated_tool_calls.append(tc.model_copy(deep=True))
                            continue

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

                    # Prepare response for next cycle
                    resp = RespToolCall(tool_calls=updated_tool_calls)
                    # Continue outer loop to re-run executor with tool results
                    continue

                # PACKET_MESSAGE_REQUEST: ask for user message then re-run executor with it
                if req.kind == PACKET_MESSAGE_REQUEST:
                    # Require a message; keep prompting until we get one
                    user_msg_packet: Optional[RespMessage] = None
                    while True:
                        if resp_packet is not None and resp_packet.kind == PACKET_MESSAGE:
                            user_msg_packet = resp_packet
                            break
                        self.status = RunnerStatus.waiting_input
                        run_event = RunEvent(
                            node=current_runtime_node.name,
                            execution=placeholder_exec,
                            event=req,
                            input_requested=True,
                        )
                        run_input = (yield run_event)
                        resp_packet = run_input.response if run_input is not None else None
                    # Record user message in history
                    step.executions.append(Activity(type=ActivityType.user, message=user_msg_packet.message))
                    # Set response for next executor cycle
                    resp = user_msg_packet
                    continue

                # Any other packet kinds: no-op; continue outer loop
                continue

