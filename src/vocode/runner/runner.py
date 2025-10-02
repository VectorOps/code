from typing import (
    AsyncIterator,
    ClassVar,
    Dict,
    Optional,
    Type,
    List,
    TYPE_CHECKING,
    Any,
    Union,
)

import asyncio
from pydantic import BaseModel
from dataclasses import dataclass

if TYPE_CHECKING:
    from vocode.project import Project
from vocode.models import Node, Confirmation, ResetPolicy, MessageMode
from vocode.graph import RuntimeGraph
from vocode.state import (
    Message,
    RunnerStatus,
    Assignment,
    ToolCallStatus,
    Step,
    Activity,
    StepStatus,
    ActivityType,
)
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
    RunnerState,
    INTERIM_PACKETS,
    PACKETS_FOR_HISTORY,
    PACKET_MESSAGE_REQUEST,
    PACKET_TOOL_CALL,
    PACKET_MESSAGE,
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

    async def run(
        self, inp: ExecRunInput
    ) -> AsyncIterator[tuple[ReqPacket, Optional[Any]]]:
        """
        Async generator from Executor to Runner. Executors yield (packet, state) tuples:
        - may yield zero or more (PACKET_MESSAGE, state) interim messages,
        - must end a cycle by yielding a non-PACKET_MESSAGE packet as (packet, state).
        They DO NOT expect responses via .asend(); runner re-invokes run() with ExecRunInput(response=...) and state.
        """
        raise NotImplementedError(
            "Executor subclasses must implement 'run' as an async generator"
        )


class Runner:
    def __init__(
        self, workflow, project: "Project", initial_message: Optional[Message] = None
    ):
        """Prepare the runner with a graph, initial message, status flags, and per-node executors."""
        self.workflow = workflow
        self.runtime_graph = RuntimeGraph(self.workflow.graph)
        self.project = project
        self.initial_message: Optional[Message] = initial_message
        self.status: RunnerStatus = RunnerStatus.idle
        self._current_exec_task: Optional[asyncio.Task] = None
        self._executors: Dict[str, Executor] = {
            n.name: Executor.create_for_node(n, project=self.project)
            for n in self.workflow.graph.nodes
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

    def _compute_node_transition(
        self, current_runtime_node, exec_activity: Activity, step: Step
    ):
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
                next_runtime_node = current_runtime_node.get_child_by_outcome(
                    selected_outcome
                )
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
                role = (
                    final_msg.role
                    if final_msg is not None
                    else (input_msgs[-1].role if input_msgs else "agent")
                )
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
        return self.runtime_graph.get_runtime_node_by_name(name)

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
                if a.type == ActivityType.executor and a.runner_state is not None:
                    rs = a.runner_state
                    if isinstance(rs, RunnerState) and rs.state is not None:
                        return rs.state
        return None

    def _get_last_final_message_for_node(
        self,
        task: Assignment,
        node_name: str,
        exclude_step: Optional[Step] = None,
    ) -> Optional[Message]:
        """
        Find the most recent executor final (is_complete=True) message for the given node
        from earlier steps (excluding the provided in-flight step). Returns None if none found.
        """
        for s in reversed(task.steps):
            if s is exclude_step:
                continue
            if s.node != node_name:
                continue
            for a in reversed(s.executions):
                if a.type == ActivityType.executor and a.is_complete:
                    return a.message
        return None

    def _build_base_input_messages(
        self,
        policy: ResetPolicy,
        task: Assignment,
        node_name: str,
        step: Step,
        initial_user_inputs: List[Message],
    ) -> List[Message]:
        base_input_messages: List[Message] = []
        if policy == ResetPolicy.keep_results:
            # Include prior user + final messages (no interim)
            base_input_messages.extend(
                self._get_previous_node_messages(task, node_name, exclude_step=step)
            )
        elif policy == ResetPolicy.keep_final:
            # Include only the immediately previous final accepted response for this node
            last_final = self._get_last_final_message_for_node(
                task, node_name, exclude_step=step
            )
            if last_final is not None:
                # Avoid duplication if the transition already provided it as an input
                if not any(m is last_final for m in initial_user_inputs):
                    base_input_messages.append(last_final)

        # Always include the inputs provided by the transition into this step
        base_input_messages.extend(initial_user_inputs)
        return base_input_messages

    def _initial_state_for_policy(
        self,
        policy: ResetPolicy,
        task: Assignment,
        node_name: str,
        step: Step,
    ) -> Optional[object]:
        if policy == ResetPolicy.keep_state:
            return self._get_previous_node_state(task, node_name, exclude_step=step)
        return None

    def _get_executor_for_policy(
        self,
        policy: ResetPolicy,
        node_name: str,
        node_model: Node,
    ) -> "Executor":
        if policy == ResetPolicy.always_reset:
            self._executors[node_name] = Executor.create_for_node(
                node_model, project=self.project
            )
        return self._executors[node_name]

    def _is_input_requested(self, req: ReqPacket, node_conf: Confirmation) -> bool:
        if req.kind == PACKET_MESSAGE_REQUEST:
            return True
        if req.kind == PACKET_TOOL_CALL:
            # Request input if any tool call is not explicitly auto-approved (None or False)
            def _needs_approval(v: Any) -> bool:
                return not (v is True)

            return any(
                _needs_approval(getattr(tc, "auto_approve", None))
                for tc in req.tool_calls
            )
        if req.kind == PACKET_FINAL_MESSAGE:
            return node_conf in (Confirmation.prompt, Confirmation.confirm)
        return False

    def _split_tool_calls(self, req: ReqToolCall):
        """Split tool calls into (auto_approved, manual) based on tc.auto_approve flag."""
        auto_calls = [tc for tc in req.tool_calls if getattr(tc, "auto_approve", False)]
        manual_calls = [
            tc for tc in req.tool_calls if not getattr(tc, "auto_approve", False)
        ]
        return auto_calls, manual_calls

    async def _run_tools(
        self,
        req: ReqToolCall,
        resp_packet: Optional[RespPacket],
    ) -> RespToolCall:
        # Default approval unless provided explicitly
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
                tc.result = {
                    "error": f"Invalid arguments for tool '{tc.name}': {str(e)}"
                }
                updated_tool_calls.append(tc.model_copy(deep=True))
                continue

            try:
                result = await tool.run(self.project, args_model)
                tc.status = ToolCallStatus.completed
                tc.result = result
            except Exception as e:
                tc.status = ToolCallStatus.rejected
                tc.result = {"error": f"Tool '{tc.name}' failed: {str(e)}"}

            updated_tool_calls.append(tc.model_copy(deep=True))

        return RespToolCall(tool_calls=updated_tool_calls)

    def _iter_retriable_activities_backward(
        self, task: Assignment, start_step_index: Optional[int] = None
    ):
        """
        Generator that yields (step_index, activity_index, activity) for retriable activities,
        iterating backwards from the end of the task or from start_step_index.
        """
        if start_step_index is None:
            start_step_index = len(task.steps) - 1

        for step_idx in range(start_step_index, -1, -1):
            step = task.steps[step_idx]
            for act_idx in range(len(step.executions) - 1, -1, -1):
                activity = step.executions[act_idx]
                if not activity.ephemeral:
                    yield step_idx, act_idx, activity

    def _trim_history(
        self, task: Assignment, step_idx: int, act_idx: int, *, keep_target: bool
    ) -> None:
        """
        Trims task history to a specific point.
        """
        # Cut steps after the one we're editing
        task.steps = task.steps[: step_idx + 1]
        target_step = task.steps[step_idx]

        # In the target step, cut activities from the target one onwards
        slice_end = act_idx + 1 if keep_target else act_idx
        target_step.executions = target_step.executions[:slice_end]

        # If the step is now effectively empty (no persisted activities), remove it.
        # This can happen if we rewind past the first non-ephemeral activity in a step.
        if not any(not ex.ephemeral for ex in target_step.executions):
            task.steps.pop(step_idx)
        else:
            target_step.status = StepStatus.running

    def _create_response_activity(self, activity, resp):
        return Activity(
            type=ActivityType.user,
            runner_state=RunnerState(
                state=activity.runner_state.state,
                req=activity.runner_state.req,
                response=resp,
                messages=list(activity.runner_state.messages),
            ),
            outcome_name=activity.outcome_name,
            is_complete=activity.is_complete,
            ephemeral=activity.ephemeral,
        )

    async def rewind(self, task: Assignment, n: int = 1) -> None:
        """
        Rewind history by removing the last n retriable activities from the assignment
        and resetting executors so the rewound nodes are as if they never executed.

        Allowed when runner is idle, stopped, or finished.
        """
        if n <= 0:
            raise ValueError("rewind 'n' must be >= 1")

        if self.status in (RunnerStatus.running, RunnerStatus.waiting_input):
            raise RuntimeError(f"Cannot rewind while runner status is '{self.status}'")

        target_activity_info = None
        activities_to_skip = n

        for activity_info in self._iter_retriable_activities_backward(task):
            print(activity_info)
            activities_to_skip -= 1
            if activities_to_skip == 0:
                target_activity_info = activity_info
                break

        if target_activity_info:
            step_idx, act_idx, _ = target_activity_info
            self._trim_history(task, step_idx, act_idx, keep_target=False)

            task.steps[-1].status = StepStatus.running
        else:
            # This happens if n is greater than the number of retriable activities.
            task.steps.clear()

        # Reset internal flags
        self._stop_requested = False
        self._internal_cancel_requested = False
        self._current_exec_task = None
        self.status = RunnerStatus.stopped

    def _node_confirmation(self, node_name: str) -> Optional[Confirmation]:
        rn = self._find_runtime_node_by_name(node_name)
        return rn.model.confirmation if rn else None

    def replace_user_input(
        self,
        task: Assignment,
        response: Union[RespMessage, RespApproval],
        n: Optional[int] = 1,
    ) -> None:
        """
        Finds a user input boundary in the history, replaces the user's response,
        and prepares the runner to resume execution from that point.
        By default it searches from the last step backwards. If step_index is provided,
        it searches from that step backwards.
        Leaves runner in stopped state ready to restart.
        """
        if n <= 0:
            raise ValueError("replace_user_input 'n' must be >= 1")

        if self.status in (RunnerStatus.running, RunnerStatus.waiting_input):
            raise RuntimeError(
                f"Cannot replace input while runner status is '{self.status}'"
            )

        target_activity_info = None
        boundaries_to_skip = n

        for step_idx, act_idx, activity in self._iter_retriable_activities_backward(
            task
        ):
            if activity.ephemeral:
                continue

            runner_state = activity.runner_state

            req = runner_state.req
            is_boundary = False
            if req.kind == PACKET_MESSAGE_REQUEST:
                is_boundary = True
            elif req.kind == PACKET_FINAL_MESSAGE:
                conf = self._node_confirmation(task.steps[step_idx].node)
                if (
                    conf
                    in (
                        Confirmation.prompt,
                        Confirmation.confirm,
                    )
                    and runner_state.response is not None
                ):
                    is_boundary = True

            if is_boundary:
                boundaries_to_skip -= 1
                if boundaries_to_skip == 0:
                    target_activity_info = (step_idx, act_idx, activity)
                    break

        if target_activity_info:
            step_idx, act_idx, boundary_activity = target_activity_info

            # Truncate history to resume from this boundary request
            self._trim_history(task, step_idx, act_idx, keep_target=True)
            target_step = task.steps[step_idx]
            target_step.status = StepStatus.running

            # Set pending response on the boundary activity
            boundary_activity = target_step.executions[act_idx]
            if boundary_activity.type == ActivityType.user:
                boundary_activity.runner_state.response = response
            else:
                target_step.executions.append(
                    self._create_response_activity(boundary_activity, response)
                )

            # Reset state for run()
            self._stop_requested = False
            self._internal_cancel_requested = False
            self._current_exec_task = None
            self.status = RunnerStatus.stopped
            return

        raise RuntimeError(
            "No previous user input or pending request to replace/respond to"
        )

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
        self.status = RunnerStatus.running
        incoming_policy_override: Optional[ResetPolicy] = None

        # Unified resume/start logic
        pending_input_messages: List[Message] = []
        resume_activity: Optional[Activity] = None
        step: Optional[Step] = None

        # If step is running, we resume the step. Otherwise just continue with a next step.
        if task.steps and task.steps[-1].status != StepStatus.finished:
            step = task.steps[-1]
            current_runtime_node = self._find_runtime_node_by_name(step.node)
            if current_runtime_node is None:
                self.status = RunnerStatus.finished
                return

            # Use last non-ephemeral activity
            last_activity = next(
                (a for a in reversed(step.executions) if not a.ephemeral),
                None,
            )

            if last_activity:
                resume_activity = last_activity

        # Natural resume from the last finished step; otherwise start from root
        if resume_activity is None:
            current_runtime_node = self.runtime_graph.root
            pending_input_messages = (
                [self.initial_message] if self.initial_message is not None else []
            )

        # Main loop
        while True:
            # Create new step for new executions
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

            # Collect initial inputs
            initial_user_inputs: List[Message] = [
                a.message
                for a in step.executions
                if a.message is not None and a.type == ActivityType.user
            ]

            base_input_messages = self._build_base_input_messages(
                policy=policy,
                task=task,
                node_name=node_name,
                step=step,
                initial_user_inputs=initial_user_inputs,
            )

            if resume_activity and resume_activity.runner_state.state is not None:
                current_state = resume_activity.runner_state.state
            else:
                current_state = self._initial_state_for_policy(
                    policy=policy,
                    task=task,
                    node_name=node_name,
                    step=step,
                )

            executor = self._get_executor_for_policy(
                policy=policy,
                node_name=node_name,
                node_model=node_model,
            )

            # Runner cycles: repeatedly invoke executor until it yields a final or request we handle externally
            resp: Optional[RespPacket] = None
            last_exec_activity: Optional[Activity] = None

            pass_messages = (
                True  # Only pass base_input_messages on the first cycle for this node
            )

            while True:
                req: Optional[ReqPacket] = None
                yielded_state = None

                if resume_activity is not None:
                    req = resume_activity.runner_state.req
                else:
                    # Invoke executor for one cycle
                    msgs_for_cycle = base_input_messages if pass_messages else []
                    pass_messages = False  # only pass base messages on the first cycle
                    agen = executor.run(
                        ExecRunInput(
                            messages=msgs_for_cycle, state=current_state, response=resp
                        )
                    )

                    # Stream interim messages; stop at first non-message packet
                    try:
                        while True:
                            try:
                                self._current_exec_task = asyncio.create_task(
                                    anext(agen)
                                )
                                pkt, yielded_state = await self._current_exec_task
                            finally:
                                self._current_exec_task = None

                            if pkt.kind in INTERIM_PACKETS:
                                # Interim events: create a temporary activity for consumer use; do not persist.
                                self.status = RunnerStatus.running
                                interim_act = Activity(
                                    type=ActivityType.executor,
                                    message=(
                                        pkt.message
                                        if pkt.kind == PACKET_MESSAGE
                                        else None
                                    ),
                                    is_complete=False,
                                    runner_state=RunnerState(
                                        state=current_state, req=pkt, response=None
                                    ),
                                    ephemeral=True,
                                )
                                run_event = RunEvent(
                                    node=current_runtime_node.name,
                                    execution=interim_act,
                                    event=pkt,
                                    input_requested=False,
                                )
                                _ = yield run_event
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
                        (
                            a
                            for a in reversed(step.executions)
                            if a.type == ActivityType.executor and a.is_complete
                        ),
                        None,
                    )
                    if last_exec_activity is None:
                        # No explicit final; allow transition with last interim if any (message_mode may handle)
                        last_interim = next(
                            (
                                a
                                for a in reversed(step.executions)
                                if a.type == ActivityType.executor
                                and a.message is not None
                            ),
                            None,
                        )
                        if last_interim is None:
                            raise RuntimeError(
                                "Executor finished without emitting any messages"
                            )
                        last_exec_activity = last_interim.clone(is_complete=True)
                        step.executions.append(last_exec_activity)

                    # Compute next node/inputs
                    next_runtime_node, next_input_messages, next_edge_policy = (
                        self._compute_node_transition(
                            current_runtime_node, last_exec_activity, step
                        )
                    )
                    if next_runtime_node is None:
                        self.status = RunnerStatus.finished
                        return
                    current_runtime_node = next_runtime_node
                    pending_input_messages = list(next_input_messages)
                    incoming_policy_override = next_edge_policy
                    step = None  # force a new Step for the next node
                    break

                # Handle completion packet kinds
                # Figure out if input is requested from UI
                input_requested = self._is_input_requested(
                    req=req, node_conf=current_runtime_node.model.confirmation
                )

                # Build a request activity; do not persist yet — we’ll persist a clone after we receive a response.
                if resume_activity is not None:
                    req_activity = resume_activity
                else:
                    req_activity = Activity(
                        type=ActivityType.executor,
                        message=(
                            req.message if req.kind == PACKET_FINAL_MESSAGE else None
                        ),
                        outcome_name=(
                            req.outcome_name
                            if req.kind == PACKET_FINAL_MESSAGE
                            else None
                        ),
                        is_complete=(req.kind == PACKET_FINAL_MESSAGE),
                        runner_state=RunnerState(
                            state=current_state,
                            req=req,
                            response=None,
                            messages=list(base_input_messages),
                        ),
                        ephemeral=(req.kind not in PACKETS_FOR_HISTORY),
                    )
                    if not req_activity.ephemeral:
                        step.executions.append(req_activity)

                run_input: Optional[RunInput] = None
                resp_packet: Optional[RespPacket] = None

                if (
                    resume_activity is not None
                    and resume_activity.type == ActivityType.user
                ):
                    resp_packet = resume_activity.runner_state.response
                    resp_activity = resume_activity
                else:
                    run_event = RunEvent(
                        node=current_runtime_node.name,
                        execution=req_activity,
                        event=req,
                        input_requested=input_requested,
                    )
                    run_input = yield run_event
                    resp_packet = run_input.response if run_input is not None else None

                    # Persist a clone with the response, only for history packets.
                    resp_activity = self._create_response_activity(
                        req_activity, resp_packet
                    )
                    if not resp_activity.ephemeral:
                        step.executions.append(resp_activity)

                resume_activity = None

                # PACKET_FINAL_MESSAGE: finalize (with confirm/prompt)
                if req.kind == PACKET_FINAL_MESSAGE:
                    node_conf = current_runtime_node.model.confirmation

                    # confirm: require explicit approval
                    if node_conf == Confirmation.confirm:
                        # Re-prompt until approval or stop on reject
                        while True:
                            if (
                                resp_packet is not None
                                and resp_packet.kind == PACKET_APPROVAL
                            ):
                                resp_activity.runner_state.response = resp_packet
                                if resp_packet.approved:
                                    break
                                # Rejected: stop runner
                                self.status = RunnerStatus.stopped
                                step.status = StepStatus.stopped
                                # consume resume marker
                                return

                            # Ask again
                            self.status = RunnerStatus.waiting_input
                            run_event = RunEvent(
                                node=current_runtime_node.name,
                                execution=resp_activity,
                                event=req,
                                input_requested=True,
                            )
                            run_input = yield run_event
                            resp_packet = (
                                run_input.response if run_input is not None else None
                            )

                    # prompt: allow one more user message; if provided, run another cycle
                    elif node_conf == Confirmation.prompt:
                        if (
                            resp_packet is not None
                            and resp_packet.kind == PACKET_MESSAGE
                        ):
                            resp_activity.message = resp_packet.message
                            resp = resp_packet
                            continue
                    else:
                        # No confirmation/prompt: accept final
                        pass

                    # Final accepted: finish step and transition
                    step.status = StepStatus.finished

                    next_runtime_node, next_input_messages, next_edge_policy = (
                        self._compute_node_transition(
                            current_runtime_node, req_activity, step
                        )
                    )

                    if next_runtime_node is None:
                        self.status = RunnerStatus.finished
                        return

                    current_runtime_node = next_runtime_node
                    pending_input_messages = list(next_input_messages)
                    incoming_policy_override = next_edge_policy
                    step = None  # force a new Step for the next node
                    break  # move to next node

                # PACKET_TOOL_CALL: execute tools and re-run executor with results
                if req.kind == PACKET_TOOL_CALL:
                    resp = await self._run_tools(req, resp_packet)
                    continue

                # PACKET_MESSAGE_REQUEST: ask for user message then re-run executor with it
                if req.kind == PACKET_MESSAGE_REQUEST:
                    # Require a message; keep prompting until we get one
                    user_msg_packet: Optional[RespMessage] = None
                    while True:
                        resp_activity.runner_state.response = resp_packet

                        if (
                            resp_packet is not None
                            and resp_packet.kind == PACKET_MESSAGE
                        ):
                            resp_activity.message = resp_packet.message
                            user_msg_packet = resp_packet
                            break

                        self.status = RunnerStatus.waiting_input
                        run_event = RunEvent(
                            node=current_runtime_node.name,
                            execution=resp_activity,
                            event=req,
                            input_requested=True,
                        )
                        run_input = yield run_event
                        resp_packet = (
                            run_input.response if run_input is not None else None
                        )

                    # Update persisted executor request with the response (mutate, do not re-append)
                    resp_activity.runner_state.response = user_msg_packet

                    # Set response for next executor cycle
                    resp = user_msg_packet
                    continue

                # Any other packet kinds: no-op; continue outer loop
                continue
