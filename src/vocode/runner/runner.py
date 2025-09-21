 
from typing import AsyncIterator, ClassVar, Dict, Optional, Type, List, TYPE_CHECKING, Any, Union

import asyncio
from pydantic import BaseModel
from dataclasses import dataclass
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
    INTERIM_PACKETS,
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

    async def run(self, inp: ExecRunInput) -> AsyncIterator[tuple[ReqPacket, Optional[Any]]]:
        """
        Async generator from Executor to Runner. Executors yield (packet, state) tuples:
        - may yield zero or more (PACKET_MESSAGE, state) interim messages,
        - must end a cycle by yielding a non-PACKET_MESSAGE packet as (packet, state).
        They DO NOT expect responses via .asend(); runner re-invokes run() with ExecRunInput(response=...) and state.
        """
        raise NotImplementedError("Executor subclasses must implement 'run' as an async generator")


@dataclass
class HistoryStep:
    node: str
    activity: Activity
    messages: List[Message]
    state: Optional[Any]
    req: ReqPacket          # will hold only ReqFinalMessage for now
    response: Optional[RespPacket]


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
        # History for retriable final messages
        self._history: List[HistoryStep] = []
        # When rewinding, the last popped history step is kept here to be replayed on resume
        self._resume_history_step: Optional[HistoryStep] = None

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
            last_final = self._get_last_final_message_for_node(task, node_name, exclude_step=step)
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
            self._executors[node_name] = Executor.create_for_node(node_model, project=self.project)
        return self._executors[node_name]

    def _is_input_requested(self, req: ReqPacket, node_conf: Confirmation) -> bool:
        if req.kind == PACKET_MESSAGE_REQUEST:
            return True
        if req.kind == PACKET_TOOL_CALL:
            # Request input if any tool call is not explicitly auto-approved (None or False)
            def _needs_approval(v: Any) -> bool:
                return not (v is True)
            return any(_needs_approval(getattr(tc, "auto_approve", None)) for tc in req.tool_calls)
        if req.kind == PACKET_FINAL_MESSAGE:
            return node_conf in (Confirmation.prompt, Confirmation.confirm)
        return False

    def _split_tool_calls(self, req: ReqToolCall):
        """Split tool calls into (auto_approved, manual) based on tc.auto_approve flag."""
        auto_calls = [tc for tc in req.tool_calls if getattr(tc, "auto_approve", False)]
        manual_calls = [tc for tc in req.tool_calls if not getattr(tc, "auto_approve", False)]
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
                tc.result = {"error": f"Invalid arguments for tool '{tc.name}': {str(e)}"}
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

        # Go back n retriable history steps (finals)
        to_remove = min(n, len(self._history))
        last_popped: Optional[HistoryStep] = None
        for _ in range(to_remove):
            last_popped = self._history.pop()
        # Align recorded steps (each final corresponds to one step)
        for _ in range(min(n, len(task.steps))):
            if task.steps:
                task.steps.pop()
        # Mark resume point at the last popped history step
        self._resume_history_step = last_popped
        # Reset internal flags
        self._stop_requested = False
        self._internal_cancel_requested = False
        self._current_exec_task = None
        self.status = RunnerStatus.stopped

    def _node_confirmation(self, node_name: str) -> Optional[Confirmation]:
        rn = self._find_runtime_node_by_name(node_name)
        return rn.model.confirmation if rn else None

    def _find_last_retriable_history_index(self) -> Optional[int]:
        """
        Find the last history step whose node would request user input on its final
        (confirmation=prompt/confirm). Returns the index in self._history or None.
        """
        for i in range(len(self._history) - 1, -1, -1):
            h = self._history[i]
            conf = self._node_confirmation(h.node)
            if conf in (Confirmation.prompt, Confirmation.confirm):
                return i
        return None

    def _find_last_user_input_step_index(self, task: Assignment) -> Optional[int]:
        """
        Find the most recent step that contains an actual user response (message.role == 'user').
        This excludes carried messages from previous nodes (which are recorded as user-type activities
        but have role != 'user').
        """
        for i in range(len(task.steps) - 1, -1, -1):
            s = task.steps[i]
            for a in s.executions:
                if a.type == ActivityType.user and a.message is not None and a.message.role == "user":
                    return i
        return None

    def replace_last_user_input(self, task: Assignment, response: Union[RespMessage, RespApproval]) -> None:
        """
        Prepare to replace the last user input:
        - If the last replaceable boundary is a final that required input (prompt/confirm),
          pop all later history/steps, set the stored response on that history step, and resume from it.
        - Otherwise, target the last step that included a user message (ReqMessageRequest case):
          pop history and steps including that step to force re-execution; UI should auto-reply with the new message.
        Leaves runner in stopped state ready to restart.
        """
        if self.status in (RunnerStatus.running, RunnerStatus.waiting_input):
            raise RuntimeError(f"Cannot replace input while runner status is '{self.status}'")

        # Case 1: last retriable final (prompt/confirm)
        hist_idx = self._find_last_retriable_history_index()
        if hist_idx is not None:
            # Pop history/steps after the target history index
            to_pop = len(self._history) - (hist_idx + 1)
            for _ in range(to_pop):
                self._history.pop()
            for _ in range(to_pop):
                if task.steps:
                    task.steps.pop()
            # Set resume point at the target and store the replacement response
            self._resume_history_step = self._history[hist_idx]
            self._resume_history_step.response = response
            # Reset internal flags
            self._stop_requested = False
            self._internal_cancel_requested = False
            self._current_exec_task = None
            self.status = RunnerStatus.stopped
            return

        # Case 2: replace last message_request input -> pop including that step to re-execute
        step_idx = self._find_last_user_input_step_index(task)
        if step_idx is None:
            raise RuntimeError("No previous user input to replace")
        # Pop history entries from that step onward (inclusive)
        to_pop_hist = max(0, len(self._history) - step_idx)
        for _ in range(to_pop_hist):
            if self._history:
                self._history.pop()
        # Pop steps down to retain only steps before the target
        while len(task.steps) > step_idx:
            task.steps.pop()
        # Force fresh execution (do not preload any saved final)
        self._resume_history_step = None
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

        # Preloaded request/response/activity when resuming from a rewound final
        preloaded_req: Optional[ReqPacket] = None
        preloaded_final_act: Optional[Activity] = None
        preloaded_resp_default: Optional[RespPacket] = None

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

        # If we have a resume history step saved from rewind(), inject its request/activity/response
        if self._resume_history_step is not None:
            preloaded_req = self._resume_history_step.req
            preloaded_final_act = self._resume_history_step.activity
            preloaded_resp_default = self._resume_history_step.response

        def _clear_resume_markers():
            nonlocal preloaded_req, preloaded_final_act, preloaded_resp_default
            self._resume_history_step = None
            preloaded_req = None
            preloaded_final_act = None
            preloaded_resp_default = None

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

            base_input_messages = self._build_base_input_messages(
                policy=policy,
                task=task,
                node_name=node_name,
                step=step,
                initial_user_inputs=initial_user_inputs,
            )

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
            placeholder_exec = Activity(type=ActivityType.executor)
            pass_messages = True  # Only pass base_input_messages on the first cycle for this node

            while True:
                # Optionally bypass executor execution when replaying a saved final from history
                replaying_history = False
                req: Optional[ReqPacket] = None
                yielded_state = None

                if preloaded_req is not None:
                    # Replay the saved request/state without running the executor
                    req = preloaded_req
                    yielded_state = current_state
                    replaying_history = True
                    # Treat the initial messages as already consumed for this node
                    pass_messages = False
                else:
                    # Invoke executor for one cycle
                    msgs_for_cycle = base_input_messages if pass_messages else []
                    agen = executor.run(ExecRunInput(messages=msgs_for_cycle, state=current_state, response=resp))
                    resp = None  # response is one-shot for each cycle
                    # After first invocation, do not pass messages again for this node
                    pass_messages = False

                    # Stream interim messages; stop at first non-message packet
                    try:
                        while True:
                            try:
                                self._current_exec_task = asyncio.create_task(anext(agen))
                                pkt, yielded_state = await self._current_exec_task
                            finally:
                                self._current_exec_task = None

                            if pkt.kind in INTERIM_PACKETS:
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
                input_requested = self._is_input_requested(
                    req=req, node_conf=current_runtime_node.model.confirmation
                )

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
                    # Prefer a preloaded final activity when replaying history
                    if preloaded_final_act is not None:
                        final_act = preloaded_final_act
                    else:
                        final_act = Activity(
                            type=ActivityType.executor,
                            message=req.message,
                            outcome_name=req.outcome_name,
                            is_complete=True,
                            state=current_state,
                        )

                    # Prepare or reuse history entry for this final
                    if replaying_history and self._resume_history_step is not None:
                        hist = self._resume_history_step
                        # restore into history list (we popped it during rewind)
                        if hist not in self._history:
                            self._history.append(hist)
                    else:
                        hist = HistoryStep(
                            node=current_runtime_node.name,
                            activity=final_act,
                            messages=list(base_input_messages),
                            state=current_state,
                            req=req,
                            response=None,
                        )
                        self._history.append(hist)

                    node_conf = current_runtime_node.model.confirmation

                    # If we have a preloaded default response for resume, apply it when appropriate
                    if preloaded_resp_default is not None and resp_packet is None:
                        resp_packet = preloaded_resp_default

                    # confirm: require explicit approval
                    if node_conf == Confirmation.confirm:
                        # Re-prompt until approval or stop on reject
                        while True:
                            if resp_packet is not None and resp_packet.kind == PACKET_APPROVAL:
                                # Record final activity in step history (and mark response on hist)
                                step.executions.append(final_act)
                                hist.response = resp_packet
                                if resp_packet.approved:
                                    break
                                # Rejected: record final and stop runner
                                self.status = RunnerStatus.stopped
                                step.status = StepStatus.stopped
                                # consume resume marker
                                if replaying_history:
                                    _clear_resume_markers()
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
                            # Record this final, store the response on hist, but continue same node with provided user message (not recorded)
                            step.executions.append(final_act)
                            hist.response = resp_packet
                            resp = resp_packet
                            # clear resume marker if we were replaying
                            if replaying_history:
                                _clear_resume_markers()
                            continue
                        # No extra message provided; accept final
                        step.executions.append(final_act)

                    else:
                        # No confirmation/prompt: accept final
                        step.executions.append(final_act)

                    # Final accepted: finish step and transition
                    step.status = StepStatus.finished
                    last_exec_activity = final_act
                    # If we were replaying history, ensure any response on hist is already set (it may be None)
                    # Clear resume marker since we've consumed it
                    if replaying_history:
                        _clear_resume_markers()

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
                    resp = await self._run_tools(req, resp_packet)
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

