
from typing import AsyncIterator, ClassVar, Dict, Optional, Type, List

import asyncio
from vocode.graph import Agent, Node
from vocode.state import Message, RunnerStatus, Task, ToolCallStatus, Step, NodeExecution, StepStatus
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

    def __init__(self, config: Node):
        """Initialize an executor instance with its corresponding Node config."""
        # Configuration object of the corresponding Node (may be a Node subclass)
        self.config = config

    @classmethod
    def register(cls, type_name: str, exec_cls: Type["Executor"]) -> None:
        """Manually register an Executor class under a node type name."""
        cls._registry[type_name] = exec_cls

    @classmethod
    def create_for_node(cls, node: Node) -> "Executor":
        """Create an Executor instance for the given Node using the registry."""
        sub = cls._registry.get(node.type)
        if sub is None:
            raise ValueError(f"No executor registered for node type '{node.type}'")
        return sub(config=node)

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
    def __init__(self, agent, initial_messages: Optional[List[Message]] = None):
        """Prepare the runner with a graph, initial messages, status flags, and per-node executors."""
        self.agent = agent
        self.initial_messages: List[Message] = list(initial_messages or [])
        self.status: RunnerStatus = RunnerStatus.idle
        self._current_exec_task: Optional[asyncio.Task] = None
        self._executors: Dict[str, Executor] = {
            n.name: Executor.create_for_node(n) for n in self.agent.graph.nodes
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

    def _compute_node_transition(self, current_runtime_node, exec_state):
        """
        Decide the next runtime node and the input messages for it based on the executed node's result.
        Returns (next_runtime_node, next_input_messages) or (None, None) if the flow should finish.
        """
        node_model = current_runtime_node.model

        # If the node has no outcomes, we're done
        if not node_model.outcomes:
            return None, None

        # Choose next outcome
        outcome_name = exec_state.outcome_name
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
            msgs = list(exec_state.input_messages)
            if exec_state.output_message is not None:
                msgs.append(exec_state.output_message)
        else:
            msgs = [exec_state.output_message] if exec_state.output_message is not None else []

        return next_runtime_node, msgs

    def _find_runtime_node_by_name(self, name: str):
        """Locate the RuntimeNode by name via BFS from the graph root."""
        root = self.agent.graph.root
        queue = [root]
        while queue:
            rn = queue.pop(0)
            if rn.name == name:
                return rn
            queue.extend(rn.children)
        return None

    def _prepare_resume(self, task: Task):
        """
        Find the last finished step and prepare to resume:
        Returns (runtime_node, base_messages_for_rerun, last_execution, step) or None.
        base_messages_for_rerun = last_execution.input_messages + [last_execution.output_message if present].
        """
        if not task.steps:
            return None

        for step in reversed(task.steps):
            if step.status == StepStatus.finished and step.executions:
                last_exec = step.executions[-1]
                if not last_exec.is_complete:
                    continue
                rn = self._find_runtime_node_by_name(step.node)
                if rn is None:
                    return None
                base_msgs = list(last_exec.input_messages)
                if last_exec.output_message is not None:
                    base_msgs.append(last_exec.output_message)
                return rn, base_msgs, last_exec, step
        return None

    async def run(
        self,
        task: Task,
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
        graph = self.agent.graph

        resume_plan = None
        if prev_status == RunnerStatus.stopped:
            resume_plan = self._prepare_resume(task)

        reuse_step: Optional[Step] = None
        if resume_plan is not None:
            rn, base_msgs, last_exec, last_step = resume_plan
            # Emit the last message as a final_message event to optionally collect user input
            req = ReqFinalMessage(message=last_exec.output_message, outcome_name=last_exec.outcome_name)
            self.status = RunnerStatus.waiting_input
            resume_event = RunEvent(
                node=rn.name,
                execution=last_exec,
                event=req,
                input_requested=True,
            )
            run_input: Optional[RunInput] = (yield resume_event)
            resp = run_input.response if run_input is not None else None

            if resp is not None and resp.kind == PACKET_MESSAGE:
                # Re-run same node using base_msgs + user message and append to the same step
                current_runtime_node = rn
                current_input_messages = list(base_msgs) + [resp.message]
                reuse_step = last_step
                reuse_step.status = StepStatus.running
            else:
                # Proceed to next node as if the final was approved/no response
                next_runtime_node, next_input_messages = self._compute_node_transition(rn, last_exec)
                if next_runtime_node is None:
                    self.status = RunnerStatus.finished
                    return
                current_runtime_node = next_runtime_node
                current_input_messages = next_input_messages
        else:
            current_runtime_node = graph.root
            current_input_messages: List[Message] = list(self.initial_messages)

        while True:
            step: Optional[Step] = reuse_step
            reuse_step = None

            rerun_same_node = True
            while rerun_same_node:
                rerun_same_node = False

                # Start a new execution for this node
                exec_state = NodeExecution(input_messages=list(current_input_messages))

                executor = self._executors[current_runtime_node.name]
                agen = executor.run(exec_state.input_messages)

                to_send: Optional[RespPacket] = None
                user_message_for_rerun: Optional[Message] = None

                while True:
                    try:
                        self._current_exec_task = asyncio.create_task(agen.asend(to_send))
                        req: ReqPacket = await self._current_exec_task
                        to_send = None
                    except StopAsyncIteration:
                        # Executor ended without explicit final; mark as completed
                        exec_state.is_complete = True
                        break
                    except asyncio.CancelledError:
                        # Runner.cancel() interrupted the in-flight executor
                        exec_state.is_canceled = True
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
                        exec_state.output_message = req.message

                    input_requested = req.kind in (
                        PACKET_MESSAGE_REQUEST,
                        PACKET_TOOL_CALL,
                        PACKET_FINAL_MESSAGE,
                    )
                    self.status = RunnerStatus.waiting_input if input_requested else RunnerStatus.running

                    # Emit event and await optional input
                    run_event = RunEvent(
                        node=current_runtime_node.name,
                        execution=exec_state,
                        event=req,
                        input_requested=input_requested,
                    )
                    run_input: Optional[RunInput] = (yield run_event)
                    resp = run_input.response if run_input is not None else None

                    # Default: no response back to executor unless specified below
                    to_send = None

                    # Handle executor requests
                    if req.kind == PACKET_FINAL_MESSAGE:
                        # Record final result of this execution
                        exec_state.output_message = req.message
                        exec_state.outcome_name = req.outcome_name
                        exec_state.is_complete = True

                        # Approval handling: None or approved=True => success => proceed to next node
                        if resp is None or (resp.kind == PACKET_APPROVAL and resp.approved):
                            # No response expected back to executor
                            break

                        # If a user message was provided instead, re-run same node with that message
                        if resp is not None and resp.kind == PACKET_MESSAGE:
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
                        to_send = resp  # guaranteed RespMessage here
                        continue

                    else:
                        # For other request kinds (e.g., interim messages), send nothing
                        to_send = None
                        continue

                # Inner processing loop (this execution) is done:
                # Persist only finalized executions
                if exec_state.is_complete:
                    if step is None:
                        step = Step(node=current_runtime_node.name, status=StepStatus.running)
                        task.steps.append(step)
                    step.executions.append(exec_state)

                # If we got a user-provided message to re-run the same node, do so
                if user_message_for_rerun is not None:
                    # Re-run same node using input_messages + output_message + RespMessage.message
                    msgs = list(exec_state.input_messages)
                    if exec_state.output_message is not None:
                        msgs.append(exec_state.output_message)
                    msgs.append(user_message_for_rerun)
                    current_input_messages = msgs
                    rerun_same_node = True
                    continue

                # NEW: mark this step as finished (no rerun for this node)
                if step is not None:
                    step.status = StepStatus.finished

                next_runtime_node, next_input_messages = self._compute_node_transition(
                    current_runtime_node, exec_state
                )
                if next_runtime_node is None:
                    self.status = RunnerStatus.finished
                    return

                current_input_messages = next_input_messages
                current_runtime_node = next_runtime_node

                break
