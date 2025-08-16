
from __future__ import annotations

from typing import Dict, Optional, List, AsyncIterator, Type, ClassVar, Tuple
import asyncio
from uuid import UUID

from ..graph.graph import Graph, RuntimeNode
from ..graph.models import Node
from ..state import Message, NodeExecution, Step, Task, RunEvent, RunInput, RunnerStatus


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

    async def run(self, messages: List[Message]) -> NodeExecution:
        """Execute node logic with messages and return a NodeExecution (override in subclasses)."""
        # Must be implemented by subclasses.
        # Receives accumulated messages and returns a NodeExecution
        # with updated messages; output_name signals readiness to move on.
        raise NotImplementedError("Executor subclasses must implement 'run'")


class Runner:
    def __init__(self, graph: Graph, initial_messages: Optional[List[Message]] = None):
        """Prepare the runner with a graph, initial messages, status flags, and per-node executors."""
        self.graph = graph
        self.initial_messages: List[Message] = list(initial_messages or [])
        self.status: RunnerStatus = RunnerStatus.idle
        self._current_exec_task: Optional[asyncio.Task] = None
        self._cancel_requested: bool = False
        self._stop_requested: bool = False
        # Construct individual executor instances per node
        self._executors: Dict[str, Executor] = {
            n.name: Executor.create_for_node(n) for n in self.graph.nodes
        }

    def executor_for(self, node_name: str) -> Executor:
        """Return the executor instance for the specified node name."""
        return self._executors[node_name]

    def cancel(self) -> None:
        """Request cancellation of the currently running execution task, if any."""
        if self._current_exec_task and not self._current_exec_task.done():
            self._cancel_requested = True
            self._current_exec_task.cancel()
        else:
            self._cancel_requested = True

    def stop(self) -> None:
        """Request a graceful stop after the current yield point."""
        self._stop_requested = True

    def rollback_current_step(self, task: Task) -> None:
        """Reset the most recent step by clearing its executions to restart it."""
        if not task.steps:
            return
        step = task.steps[-1]
        if not step.executions:
            return
        # Determine the earliest input to restart this step from scratch
        first_in = list(step.executions[0].input_messages)
        step.executions.clear()
        # Preserve the original starting input for the step as a hint by seeding a placeholder execution
        # The real run will overwrite with actual executions; we keep no placeholder to avoid confusion.
        # Instead, rely on caller passing initial_messages=None to resume; we’ll use this cleared state.

    def rollback_steps(self, task: Task, to_step_id: UUID) -> None:
        """Remove all steps after the specified step (keep the step with to_step_id)."""
        idx = next((i for i, s in enumerate(task.steps) if s.id == to_step_id), -1)
        if idx < 0:
            raise ValueError(f"Step id {to_step_id} not found")
        del task.steps[idx + 1:]

    def _find_runtime_node_by_name(self, name: str) -> RuntimeNode:
        """Find a RuntimeNode by name via DFS; raise KeyError if not found."""
        root = self.graph.root
        stack = [root]
        while stack:
            rn = stack.pop()
            if rn.name == name:
                return rn
            stack.extend(rn.children)
        raise KeyError(f"Runtime node '{name}' not found")

    def _resolve_resume_point(
        self, task: Task
    ) -> Tuple["RuntimeNode", List[Message], Optional[Step]]:
        """Determine next RuntimeNode, messages to use, and current step based on task history."""
        # No history: start at root
        if not task.steps:
            start_messages: List[Message] = list(self.initial_messages or [])
            return (self.graph.root, start_messages, None)
        # Has history: use the last step
        step = task.steps[-1]
        current = self._find_runtime_node_by_name(step.node)
        if not step.executions:
            # Step exists but was reset (rollback to start); use runner's initial_messages
            start_messages = list(self.initial_messages or [])
            return (current, start_messages, step)
        last_exec = step.executions[-1]
        # If the node still needs input, stay on the same node
        if last_exec.output_name is None:
            return (current, list(last_exec.messages), step)
        # Otherwise, proceed to the next node based on output_name
        next_node = current.get_child_by_output(last_exec.output_name)  # type: ignore[arg-type]
        if next_node is None:
            # Terminal reached previously; resume has nothing to do from here
            return (current, list(last_exec.messages), step)
        if getattr(current.model, "pass_all_messages", True):
            msgs = list(last_exec.messages)
        else:
            msgs = list(last_exec.messages[-1:])
        return (next_node, msgs, None)

    async def run(
        self,
        task: Task,
    ) -> AsyncIterator[RunEvent]:
        """Async generator that executes the graph from current task state; yields RunEvent and expects RunInput after each yield. Supports loop re-runs, input prompts, stop/cancel, and records task history."""
        self._cancel_requested = False
        self._stop_requested = False
        self.status = RunnerStatus.running
        current, messages, current_step = self._resolve_resume_point(task)
        while True:
            if self._stop_requested:
                self.status = RunnerStatus.stopped
                break
            executor = self.executor_for(current.name)
            # Create or reuse step for this node
            if current_step is None or current_step.node != current.name:
                current_step = Step(node=current.name)
                task.steps.append(current_step)
            # Capture input for this execution
            exec_input = list(messages)
            # Run with cancellation support
            self._current_exec_task = asyncio.create_task(executor.run(messages))
            try:
                execution = await self._current_exec_task
            except asyncio.CancelledError:
                self._current_exec_task = None
                self.status = RunnerStatus.canceled
                break
            finally:
                self._current_exec_task = None
            # Persist input and update messages from result
            execution.input_messages = exec_input
            messages = execution.messages
            # Log under current step
            current_step.executions.append(execution)
            # Yield event and possibly request input
            evt = RunEvent(node=current.name, execution=execution)
            incoming = yield evt
            if not isinstance(incoming, RunInput):
                raise TypeError(f"Runner.run expects RunInput after RunEvent; got {type(incoming).__name__}")
            # Retry loop: explicit loop or missing output
            while (execution.output_name is None) or incoming.loop:
                if self._stop_requested:
                    self.status = RunnerStatus.stopped
                    return
                if incoming.loop:
                    if incoming.messages:
                        messages = messages + incoming.messages
                    # Re-run
                    exec_input = list(messages)
                    self._current_exec_task = asyncio.create_task(executor.run(messages))
                    try:
                        execution = await self._current_exec_task
                    except asyncio.CancelledError:
                        self._current_exec_task = None
                        self.status = RunnerStatus.canceled
                        return
                    finally:
                        self._current_exec_task = None
                    execution.input_messages = exec_input
                    messages = execution.messages
                    current_step.executions.append(execution)
                    incoming = yield RunEvent(node=current.name, execution=execution)
                    if not isinstance(incoming, RunInput):
                        raise TypeError(f"Runner.run expects RunInput after RunEvent; got {type(incoming).__name__}")
                    continue
                # No explicit loop; if missing output, request input
                if execution.output_name is None:
                    self.status = RunnerStatus.waiting_input
                    incoming = yield RunEvent(node=current.name, need_input=True)
                    self.status = RunnerStatus.running
                    if not isinstance(incoming, RunInput):
                        raise TypeError(f"Runner.run expects RunInput after RunEvent; got {type(incoming).__name__}")
                else:
                    break
            # Decide next node
            if len(current.outputs) == 0:
                self.status = RunnerStatus.finished
                break
            next_node = current.get_child_by_output(execution.output_name)  # type: ignore[arg-type]
            if next_node is None:
                self.status = RunnerStatus.finished
                break
            # Pass messages to the next node and move forward
            if getattr(current.model, "pass_all_messages", True):
                messages = execution.messages
            else:
                messages = execution.messages[-1:]
            current = next_node
            # Force a new step for the next node
            current_step = None
