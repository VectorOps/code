
from __future__ import annotations

from typing import Dict, Optional, List, AsyncIterator, Type, ClassVar, Tuple
import asyncio
from contextlib import suppress
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

    async def run(self, messages: List[Message]) -> AsyncIterator[NodeExecution]:
        """Async generator: yield intermediate NodeExecution updates; close when complete."""
        raise NotImplementedError("Executor subclasses must implement 'run' as an async generator")


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

    async def _drive_executor_stream(
        self,
        executor: Executor,
        execution: NodeExecution,
        queue: "asyncio.Queue[Optional[NodeExecution]]",
    ) -> None:
        """Run executor async generator, update execution in place, and enqueue snapshots with input_messages."""
        agen = executor.run(execution.input_messages)
        try:
            async for part in agen:
                if part.messages:
                    execution.messages = part.messages
                if part.output_name is not None:
                    execution.output_name = part.output_name
                await queue.put(
                    NodeExecution(
                        id=execution.id,
                        input_messages=list(execution.input_messages),
                        messages=list(execution.messages),
                        output_name=execution.output_name,
                        is_complete=False,
                        is_canceled=execution.is_canceled,
                    )
                )
        except asyncio.CancelledError:
            with suppress(Exception):
                await agen.aclose()
            execution.is_canceled = True
            # runner will handle cancellation state; just finish
            return
        else:
            execution.is_complete = True
            # Do not enqueue a final snapshot; runner will prompt for input or proceed to next node.
        finally:
            await queue.put(None)

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

            # Unified execution loop for this node
            while True:
                exec_input = list(messages)

                execution = NodeExecution(input_messages=exec_input, messages=list(messages))
                # Only record execution after first yield
                execution_recorded = False

                stream_queue: "asyncio.Queue[Optional[NodeExecution]]" = asyncio.Queue()
                self._current_exec_task = asyncio.create_task(
                    self._drive_executor_stream(executor, execution, stream_queue)
                )

                incoming_after_final: Optional[RunInput] = None
                try:
                    while True:
                        part = await stream_queue.get()
                        if part is None:
                            break

                        if not execution_recorded:
                            current_step.executions.append(execution)
                            execution_recorded = True

                        incoming = yield RunEvent(node=current.name, execution=part)
                        if not isinstance(incoming, RunInput):
                            raise TypeError(f"Runner.run expects RunInput after RunEvent; got {type(incoming).__name__}")

                        if self._stop_requested:
                            self._current_exec_task.cancel()
                            with suppress(Exception):
                                await self._current_exec_task
                            self._current_exec_task = None
                            self.status = RunnerStatus.stopped
                            return

                        incoming_after_final = incoming
                finally:
                    if self._current_exec_task and self._current_exec_task.done():
                        self._current_exec_task = None

                if self._cancel_requested:
                    self.status = RunnerStatus.canceled
                    return

                messages = execution.messages
                incoming = incoming_after_final or RunInput()
                if incoming.loop:
                    if incoming.messages:
                        messages = messages + incoming.messages
                    continue  # re-run same node (new execution instance will be created)

                if execution.output_name is None:
                    self.status = RunnerStatus.waiting_input
                    incoming = yield RunEvent(node=current.name, need_input=True)
                    self.status = RunnerStatus.running
                    if not isinstance(incoming, RunInput):
                        raise TypeError(f"Runner.run expects RunInput after RunEvent; got {type(incoming).__name__}")
                    if incoming.messages:
                        messages = messages + incoming.messages
                    continue  # re-run same node after collecting input

                break  # finished this node; proceed to next

            # Decide next node
            if len(current.outputs) == 0:
                self.status = RunnerStatus.finished
                break

            next_node = current.get_child_by_output(execution.output_name)  # type: ignore[arg-type]
            if next_node is None:
                self.status = RunnerStatus.finished
                break

            # Pass messages to the next node and move forward
            if current.model.pass_all_messages:
                messages = execution.messages
            else:
                messages = execution.messages[-1:]

            current = next_node
            # Force a new step for the next node
            current_step = None
