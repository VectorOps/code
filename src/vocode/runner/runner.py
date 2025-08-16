
from __future__ import annotations

from typing import Dict, Optional, List, AsyncIterator, Type, ClassVar

from ..graph.graph import Graph, RuntimeNode
from ..graph.models import Node
from ..state import Message, NodeExecution, Step, Task, RunEvent, RunInput


class Executor:
    # Subclasses must set 'type' to the Node.type they handle
    type: ClassVar[Optional[str]] = None
    _registry: ClassVar[Dict[str, Type["Executor"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        t = getattr(cls, "type", None)
        if isinstance(t, str) and t:
            Executor._registry[t] = cls

    def __init__(self, config: Node):
        # Configuration object of the corresponding Node (may be a Node subclass)
        self.config = config

    @classmethod
    def register(cls, type_name: str, exec_cls: Type["Executor"]) -> None:
        cls._registry[type_name] = exec_cls

    @classmethod
    def create_for_node(cls, node: Node) -> "Executor":
        sub = cls._registry.get(node.type)
        if sub is None:
            raise ValueError(f"No executor registered for node type '{node.type}'")
        return sub(config=node)

    async def run(self, messages: List[Message]) -> NodeExecution:
        # Must be implemented by subclasses.
        # Receives accumulated messages and returns a NodeExecution
        # with updated messages; output_name signals readiness to move on.
        raise NotImplementedError("Executor subclasses must implement 'run'")


class Runner:
    def __init__(self, graph: Graph):
        self.graph = graph
        # Construct individual executor instances per node
        self._executors: Dict[str, Executor] = {
            n.name: Executor.create_for_node(n) for n in self.graph.nodes
        }

    def executor_for(self, node_name: str) -> Executor:
        return self._executors[node_name]

    async def run(
        self,
        task: Task,
        initial_messages: List[Message],
    ) -> AsyncIterator[RunEvent]:
        """
        Async generator that:
          - starts from the graph root
          - executes nodes sequentially following output_name -> edge mapping
          - yields RunEvent(node=..., execution=NodeExecution) after each execution
          - After every yield, the caller MUST resume with RunInput; use loop=True to re-run the same node
            (optionally providing messages), or loop=False to proceed.
          - Retry loop if node requests more input (no output_name) OR caller explicitly sets loop=True
          - stops when a terminal node (no outputs) has been executed
        """
        current: RuntimeNode = self.graph.root
        messages: List[Message] = list(initial_messages)

        while True:
            executor = self.executor_for(current.name)
            # Create a new step for this node; all executions for this node append here
            step = Step()
            task.steps.append(step)

            # Run current node once
            execution = await executor.run(messages)

            messages = execution.messages

            # Log under the current step for this node
            step.executions.append(execution)

            incoming = yield RunEvent(node=current.name, execution=execution)
            if not isinstance(incoming, RunInput):
                raise TypeError(f"Runner.run expects RunInput after RunEvent; got {type(incoming).__name__}")

            # Retry loop if node did not produce output OR caller explicitly requested a loop
            while (execution.output_name is None) or incoming.loop:
                if incoming.loop:
                    if incoming.messages:
                        messages = messages + incoming.messages

                    execution = await executor.run(messages)

                    messages = execution.messages
                    step.executions.append(execution)

                    incoming = yield RunEvent(node=current.name, execution=execution)
                    if not isinstance(incoming, RunInput):
                        raise TypeError(f"Runner.run expects RunInput after RunEvent; got {type(incoming).__name__}")

                    continue

                # No explicit loop requested
                if execution.output_name is None:
                    # Ask the caller to provide more input for the same node
                    incoming = yield RunEvent(node=current.name, need_input=True)
                    if not isinstance(incoming, RunInput):
                        raise TypeError(f"Runner.run expects RunInput after RunEvent; got {type(incoming).__name__}")
                else:
                    # We have an output and no loop requested; exit retry loop
                    break

            # Decide next node by output_name
            if len(current.outputs) == 0:
                # Terminal node reached and executed
                break

            next_node = current.get_child_by_output(execution.output_name)  # type: ignore[arg-type]
            if next_node is None:
                # Should not happen if graph is valid; treat as terminal
                break

            # Pass the returned messages to the next node
            if getattr(current.model, "pass_all_messages", True):
                messages = execution.messages
            else:
                # Only pass the last message (or none if there are no messages)
                messages = execution.messages[-1:]
            current = next_node
