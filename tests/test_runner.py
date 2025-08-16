import pytest

from vocode.graph.graph import Graph
from vocode.graph.models import Node, OutputSlot, Edge
from vocode.runner.runner import Runner, Executor
from typing import List
from vocode.state import Message, Task, NodeExecution, RunEvent, RunInput


# Define Node subclasses to be used in the test
class AskNode(Node):
    # Requests more input until it sees a "go" message
    type: str = "ask"


class EchoNode(Node):
    # Terminal node
    type: str = "echo"


# Define matching Executors registered by type
class AskExecutor(Executor):
    type = "ask"

    async def run(self, messages: List[Message]) -> NodeExecution:
        has_go = any(m.raw == "go" for m in messages)
        if not has_go:
            # Request more input
            return NodeExecution(messages=list(messages), output_name=None)
        # Proceed to next node once "go" is present
        messages_out = list(messages) + [Message(role="agent", raw="ack:go")]
        return NodeExecution(messages=messages_out, output_name="done")


class EchoExecutor(Executor):
    type = "echo"

    async def run(self, messages: List[Message]) -> NodeExecution:
        # Terminal node — set some output_name so Runner won't prompt for more input
        messages_out = list(messages) + [Message(role="agent", raw="echo")]
        return NodeExecution(messages=messages_out, output_name="done")


@pytest.mark.asyncio
async def test_runner_async_flow_and_callback():
    # Build graph: AskNode (root) -> EchoNode (terminal)
    ask = AskNode(name="ask", outputs=[OutputSlot(name="done")])
    echo = EchoNode(name="echo")  # terminal (no outputs)
    graph = Graph.build(
        nodes=[ask, echo],
        edges=[Edge(source_node="ask", source_slot="done", target_node="echo")],
    )

    runner = Runner(graph)
    # Sanity: executors were created from the registry
    assert type(runner.executor_for("ask")) is AskExecutor
    assert type(runner.executor_for("echo")) is EchoExecutor

    task = Task()
    initial = [Message(role="user", raw="hi")]

    gen = runner.run(task=task, initial_messages=initial)

    # 1) First execution of ask node: no "go" yet => needs more input
    evt1 = await gen.__anext__()
    assert evt1.node == "ask"
    assert evt1.execution is not None
    assert evt1.execution.output_name is None
    assert [m.raw for m in evt1.execution.messages] == ["hi"]

    # 2) Demonstrate callback prompt by not providing input yet
    prompt = await gen.asend(RunInput())  # loop defaults to False
    assert isinstance(prompt, RunEvent)
    assert prompt.need_input is True
    assert prompt.node == "ask"
    assert prompt.execution is None

    # 3) Provide additional input to retry the same node
    evt2 = await gen.asend(RunInput(loop=True, messages=[Message(role="user", raw="go")]))
    assert evt2.execution is not None
    assert evt2.execution.output_name == "done"
    # Messages include the ack added by AskExecutor
    assert [m.raw for m in evt2.execution.messages] == ["hi", "go", "ack:go"]

    # 3b) Even if output was provided, we can explicitly request another loop on the same node
    evt3 = await gen.asend(RunInput(loop=True, messages=[Message(role="user", raw="more")]))
    assert evt3.node == "ask"
    assert evt3.execution is not None
    assert evt3.execution.output_name == "done"
    assert [m.raw for m in evt3.execution.messages] == ["hi", "go", "ack:go", "more", "ack:go"]

    # 4) Next node (echo) executes and terminates the run
    evt4 = await gen.asend(RunInput())  # proceed without looping
    assert evt4.node == "echo"
    assert evt4.execution is not None
    assert evt4.execution.output_name == "done"
    assert evt4.execution.messages[-1].raw == "echo"

    # 5) Runner should now be finished
    with pytest.raises(StopAsyncIteration):
        await gen.asend(RunInput())

    # 6) Task logging: one step per node (Ask has 3 executions due to explicit loop, Echo has 1)
    assert len(task.steps) == 2
    assert len(task.steps[0].executions) == 3  # Ask node: first run + re-run + explicit loop after done
    assert len(task.steps[1].executions) == 1  # Echo node: single terminal run
