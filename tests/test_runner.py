import pytest
import asyncio

from vocode.graph.graph import Graph
from vocode.graph.models import Node, OutputSlot, Edge
from vocode.runner.runner import Runner, Executor
from typing import List, AsyncIterator
from vocode.state import Message, Task, NodeExecution, RunEvent, RunInput, RunnerStatus


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

    async def run(self, messages: List[Message]) -> AsyncIterator[NodeExecution]:
        has_go = any(m.raw == "go" for m in messages)
        if not has_go:
            # Request more input
            yield NodeExecution(messages=list(messages), output_name=None)
            return
        # Proceed to next node once "go" is present
        messages_out = list(messages) + [Message(role="agent", raw="ack:go")]
        yield NodeExecution(messages=messages_out, output_name="done")


class EchoExecutor(Executor):
    type = "echo"

    async def run(self, messages: List[Message]) -> AsyncIterator[NodeExecution]:
        # Terminal node — set some output_name so Runner won't prompt for more input
        messages_out = list(messages) + [Message(role="agent", raw="echo")]
        yield NodeExecution(messages=messages_out, output_name="done")


class SleepNode(Node):
    type: str = "sleep"

class SleepExecutor(Executor):
    type = "sleep"

    async def run(self, messages: List[Message]) -> AsyncIterator[NodeExecution]:
        # Simulate long-running work to allow cancellation
        await asyncio.sleep(0.2)
        out = list(messages) + [Message(role="agent", raw="slept")]
        yield NodeExecution(messages=out, output_name="done")


@pytest.mark.asyncio
async def test_runner_async_flow_and_callback():
    # Build graph: AskNode (root) -> EchoNode (terminal)
    ask = AskNode(name="ask", outputs=[OutputSlot(name="done")])
    echo = EchoNode(name="echo")  # terminal (no outputs)
    graph = Graph.build(
        nodes=[ask, echo],
        edges=[Edge(source_node="ask", source_slot="done", target_node="echo")],
    )

    runner = Runner(graph, initial_messages=[Message(role="user", raw="hi")])
    # Sanity: executors were created from the registry
    assert type(runner.executor_for("ask")) is AskExecutor
    assert type(runner.executor_for("echo")) is EchoExecutor

    task = Task()

    gen = runner.run(task=task)

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


@pytest.mark.asyncio
async def test_runner_cancel_mid_execution_and_resume():
    # Build graph: SleepNode -> EchoNode (terminal)
    sleep = SleepNode(name="sleep", outputs=[OutputSlot(name="done")])
    echo = EchoNode(name="echo")
    graph = Graph.build(
        nodes=[sleep, echo],
        edges=[Edge(source_node="sleep", source_slot="done", target_node="echo")],
    )
    runner = Runner(graph, initial_messages=[Message(role="user", raw="init")])
    task = Task()
    gen = runner.run(task=task)
    # Start first execution and cancel while running
    first_event_fut = asyncio.create_task(gen.__anext__())
    await asyncio.sleep(0.01)
    runner.cancel()
    with pytest.raises(StopAsyncIteration):
        await first_event_fut
    assert runner.status == RunnerStatus.canceled
    # Step for 'sleep' should exist but contain no executions
    assert len(task.steps) == 1
    assert task.steps[-1].node == "sleep"
    assert len(task.steps[-1].executions) == 0
    # Resume run from the same node by providing initial messages again
    gen2 = runner.run(task=task)
    evt1 = await gen2.__anext__()
    assert evt1.node == "sleep"
    assert evt1.execution is not None
    assert [m.raw for m in evt1.execution.input_messages] == ["init"]
    assert evt1.execution.output_name == "done"
    # Proceed to echo
    evt2 = await gen2.asend(RunInput())
    assert evt2.node == "echo"
    assert evt2.execution is not None
    assert evt2.execution.output_name == "done"
    with pytest.raises(StopAsyncIteration):
        await gen2.asend(RunInput())
    assert runner.status == RunnerStatus.finished


@pytest.mark.asyncio
async def test_runner_stop_and_resume_next_node():
    ask = AskNode(name="ask", outputs=[OutputSlot(name="done")])
    echo = EchoNode(name="echo")
    graph = Graph.build(
        nodes=[ask, echo],
        edges=[Edge(source_node="ask", source_slot="done", target_node="echo")],
    )
    runner = Runner(graph, initial_messages=[Message(role="user", raw="go")])
    task = Task()
    gen = runner.run(task=task)
    evt1 = await gen.__anext__()
    assert evt1.node == "ask"
    assert evt1.execution is not None and evt1.execution.output_name == "done"
    # Request stop before proceeding to the next node
    runner.stop()
    with pytest.raises(StopAsyncIteration):
        await gen.asend(RunInput())
    assert runner.status == RunnerStatus.stopped
    # Resume; should continue at 'echo'
    gen2 = runner.run(task=task)
    evt2 = await gen2.__anext__()
    assert evt2.node == "echo"
    assert evt2.execution is not None and evt2.execution.output_name == "done"
    with pytest.raises(StopAsyncIteration):
        await gen2.asend(RunInput())
    assert runner.status == RunnerStatus.finished
    # History: one step per node
    assert [s.node for s in task.steps] == ["ask", "echo"]


@pytest.mark.asyncio
async def test_runner_rollback_current_step_and_steps():
    ask = AskNode(name="ask", outputs=[OutputSlot(name="done")])
    echo = EchoNode(name="echo")
    graph = Graph.build(
        nodes=[ask, echo],
        edges=[Edge(source_node="ask", source_slot="done", target_node="echo")],
    )
    runner = Runner(graph, initial_messages=[Message(role="user", raw="hi")])
    task = Task()
    gen = runner.run(task=task)
    # First run: ask needs input
    evt1 = await gen.__anext__()
    assert evt1.node == "ask" and evt1.execution is not None and evt1.execution.output_name is None
    # Provide 'go' -> done
    evt2 = await gen.asend(RunInput(loop=True, messages=[Message(role="user", raw="go")]))
    assert evt2.node == "ask" and evt2.execution is not None and evt2.execution.output_name == "done"
    # Explicit extra loop -> another execution on same step
    evt3 = await gen.asend(RunInput(loop=True, messages=[Message(role="user", raw="more")]))
    assert evt3.node == "ask" and evt3.execution is not None and evt3.execution.output_name == "done"
    # Stop before proceeding to the next node
    runner.stop()
    with pytest.raises(StopAsyncIteration):
        await gen.asend(RunInput())  # attempt to proceed
    assert runner.status == RunnerStatus.stopped
    # Roll back the current step (ask) to its beginning
    runner.rollback_current_step(task)
    assert len(task.steps) == 1 and task.steps[0].node == "ask" and len(task.steps[0].executions) == 0
    # Resume from start of step with fresh input
    runner2 = Runner(graph, initial_messages=[Message(role="user", raw="fresh")])
    gen2 = runner2.run(task=task)
    evt4 = await gen2.__anext__()
    assert evt4.node == "ask" and evt4.execution is not None and evt4.execution.output_name is None
    assert [m.raw for m in evt4.execution.input_messages] == ["fresh"]
    # Provide go to finish ask, then proceed to echo
    evt5 = await gen2.asend(RunInput(loop=True, messages=[Message(role="user", raw="go")]))
    assert evt5.execution is not None and evt5.execution.output_name == "done"
    evt6 = await gen2.asend(RunInput())
    assert evt6.node == "echo" and evt6.execution is not None and evt6.execution.output_name == "done"
    with pytest.raises(StopAsyncIteration):
        await gen2.asend(RunInput())
    assert [s.node for s in task.steps] == ["ask", "echo"]
    # Now roll back the last step (echo) entirely, and re-run it
    runner.rollback_steps(task, task.steps[0].id)
    assert [s.node for s in task.steps] == ["ask"]
    gen3 = runner.run(task=task)
    evt7 = await gen3.__anext__()
    assert evt7.node == "echo" and evt7.execution is not None and evt7.execution.output_name == "done"
    with pytest.raises(StopAsyncIteration):
        await gen3.asend(RunInput())
    assert [s.node for s in task.steps] == ["ask", "echo"]
