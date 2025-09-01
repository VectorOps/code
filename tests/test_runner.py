import asyncio
import pytest
from pathlib import Path
from vocode.project import Project
from vocode.graph import Graph, Node, Edge, OutcomeSlot, Workflow
from vocode.runner.runner import Runner, Executor
from vocode.runner.models import (
    ReqMessageRequest,
    ReqToolCall,
    ReqInterimMessage,
    ReqFinalMessage,
    RespMessage,
    RespApproval,
    RunInput,
)
from vocode.state import Message, ToolCall, ToolCallStatus, Assignment


def msg(role: str, text: str) -> Message:
    return Message(role=role, text=text)


@pytest.mark.asyncio
async def test_message_request_reprompt_and_finish():
    # Single-node graph with type 'ask'
    nodes = [{"name": "Ask", "type": "ask", "outcomes": []}]
    g = Graph.build(nodes=nodes, edges=[])
    workflow = Workflow(name="workflow", graph=g)

    # Executor that requires a message, then emits interim and final
    class AskExecutor(Executor):
        type = "ask"

        async def run(self, messages):
            # Request a message
            resp = yield ReqMessageRequest()
            assert resp.kind == "message"  # runner should re-prompt until message is provided
            # Emit an interim message (no input expected)
            interim = msg("agent", "got it")
            r = yield ReqInterimMessage(message=interim)
            assert r is None
            # Finish
            final = msg("agent", f"final:{resp.message.text}")
            yield ReqFinalMessage(message=final)

    project = Project(base_path=Path("."))
    runner = Runner(workflow, project)
    task = Assignment()
    it = runner.run(task)

    # First: message_request
    ev1 = await it.__anext__()
    assert ev1.node == "Ask"
    assert ev1.event.kind == "message_request"
    assert ev1.input_requested is True

    # Send None to trigger re-prompt
    ev2 = await it.asend(RunInput(response=None))
    # Should be the same event object (reused)
    assert ev2 is ev1

    # Now provide a message
    user_msg = msg("user", "hi")
    ev3 = await it.asend(RunInput(response=RespMessage(message=user_msg)))
    assert ev3.event.kind == "message"
    assert ev3.input_requested is False
    # Execution output should reflect interim message
    assert ev3.execution.message is not None
    assert ev3.execution.message.text == "got it"

    # Final message
    ev4 = await it.__anext__()
    assert ev4.event.kind == "final_message"
    assert ev4.input_requested is True
    # Implicit approval by sending no response (None)
    with pytest.raises(StopAsyncIteration):
        await it.asend(None)

    # Runner state and task state
    assert runner.status.name == "finished"
    assert len(task.steps) == 1
    step = task.steps[0]
    assert step.node == "Ask"
    # Three activities: user input, interim, final
    assert len(step.executions) == 3
    assert step.executions[0].type.value == "user"
    assert step.executions[0].message.text == "hi"
    assert step.executions[1].type.value == "executor"
    assert step.executions[1].message.text == "got it"
    assert step.executions[2].type.value == "executor"
    assert step.executions[2].is_complete is True
    assert step.executions[2].message is not None
    assert step.executions[2].message.text == "final:hi"


@pytest.mark.asyncio
async def test_tool_call_approved_and_rejected():
    nodes = [{"name": "Tool", "type": "tool", "outcomes": []}]
    g = Graph.build(nodes=nodes, edges=[])
    workflow = Workflow(name="workflow", graph=g)

    class ToolExecutor(Executor):
        type = "tool"

        async def run(self, messages):
            # Approved tool call
            tc1 = ToolCall(name="one", arguments="{}")
            resp1 = yield ReqToolCall(tool_calls=[tc1])
            assert resp1.kind == "tool_call"
            assert len(resp1.tool_calls) == 1
            assert resp1.tool_calls[0].status == ToolCallStatus.completed
            assert resp1.tool_calls[0].result == "{}"
            # Rejected tool call
            tc2 = ToolCall(name="two", arguments="{}")
            resp2 = yield ReqToolCall(tool_calls=[tc2])
            assert resp2.kind == "tool_call"
            assert resp2.tool_calls[0].status == ToolCallStatus.rejected
            assert resp2.tool_calls[0].result is None
            # Finish
            yield ReqFinalMessage(message=msg("agent", "done"))

    project = Project(base_path=Path("."))
    runner = Runner(workflow, project)
    task = Assignment()
    it = runner.run(task)

    # First tool call (approve)
    ev1 = await it.__anext__()
    assert ev1.event.kind == "tool_call"
    # Respond and capture the next event (this returns the next yielded request)
    ev2 = await it.asend(RunInput(response=RespApproval(approved=True)))
    # After approval the first req object should be updated
    assert ev1.event.tool_calls[0].status == ToolCallStatus.completed
    assert ev1.event.tool_calls[0].result == "{}"

    # Second tool call (reject)
    assert ev2.event.kind == "tool_call"
    ev3 = await it.asend(RunInput(response=RespApproval(approved=False)))
    assert ev2.event.tool_calls[0].status == ToolCallStatus.rejected
    assert ev2.event.tool_calls[0].result is None

    # Finalize (returned by the previous asend)
    assert ev3.event.kind == "final_message"
    with pytest.raises(StopAsyncIteration):
        await it.asend(None)

    # State
    assert runner.status.name == "finished"
    assert len(task.steps) == 1
    ex = task.steps[0].executions[0]
    assert ex.is_complete is True
    assert ex.message.text == "done"


@pytest.mark.asyncio
async def test_final_message_rerun_same_node_with_user_message():
    nodes = [{"name": "Echo", "type": "echo", "outcomes": []}]
    g = Graph.build(nodes=nodes, edges=[])
    workflow = Workflow(name="workflow", graph=g)

    class EchoExecutor(Executor):
        type = "echo"

        async def run(self, messages):
            # First run: no inputs; emit final to trigger rerun
            if len(messages) == 0:
                yield ReqFinalMessage(message=msg("agent", "ask"))
            else:
                # Rerun should pass previous output + user message
                assert len(messages) == 2
                assert messages[0].role in ("agent", "tool")
                assert messages[0].text == "ask"
                assert messages[1].role == "user"
                assert messages[1].text == "more"
                yield ReqFinalMessage(message=msg("agent", "done"))

    project = Project(base_path=Path("."))
    runner = Runner(workflow, project)
    task = Assignment()
    it = runner.run(task)

    # First: final_message (requesting input)
    ev1 = await it.__anext__()
    assert ev1.event.kind == "final_message"
    # Provide a user message to trigger rerun
    ev2 = await it.asend(RunInput(response=RespMessage(message=msg("user", "more"))))
    # Now we are in second execution; final again
    assert ev2.event.kind == "final_message"
    with pytest.raises(StopAsyncIteration):
        await it.asend(None)

    # Verify two executions within the same step
    assert runner.status.name == "finished"
    assert len(task.steps) == 1
    step = task.steps[0]
    # Activities: final 'ask', user 'more', final 'done'
    assert len(step.executions) == 3
    assert step.executions[0].type.value == "executor"
    assert step.executions[0].message.text == "ask"
    assert step.executions[1].type.value == "user"
    assert step.executions[1].message.text == "more"
    assert step.executions[2].type.value == "executor"
    assert step.executions[2].is_complete is True
    assert step.executions[2].message.text == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("pass_all", [True, False])
async def test_transition_to_next_node_and_pass_all_messages(pass_all):
    nodes = [
        {"name": "A", "type": "a", "outcomes": [OutcomeSlot(name="toB")], "pass_all_messages": pass_all},
        {"name": "B", "type": "b", "outcomes": []},
    ]
    edges = [Edge(source_node="A", source_outcome="toB", target_node="B")]
    g = Graph.build(nodes=nodes, edges=edges)
    workflow = Workflow(name="workflow", graph=g)

    class AExec(Executor):
        type = "a"

        async def run(self, messages):
            # Emit final with outcome to go to B
            yield ReqFinalMessage(message=msg("agent", "Aout"), outcome_name="toB")

    class BExec(Executor):
        type = "b"

        async def run(self, messages):
            # Expect messages based on pass_all flag and initial messages
            if pass_all:
                assert len(messages) == 2  # initial system + A's output
                assert messages[0].role == "system"
                assert messages[0].text == "sys"
                assert messages[1].text == "Aout"
            else:
                assert len(messages) == 1
                assert messages[0].text == "Aout"
            yield ReqFinalMessage(message=msg("agent", "Bout"))

    project = Project(base_path=Path("."))
    runner = Runner(workflow, project, initial_message=msg("system", "sys"))
    task = Assignment()
    it = runner.run(task)

    # A final -> approve
    ev1 = await it.__anext__()
    assert ev1.event.kind == "final_message" and ev1.node == "A"
    ev2 = await it.asend(None)
    # Now B final -> approve
    assert ev2.event.kind == "final_message" and ev2.node == "B"
    with pytest.raises(StopAsyncIteration):
        await it.asend(None)

    # Steps per node
    assert runner.status.name == "finished"
    assert [s.node for s in task.steps] == ["A", "B"]
    assert len(task.steps[0].executions) == 2
    assert task.steps[0].executions[0].message.text == "sys"
    assert task.steps[0].executions[0].type.value == "user"
    assert task.steps[0].executions[1].message.text == "Aout"
    assert task.steps[0].executions[1].type.value == "executor"
    if pass_all:
        # B step includes carried inputs (system + A's output) and its final
        assert len(task.steps[1].executions) == 3
        assert task.steps[1].executions[0].type.value == "user"
        assert task.steps[1].executions[0].message.text == "sys"
        assert task.steps[1].executions[1].type.value == "user"
        assert task.steps[1].executions[1].message.text == "Aout"
        assert task.steps[1].executions[2].type.value == "executor"
        assert task.steps[1].executions[2].message.text == "Bout"
    else:
        # B step includes carried input (A's output) and its final
        assert len(task.steps[1].executions) == 2
        assert task.steps[1].executions[0].type.value == "user"
        assert task.steps[1].executions[0].message.text == "Aout"
        assert task.steps[1].executions[1].type.value == "executor"
        assert task.steps[1].executions[1].message.text == "Bout"


@pytest.mark.asyncio
async def test_error_multiple_outcomes_without_outcome_name():
    nodes = [
        {"name": "A", "type": "errA", "outcomes": [OutcomeSlot(name="x"), OutcomeSlot(name="y")]},
        {"name": "Bx", "type": "b1", "outcomes": []},
        {"name": "By", "type": "b2", "outcomes": []},
    ]
    edges = [
        Edge(source_node="A", source_outcome="x", target_node="Bx"),
        Edge(source_node="A", source_outcome="y", target_node="By"),
    ]
    g = Graph.build(nodes=nodes, edges=edges)
    workflow = Workflow(name="workflow", graph=g)

    class ErrExecA(Executor):
        type = "errA"

        async def run(self, messages):
            # Emit final without outcome_name -> error since multiple outcomes
            yield ReqFinalMessage(message=msg("agent", "no outcome"))

    # Child executors (won't be reached)
    class B1(Executor):
        type = "b1"
        async def run(self, messages):
            yield ReqFinalMessage(message=msg("agent", "Bx"))

    class B2(Executor):
        type = "b2"
        async def run(self, messages):
            yield ReqFinalMessage(message=msg("agent", "By"))

    project = Project(base_path=Path("."))
    runner = Runner(workflow, project)
    task = Assignment()
    it = runner.run(task)

    ev = await it.__anext__()
    assert ev.event.kind == "final_message" and ev.node == "A"
    with pytest.raises(ValueError, match="did not provide an outcome and has 2 outcomes"):
        await it.asend(None)


@pytest.mark.asyncio
async def test_error_unknown_outcome_name():
    nodes = [
        {"name": "A", "type": "err2", "outcomes": [OutcomeSlot(name="go")]},
        {"name": "B", "type": "b", "outcomes": []},
    ]
    edges = [Edge(source_node="A", source_outcome="go", target_node="B")]
    g = Graph.build(nodes=nodes, edges=edges)
    workflow = Workflow(name="workflow", graph=g)

    class Err2Exec(Executor):
        type = "err2"
        async def run(self, messages):
            yield ReqFinalMessage(message=msg("agent", "oops"), outcome_name="unknown")

    class BExec(Executor):
        type = "b"
        async def run(self, messages):
            yield ReqFinalMessage(message=msg("agent", "B"))

    project = Project(base_path=Path("."))
    runner = Runner(workflow, project)
    task = Assignment()
    it = runner.run(task)

    ev = await it.__anext__()
    assert ev.event.kind == "final_message"
    with pytest.raises(ValueError, match="No edge defined from node 'A' via outcome 'unknown'"):
        await it.asend(None)


@pytest.mark.asyncio
async def test_cancel_before_first_yield():
    nodes = [{"name": "Slow", "type": "slow", "outcomes": []}]
    g = Graph.build(nodes=nodes, edges=[])
    workflow = Workflow(name="workflow", graph=g)

    class SlowExec(Executor):
        type = "slow"
        closed = False
        async def run(self, messages):
            try:
                # Block indefinitely until canceled
                await asyncio.sleep(60)
                # Would yield if not canceled
                yield ReqInterimMessage(message=msg("agent", "never"))
            finally:
                type(self).closed = True

    project = Project(base_path=Path("."))
    runner = Runner(workflow, project)
    task = Assignment()
    it = runner.run(task)

    # Start the run loop; it will block inside agen.asend(None) awaiting the sleep
    first = asyncio.create_task(it.__anext__())
    await asyncio.sleep(0.01)

    # Cancel in-flight executor step
    runner.cancel()

    # The generator should terminate
    with pytest.raises(StopAsyncIteration):
        await first

    assert runner.status.name == "canceled"
    assert SlowExec.closed is True
    # A Step is created and marked canceled, but no executions
    assert len(task.steps) == 1
    s = task.steps[0]
    assert s.node == "Slow"
    assert s.status.name == "canceled"
    assert len(s.executions) == 0


@pytest.mark.asyncio
async def test_cancel_during_asend_after_tool_response():
    nodes = [{"name": "ToolSlow", "type": "toolslow", "outcomes": []}]
    g = Graph.build(nodes=nodes, edges=[])
    workflow = Workflow(name="workflow", graph=g)

    class ToolThenSlow(Executor):
        type = "toolslow"
        closed = False
        async def run(self, messages):
            try:
                resp = yield ReqToolCall(tool_calls=[ToolCall(name="sleep", arguments="{}")])
                assert resp.kind == "tool_call"
                # Block after receiving tool response
                await asyncio.sleep(60)
                yield ReqFinalMessage(message=msg("agent", "done"))
            finally:
                type(self).closed = True

    project = Project(base_path=Path("."))
    runner = Runner(workflow, project)
    task = Assignment()
    it = runner.run(task)

    # First event: tool_call
    ev1 = await it.__anext__()
    assert ev1.event.kind == "tool_call"

    # Send approval and let runner drive agen.asend(response) in background
    pending = asyncio.create_task(it.asend(RunInput(response=RespApproval(approved=True))))
    await asyncio.sleep(0.01)

    # Cancel in-flight executor step (waiting inside agen.asend(...))
    runner.cancel()

    # The pending asend should terminate the generator
    with pytest.raises(StopAsyncIteration):
        await pending

    assert runner.status.name == "canceled"
    assert ToolThenSlow.closed is True
    # A Step is created and marked canceled, but no executions
    assert len(task.steps) == 1
    s = task.steps[0]
    assert s.node == "ToolSlow"
    assert s.status.name == "canceled"
    assert len(s.executions) == 0
@pytest.mark.asyncio
async def test_stop_before_first_yield():
    nodes = [{"name": "Slow", "type": "slow", "outcomes": []}]
    g = Graph.build(nodes=nodes, edges=[])
    workflow = Workflow(name="workflow", graph=g)

    class SlowExec(Executor):
        type = "slow"
        closed = False
        async def run(self, messages):
            try:
                await asyncio.sleep(60)
                yield ReqInterimMessage(message=msg("agent", "never"))
            finally:
                type(self).closed = True

    project = Project(base_path=Path("."))
    runner = Runner(workflow, project)
    task = Assignment()
    it = runner.run(task)

    first = asyncio.create_task(it.__anext__())
    await asyncio.sleep(0.01)

    # Stop in-flight executor step
    runner.stop()

    # The generator should terminate
    with pytest.raises(StopAsyncIteration):
        await first

    assert runner.status.name == "stopped"
    assert SlowExec.closed is True
    # A Step is created and marked stopped, but no executions
    assert len(task.steps) == 1
    s = task.steps[0]
    assert s.node == "Slow"
    assert s.status.name == "stopped"
    assert len(s.executions) == 0

@pytest.mark.asyncio
async def test_stop_and_resume_from_last_good_step():
    # Graph: A -> B
    nodes = [
        {"name": "A", "type": "a", "outcomes": [OutcomeSlot(name="toB")]},
        {"name": "B", "type": "b", "outcomes": []},
    ]
    edges = [Edge(source_node="A", source_outcome="toB", target_node="B")]
    g = Graph.build(nodes=nodes, edges=edges)
    workflow = Workflow(name="workflow", graph=g)

    class AExec(Executor):
        type = "a"
        runs = 0
        async def run(self, messages):
            # First run: no inputs; emit final to go to B
            if type(self).runs == 0:
                type(self).runs += 1
                assert len(messages) == 0
                yield ReqFinalMessage(message=msg("agent", "A1"), outcome_name="toB")
            else:
                # Rerun after resume: should receive previous output + user message
                assert len(messages) == 2
                assert messages[0].role in ("agent", "tool")
                assert messages[0].text == "A1"
                assert messages[1].role == "user"
                assert messages[1].text == "followup"
                yield ReqFinalMessage(message=msg("agent", "A2"), outcome_name="toB")

    class BExec(Executor):
        type = "b"
        calls = 0
        async def run(self, messages):
            # First time: block until runner.stop() is invoked (no yield)
            type(self).calls += 1
            if type(self).calls == 1:
                await asyncio.sleep(60)
                # Would yield if not stopped
                yield ReqInterimMessage(message=msg("agent", "never"))
            else:
                # After resume and rerun of A, complete immediately
                yield ReqFinalMessage(message=msg("agent", "Bdone"))

    project = Project(base_path=Path("."))
    runner = Runner(workflow, project)
    task = Assignment()
    it = runner.run(task)

    # A emits final; approve to move to B (which will block before first yield)
    ev_a1 = await it.__anext__()
    assert ev_a1.node == "A" and ev_a1.event.kind == "final_message"
    pending = asyncio.create_task(it.asend(None))
    await asyncio.sleep(0.01)

    # Stop while B is pending inside agen.asend(None)
    runner.stop()
    with pytest.raises(StopAsyncIteration):
        await pending

    # Runner stopped; last good step (A) should be recorded
    assert runner.status.name == "stopped"
    # A finished; B recorded as stopped (no executor yields) with its carried input
    assert [s.node for s in task.steps] == ["A", "B"]
    assert task.steps[0].status.name == "finished"
    assert len(task.steps[0].executions) == 1
    assert task.steps[0].executions[0].is_complete is True
    assert task.steps[0].executions[0].message is not None
    assert task.steps[0].executions[0].message.text == "A1"
    b_stopped = task.steps[1]
    assert b_stopped.status.name == "stopped"
    assert len(b_stopped.executions) == 1
    assert b_stopped.executions[0].type.value == "user"
    assert b_stopped.executions[0].message.text == "A1"

    # Resume: helper should yield last message from A and allow user input to rerun A
    it2 = runner.run(task)
    ev_resume_prompt = await it2.__anext__()
    assert ev_resume_prompt.node == "A"
    assert ev_resume_prompt.event.kind == "final_message"
    assert ev_resume_prompt.event.message is not None
    assert ev_resume_prompt.event.message.text == "A1"
    assert ev_resume_prompt.input_requested is True

    # Provide user message -> reruns A with input_messages + output_message + user input
    ev_a2 = await it2.asend(RunInput(response=RespMessage(message=msg("user", "followup"))))
    assert ev_a2.node == "A" and ev_a2.event.kind == "final_message"

    # Approve A's second final -> proceed to B (now completes)
    ev_b = await it2.asend(None)
    assert ev_b.node == "B" and ev_b.event.kind == "final_message"

    # Approve B's final -> finish
    with pytest.raises(StopAsyncIteration):
        await it2.asend(None)

    # Validate final task state
    assert runner.status.name == "finished"
    # We keep the stopped B step and add a new finished B step
    assert [s.node for s in task.steps] == ["A", "B", "B"]
    # A step has three activities: final A1, user followup, final A2
    assert len(task.steps[0].executions) == 3
    assert task.steps[0].executions[0].type.value == "executor"
    assert task.steps[0].executions[0].message.text == "A1"
    assert task.steps[0].executions[1].type.value == "user"
    assert task.steps[0].executions[1].message.text == "followup"
    assert task.steps[0].executions[2].type.value == "executor"
    assert task.steps[0].executions[2].message.text == "A2"
    # First B step is the stopped one with carried input from A1
    assert task.steps[1].status.name == "stopped"
    assert len(task.steps[1].executions) == 1
    assert task.steps[1].executions[0].type.value == "user"
    assert task.steps[1].executions[0].message.text == "A1"
    # Second B step (after resume) includes carried input (A2) and its final
    assert task.steps[2].status.name == "finished"
    assert len(task.steps[2].executions) == 2
    assert task.steps[2].executions[0].type.value == "user"
    assert task.steps[2].executions[0].message.text == "A2"
    assert task.steps[2].executions[1].type.value == "executor"
    assert task.steps[2].executions[1].message.text == "Bdone"


@pytest.mark.asyncio
async def test_run_disallowed_when_running():
    nodes = [{"name": "Block", "type": "block", "outcomes": []}]
    g = Graph.build(nodes=nodes, edges=[])
    workflow = Workflow(name="workflow", graph=g)

    class BlockExecutor(Executor):
        type = "block"

        async def run(self, messages):
            await asyncio.sleep(60)
            yield ReqFinalMessage(message=msg("agent", "done"))

    project = Project(base_path=Path("."))
    runner = Runner(workflow, project)
    task = Assignment()
    it = runner.run(task)

    run_task = asyncio.create_task(it.__anext__())
    await asyncio.sleep(0.01)  # allow runner to start

    assert runner.status == "running"
    with pytest.raises(RuntimeError, match=".*not allowed when runner status"):
        # Create a new run iterator and try to advance it; this should fail.
        await runner.run(task).__anext__()

    # Cleanup
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task


@pytest.mark.asyncio
async def test_reset_policy_auto_confirmation_and_single_outcome_transition():
    # Graph: A -> B -> A -> C.
    # - A has never_reset policy, so its second run accumulates messages.
    # - B has auto confirmation, so it transitions without user input.
    # - B has one outcome, its executor gives no outcome_name, testing single outcome transition.
    nodes = [
        {
            "name": "A",
            "type": "a",
            "outcomes": [OutcomeSlot(name="toB"), OutcomeSlot(name="toC")],
            "reset_policy": "never_reset",
            "pass_all_messages": True,  # Pass all to B
        },
        {
            "name": "B",
            "type": "b",
            "outcomes": [OutcomeSlot(name="toA")],
            "confirmation": "auto",
        },
        {"name": "C", "type": "c", "outcomes": []},
    ]
    edges = [
        Edge(source_node="A", source_outcome="toB", target_node="B"),
        Edge(source_node="B", source_outcome="toA", target_node="A"),
        Edge(source_node="A", source_outcome="toC", target_node="C"),
    ]
    g = Graph.build(nodes=nodes, edges=edges)
    workflow = Workflow(name="workflow", graph=g)

    class AExec(Executor):
        type = "a"
        runs = 0

        async def run(self, messages):
            type(self).runs += 1
            if type(self).runs == 1:
                # First run, gets initial message
                assert len(messages) == 1
                assert messages[0].role == "user" and messages[0].text == "start"
                yield ReqFinalMessage(message=msg("agent", "A1"), outcome_name="toB")
            else:
                # Second run, should accumulate messages from its first run and B's output
                # Initial "start", A's "A1", and B's output "B1" (which is the input to this step)
                assert len(messages) == 3
                texts = {m.text for m in messages}
                assert texts == {"start", "A1", "B1"}
                yield ReqFinalMessage(message=msg("agent", "A2"), outcome_name="toC")

    class BExec(Executor):
        type = "b"

        async def run(self, messages):
            # Should get all messages from A due to pass_all_messages
            assert len(messages) == 2
            assert messages[0].text == "start"
            assert messages[1].text == "A1"
            # No outcome_name -> auto-transition due to single outcome
            yield ReqFinalMessage(message=msg("agent", "B1"))

    class CExec(Executor):
        type = "c"

        async def run(self, messages):
            yield ReqFinalMessage(message=msg("agent", "C1"))

    project = Project(base_path=Path("."))
    runner = Runner(workflow, project, initial_message=msg("user", "start"))
    task = Assignment()
    it = runner.run(task)

    # A's first run
    ev_a1 = await it.__anext__()
    assert ev_a1.node == "A"
    assert ev_a1.event.kind == "final_message"
    assert ev_a1.event.message.text == "A1"
    assert ev_a1.input_requested is True

    # Approve A1. This triggers B. B is auto-confirmed, so it will yield its final
    # event with input_requested=False.
    ev_b1 = await it.asend(None)
    assert ev_b1.node == "B"
    assert ev_b1.event.kind == "final_message"
    assert ev_b1.event.message.text == "B1"
    assert ev_b1.input_requested is False

    # Since B did not request input, we can immediately advance the runner. This transitions
    # from B back to A for its second run, which yields its final message.
    ev_a2 = await it.asend(None)
    assert ev_a2.node == "A"
    assert ev_a2.event.kind == "final_message"
    assert ev_a2.event.message.text == "A2"
    assert ev_a2.input_requested is True

    # Approve A2. This triggers C.
    ev_c1 = await it.asend(None)
    assert ev_c1.node == "C"
    assert ev_c1.event.kind == "final_message"
    assert ev_c1.event.message.text == "C1"

    # Approve C to finish.
    with pytest.raises(StopAsyncIteration):
        await it.asend(None)

    assert runner.status == "finished"
    assert [s.node for s in task.steps] == ["A", "B", "A", "C"]
