import pytest
import asyncio
from pathlib import Path
from vocode.testing import ProjectSandbox
from vocode.graph import Graph, Node, Edge, OutcomeSlot, Workflow
from vocode.runner.runner import Runner, Executor
from vocode.runner.models import (
    ReqMessageRequest,
    ReqToolCall,
    ReqInterimMessage,
    ReqLogMessage,
    ReqFinalMessage,
    RespMessage,
    RespApproval,
    RunInput,
)
from vocode.state import Message, ToolCall, ToolCallStatus, Assignment, RunnerStatus


def msg(role: str, text: str) -> Message:
    return Message(role=role, text=text)


@pytest.mark.asyncio
async def test_message_request_reprompt_and_finish(tmp_path: Path):
    # Single-node graph with type 'ask'
    nodes = [{"name": "Ask", "type": "ask", "outcomes": [], "confirmation": "confirm"}]
    g = Graph.build(nodes=nodes, edges=[])
    workflow = Workflow(name="workflow", graph=g)

    # Executor that requires a message, then emits interim and final
    class AskExecutor(Executor):
        type = "ask"
        async def run(self, inp):
            # First cycle: request a message
            if inp.response is None:
                yield (ReqMessageRequest(), None)
                return
            # Second cycle: got user message response -> emit interim, then final
            assert inp.response is not None and inp.response.kind == "message"
            interim = msg("agent", "got it")
            yield (ReqInterimMessage(message=interim), None)
            final = msg("agent", f"final:{inp.response.message.text}")
            yield (ReqFinalMessage(message=final), None)

    async with ProjectSandbox.create(tmp_path) as project:
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
        assert ev2.node == "Ask"
        assert ev2.event.kind == "message_request"
        assert ev2.input_requested is True

        # Now provide a message
        user_msg = msg("user", "hi")
        ev3 = await it.asend(RunInput(response=RespMessage(message=user_msg)))
        assert ev3.event.kind == "message"
        assert ev3.input_requested is False
        assert ev3.event.message is not None
        assert ev3.event.message.text == "got it"

        # Final message
        ev4 = await it.__anext__()
        assert ev4.event.kind == "final_message"
        assert ev4.input_requested is True
        # Explicit approval to finish
        with pytest.raises(StopAsyncIteration):
            await it.asend(RunInput(response=RespApproval(approved=True)))

        assert runner.status.name == "finished"
        assert len(task.steps) == 1
        step = task.steps[0]
        assert step.node == "Ask"
        # Two activities: user input, final
        assert len(step.executions) == 2
        assert step.executions[0].type.value == "user"
        assert step.executions[0].message.text == "hi"
        assert step.executions[1].type.value == "executor"
        assert step.executions[1].is_complete is True
        assert step.executions[1].message is not None
        assert step.executions[1].message.text == "final:hi"


@pytest.mark.asyncio
async def test_rewind_one_step_and_resume(tmp_path: Path):
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
        async def run(self, inp):
            yield (ReqFinalMessage(message=msg("agent", "A1"), outcome_name="toB"), None)

    class BExec(Executor):
        type = "b"
        runs = 0
        async def run(self, inp):
            type(self).runs += 1
            yield (ReqFinalMessage(message=msg("agent", f"B{type(self).runs}")), None)

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        # Complete A then B
        ev_a = await it.__anext__()
        assert ev_a.node == "A" and ev_a.event.kind == "final_message"
        ev_b = await it.asend(None)
        assert ev_b.node == "B" and ev_b.event.kind == "final_message"
        with pytest.raises(StopAsyncIteration):
            await it.asend(None)
        assert runner.status == RunnerStatus.finished
        assert [s.node for s in task.steps] == ["A", "B"]

        # Rewind last step (B)
        await runner.rewind(task, 1)
        assert runner.status == RunnerStatus.stopped
        assert [s.node for s in task.steps] == ["A"]

        # Resume -> should start at B again, fresh
        it2 = runner.run(task)
        ev_b2 = await it2.__anext__()
        assert ev_b2.node == "B" and ev_b2.event.kind == "final_message"
        # Ensure the executor was not re-executed on resume (replayed from history)
        assert BExec.runs == 1  # resumed from history (no re-execution)
        with pytest.raises(StopAsyncIteration):
            await it2.asend(None)
        assert runner.status == RunnerStatus.finished
        assert [s.node for s in task.steps] == ["A", "B"]


@pytest.mark.asyncio
async def test_rewind_multiple_steps_and_resume(tmp_path: Path):
    # Graph: A -> B -> C
    nodes = [
        {"name": "A", "type": "a", "outcomes": [OutcomeSlot(name="toB")]},
        {"name": "B", "type": "b", "outcomes": [OutcomeSlot(name="toC")]},
        {"name": "C", "type": "c", "outcomes": []},
    ]
    edges = [
        Edge(source_node="A", source_outcome="toB", target_node="B"),
        Edge(source_node="B", source_outcome="toC", target_node="C"),
    ]
    g = Graph.build(nodes=nodes, edges=edges)
    workflow = Workflow(name="workflow", graph=g)

    class AExec(Executor):
        type = "a"
        async def run(self, inp):
            yield (ReqFinalMessage(message=msg("agent", "A"), outcome_name="toB"), None)

    class BExec(Executor):
        type = "b"
        async def run(self, inp):
            yield (ReqFinalMessage(message=msg("agent", "B"), outcome_name="toC"), None)

    class CExec(Executor):
        type = "c"
        runs = 0
        async def run(self, inp):
            type(self).runs += 1
            yield (ReqFinalMessage(message=msg("agent", f"C{type(self).runs}")), None)

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        # Finish A -> B -> C
        ev_a = await it.__anext__()
        assert ev_a.node == "A" and ev_a.event.kind == "final_message"
        ev_b = await it.asend(None)
        assert ev_b.node == "B" and ev_b.event.kind == "final_message"
        ev_c = await it.asend(None)
        assert ev_c.node == "C" and ev_c.event.kind == "final_message"
        with pytest.raises(StopAsyncIteration):
            await it.asend(None)
        assert runner.status == RunnerStatus.finished
        assert [s.node for s in task.steps] == ["A", "B", "C"]

        # Rewind last two steps (B, C)
        await runner.rewind(task, 2)
        assert runner.status == RunnerStatus.stopped
        assert [s.node for s in task.steps] == ["A"]

        # Resume -> should execute B then C again
        it2 = runner.run(task)
        ev_b2 = await it2.__anext__()
        assert ev_b2.node == "B" and ev_b2.event.kind == "final_message"
        ev_c2 = await it2.asend(None)
        assert ev_c2.node == "C" and ev_c2.event.kind == "final_message"
        with pytest.raises(StopAsyncIteration):
            await it2.asend(None)
        assert runner.status == RunnerStatus.finished
        assert [s.node for s in task.steps] == ["A", "B", "C"]



@pytest.mark.asyncio
async def test_tool_call_approved_and_rejected(tmp_path: Path):
    nodes = [{"name": "Tool", "type": "tool", "outcomes": []}]
    g = Graph.build(nodes=nodes, edges=[])
    workflow = Workflow(name="workflow", graph=g)

    class ToolExecutor(Executor):
        type = "tool"
        async def run(self, inp):
            # First cycle: emit tool call "one"
            if inp.response is None:
                tc1 = ToolCall(name="one", arguments={})
                yield (ReqToolCall(tool_calls=[tc1]), None)
                return
            # Second cycle: after "one" response, emit tool call "two"
            if inp.response.kind == "tool_call" and inp.response.tool_calls and inp.response.tool_calls[0].name == "one":
                tc2 = ToolCall(name="two", arguments={})
                yield (ReqToolCall(tool_calls=[tc2]), None)
                return
            # Third cycle: after "two" response, finish
            yield (ReqFinalMessage(message=msg("agent", "done")), None)

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        # First tool call (approve)
        ev1 = await it.__anext__()
        assert ev1.event.kind == "tool_call"
        # Respond and capture the next event (this returns the next yielded request)
        ev2 = await it.asend(RunInput(response=RespApproval(approved=True)))
        # After approval the first req object should be updated
        assert ev1.event.tool_calls[0].status == ToolCallStatus.rejected
        err = ev1.event.tool_calls[0].result
        assert isinstance(err, dict)
        assert "Unknown tool" in (err.get("error") or "")

        # Second tool call (reject)
        assert ev2.event.kind == "tool_call"
        ev3 = await it.asend(RunInput(response=RespApproval(approved=False)))
        assert ev2.event.tool_calls[0].status == ToolCallStatus.rejected
        err2 = ev2.event.tool_calls[0].result
        assert isinstance(err2, dict)
        assert "rejected by user" in (err2.get("error") or "")

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
async def test_final_message_rerun_same_node_with_user_message(tmp_path: Path):
    nodes = [{"name": "Echo", "type": "echo", "outcomes": [], "confirmation": "prompt"}]
    g = Graph.build(nodes=nodes, edges=[])
    workflow = Workflow(name="workflow", graph=g)

    class EchoExecutor(Executor):
        type = "echo"
        async def run(self, inp):
            if not inp.response or inp.response.kind != "message":
                # First final: trigger post-final user message
                yield (ReqFinalMessage(message=msg("agent", "ask")), None)
                return
            # Second cycle: user provided a message
            assert inp.response.message.role == "user" and inp.response.message.text == "more"
            yield (ReqFinalMessage(message=msg("agent", "done")), None)

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        # First: final_message (requesting input)
        ev1 = await it.__anext__()
        assert ev1.event.kind == "final_message"
        # Provide a user message to continue same executor (not added to history)
        ev2 = await it.asend(RunInput(response=RespMessage(message=msg("user", "more"))))
        assert ev2.event.kind == "final_message"
        with pytest.raises(StopAsyncIteration):
            await it.asend(None)

        # Verify two executions within the same step; no user message recorded
        assert runner.status.name == "finished"
        assert len(task.steps) == 1
        step = task.steps[0]
        assert len(step.executions) == 2
        assert step.executions[0].type.value == "executor"
        assert step.executions[0].message.text == "ask"
        assert step.executions[1].type.value == "executor"
        assert step.executions[1].is_complete is True
        assert step.executions[1].message.text == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["all_messages", "final_response"])
async def test_transition_to_next_node_and_pass_all_messages(mode, tmp_path: Path):
    nodes = [
        {"name": "A", "type": "a", "outcomes": [OutcomeSlot(name="toB")], "message_mode": mode},
        {"name": "B", "type": "b", "outcomes": []},
    ]
    edges = [Edge(source_node="A", source_outcome="toB", target_node="B")]
    g = Graph.build(nodes=nodes, edges=edges)
    workflow = Workflow(name="workflow", graph=g)

    class AExec(Executor):
        type = "a"

        async def run(self, inp):
            # Emit final with outcome to go to B
            yield (ReqFinalMessage(message=msg("agent", "Aout"), outcome_name="toB"), None)

    class BExec(Executor):
        type = "b"

        async def run(self, inp):
            messages = inp.messages
            # Expect messages based on mode and initial messages
            if mode == "all_messages":
                assert len(messages) == 2  # initial system + A's output
                assert messages[0].role == "system"
                assert messages[0].text == "sys"
                assert messages[1].text == "Aout"
            else:
                assert len(messages) == 1
                assert messages[0].text == "Aout"
            yield (ReqFinalMessage(message=msg("agent", "Bout")), None)

    async with ProjectSandbox.create(tmp_path) as project:
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
    if mode == "all_messages":
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
async def test_error_multiple_outcomes_without_outcome_name(tmp_path: Path):
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

        async def run(self, inp):
            # Emit final without outcome_name -> error since multiple outcomes
            yield (ReqFinalMessage(message=msg("agent", "no outcome")), None)

    # Child executors (won't be reached)
    class B1(Executor):
        type = "b1"
        async def run(self, inp):
            yield (ReqFinalMessage(message=msg("agent", "Bx")), None)

    class B2(Executor):
        type = "b2"
        async def run(self, inp):
            yield (ReqFinalMessage(message=msg("agent", "By")), None)

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        ev = await it.__anext__()
        assert ev.event.kind == "final_message" and ev.node == "A"
        with pytest.raises(ValueError, match="did not provide an outcome and has 2 outcomes"):
            await it.asend(None)


@pytest.mark.asyncio
async def test_error_unknown_outcome_name(tmp_path: Path):
    nodes = [
        {"name": "A", "type": "err2", "outcomes": [OutcomeSlot(name="go")]},
        {"name": "B", "type": "b", "outcomes": []},
    ]
    edges = [Edge(source_node="A", source_outcome="go", target_node="B")]
    g = Graph.build(nodes=nodes, edges=edges)
    workflow = Workflow(name="workflow", graph=g)

    class Err2Exec(Executor):
        type = "err2"
        async def run(self, inp):
            yield (ReqFinalMessage(message=msg("agent", "oops"), outcome_name="unknown"), None)

    class BExec(Executor):
        type = "b"
        async def run(self, inp):
            yield (ReqFinalMessage(message=msg("agent", "B")), None)

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        ev = await it.__anext__()
        assert ev.event.kind == "final_message"
        with pytest.raises(ValueError, match="No edge defined from node 'A' via outcome 'unknown'"):
            await it.asend(None)







@pytest.mark.asyncio
async def test_run_disallowed_when_running(tmp_path: Path):
    nodes = [{"name": "Block", "type": "block", "outcomes": []}]
    g = Graph.build(nodes=nodes, edges=[])
    workflow = Workflow(name="workflow", graph=g)

    class BlockExecutor(Executor):
        type = "block"

        async def run(self, messages):
            await asyncio.sleep(60)
            yield ReqFinalMessage(message=msg("agent", "done"))

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        run_task = asyncio.create_task(it.__anext__())
        await asyncio.sleep(0.01)  # allow runner to start

        assert runner.status == RunnerStatus.running
        with pytest.raises(RuntimeError, match=".*not allowed when runner status"):
            # Create a new run iterator and try to advance it; this should fail.
            await runner.run(task).__anext__()

        # Cleanup
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task


@pytest.mark.asyncio
async def test_reset_policy_auto_confirmation_and_single_outcome_transition(tmp_path: Path):
    # Graph: A -> B -> A -> C.
    # - A has keep_results policy, so its second run accumulates messages.
    # - B has auto confirmation, so it transitions without user input.
    # - B has one outcome, its executor gives no outcome_name, testing single outcome transition.
    nodes = [
        {
            "name": "A",
            "type": "a",
            "outcomes": [OutcomeSlot(name="toB"), OutcomeSlot(name="toC")],
            "reset_policy": "keep_results",
            "message_mode": "all_messages",  # Pass all to B
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

        async def run(self, inp):
            messages = inp.messages
            type(self).runs += 1
            if type(self).runs == 1:
                # First run, gets initial message
                assert len(messages) == 1
                assert messages[0].role == "user" and messages[0].text == "start"
                yield (ReqFinalMessage(message=msg("agent", "A1"), outcome_name="toB"), None)
            else:
                # Second run via keep_results: receives all prior A messages + new input from B
                assert [m.text for m in messages] == ["start", "A1", "B1"]
                yield (ReqFinalMessage(message=msg("agent", "A2"), outcome_name="toC"), None)

    class BExec(Executor):
        type = "b"

        async def run(self, inp):
            messages = inp.messages
            # Should get all messages from A due to pass_all_messages
            assert len(messages) == 2
            assert messages[0].text == "start"
            assert messages[1].text == "A1"
            # No outcome_name -> auto-transition due to single outcome
            yield (ReqFinalMessage(message=msg("agent", "B1")), None)

    class CExec(Executor):
        type = "c"

        async def run(self, inp):
            yield (ReqFinalMessage(message=msg("agent", "C1")), None)

    async with ProjectSandbox.create(tmp_path) as project:
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

        assert runner.status == RunnerStatus.finished
        assert [s.node for s in task.steps] == ["A", "B", "A", "C"]

@pytest.mark.asyncio
async def test_edge_reset_policy_override_short_syntax(tmp_path: Path):
    # Graph:
    #   B (default keep_state) --again--> B:always_reset (short syntax override)
    #   B --done--> C
    nodes = [
        {
            "name": "B",
            "type": "bshort",
            "outcomes": [OutcomeSlot(name="again"), OutcomeSlot(name="done")],
            "reset_policy": "keep_state",
        },
        {"name": "C", "type": "cshort", "outcomes": []},
    ]
    edges = [
        "B.again -> B:always_reset",  # short syntax override
        "B.done -> C",
    ]
    g = Graph.build(nodes=nodes, edges=edges)
    workflow = Workflow(name="workflow", graph=g)

    class BShortExec(Executor):
        type = "bshort"
        next_id = 1
        global_runs = 0  # track runs across instances for this test

        def __init__(self, config, project):
            super().__init__(config, project)
            self.instance_id = type(self).next_id
            type(self).next_id += 1

        async def run(self, inp):
            if type(self).global_runs == 0:
                type(self).global_runs += 1
                # First final: loop back to B (new instance due to always_reset override)
                yield (ReqFinalMessage(message=msg("agent", f"B1:{self.instance_id}"), outcome_name="again"), None)
            else:
                type(self).global_runs += 1
                # Second final: go to C
                yield (ReqFinalMessage(message=msg("agent", f"B2:{self.instance_id}"), outcome_name="done"), None)

    class CShortExec(Executor):
        type = "cshort"
        async def run(self, inp):
            yield (ReqFinalMessage(message=msg("agent", "C")), None)

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        # First B final
        ev_b1 = await it.__anext__()
        assert ev_b1.node == "B" and ev_b1.event.kind == "final_message"
        id1 = int(ev_b1.event.message.text.split(":", 1)[1])

        # Approve -> B again (override always_reset => new instance)
        ev_b2 = await it.asend(None)
        assert ev_b2.node == "B" and ev_b2.event.kind == "final_message"
        id2 = int(ev_b2.event.message.text.split(":", 1)[1])
        assert id2 != id1  # instance replaced due to edge override

        # Approve -> C
        ev_c = await it.asend(None)
        assert ev_c.node == "C" and ev_c.event.kind == "final_message"

        # Approve -> finish
        with pytest.raises(StopAsyncIteration):
            await it.asend(None)

@pytest.mark.asyncio
async def test_log_stream_then_final(tmp_path: Path):
    nodes = [{"name": "Log", "type": "logg", "outcomes": []}]
    g = Graph.build(nodes=nodes, edges=[])
    workflow = Workflow(name="workflow", graph=g)

    class LogExec(Executor):
        type = "logg"
        async def run(self, inp):
            # Emit a log, then a final message in the same cycle
            yield (ReqLogMessage(text="hello"), None)
            yield (ReqFinalMessage(message=msg("agent", "done")), None)

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        # First event is a log; should not request input and should not finish the step
        ev1 = await it.__anext__()
        assert ev1.node == "Log"
        assert ev1.event.kind == "log"
        assert ev1.input_requested is False

        # Next event is the final message for this node
        ev2 = await it.__anext__()
        assert ev2.node == "Log"
        assert ev2.event.kind == "final_message"

        with pytest.raises(StopAsyncIteration):
            await it.asend(None)

        # Verify only the final executor message is recorded (logs are not recorded)
        assert runner.status.name == "finished"
        assert len(task.steps) == 1
        step = task.steps[0]
        assert step.node == "Log"
        assert len(step.executions) == 1
        assert step.executions[0].type.value == "executor"
        assert step.executions[0].is_complete is True
        assert step.executions[0].message is not None
        assert step.executions[0].message.text == "done"


@pytest.mark.asyncio
async def test_final_message_confirm_requires_explicit_approval_and_reprompts(tmp_path: Path):
    # Single-node graph with confirmation=confirm
    nodes = [{"name": "C", "type": "conf", "outcomes": [], "confirmation": "confirm"}]
    g = Graph.build(nodes=nodes, edges=[])
    workflow = Workflow(name="workflow", graph=g)

    class ConfirmExec(Executor):
        type = "conf"
        async def run(self, inp):
            yield (ReqFinalMessage(message=msg("agent", "done")), None)

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        # First: final_message requests explicit approval
        ev1 = await it.__anext__()
        assert ev1.node == "C"
        assert ev1.event.kind == "final_message"
        assert ev1.input_requested is True

        # Sending a user message should NOT be accepted; runner should re-prompt final_message
        ev2 = await it.asend(RunInput(response=RespMessage(message=msg("user", "not approval"))))
        assert ev2.node == "C"
        assert ev2.event.kind == "final_message"
        assert ev2.input_requested is True
        assert ev2.event.message is not None and ev2.event.message.text == "done"

        # Now approve explicitly -> runner should finish
        with pytest.raises(StopAsyncIteration):
            await it.asend(RunInput(response=RespApproval(approved=True)))

        # Runner state and recorded activities: only the executor final is recorded
        assert runner.status.name == "finished"
        assert len(task.steps) == 1
        step = task.steps[0]
        assert step.node == "C"
        assert step.status.name == "finished"
        assert len(step.executions) == 1
        assert step.executions[0].type.value == "executor"
        assert step.executions[0].is_complete is True
        assert step.executions[0].message is not None
        assert step.executions[0].message.text == "done"


@pytest.mark.asyncio
async def test_final_message_confirm_reject_stops_runner(tmp_path: Path):
    nodes = [{"name": "C", "type": "conf", "outcomes": [], "confirmation": "confirm"}]
    g = Graph.build(nodes=nodes, edges=[])
    workflow = Workflow(name="workflow", graph=g)

    class ConfirmExec(Executor):
        type = "conf"
        async def run(self, inp):
            yield (ReqFinalMessage(message=msg("agent", "done")), None)

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        ev1 = await it.__anext__()
        assert ev1.event.kind == "final_message" and ev1.input_requested is True

        # Reject explicitly -> runner should stop
        with pytest.raises(StopAsyncIteration):
            await it.asend(RunInput(response=RespApproval(approved=False)))

        assert runner.status.name == "stopped"
        assert len(task.steps) == 1
        step = task.steps[0]
        assert step.node == "C"
        assert step.status.name == "stopped"
        # Final executor message should be recorded before stop
        assert len(step.executions) == 1
        ex = step.executions[0]
        assert ex.type.value == "executor"
        assert ex.is_complete is True
        assert ex.message is not None and ex.message.text == "done"


@pytest.mark.asyncio
async def test_edge_reset_policy_override_full_syntax(tmp_path: Path):
    # Graph:
    #   A --toB--> B (edge override keep_state)
    #   B --again--> B (edge override keep_state)
    #   B --done--> C
    # B default policy is always_reset, but both incoming edges override to keep_state so the same
    # B executor instance should be reused across the two B runs.
    nodes = [
        {"name": "A", "type": "afull", "outcomes": [OutcomeSlot(name="toB")]},
        {
            "name": "B",
            "type": "bfull",
            "outcomes": [OutcomeSlot(name="again"), OutcomeSlot(name="done")],
            "reset_policy": "always_reset",
        },
        {"name": "C", "type": "cfull", "outcomes": []},
    ]
    edges = [
        Edge(source_node="A", source_outcome="toB", target_node="B", reset_policy="keep_state"),
        Edge(source_node="B", source_outcome="again", target_node="B", reset_policy="keep_state"),
        Edge(source_node="B", source_outcome="done", target_node="C"),
    ]
    g = Graph.build(nodes=nodes, edges=edges)
    workflow = Workflow(name="workflow", graph=g)

    class AFullExec(Executor):
        type = "afull"
        async def run(self, inp):
            yield (ReqFinalMessage(message=msg("agent", "A1"), outcome_name="toB"), None)

    class BFullExec(Executor):
        type = "bfull"
        next_id = 1
        def __init__(self, config, project):
            super().__init__(config, project)
            self.instance_id = type(self).next_id
            type(self).next_id += 1
            self._runs = 0

        async def run(self, inp):
            if self._runs == 0:
                self._runs += 1
                yield (
                    ReqFinalMessage(
                        message=msg("agent", f"B1:{self.instance_id}"),
                        outcome_name="again",
                    ),
                    None,
                )
            else:
                self._runs += 1
                yield (
                    ReqFinalMessage(
                        message=msg("agent", f"B2:{self.instance_id}"),
                        outcome_name="done",
                    ),
                    None,
                )

    class CFullExec(Executor):
        type = "cfull"
        async def run(self, inp):
            yield (ReqFinalMessage(message=msg("agent", "C")), None)

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        # A final -> approve to move to B
        ev_a = await it.__anext__()
        assert ev_a.node == "A" and ev_a.event.kind == "final_message"
        ev_b1 = await it.asend(None)

        # First B final
        assert ev_b1.node == "B" and ev_b1.event.kind == "final_message"
        id1 = int(ev_b1.event.message.text.split(":", 1)[1])

        # Approve -> B again (override keep_state => reuse same instance)
        ev_b2 = await it.asend(None)
        assert ev_b2.node == "B" and ev_b2.event.kind == "final_message"
        id2 = int(ev_b2.event.message.text.split(":", 1)[1])
        assert id2 == id1  # instance reused due to edge override

        # Approve -> C
        ev_c = await it.asend(None)
        assert ev_c.node == "C" and ev_c.event.kind == "final_message"

        # Approve -> finish
        with pytest.raises(StopAsyncIteration):
            await it.asend(None)
