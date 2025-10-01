import pytest
import asyncio
from pathlib import Path
from vocode.testing import ProjectSandbox
from vocode.models import Graph, Node, Edge, OutcomeSlot, Workflow
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
from vocode.state import (
    Message,
    ToolCall,
    ToolCallStatus,
    Assignment,
    RunnerStatus,
    StepStatus,
)
from vocode.ui.base import UIState
from vocode.ui.proto import UIPacketRunEvent, UIPacketStatus
from vocode.runner.models import PACKET_FINAL_MESSAGE
from vocode.runner.executors.file_state import FileStateNode
from vocode.commands import CommandContext


def msg(role: str, text: str) -> Message:
    return Message(role=role, text=text)


@pytest.mark.asyncio
async def test_tool_call_auto_approved_no_prompt(tmp_path: Path):
    # Executor emits a tool call with auto_approve=True; runner should not prompt
    nodes = [{"name": "Tool", "type": "tool_auto", "outcomes": []}]
    g = Graph(nodes=nodes, edges=[])
    workflow = Workflow(name="workflow", graph=g)

    class ToolAutoExec(Executor):
        type = "tool_auto"

        async def run(self, inp):
            if inp.response is None:
                tc = ToolCall(name="nonexistent", arguments={}, auto_approve=True)
                yield (ReqToolCall(tool_calls=[tc]), None)
                return
            # After tool response, finish
            yield (ReqFinalMessage(message=msg("agent", "done")), None)

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        ev1 = await it.__anext__()
        assert ev1.event.kind == "tool_call"
        assert ev1.input_requested is False  # auto-approved => no prompt
        ev2 = await it.asend(RunInput(response=None))  # no approval needed
        assert ev2.event.kind == "final_message"


@pytest.mark.asyncio
async def test_message_request_reprompt_and_finish(tmp_path: Path):
    # Single-node graph with type 'ask'
    nodes = [{"name": "Ask", "type": "ask", "outcomes": [], "confirmation": "confirm"}]
    g = Graph(nodes=nodes, edges=[])
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
        # Four activities: executor message_request, user input, final, user approval
        assert len(step.executions) == 4
        assert step.executions[0].type.value == "executor"  # message_request
        assert step.executions[0].message is None
        assert step.executions[1].type.value == "user"
        assert step.executions[1].message.text == "hi"
        assert step.executions[2].type.value == "executor"
        assert step.executions[2].is_complete is True
        assert step.executions[2].message is not None
        assert step.executions[2].message.text == "final:hi"
        assert step.executions[3].type.value == "user"  # approval (no message payload)
        assert step.executions[3].message is None


@pytest.mark.asyncio
async def test_rewind_one_step_and_resume(tmp_path: Path):
    # Graph: A -> B
    nodes = [
        {"name": "A", "type": "a", "outcomes": [OutcomeSlot(name="toB")]},
        {"name": "B", "type": "b", "outcomes": []},
    ]
    edges = [Edge(source_node="A", source_outcome="toB", target_node="B")]
    g = Graph(nodes=nodes, edges=edges)
    workflow = Workflow(name="workflow", graph=g)

    class AExec(Executor):
        type = "a"

        async def run(self, inp):
            yield (
                ReqFinalMessage(message=msg("agent", "A1"), outcome_name="toB"),
                None,
            )

    class BExec(Executor):
        type = "b"
        runs = 0

        async def run(self, inp):
            type(self).runs += 1
            yield (ReqFinalMessage(message=msg("agent", f"B{type(self).runs}")), None)

    async with ProjectSandbox.create(tmp_path) as project:
        BExec.runs = 0  # Reset for idempotency
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        # Complete A then B
        ev_a = await it.__anext__()
        assert ev_a.node == "A" and ev_a.event.kind == "final_message"
        ev_b = await it.asend(None)
        assert ev_b.node == "B" and ev_b.event.kind == "final_message"
        assert ev_b.event.message.text == "B1"
        with pytest.raises(StopAsyncIteration):
            await it.asend(None)

        assert runner.status == RunnerStatus.finished
        assert BExec.runs == 1

        # Rewind 2 activities to undo step B completely
        await runner.rewind(task, n=2)
        assert runner.status == RunnerStatus.stopped
        assert len(task.steps) == 1  # Step B removed

        # Resume, B should run again
        it2 = runner.run(task)
        ev_b2 = await it2.__anext__()

        import devtools

        devtools.pprint(task)

        assert ev_b2.node == "B" and ev_b2.event.kind == "final_message"
        assert ev_b2.event.message.text == "B2"
        with pytest.raises(StopAsyncIteration):
            await it2.asend(None)

        assert runner.status == RunnerStatus.finished
        assert BExec.runs == 2


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_natural_resume_after_stop(tmp_path: Path):
    # Graph: A -> B -> C
    nodes = [
        {"name": "A", "type": "a_resume", "outcomes": [OutcomeSlot(name="toB")]},
        {"name": "B", "type": "b_resume", "outcomes": [OutcomeSlot(name="toC")]},
        {"name": "C", "type": "c_resume", "outcomes": []},
    ]
    edges = [
        Edge(source_node="A", source_outcome="toB", target_node="B"),
        Edge(source_node="B", source_outcome="toC", target_node="C"),
    ]
    g = Graph(nodes=nodes, edges=edges)
    workflow = Workflow(name="workflow", graph=g)

    # Use class attributes to track runs across executor instances
    class AExec(Executor):
        type = "a_resume"
        runs = 0

        async def run(self, inp):
            type(self).runs += 1
            yield (ReqFinalMessage(message=msg("agent", "A"), outcome_name="toB"), None)

    class BExec(Executor):
        type = "b_resume"
        runs = 0

        async def run(self, inp):
            type(self).runs += 1
            yield (ReqFinalMessage(message=msg("agent", "B"), outcome_name="toC"), None)

    class CExec(Executor):
        type = "c_resume"
        runs = 0

        async def run(self, inp):
            type(self).runs += 1
            yield (ReqFinalMessage(message=msg("agent", "C")), None)

    async with ProjectSandbox.create(tmp_path) as project:
        # Reset counters for clean test run
        AExec.runs = 0
        BExec.runs = 0
        CExec.runs = 0

        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        # Run A, then B. Stop after B yields its final message but before it's processed.
        ev_a = await it.__anext__()
        assert ev_a.node == "A"
        ev_b = await it.asend(None)
        assert ev_b.node == "B"

        runner.stop()
        await it.aclose()  # Simulates UI driver stopping/closing the generator

        assert runner.status == RunnerStatus.stopped
        assert AExec.runs == 1
        assert BExec.runs == 1
        assert CExec.runs == 0
        assert len(task.steps) == 2
        assert task.steps[0].status == "finished"
        assert task.steps[1].status == "running"  # B was started but not finished

        # Resume. Natural resume should find that B request was started, but not finished
        # and should start continue execution with B.
        it2 = runner.run(task)
        ev_b2 = await it2.__anext__()
        assert ev_b2.node == "B" and ev_b2.event.kind == "final_message"

        ev_c2 = await it2.asend(None)
        assert ev_c2.node == "C" and ev_c2.event.kind == "final_message"

        with pytest.raises(StopAsyncIteration):
            await it2.asend(None)

        assert runner.status == RunnerStatus.finished
        # A ran once, B ran twice (once interrupted, once on resume), C ran once.
        assert AExec.runs == 1
        assert BExec.runs == 1
        assert CExec.runs == 1
        assert len(task.steps) == 3
        assert [s.node for s in task.steps] == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_replace_user_input_with_step_index(tmp_path: Path):
    """Scenario: Run a multi-step workflow, then replace an input from an early step."""
    nodes = [
        {"name": "A", "type": "a_replace", "outcomes": [OutcomeSlot(name="next")]},
        {"name": "B", "type": "b_replace", "outcomes": []},
    ]
    edges = [Edge(source_node="A", source_outcome="next", target_node="B")]
    g = Graph(nodes=nodes, edges=edges)
    workflow = Workflow(name="w", graph=g)

    class AExec(Executor):
        type = "a_replace"

        async def run(self, inp):
            if inp.response and inp.response.kind == "message":
                yield (
                    ReqFinalMessage(
                        message=msg("agent", f"A saw:{inp.response.message.text}"),
                        outcome_name="next",
                    ),
                    None,
                )
            else:
                yield (ReqMessageRequest(), None)

    class BExec(Executor):
        type = "b_replace"

        async def run(self, inp):
            yield (ReqFinalMessage(message=msg("agent", "B done")), None)

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        # Run 1: A asks, user says "A1"
        ev1 = await it.__anext__()
        assert ev1.event.kind == "message_request" and ev1.node == "A"
        ev2 = await it.asend(RunInput(response=RespMessage(message=msg("user", "A1"))))
        assert ev2.event.kind == "final_message" and ev2.node == "A"
        assert "A saw:A1" in ev2.event.message.text

        # Run 1: B runs
        ev3 = await it.asend(None)
        assert ev3.event.kind == "final_message" and ev3.node == "B"
        with pytest.raises(StopAsyncIteration):
            await it.asend(None)

        assert runner.status == RunnerStatus.finished
        assert len(task.steps) == 2

        # Replace input at step 0 (node A)
        runner.replace_user_input(
            task, RespMessage(message=msg("user", "A2")), step_index=0
        )
        assert runner.status == RunnerStatus.stopped

        # Run 2: starts from A with new input
        it2 = runner.run(task)
        ev2_1 = await it2.__anext__()
        assert ev2_1.event.kind == "final_message" and ev2_1.node == "A"
        assert "A saw:A2" in ev2_1.event.message.text

        # Run 2: B runs
        ev2_2 = await it2.asend(None)
        assert ev2_2.event.kind == "final_message" and ev2_2.node == "B"
        with pytest.raises(StopAsyncIteration):
            await it2.asend(None)

        assert runner.status == RunnerStatus.finished
        assert len(task.steps) == 2


@pytest.mark.asyncio
async def test_replace_input_for_pending_request(tmp_path: Path):
    """Scenario: Input node is waiting for input, runner is stopped.
    A new message should be provided to the input node on restart."""
    nodes = [{"name": "In", "type": "input", "outcomes": []}]
    g = Graph(nodes=nodes, edges=[])
    workflow = Workflow(name="w", graph=g)

    class InputExec(Executor):
        type = "input"

        async def run(self, inp):
            if inp.response and inp.response.kind == "message":
                assert inp.response.message.text == "replaced"
                yield (ReqFinalMessage(message=msg("agent", "ok")), None)
            else:
                yield (ReqMessageRequest(), None)

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        ev1 = await it.__anext__()
        assert ev1.event.kind == "message_request"

        runner.stop()
        await it.aclose()
        assert runner.status == RunnerStatus.stopped

        runner.replace_user_input(task, RespMessage(message=msg("user", "replaced")))

        it2 = runner.run(task)
        ev2 = await it2.__anext__()
        assert ev2.event.kind == "final_message"
        assert ev2.event.message.text == "ok"
        with pytest.raises(StopAsyncIteration):
            await it2.asend(None)
        assert runner.status == RunnerStatus.finished


@pytest.mark.asyncio
async def test_replace_input_for_pending_request_with_approval_is_ignored(
    tmp_path: Path,
):
    """Scenario: Input node waiting, stopped, then an approval is provided.
    The approval should be ignored, and the input node should re-request input."""
    nodes = [{"name": "In", "type": "input", "outcomes": []}]
    g = Graph(nodes=nodes, edges=[])
    workflow = Workflow(name="w", graph=g)

    class InputExec(Executor):
        type = "input"
        runs = 0

        async def run(self, inp):
            print("RUN")
            type(self).runs += 1
            # Should not receive approval; if it does, it will be ignored by logic
            assert not (inp.response and inp.response.kind == "approval")
            yield (ReqMessageRequest(), None)

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)
        await it.__anext__()  # Get message_request

        runner.stop()
        await it.aclose()

        runner.replace_user_input(task, RespApproval(approved=True))

        it2 = runner.run(task)
        ev2 = await it2.__anext__()
        assert ev2.event.kind == "message_request"
        assert InputExec.runs == 1

        runner.stop()
        await it2.aclose()


@pytest.mark.asyncio
async def test_replace_input_at_retriable_final_message(tmp_path: Path):
    """Scenario: Stop after a retriable final message.
    Replacement with approval should resume from that final message."""
    nodes = [
        {
            "name": "In1",
            "type": "in1",
            "outcomes": [OutcomeSlot(name="next")],
            "confirmation": "prompt",
        },
        {"name": "Other", "type": "other", "outcomes": []},
    ]
    edges = [Edge(source_node="In1", source_outcome="next", target_node="Other")]
    g = Graph(nodes=nodes, edges=edges)
    workflow = Workflow(name="w", graph=g)

    class Input1Exec(Executor):
        type = "in1"

        async def run(self, inp):
            yield (
                ReqFinalMessage(message=msg("agent", "in1 done"), outcome_name="next"),
                None,
            )

    class OtherExec(Executor):
        type = "other"

        async def run(self, inp):
            yield (ReqMessageRequest(), None)

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project, initial_message=msg("user", "start"))
        task = Assignment()
        it = runner.run(task)
        ev_in1_final = await it.__anext__()
        assert ev_in1_final.node == "In1" and ev_in1_final.event.kind == "final_message"

        # No user message provided for prompt, so transition to Other
        ev_other_req = await it.asend(RunInput(response=None))
        assert (
            ev_other_req.node == "Other"
            and ev_other_req.event.kind == "message_request"
        )

        runner.stop()
        await it.aclose()

        runner.replace_user_input(task, RespApproval(approved=True))
        it2 = runner.run(task)

        # Since we provided an approval, it should now transition and run Other.
        # We .asend(None) because the replay does not request new input from the UI.
        ev_other_req_again = await it2.asend(None)
        assert (
            ev_other_req_again.node == "Other"
            and ev_other_req_again.event.kind == "message_request"
        )


@pytest.mark.asyncio
async def test_replace_input_with_auto_confirm_node(tmp_path: Path):
    """Scenario: An auto-confirm input node runs, then next node starts.
    Replacement should rewind to the auto-confirm node."""
    nodes = [
        {
            "name": "In",
            "type": "in_auto",
            "outcomes": [OutcomeSlot(name="next")],
            "confirmation": "auto",
        },
        {"name": "B", "type": "b_auto", "outcomes": []},
    ]
    edges = [Edge(source_node="In", source_outcome="next", target_node="B")]
    g = Graph(nodes=nodes, edges=edges)
    workflow = Workflow(name="w", graph=g)

    class InputAutoExec(Executor):
        type = "in_auto"

        async def run(self, inp):
            if inp.response and inp.response.kind == "message":
                yield (
                    ReqFinalMessage(
                        message=msg("agent", f"in:{inp.response.message.text}"),
                        outcome_name="next",
                    ),
                    None,
                )
            else:
                yield (ReqMessageRequest(), None)

    class BExec(Executor):
        type = "b_auto"

        async def run(self, inp):
            yield (ReqMessageRequest(), None)

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        await it.__anext__()  # In reqs
        ev_in_final = await it.asend(
            RunInput(response=RespMessage(message=msg("user", "A")))
        )
        assert ev_in_final.node == "In" and ev_in_final.event.message.text == "in:A"
        assert ev_in_final.input_requested is False

        runner.stop()
        await it.aclose()

        runner.replace_user_input(task, RespMessage(message=msg("user", "B")))
        it2 = runner.run(task)

        # The runner should auto-respond with "B", causing "In" to emit its final message immediately.
        ev_in_final_again = await it2.__anext__()
        assert ev_in_final_again.node == "In"
        assert (
            ev_in_final_again.node == "In"
            and ev_in_final_again.event.message.text == "in:B"
        )
        assert not ev_in_final_again.input_requested  # auto-confirm node

        # Now we can transition to B
        ev_b_req = await it2.asend(None)
        assert ev_b_req.node == "B" and ev_b_req.event.kind == "message_request"


@pytest.mark.asyncio
async def test_reset_policy_keep_final_self_loop_only_final_carried(tmp_path: Path):
    # Graph: K (keep_final, self-loop once) -> End
    nodes = [
        {
            "name": "K",
            "type": "kf",
            "outcomes": [OutcomeSlot(name="again"), OutcomeSlot(name="done")],
            "reset_policy": "keep_final",
            "confirmation": "auto",
        },
        {"name": "End", "type": "kend", "outcomes": []},
    ]
    edges = [
        Edge(source_node="K", source_outcome="again", target_node="K"),
        Edge(source_node="K", source_outcome="done", target_node="End"),
    ]
    g = Graph(nodes=nodes, edges=edges)
    workflow = Workflow(name="workflow", graph=g)

    class KeepFinalExec(Executor):
        type = "kf"
        runs = 0

        async def run(self, inp):
            type(self).runs += 1
            if type(self).runs == 1:
                # First run: should receive only initial user message
                assert [m.text for m in inp.messages] == ["start"]
                # Emit an interim, which must NOT persist, then final to loop
                yield (ReqInterimMessage(message=msg("agent", "INT1")), None)
                yield (
                    ReqFinalMessage(message=msg("agent", "F1"), outcome_name="again"),
                    None,
                )
            else:
                # Second run: keep_final should only pass the previous final "F1"
                assert [m.text for m in inp.messages] == ["F1"]
                yield (
                    ReqFinalMessage(message=msg("agent", "F2"), outcome_name="done"),
                    None,
                )

    class EndExec(Executor):
        type = "kend"

        async def run(self, inp):
            yield (ReqFinalMessage(message=msg("agent", "END")), None)

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project, initial_message=msg("user", "start"))
        task = Assignment()
        it = runner.run(task)
        # K first run: interim, then final (auto) -> transitions to K again
        ev_k_int1 = await it.__anext__()
        assert ev_k_int1.node == "K"
        assert ev_k_int1.event.kind == "message"
        assert ev_k_int1.event.message.text == "INT1"

        ev_k1 = await it.__anext__()
        assert (
            ev_k1.node == "K"
            and ev_k1.event.kind == "final_message"
            and ev_k1.event.message.text == "F1"
        )

        ev_k2 = await it.asend(None)
        assert (
            ev_k2.node == "K"
            and ev_k2.event.kind == "final_message"
            and ev_k2.event.message.text == "F2"
        )
        # End
        ev_end = await it.asend(None)
        assert ev_end.node == "End" and ev_end.event.kind == "final_message"
        with pytest.raises(StopAsyncIteration):
            await it.asend(None)


@pytest.mark.asyncio
async def test_keep_results_excludes_interim_messages(tmp_path: Path):
    # Graph: A(keep_results, all_messages) -> B(auto) -> A -> C
    nodes = [
        {
            "name": "A",
            "type": "a_int",
            "outcomes": [OutcomeSlot(name="toB"), OutcomeSlot(name="toC")],
            "reset_policy": "keep_results",
            "message_mode": "all_messages",  # pass inputs + final to B
        },
        {
            "name": "B",
            "type": "b_int",
            "outcomes": [OutcomeSlot(name="toA")],
            "confirmation": "auto",
        },
        {"name": "C", "type": "c_int", "outcomes": []},
    ]
    edges = [
        Edge(source_node="A", source_outcome="toB", target_node="B"),
        Edge(source_node="B", source_outcome="toA", target_node="A"),
        Edge(source_node="A", source_outcome="toC", target_node="C"),
    ]
    g = Graph(nodes=nodes, edges=edges)
    workflow = Workflow(name="workflow", graph=g)

    class AExec(Executor):
        type = "a_int"
        runs = 0

        async def run(self, inp):
            type(self).runs += 1
            if type(self).runs == 1:
                # First run: initial message only
                assert [m.text for m in inp.messages] == ["start"]
                # Emit interim (should not persist), then final to B
                yield (ReqInterimMessage(message=msg("agent", "INT_A1")), None)
                yield (
                    ReqFinalMessage(message=msg("agent", "A1"), outcome_name="toB"),
                    None,
                )
            else:
                # Second run via keep_results: should include prior A user+final plus B's final; no interim
                texts = [m.text for m in inp.messages]
                assert texts == [
                    "start",
                    "A1",
                    "B1",
                ], f"unexpected carried messages: {texts}"
                yield (
                    ReqFinalMessage(message=msg("agent", "A2"), outcome_name="toC"),
                    None,
                )

    class BExec(Executor):
        type = "b_int"
        runs = 0

        async def run(self, inp):
            type(self).runs += 1
            # Receives initial + A1 due to all_messages
            assert [m.text for m in inp.messages] == ["start", "A1"]
            yield (ReqFinalMessage(message=msg("agent", "B1")), None)

    class CExec(Executor):
        type = "c_int"
        runs = 0

        async def run(self, inp):
            type(self).runs += 1
            yield (ReqFinalMessage(message=msg("agent", "C1")), None)

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project, initial_message=msg("user", "start"))
        task = Assignment()
        it = runner.run(task)
        # A1 -> B1(auto) -> A2 -> C
        ev_a1_int = await it.__anext__()
        assert (
            ev_a1_int.node == "A"
            and ev_a1_int.event.kind == "message"
            and ev_a1_int.event.message.text == "INT_A1"
        )

        ev_a1 = await it.__anext__()
        assert ev_a1.node == "A" and ev_a1.event.kind == "final_message"
        ev_b1 = await it.asend(None)
        assert ev_b1.node == "B" and ev_b1.event.kind == "final_message"
        ev_a2 = await it.asend(None)
        assert ev_a2.node == "A" and ev_a2.event.kind == "final_message"
        ev_c1 = await it.asend(None)
        assert ev_c1.node == "C" and ev_c1.event.kind == "final_message"
        with pytest.raises(StopAsyncIteration):
            await it.asend(None)
        assert runner.status == RunnerStatus.finished
        assert [s.node for s in task.steps] == ["A", "B", "A", "C"]

        # Rewind last step (C)
        await runner.rewind(task, 2)
        assert runner.status == RunnerStatus.stopped
        assert [s.node for s in task.steps] == ["A", "B", "A"]

        # Resume -> should start at C again, re-executing it.
        it2 = runner.run(task)
        ev_c2 = await it2.__anext__()
        assert ev_c2.node == "C" and ev_c2.event.kind == "final_message"
        # Ensure only C's executor was re-executed on resume.
        assert AExec.runs == 2
        assert BExec.runs == 1
        assert CExec.runs == 2

        with pytest.raises(StopAsyncIteration):
            await it2.asend(None)
        assert runner.status == RunnerStatus.finished
        assert [s.node for s in task.steps] == ["A", "B", "A", "C"]


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
    g = Graph(nodes=nodes, edges=edges)
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
        CExec.runs = 0
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        # Finish A -> B -> C
        ev_a = await it.__anext__()
        assert ev_a.node == "A" and ev_a.event.kind == "final_message"
        ev_b = await it.asend(None)
        assert ev_b.node == "B" and ev_b.event.kind == "final_message"
        ev_c = await it.asend(None)
        assert (
            ev_c.node == "C"
            and ev_c.event.kind == "final_message"
            and ev_c.event.message.text == "C1"
        )
        with pytest.raises(StopAsyncIteration):
            await it.asend(None)
        assert runner.status == RunnerStatus.finished
        assert [s.node for s in task.steps] == ["A", "B", "C"]

        # Rewind last two steps (B, C)
        await runner.rewind(task, 4)
        assert runner.status == RunnerStatus.stopped
        assert [s.node for s in task.steps] == ["A"]

        # Resume -> should execute B then C again
        it2 = runner.run(task)
        ev_b2 = await it2.__anext__()
        assert ev_b2.node == "B" and ev_b2.event.kind == "final_message"
        ev_c2 = await it2.asend(None)
        assert (
            ev_c2.node == "C"
            and ev_c2.event.kind == "final_message"
            and ev_c2.event.message.text == "C2"
        )
        with pytest.raises(StopAsyncIteration):
            await it2.asend(None)
        assert runner.status == RunnerStatus.finished
        assert [s.node for s in task.steps] == ["A", "B", "C"]
        assert CExec.runs == 2


@pytest.mark.asyncio
async def test_rewind_within_step_and_resume(tmp_path: Path):
    """Scenario: A single step has multiple retriable activities.
    Rewinding by 1 should remove only the last activity, not the whole step."""
    nodes = [{"name": "A", "type": "a_rewind_step", "outcomes": []}]
    g = Graph(nodes=nodes, edges=[])
    workflow = Workflow(name="workflow", graph=g)

    class AExec(Executor):
        type = "a_rewind_step"

        async def run(self, inp):
            count = (inp.state or {}).get("count", 0)

            if inp.response:
                count += 1

            if count < 2:
                yield (ReqMessageRequest(), {"count": count})
            else:
                yield (
                    ReqFinalMessage(message=msg("agent", "A done")),
                    {"count": count},
                )

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        # Run through to completion
        ev1 = await it.__anext__()  # req1
        assert ev1.event.kind == "message_request"
        ev2 = await it.asend(
            RunInput(response=RespMessage(message=msg("user", "msg1")))
        )
        assert ev2.event.kind == "message_request"
        ev3 = await it.asend(
            RunInput(response=RespMessage(message=msg("user", "msg2")))
        )
        assert ev3.event.kind == "final_message"
        with pytest.raises(StopAsyncIteration):
            await it.asend(None)

        assert runner.status == RunnerStatus.finished
        assert len(task.steps) == 1
        step = task.steps[0]
        retriable_count = sum(1 for ex in step.executions if not ex.ephemeral)
        assert (
            retriable_count == 6
        )  # req1, user_resp1, req2, user_resp2, final1, approve

        # Rewind 2 to undo the last user message and the executor's final response
        await runner.rewind(task, n=3)
        assert runner.status == RunnerStatus.stopped
        assert len(task.steps) == 1  # Step should NOT be removed
        step = task.steps[0]
        assert step.status == StepStatus.running
        retriable_count_after = sum(1 for ex in step.executions if not ex.ephemeral)
        assert (
            retriable_count_after == 3
        )  # req1, user_resp1, req2, approve should remain

        # Resume. The runner will find the step for node A in a 'running' state
        # and will continue execution from within that step.
        it2 = runner.run(task)
        ev4 = await it2.__anext__()
        assert ev4.event.kind == "message_request"


@pytest.mark.asyncio
async def test_tool_call_approved_and_rejected(tmp_path: Path):
    nodes = [{"name": "Tool", "type": "tool", "outcomes": []}]
    g = Graph(nodes=nodes, edges=[])
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
            if (
                inp.response.kind == "tool_call"
                and inp.response.tool_calls
                and inp.response.tool_calls[0].name == "one"
            ):
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
    g = Graph(nodes=nodes, edges=[])
    workflow = Workflow(name="workflow", graph=g)

    class EchoExecutor(Executor):
        type = "echo"

        async def run(self, inp):
            if not inp.response or inp.response.kind != "message":
                # First final: trigger post-final user message
                yield (ReqFinalMessage(message=msg("agent", "ask")), None)
                return
            # Second cycle: user provided a message
            assert (
                inp.response.message.role == "user"
                and inp.response.message.text == "more"
            )
            yield (ReqFinalMessage(message=msg("agent", "done")), None)

    async with ProjectSandbox.create(tmp_path) as project:
        runner = Runner(workflow, project)
        task = Assignment()
        it = runner.run(task)

        # First: final_message (requesting input)
        ev1 = await it.__anext__()
        assert ev1.event.kind == "final_message"
        # Provide a user message to continue same executor (not added to history)
        ev2 = await it.asend(
            RunInput(response=RespMessage(message=msg("user", "more")))
        )
        assert ev2.event.kind == "final_message"
        with pytest.raises(StopAsyncIteration):
            await it.asend(None)

        # Verify two executions within the same step; no user message recorded
        assert runner.status.name == "finished"
        assert len(task.steps) == 1
        step = task.steps[0]
        assert len(step.executions) == 4
        assert step.executions[0].type.value == "executor"
        assert step.executions[0].message.text == "ask"
        assert step.executions[1].type.value == "user"
        assert step.executions[1].message is not None
        assert step.executions[1].message.text == "more"
        assert step.executions[2].type.value == "executor"
        assert step.executions[2].is_complete is True
        assert step.executions[2].message.text == "done"
        assert step.executions[3].type.value == "user"
        assert step.executions[3].is_complete is True
        assert step.executions[3].message is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["all_messages", "final_response"])
async def test_transition_to_next_node_and_pass_all_messages(mode, tmp_path: Path):
    nodes = [
        {
            "name": "A",
            "type": "a",
            "outcomes": [OutcomeSlot(name="toB")],
            "message_mode": mode,
        },
        {"name": "B", "type": "b", "outcomes": []},
    ]
    edges = [Edge(source_node="A", source_outcome="toB", target_node="B")]
    g = Graph(nodes=nodes, edges=edges)
    workflow = Workflow(name="workflow", graph=g)

    class AExec(Executor):
        type = "a"

        async def run(self, inp):
            # Emit final with outcome to go to B
            yield (
                ReqFinalMessage(message=msg("agent", "Aout"), outcome_name="toB"),
                None,
            )

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

    # Step A: initial user message, A final, and an implicit user approval for 'auto' confirmation.
    assert len(task.steps[0].executions) == 3
    assert task.steps[0].executions[0].type.value == "user"
    assert task.steps[0].executions[0].message.text == "sys"
    assert task.steps[0].executions[1].type.value == "executor"
    assert task.steps[0].executions[1].message.text == "Aout"
    assert task.steps[0].executions[2].type.value == "user"
    assert task.steps[0].executions[2].message is None

    if mode == "all_messages":
        # Step B: carried inputs (sys + Aout), B final, and implicit user approval.
        assert len(task.steps[1].executions) == 4
        assert task.steps[1].executions[0].type.value == "user"
        assert task.steps[1].executions[0].message.text == "sys"
        assert task.steps[1].executions[1].type.value == "user"
        assert task.steps[1].executions[1].message.text == "Aout"
        assert task.steps[1].executions[2].type.value == "executor"
        assert task.steps[1].executions[2].message.text == "Bout"
        assert task.steps[1].executions[3].type.value == "user"
        assert task.steps[1].executions[3].message is None
    else:
        # Step B: carried input (Aout), B final, and implicit user approval.
        assert len(task.steps[1].executions) == 3
        assert task.steps[1].executions[0].type.value == "user"
        assert task.steps[1].executions[0].message.text == "Aout"
        assert task.steps[1].executions[1].type.value == "executor"
        assert task.steps[1].executions[1].message.text == "Bout"
        assert task.steps[1].executions[2].type.value == "user"
        assert task.steps[1].executions[2].message is None


@pytest.mark.asyncio
async def test_error_multiple_outcomes_without_outcome_name(tmp_path: Path):
    nodes = [
        {
            "name": "A",
            "type": "errA",
            "outcomes": [OutcomeSlot(name="x"), OutcomeSlot(name="y")],
        },
        {"name": "Bx", "type": "b1", "outcomes": []},
        {"name": "By", "type": "b2", "outcomes": []},
    ]
    edges = [
        Edge(source_node="A", source_outcome="x", target_node="Bx"),
        Edge(source_node="A", source_outcome="y", target_node="By"),
    ]
    g = Graph(nodes=nodes, edges=edges)
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
        with pytest.raises(
            ValueError, match="did not provide an outcome and has 2 outcomes"
        ):
            await it.asend(None)


@pytest.mark.asyncio
async def test_error_unknown_outcome_name(tmp_path: Path):
    nodes = [
        {"name": "A", "type": "err2", "outcomes": [OutcomeSlot(name="go")]},
        {"name": "B", "type": "b", "outcomes": []},
    ]
    edges = [Edge(source_node="A", source_outcome="go", target_node="B")]
    g = Graph(nodes=nodes, edges=edges)
    workflow = Workflow(name="workflow", graph=g)

    class Err2Exec(Executor):
        type = "err2"

        async def run(self, inp):
            yield (
                ReqFinalMessage(message=msg("agent", "oops"), outcome_name="unknown"),
                None,
            )

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
        with pytest.raises(
            ValueError, match="No edge defined from node 'A' via outcome 'unknown'"
        ):
            await it.asend(None)


@pytest.mark.asyncio
async def test_run_disallowed_when_running(tmp_path: Path):
    nodes = [{"name": "Block", "type": "block", "outcomes": []}]
    g = Graph(nodes=nodes, edges=[])
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
async def test_reset_policy_auto_confirmation_and_single_outcome_transition(
    tmp_path: Path,
):
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
    g = Graph(nodes=nodes, edges=edges)
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
                yield (
                    ReqFinalMessage(message=msg("agent", "A1"), outcome_name="toB"),
                    None,
                )
            else:
                # Second run via keep_results: receives all prior A messages + new input from B
                assert [m.text for m in messages] == ["start", "A1", "B1"]
                yield (
                    ReqFinalMessage(message=msg("agent", "A2"), outcome_name="toC"),
                    None,
                )

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
    g = Graph(nodes=nodes, edges=edges)
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
                yield (
                    ReqFinalMessage(
                        message=msg("agent", f"B1:{self.instance_id}"),
                        outcome_name="again",
                    ),
                    None,
                )
            else:
                type(self).global_runs += 1
                # Second final: go to C
                yield (
                    ReqFinalMessage(
                        message=msg("agent", f"B2:{self.instance_id}"),
                        outcome_name="done",
                    ),
                    None,
                )

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
    g = Graph(nodes=nodes, edges=[])
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

        # Verify the final executor message and its implicit approval are recorded (logs are ephemeral).
        assert runner.status.name == "finished"
        assert len(task.steps) == 1
        step = task.steps[0]
        assert step.node == "Log"
        assert len(step.executions) == 2
        assert step.executions[0].type.value == "executor"
        assert step.executions[0].is_complete is True
        assert step.executions[0].message is not None
        assert step.executions[0].message.text == "done"
        assert step.executions[1].type.value == "user"
        assert step.executions[1].is_complete is True
        assert step.executions[1].message is None


@pytest.mark.asyncio
async def test_final_message_confirm_requires_explicit_approval_and_reprompts(
    tmp_path: Path,
):
    # Single-node graph with confirmation=confirm
    nodes = [{"name": "C", "type": "conf", "outcomes": [], "confirmation": "confirm"}]
    g = Graph(nodes=nodes, edges=[])
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
        ev2 = await it.asend(
            RunInput(response=RespMessage(message=msg("user", "not approval")))
        )
        assert ev2.node == "C"
        assert ev2.event.kind == "final_message"
        assert ev2.input_requested is True
        assert ev2.event.message is not None and ev2.event.message.text == "done"

        # Now approve explicitly -> runner should finish
        with pytest.raises(StopAsyncIteration):
            await it.asend(RunInput(response=RespApproval(approved=True)))

        # Runner state and recorded activities: executor final + user approval are recorded
        assert runner.status.name == "finished"
        assert len(task.steps) == 1
        step = task.steps[0]
        assert step.node == "C"
        assert step.status.name == "finished"
        assert len(step.executions) == 2
        assert step.executions[0].type.value == "executor"
        assert step.executions[0].is_complete is True
        assert step.executions[0].message is not None
        assert step.executions[0].message.text == "done"
        assert step.executions[1].type.value == "user"
        assert step.executions[1].message is None


@pytest.mark.asyncio
async def test_final_message_confirm_reject_stops_runner(tmp_path: Path):
    nodes = [{"name": "C", "type": "conf", "outcomes": [], "confirmation": "confirm"}]
    g = Graph(nodes=nodes, edges=[])
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
        # Final executor message and user rejection are recorded before stop
        assert len(step.executions) == 2
        ex0 = step.executions[0]
        assert ex0.type.value == "executor"
        assert ex0.is_complete is True
        assert ex0.message is not None and ex0.message.text == "done"
        ex1 = step.executions[1]
        assert ex1.type.value == "user"
        assert ex1.message is None


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
        Edge(
            source_node="A",
            source_outcome="toB",
            target_node="B",
            reset_policy="keep_state",
        ),
        Edge(
            source_node="B",
            source_outcome="again",
            target_node="B",
            reset_policy="keep_state",
        ),
        Edge(source_node="B", source_outcome="done", target_node="C"),
    ]
    g = Graph(nodes=nodes, edges=edges)
    workflow = Workflow(name="workflow", graph=g)

    class AFullExec(Executor):
        type = "afull"

        async def run(self, inp):
            yield (
                ReqFinalMessage(message=msg("agent", "A1"), outcome_name="toB"),
                None,
            )

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
