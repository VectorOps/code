import asyncio
import pytest
from vocode.ui.base import UIState
from vocode.runner.runner import Executor
from vocode.runner.models import ReqFinalMessage, ReqStartWorkflow, RespMessage, RespApproval, RunInput
from vocode.models import Node, Workflow, Graph
from vocode.state import Message, RunnerStatus
from vocode.runner.executors.noop import NoopNode
from vocode.ui.proto import UIPacketEnvelope, UIPacketRunInput, PACKET_RUN_EVENT, PACKET_STATUS
from vocode.testing.ui import FakeProject


#
# TestProject helper for tests
# Use TestProject in tests below via TestProject.with_workflows(...)
#


class StartWorkflowExecutor(Executor):
    type = "start_workflow_test"

    async def run(self, inp):
        # Match InputExecutor semantics:
        # - If we have a response (child final), finalize with it.
        # - Otherwise, request starting the child workflow once.
        if isinstance(inp.response, RespMessage):
            yield ReqFinalMessage(message=inp.response.message), inp.state
            return
        if not inp.state:
            yield ReqStartWorkflow(
                workflow="child", initial_message=Message(role="user", text="hi child")
            ), True
            return
        # Already started; runner will continue waiting for the child response.


# Register test executor
Executor.register("start_workflow_test", StartWorkflowExecutor)


@pytest.mark.asyncio
async def test_stacked_runner_end_to_end():
    parent_nodes = [Node(name="parent", type="start_workflow_test", outcomes=[])]
    child_nodes = [NoopNode(name="noop", type="noop", outcomes=[])]

    class WF:
        def __init__(self, nodes, edges):
            self.nodes = nodes
            self.edges = edges

    project = FakeProject.with_workflows({"parent": WF(parent_nodes, []), "child": WF(child_nodes, [])})

    ui = UIState(project)
    await ui.start_by_name("parent")

    parent_final = None
    while True:
        env = await asyncio.wait_for(ui.recv(), timeout=2)
        pkt = env.payload
        # Reply to input requests (confirmation/prompt) to unblock the runner
        if pkt.kind == "run_event" and pkt.event.input_requested:
            # Approve final to accept it without additional user message
            await ui.send(
                UIPacketEnvelope(
                    msg_id=0,
                    source_msg_id=env.msg_id,
                    payload=UIPacketRunInput(
                        input=RunInput(response=RespApproval(approved=True))
                    ),
                )
            )
        if pkt.kind == "status" and pkt.curr == RunnerStatus.finished:
            break
        if pkt.kind == "run_event":
            ev = pkt.event
            if (
                ev.node == "parent"
                and ev.event.kind == "final_message"
                and ev.event.message
            ):
                parent_final = ev.event.message.text
    # Parent final should be the child's final (empty string from Noop)
    assert parent_final is not None


@pytest.mark.asyncio
async def test_stop_restart_top_only():
    parent_nodes = [Node(name="parent", type="start_workflow_test", outcomes=[])]
    child_nodes = [NoopNode(name="noop", type="noop", outcomes=[])]
    WF = lambda n, e: type("WF", (), {"nodes": n, "edges": e})
    project = FakeProject.with_workflows({"parent": WF(parent_nodes, []), "child": WF(child_nodes, [])})
    ui = UIState(project)
    await ui.start_by_name("parent")
    # Allow child to be pushed
    await asyncio.sleep(0.05)
    # Stop top (child)
    await ui.stop(wait=True)
    assert ui.status in (RunnerStatus.stopped, RunnerStatus.finished)
    # Restart top (child)
    await ui.restart()
    # Wait for finish
    done = False
    while not done:
        env = await asyncio.wait_for(ui.recv(), timeout=5)
        print(env)
        pkt = env.payload
        # Reply to any input requests after restart
        if pkt.kind == "run_event" and pkt.event.input_requested:
            await ui.send(
                UIPacketEnvelope(
                    msg_id=0,
                    source_msg_id=env.msg_id,
                    payload=UIPacketRunInput(
                        input=RunInput(response=RespApproval(approved=True))
                    ),
                )
            )
        if pkt.kind == "status" and pkt.curr == RunnerStatus.finished:
            done = True
    assert done


@pytest.mark.asyncio
async def test_replace_user_input_raises_when_not_possible():
    project = FakeProject.with_workflows({})
    ui = UIState(project)
    with pytest.raises(RuntimeError):
        await ui.replace_user_input(RespMessage(message=Message(role="user", text="x")))
