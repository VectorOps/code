import asyncio
import pytest

from vocode.ui.base import UIState
from vocode.runner.models import RunInput, RespApproval
from vocode.models import StartWorkflowNode
from vocode.state import RunnerStatus
from vocode.runner.executors.noop import NoopNode
from vocode.runner.executors.start_workflow import StartWorkflowExecutor  # ensure import/registration
from vocode.ui.proto import UIPacketEnvelope, UIPacketRunInput


class DummyLLMUsage:
    prompt_tokens = 0
    completion_tokens = 0
    cost_dollars = 0.0


class DummyCommands:
    def subscribe(self):
        return asyncio.Queue()

    async def execute(self, name, ctx, args):
        return ""

    def clear(self):
        pass


class DummyProject:
    def __init__(self, workflows: dict):
        class Settings:
            def __init__(self, workflows):
                self.workflows = workflows

        self.settings = Settings(workflows)
        self.llm_usage = DummyLLMUsage()
        self.commands = DummyCommands()
        self.project_state = {}
        self.base_path = "."

    async def message_generator(self):
        if False:
            yield None

    async def shutdown(self):
        pass


@pytest.mark.asyncio
async def test_start_workflow_executor_stack_and_complete():
    # Parent: Start the "child" workflow, passing initial text
    parent_nodes = [
        StartWorkflowNode(name="parent", workflow="child", initial_text="hi child", outcomes=[])
    ]
    # Child: simple noop which completes immediately with a final message
    child_nodes = [NoopNode(name="noop", type="noop", outcomes=[])]

    class WF:
        def __init__(self, nodes, edges):
            self.nodes = nodes
            self.edges = edges

    project = DummyProject(
        {"parent": WF(parent_nodes, []), "child": WF(child_nodes, [])}
    )

    ui = UIState(project)
    await ui.start_by_name("parent")

    parent_final = None
    done = False
    while not done:
        env = await asyncio.wait_for(ui.recv(), timeout=5)
        pkt = env.payload

        # Approve final confirmations to unblock the runner
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
        if pkt.kind == "run_event":
            ev = pkt.event
            if ev.node == "parent" and ev.event.kind == "final_message" and ev.event.message:
                parent_final = ev.event.message.text

    assert parent_final is not None