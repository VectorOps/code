import asyncio
import pytest

from vocode.ui.base import UIState
from vocode.runner.models import RunInput, RespApproval
from vocode.state import RunnerStatus
from vocode.runner.executors.noop import NoopNode
from vocode.runner.executors.start_workflow import (
    StartWorkflowExecutor,
    StartWorkflowNode,
)  # ensure import/registration
from vocode.ui.proto import UIPacketEnvelope, UIPacketRunInput
from vocode.runner.runner import Executor
from vocode.state import Message, ToolCall
from vocode.runner.models import ReqToolCall, ReqFinalMessage
from vocode.tools import BaseTool, ToolStartWorkflowResponse
from vocode.settings import Settings, WorkflowConfig, LoggingSettings


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
        # Convert provided workflow stubs to real WorkflowConfig entries
        wf_map = {name: WorkflowConfig(nodes=wf.nodes, edges=wf.edges) for name, wf in workflows.items()}
        self.settings = Settings(workflows=wf_map, logging=LoggingSettings())
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
        StartWorkflowNode(
            name="parent", workflow="child", initial_text="hi child", outcomes=[]
        )
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
            if (
                ev.node == "parent"
                and ev.event.kind == "final_message"
                and ev.event.message
            ):
                parent_final = ev.event.message.text

    assert parent_final is not None


@pytest.mark.asyncio
async def test_tool_start_workflow_initiates_child_and_returns_result():
    # Define a dummy tool that requests starting the 'child' workflow
    class StartChildTool(BaseTool):
        name = "start_child_tool"

        async def run(self, project, spec, args):
            # Forward initial text from args, if present
            initial_text = args.get("text") if isinstance(args, dict) else None
            return ToolStartWorkflowResponse(
                workflow="child", initial_text=initial_text
            )

        def openapi_spec(self):
            return {
                "name": self.name,
                "description": "Starts a child workflow",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            }

    # Executor that calls the tool and then returns the tool result as final text
    class ToolExec(Executor):
        type = "tool_exec"

        async def run(self, inp):
            if inp.response is None:
                from vocode.settings import ToolSpec

                tc = ToolCall(
                    name="start_child_tool",
                    arguments={"text": "seed"},
                    tool_spec=ToolSpec(
                        name="start_child_tool", enabled=True, auto_approve=True
                    ),
                )
                yield (ReqToolCall(tool_calls=[tc]), None)
                return
            # After tool call completes, echo the result as the final agent message
            # RespToolCall is in inp.response; runner ensures tool_calls[0].result is set to child's final text
            tool_result_text = None
            try:
                tool_result_text = inp.response.tool_calls[0].result  # type: ignore[attr-defined]
            except Exception:
                tool_result_text = None
            yield (
                ReqFinalMessage(
                    message=Message(role="agent", text=tool_result_text or "")
                ),
                None,
            )

    Executor.register("tool_exec", ToolExec)

    # Child executor that emits a known final message text
    class ChildExec(Executor):
        type = "child_exec"

        async def run(self, inp):
            yield (
                ReqFinalMessage(message=Message(role="agent", text="child says hi")),
                None,
            )

    Executor.register("child_exec", ChildExec)

    # Parent and child workflows
    # Set confirmation to 'auto' to avoid prompt/approval flow and ensure the final is forwarded
    parent_nodes = [
        {"name": "parent", "type": "tool_exec", "outcomes": [], "confirmation": "auto"}
    ]
    child_nodes = [{"name": "child", "type": "child_exec", "outcomes": []}]

    class WF:
        def __init__(self, nodes, edges):
            self.nodes = nodes
            self.edges = edges

    project = DummyProject(
        {"parent": WF(parent_nodes, []), "child": WF(child_nodes, [])}
    )
    # Provide the tool via the project's tool registry
    project.tools = {"start_child_tool": StartChildTool()}

    ui = UIState(project)
    await ui.start_by_name("parent")

    parent_final = None
    done = False
    while not done:
        env = await asyncio.wait_for(ui.recv(), timeout=5)
        pkt = env.payload

        # Auto-approve any prompted confirmations
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
            if (
                ev.node == "parent"
                and ev.event.kind == "final_message"
                and ev.event.message
            ):
                parent_final = ev.event.message.text

    # The parent final text should be the child's final message text
    assert parent_final == "child says hi"
