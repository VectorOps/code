import asyncio
from typing import Any

import pytest

from vocode.models import Node
from vocode.graph import Graph
from vocode.runner.runner import Executor, Runner
from vocode.runner.models import ReqToolCall, ReqFinalMessage, RunInput
from vocode.settings import Settings, WorkflowConfig, ToolSpec, ToolRuntimeSettings
from vocode.state import Message, Assignment, Step, Activity, RunStatus, ToolCall
from vocode.tools import BaseTool, ToolTextResponse


class _ConcurrencyTool(BaseTool):
    name = "concurrency_tool"

    def __init__(self, project, counter: "_Counter") -> None:
        super().__init__(project)
        self._counter = counter

    async def run(self, spec: ToolSpec, args: Any) -> ToolTextResponse:
        async with self._counter.track():
            await asyncio.sleep(args.get("delay", 0.01))
        return ToolTextResponse(text="ok")
    
    async def openapi_spec(self, spec: ToolSpec) -> dict:
        return {
            "name": self.name,
            "description": "Test concurrency tool",
            "parameters": {"type": "object", "properties": {}},
        }


class _Counter:
    def __init__(self) -> None:
        self.current = 0
        self.max_seen = 0
        self._lock = asyncio.Lock()

    async def _inc(self) -> None:
        async with self._lock:
            self.current += 1
            if self.current > self.max_seen:
                self.max_seen = self.current

    async def _dec(self) -> None:
        async with self._lock:
            self.current -= 1

    def track(self):
        counter = self

        class _Ctx:
            async def __aenter__(self_inner):  # type: ignore[override]
                await counter._inc()

            async def __aexit__(self_inner, exc_type, exc, tb):  # type: ignore[override]
                await counter._dec()

        return _Ctx()


class _ToolExecNode(Node):
    type: str = "tool_exec_test"


class _ToolExecExecutor(Executor):
    type = "tool_exec_test"

    async def run(self, inp):
        if inp.response is None:
            tool_calls = [
                ToolCall(
                    name="concurrency_tool",
                    arguments={"delay": 0.05},
                    tool_spec=ToolSpec(
                        name="concurrency_tool",
                        enabled=True,
                        auto_approve=True,
                    ),
                )
                for _ in range(4)
            ]
            yield (ReqToolCall(tool_calls=tool_calls), None)
            return

        yield (
            ReqFinalMessage(message=Message(role="agent", text="done")),
            None,
        )


Executor.register("tool_exec_test", _ToolExecExecutor)


class _DummyProject:
    def __init__(self, settings: Settings, tool):
        self.settings = settings
        self.tools = {"concurrency_tool": tool}


@pytest.mark.asyncio
async def test_runner_tool_concurrency_respects_setting():
    counter = _Counter()

    wf_nodes = [_ToolExecNode(name="root", outcomes=[])]
    graph = Graph(nodes=wf_nodes, edges=[])
    wf_conf = WorkflowConfig(nodes=wf_nodes, edges=[])
    settings = Settings(
        workflows={"test": wf_conf},
        tools_runtime=ToolRuntimeSettings(max_concurrent=2),
    )

    project = _DummyProject(settings=settings, tool=_ConcurrencyTool(None, counter))

    class _WF:
        def __init__(self, g):
            self.graph = g

    runner = Runner(_WF(graph), project)
    assignment = Assignment(steps=[], status=RunStatus.running)

    agen = runner.run(assignment)

    # First event should be a tool_call request
    ev = await anext(agen)
    assert ev.event.kind == "tool_call"
    assert len(ev.event.tool_calls) == 4

    # Runner auto-approves tools (auto_approve=True), so we do not send a response
    # here. Instead, advance the generator with None so _run_tools executes.
    ev = await agen.asend(None)
    assert ev.event.kind == "tool_result"

    # Advance once more to get the executor's final message
    ev = await agen.asend(RunInput(response=None))
    assert ev.event.kind == "final_message"

    # Concurrency counter should never exceed max_concurrent
    assert counter.max_seen <= 2
