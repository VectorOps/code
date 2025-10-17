import asyncio
import pytest
from pathlib import Path

from vocode.ui.base import UIState
from vocode.ui.proto import (
    PACKET_PROJECT_OP_START,
    PACKET_PROJECT_OP_PROGRESS,
    PACKET_PROJECT_OP_FINISH,
)
from vocode.proto import (
    PacketProjectOpStart,
    PacketProjectOpProgress,
    PacketProjectOpFinish,
)


class _FakeCommands:
    def __init__(self):
        self._q = asyncio.Queue()
    def subscribe(self):
        return self._q
    def clear(self):
        pass


class _FakeLLMUsage:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_dollars = 0.0


class FakeProject:
    def __init__(self):
        self.base_path = Path(".")
        self.settings = None
        self.commands = _FakeCommands()
        self.llm_usage = _FakeLLMUsage()
        self.project_state = type("PS", (), {"clear": lambda self: None})()
        self._q = asyncio.Queue()
    async def message_generator(self):
        while True:
            yield await self._q.get()
    async def send_message(self, pack):
        await self._q.put(pack)


@pytest.mark.asyncio
async def test_ui_state_forwards_project_op_packets():
    proj = FakeProject()
    ui = UIState(proj)

    await proj.send_message(PacketProjectOpStart(message="Doing work"))
    await proj.send_message(PacketProjectOpProgress(progress=3, total=10))
    await proj.send_message(PacketProjectOpFinish())

    env1 = await asyncio.wait_for(ui.recv(), timeout=1.0)
    env2 = await asyncio.wait_for(ui.recv(), timeout=1.0)
    env3 = await asyncio.wait_for(ui.recv(), timeout=1.0)

    assert env1.payload.kind == PACKET_PROJECT_OP_START
    assert getattr(env1.payload, "message", None) == "Doing work"
    assert env2.payload.kind == PACKET_PROJECT_OP_PROGRESS
    assert getattr(env2.payload, "progress", None) == 3
    assert getattr(env2.payload, "total", None) == 10
    assert env3.payload.kind == PACKET_PROJECT_OP_FINISH