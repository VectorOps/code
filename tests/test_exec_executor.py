import asyncio
import os
from pathlib import Path

import pytest

from vocode.runner.executors.exec import ExecNode, ExecExecutor
from vocode.proc.manager import ProcessManager
from vocode.proc.shell import ShellManager
from vocode.settings import ProcessSettings, Settings
from vocode.state import Message
from vocode.runner.models import ExecRunInput, PACKET_FINAL_MESSAGE


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX-only tests")


class _DummyProject:
    def __init__(self, base: Path):
        self.base_path = base
        # Minimal settings so ExecExecutor and ShellManager can resolve defaults.
        self.settings = Settings(process=ProcessSettings())
        self.processes = ProcessManager(backend_name="local", default_cwd=base)
        self.shells = ShellManager(
            process_manager=self.processes,
            settings=self.settings.process.shell,
            default_cwd=base,
        )


def test_exec_executor_success_and_timeout(tmp_path: Path):
    async def scenario():
        proj = _DummyProject(tmp_path)

        # Success case with expected_return_code and custom message
        node_ok = ExecNode(
            name="run_echo",
            type="exec",
            command="echo hi",
            message="prefix",
            expected_return_code=0,
            outcomes=[{"name": "success"}, {"name": "fail"}],
        )
        exec_ok = ExecExecutor(config=node_ok, project=proj)

        agen = exec_ok.run(ExecRunInput(messages=[], state=None, response=None))
        pkt, _ = await anext(agen)
        assert pkt.kind == PACKET_FINAL_MESSAGE
        assert pkt.outcome_name == "success"
        assert pkt.message is not None
        lines = pkt.message.text.splitlines()
        # Expected order: message, command header, output line
        assert lines[0] == "prefix"
        assert lines[1] == "> echo hi"
        assert lines[2] == "hi"

        # Timeout case without expected_return_code (no/at most one outcome)
        node_to = ExecNode(
            name="sleep_short",
            type="exec",
            command="sleep 5",
            timeout_s=0.1,
            outcomes=[],  # allowed: at most one outcome
        )
        exec_to = ExecExecutor(config=node_to, project=proj)
        agen2 = exec_to.run(ExecRunInput(messages=[], state=None, response=None))
        pkt2, _ = await anext(agen2)
        assert pkt2.kind == PACKET_FINAL_MESSAGE
        # No expected_return_code and no outcomes => no outcome_name
        assert pkt2.outcome_name is None
        assert pkt2.message is not None
        # Header must include command
        assert pkt2.message.text.splitlines()[0] == "> sleep 5"

        # cleanup
        await proj.shells.stop()
        await proj.processes.shutdown()

    asyncio.run(scenario())