import asyncio
import json
import os
from pathlib import Path

import pytest

from vocode.proc.manager import ProcessManager
from vocode.tools.exec_tool import ExecTool
from vocode.settings import ToolSpec

pytestmark = [
    pytest.mark.skipif(os.name != "posix", reason="POSIX-only tests"),
]


class _DummyProject:
    def __init__(self, pm: ProcessManager):
        self.processes = pm
        self.settings = None


def test_exec_tool_basic_stderr_and_timeout(tmp_path: Path):
    async def scenario():
        pm = ProcessManager(backend_name="local", default_cwd=tmp_path)
        proj = _DummyProject(pm)
        tool = ExecTool()
        # Use a small timeout for CI speed; default is 60s for production use.
        spec = ToolSpec(name="exec", config={"timeout_s": 0.1})

        # Basic echo
        resp1 = await tool.run(proj, spec, {"command": "echo hi"})
        data1 = json.loads(resp1.text or "{}")
        assert data1["timed_out"] is False
        assert data1["exit_code"] == 0
        assert data1["output"] == "hi\n"

        # Non-zero exit code
        resp2 = await tool.run(proj, spec, {"command": "false"})
        data2 = json.loads(resp2.text or "{}")
        assert data2["timed_out"] is False
        assert isinstance(data2["exit_code"], int) and data2["exit_code"] != 0
        assert data2["output"] == ""

        # Combined stdout + stderr
        resp3 = await tool.run(proj, spec, {"command": "echo out; echo err 1>&2"})
        data3 = json.loads(resp3.text or "{}")
        assert "out\n" in data3["output"]
        assert "err\n" in data3["output"]
        assert data3["exit_code"] == 0
        assert data3["timed_out"] is False

        # Timeout handling (fixed timeout in tool)
        resp4 = await tool.run(proj, spec, {"command": "sleep 5"})
        data4 = json.loads(resp4.text or "{}")
        assert data4["timed_out"] is True
        assert data4["exit_code"] is None
        assert data4["output"] == ""

        await pm.shutdown()

    asyncio.run(scenario())