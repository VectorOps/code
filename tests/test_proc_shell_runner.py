import asyncio
import os
import shutil
from pathlib import Path
import pytest

from vocode.proc.manager import ProcessManager
from vocode.proc.shell import ShellRunner, ShellTimeoutError

# POSIX-only; require bash for ShellRunner default spec
pytestmark = [
    pytest.mark.skipif(os.name != "posix", reason="POSIX-only tests"),
    pytest.mark.skipif(
        shutil.which("bash") is None, reason="bash required for ShellRunner tests"
    ),
]


def test_shell_runner_basic_run_and_restart(tmp_path: Path):
    async def scenario():
        pm = ProcessManager(backend_name="local", default_cwd=tmp_path)
        runner = ShellRunner(pm, default_timeout_s=2.0)
        await runner.start()
        out1 = await runner.run("echo hi")
        assert out1 == "hi\n"
        # Explicit stop, then ensure run auto-starts the shell again
        await runner.aclose()
        out2 = await runner.run("echo bye")
        assert out2 == "bye\n"
        await runner.aclose()
        await pm.shutdown()

    asyncio.run(scenario())


def test_shell_runner_env_and_cwd(tmp_path: Path):
    async def scenario():
        pm = ProcessManager(backend_name="local", default_cwd=tmp_path)
        subdir = tmp_path / "shelldir"
        subdir.mkdir()
        runner = ShellRunner(
            pm, cwd=subdir, env_overlay={"FOO": "BAR"}, default_timeout_s=5.0
        )
        await runner.start()
        out = await runner.run("pwd; printf '%s\\n' \"$FOO\"")
        lines = [ln for ln in out.splitlines()]
        assert lines[0] == str(subdir)
        assert lines[1] == "BAR"
        await runner.aclose()
        await pm.shutdown()

    asyncio.run(scenario())


def test_shell_runner_timeout_and_auto_restart(tmp_path: Path):
    async def scenario():
        pm = ProcessManager(backend_name="local", default_cwd=tmp_path)
        runner = ShellRunner(pm, default_timeout_s=1.0)
        await runner.start()
        with pytest.raises(ShellTimeoutError):
            await runner.run("sleep 5", timeout_s=0.1)
        # After timeout, runner restarts; next command should succeed
        out = await runner.run("echo ok")
        assert out == "ok\n"
        await runner.aclose()
        await pm.shutdown()

    asyncio.run(scenario())


def test_shell_runner_combined_output_includes_stderr(tmp_path: Path):
    async def scenario():
        pm = ProcessManager(backend_name="local", default_cwd=tmp_path)
        runner = ShellRunner(pm, default_timeout_s=2.0)
        await runner.start()
        # stderr redirected to stdout by wrapper; both lines should appear
        out = await runner.run("echo out; echo err 1>&2")
        assert "out\n" in out
        assert "err\n" in out
        await runner.aclose()
        await pm.shutdown()

    asyncio.run(scenario())
