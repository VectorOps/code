import asyncio
import os
import shutil
from pathlib import Path

import pytest

from vocode.proc.manager import ProcessManager
from vocode.proc.docker_shell import DockerShellRunner, DockerShellTimeoutError


pytestmark = [
    pytest.mark.skipif(os.name != "posix", reason="POSIX-only tests"),
    pytest.mark.skipif(
        shutil.which("docker") is None,
        reason="docker required for DockerShellRunner tests",
    ),
]


def test_docker_shell_runner_basic_run_and_restart(tmp_path: Path):
    async def scenario():
        pm = ProcessManager(backend_name="local", default_cwd=tmp_path)
        runner = DockerShellRunner(pm, default_timeout_s=10.0)
        await runner.start()
        out1, rc1 = await runner.run("echo hi")
        assert out1 == "hi\n"
        assert rc1 == 0
        await runner.aclose()
        out2, rc2 = await runner.run("echo bye")
        assert out2 == "bye\n"
        assert rc2 == 0
        await runner.aclose()
        await pm.shutdown()

    asyncio.run(scenario())


def test_docker_shell_runner_env_and_cwd(tmp_path: Path):
    async def scenario():
        pm = ProcessManager(backend_name="local", default_cwd=tmp_path)
        subdir = tmp_path / "dockdir"
        subdir.mkdir()
        runner = DockerShellRunner(
            pm,
            project_dir=subdir,
            env_overlay={"FOO": "BAR"},
            default_timeout_s=10.0,
        )
        await runner.start()
        out, rc = await runner.run("pwd")
        lines = [ln for ln in out.splitlines()]
        # Inside container we expect the configured workdir; exact host path
        # is not preserved but workdir suffix should match.
        assert lines[0].endswith("/workspace")
        assert rc == 0
        await runner.aclose()
        await pm.shutdown()

    asyncio.run(scenario())


def test_docker_shell_runner_timeout_and_auto_restart(tmp_path: Path):
    async def scenario():
        pm = ProcessManager(backend_name="local", default_cwd=tmp_path)
        runner = DockerShellRunner(pm, default_timeout_s=5.0)
        await runner.start()
        with pytest.raises(DockerShellTimeoutError):
            await runner.run("sleep 5", timeout_s=0.1)
        out, rc = await runner.run("echo ok")
        assert out == "ok\n"
        assert rc == 0
        await runner.aclose()
        await pm.shutdown()

    asyncio.run(scenario())


def test_docker_shell_runner_combined_output_includes_stderr(tmp_path: Path):
    async def scenario():
        pm = ProcessManager(backend_name="local", default_cwd=tmp_path)
        runner = DockerShellRunner(pm, default_timeout_s=10.0)
        await runner.start()
        out, rc = await runner.run("echo out; echo err 1>&2")
        assert "out\n" in out
        assert "err\n" in out
        assert rc == 0
        await runner.aclose()
        await pm.shutdown()

    asyncio.run(scenario())


def test_docker_shell_runner_exit_code_and_marker_on_failure(tmp_path: Path):
    async def scenario():
        pm = ProcessManager(backend_name="local", default_cwd=tmp_path)
        runner = DockerShellRunner(pm, default_timeout_s=10.0)
        await runner.start()
        out, rc = await runner.run("false")
        assert out == ""
        assert rc != 0
        await runner.aclose()
        await pm.shutdown()

    asyncio.run(scenario())


def test_docker_shell_runner_handles_special_chars_and_syntax_error(tmp_path: Path):
    async def scenario():
        pm = ProcessManager(backend_name="local", default_cwd=tmp_path)
        runner = DockerShellRunner(pm, default_timeout_s=10.0)
        await runner.start()
        out, rc = await runner.run("echo 'a\"b' && echo end")
        assert out.splitlines() == ['a"b', "end"]
        out2, rc2 = await runner.run(")")
        assert "syntax error" in out2.lower()
        assert rc2 != 0
        await runner.aclose()
        await pm.shutdown()

    asyncio.run(scenario())
