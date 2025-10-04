import asyncio
import os
import shlex
import sys
from pathlib import Path
import pytest

from vocode.proc.local import LocalSubprocessBackend
from vocode.proc.base import SpawnOptions, EnvPolicy

# POSIX-only tests
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX-only tests")


async def _read_all_stdout(handle) -> str:
    chunks = []
    async for line in handle.iter_stdout():
        chunks.append(line)
    return "".join(chunks)


def test_local_backend_handle_io_and_wait(tmp_path: Path):
    async def scenario():
        backend = LocalSubprocessBackend()
        handle = await backend.spawn(SpawnOptions(command="cat", cwd=tmp_path))
        await handle.write("hello\n")
        await handle.write(b"world\n")
        await handle.close_stdin()
        out = await _read_all_stdout(handle)
        rc = await handle.wait()
        assert rc == 0
        assert out == "hello\nworld\n"

    asyncio.run(scenario())


def test_local_backend_nonzero_exit(tmp_path: Path):
    async def scenario():
        backend = LocalSubprocessBackend()
        # 'false' is a POSIX command that exits with status 1
        handle = await backend.spawn(SpawnOptions(command="false", cwd=tmp_path))
        rc = await handle.wait()
        assert rc == 1
        # Ensure the handle reflects the return code as well
        assert handle.returncode == 1

    asyncio.run(scenario())


def test_local_backend_env_policy_denylist(monkeypatch, tmp_path: Path):
    async def scenario():
        # Ensure a variable exists in parent env
        monkeypatch.setenv("ZREM", "abc")
        backend = LocalSubprocessBackend()
        backend.env_policy = EnvPolicy(inherit_parent=True, denylist=["ZREM"])
        # If denylist applied, $ZREM should not be present in child
        handle = await backend.spawn(
            SpawnOptions(command="printf '%s\\n' \"$ZREM\"", cwd=tmp_path)
        )
        out = await _read_all_stdout(handle)
        print(out)
        rc = await handle.wait()
        assert rc == 0
        assert out == "\n"

    asyncio.run(scenario())


def test_local_backend_terminate_then_kill_path(tmp_path: Path):
    async def scenario():
        backend = LocalSubprocessBackend()
        py = shlex.quote(sys.executable)
        code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
        cmd = f"{py} -c {shlex.quote(code)}"
        handle = await backend.spawn(SpawnOptions(command=cmd, cwd=tmp_path))
        # Terminate with short grace; process ignores SIGTERM so kill should be issued
        await handle.terminate(grace_s=0.05)
        rc = await asyncio.wait_for(handle.wait(), timeout=2.0)
        assert rc is not None

    asyncio.run(scenario())


def test_local_backend_pty_not_supported(tmp_path: Path):
    async def scenario():
        backend = LocalSubprocessBackend()
        with pytest.raises(NotImplementedError):
            await backend.spawn(
                SpawnOptions(command="true", cwd=tmp_path, use_pty=True)
            )

    asyncio.run(scenario())
