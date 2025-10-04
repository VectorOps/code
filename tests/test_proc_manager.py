import asyncio
import os
import pytest
from pathlib import Path

from vocode.proc.manager import ProcessManager

# POSIX-only tests
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX-only tests")


async def _read_all_stdout(handle) -> str:
    out_chunks = []
    async for line in handle.iter_stdout():
        out_chunks.append(line)
    return "".join(out_chunks)


def test_process_manager_spawn_list_get_env_cwd_and_shutdown(tmp_path: Path):
    async def scenario():
        pm = ProcessManager(backend_name="local", default_cwd=tmp_path)

        # 1) Spawn with env overlay; verify output
        h1 = await pm.spawn(command="printf '%s\\n' \"$FOO\"", env_overlay={"FOO": "BAR"})
        assert pm.get(h1.id) is h1
        assert h1 in pm.list()
        out1 = await _read_all_stdout(h1)
        rc1 = await h1.wait()
        assert rc1 == 0
        assert out1 == "BAR\n"

        # 2) Spawn with explicit cwd; verify pwd output
        subdir = tmp_path / "cwdtest"
        subdir.mkdir()
        h2 = await pm.spawn(command="pwd", cwd=subdir)
        out2 = await _read_all_stdout(h2)
        rc2 = await h2.wait()
        assert rc2 == 0
        assert out2.strip() == str(subdir)

        # 3) PTY requested should be unsupported on local backend
        with pytest.raises(NotImplementedError):
            await pm.spawn(command="true", use_pty=True)

        # 4) Shutdown should terminate lingering processes and clear registry
        h3 = await pm.spawn(command="sleep 60")
        await pm.shutdown(grace_s=0.1)
        # Handle reference should be completed after shutdown
        # (wait returns immediately if already finished)
        rc3 = await asyncio.wait_for(h3.wait(), timeout=2.0)
        assert rc3 is not None
        assert pm.list() == []

    asyncio.run(scenario())
