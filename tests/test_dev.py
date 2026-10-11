import asyncio
from pathlib import Path
import sys

from runner_desktop.dev import stop_process, watch_paths


def test_watch_paths(tmp_path: Path) -> None:
    root = tmp_path / "desktop"
    for path in [root / "src", tmp_path / "runner/src", tmp_path / "connect/src"]:
        path.mkdir(parents=True)
    paths = watch_paths(root, [root / "src", tmp_path / "missing"])
    assert paths == [
        (root / "src").resolve(),
        (tmp_path / "runner/src").resolve(),
        (tmp_path / "connect/src").resolve(),
    ]


async def test_stop_process() -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(60)"
    )
    await stop_process(process)
    assert process.returncode is not None
    await stop_process(process)
