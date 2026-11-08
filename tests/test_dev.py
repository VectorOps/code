import asyncio
from pathlib import Path
import sys
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vectorops_code.dev import run, stop_process, watch_paths


def test_watch_paths(tmp_path: Path) -> None:
    root = tmp_path / "code"
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


async def test_dev_launcher_marks_worker_as_development(tmp_path: Path) -> None:
    vite = tmp_path / "frontend/node_modules/vite/bin/vite.js"
    vite.parent.mkdir(parents=True)
    vite.touch()
    process = MagicMock(spec=asyncio.subprocess.Process)
    process.returncode = 0
    process.wait = AsyncMock(return_value=0)

    async def watch(
        *args: object, **kwargs: object
    ) -> AsyncIterator[set[tuple[int, str]]]:
        await asyncio.Event().wait()
        yield set()

    with (
        patch("vectorops_code.dev.shutil.which", return_value="node"),
        patch("vectorops_code.dev.TemplateServer"),
        patch("watchfiles.awatch", watch),
        patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=process,
        ) as create,
    ):
        with pytest.raises(RuntimeError, match="Development frontend exited"):
            await run(tmp_path, tmp_path, 8765, 5173, [])
    assert "--development" in create.call_args_list[1].args
    assert (
        "from vectorops_code.cli import main; main()" in create.call_args_list[1].args
    )
    assert "VECTOROPS_CODE_TOKEN" in create.call_args_list[1].kwargs["env"]
    assert "--strictPort" in create.call_args_list[0].args


def test_watch_paths_without_sibling_checkouts(tmp_path: Path) -> None:
    root = tmp_path / "code"
    source = root / "src"
    source.mkdir(parents=True)
    assert watch_paths(root, []) == [source.resolve()]
