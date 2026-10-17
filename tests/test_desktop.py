import asyncio
import json
from pathlib import Path
import secrets
from unittest.mock import MagicMock, patch

import pytest
from websockets.asyncio.client import connect

from runner_desktop.desktop import launch, validate_endpoint
from runner_desktop.worker import Worker


@pytest.mark.parametrize(
    "endpoint",
    [
        "ws://example.com",
        "https://example.com",
        "wss://user:password@example.com",
        "ws://localhost/#secret",
    ],
)
def test_unsafe_endpoint(endpoint: str) -> None:
    with pytest.raises(ValueError):
        validate_endpoint(endpoint)


@pytest.mark.parametrize(
    "endpoint",
    ["ws://127.0.0.1:8765", "ws://localhost:8765", "wss://example.com/runner"],
)
def test_safe_endpoint(endpoint: str) -> None:
    assert validate_endpoint(endpoint) == endpoint


@pytest.mark.parametrize("relative", [False, True])
def test_window_title_includes_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace = Path("project workspace")
    if not relative:
        workspace = tmp_path / workspace
    webview = MagicMock()
    with (
        patch.dict("sys.modules", {"webview": webview}),
        patch("runner_desktop.desktop.AssetServer") as assets,
        patch("runner_desktop.desktop.Worker") as worker,
    ):
        assets.return_value.__enter__.return_value.origin = "http://127.0.0.1:8766"
        worker.return_value.start.return_value = "ws://127.0.0.1:8765"
        launch(workspace)

    assert (
        webview.create_window.call_args.args[0] == f"VectorOps - {workspace.resolve()}"
    )
    webview.start.assert_called_once_with()
    worker.return_value.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_owned_worker_stops_and_joins(tmp_path: Path) -> None:
    origin = "http://127.0.0.1:8766"
    token = secrets.token_urlsafe(32)
    worker = Worker(tmp_path, token, origin)
    endpoint = await asyncio.to_thread(worker.start)
    try:
        async with connect(endpoint, origin=origin) as socket:
            await socket.send(
                json.dumps({"kind": "authenticate", "version": 2, "token": token})
            )
            assert json.loads(await socket.recv())["kind"] == "authenticated"
    finally:
        await asyncio.to_thread(worker.close)
    assert not worker.thread.is_alive()
