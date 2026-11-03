import asyncio
from contextlib import ExitStack
import errno
import json
from pathlib import Path
import secrets
import socket
from threading import Event
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest
from websockets.asyncio.client import connect

from runner_desktop.desktop import launch, validate_endpoint
from runner_desktop.assets import AssetServer
from runner_desktop import worker as worker_module
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
    workspace.mkdir()
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
    webview.start.assert_called_once()
    assert callable(webview.start.call_args.args[0])
    worker.assert_called_once_with(
        workspace.resolve(), worker.call_args.args[1], "http://127.0.0.1:8766"
    )
    assets.assert_called_once_with(port=0)
    worker.return_value.close.assert_called_once_with(raise_failure=True)


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


def test_asset_servers_allocate_distinct_available_ports(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("ready")
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        with AssetServer(tmp_path) as first, AssetServer(tmp_path) as second:
            ports = {
                first.server.server_port,
                second.server.server_port,
                occupied.getsockname()[1],
            }
            assert len(ports) == 3
            assert 0 not in ports


def test_local_launch_ignores_occupied_conventional_ports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOPS_RUNNER_CONFIG_DIR", str(tmp_path / "config"))
    (tmp_path / "index.html").write_text("ready")
    webview = MagicMock()
    with ExitStack() as stack:
        for port in (8765, 8766):
            listener = stack.enter_context(socket.socket())
            try:
                listener.bind(("127.0.0.1", port))
                listener.listen()
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise
        stack.enter_context(patch.dict("sys.modules", {"webview": webview}))
        stack.enter_context(
            patch(
                "runner_desktop.desktop.AssetServer",
                side_effect=lambda port: AssetServer(tmp_path, port),
            )
        )
        launch(tmp_path)
    url = urlsplit(webview.create_window.call_args.args[1])
    endpoint = urlsplit(parse_qs(url.fragment)["endpoint"][0])
    assert url.port not in (0, 8765, 8766)
    assert endpoint.port not in (0, 8765, 8766, url.port)


def test_attached_desktop_does_not_own_worker(tmp_path: Path) -> None:
    webview = MagicMock()
    with (
        patch.dict("sys.modules", {"webview": webview}),
        patch("runner_desktop.desktop.AssetServer") as assets,
        patch("runner_desktop.desktop.Worker") as worker,
    ):
        launch(tmp_path, "ws://localhost:1234", secrets.token_urlsafe(32), 8766)
    assets.assert_called_once_with(port=8766)
    worker.assert_not_called()
    webview.start.assert_called_once_with()


def test_worker_failure_is_shown_in_window(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    webview = MagicMock()
    webview.start.side_effect = lambda monitor: monitor()
    with (
        patch.dict("sys.modules", {"webview": webview}),
        patch("runner_desktop.desktop.AssetServer"),
        patch("runner_desktop.desktop.Worker") as worker,
    ):
        worker.return_value.finished.is_set.return_value = True
        worker.return_value.failure = RuntimeError("worker stopped unexpectedly")
        launch(tmp_path)
    webview.create_window.return_value.load_html.assert_called_once()
    assert (
        "Local runner stopped"
        in webview.create_window.return_value.load_html.call_args.args[0]
    )
    assert "worker stopped unexpectedly" in capsys.readouterr().err


def test_startup_failure_is_not_masked_by_cleanup(tmp_path: Path) -> None:
    webview = MagicMock()
    with (
        patch.dict("sys.modules", {"webview": webview}),
        patch("runner_desktop.desktop.AssetServer"),
        patch("runner_desktop.desktop.Worker") as worker,
    ):
        worker.return_value.start.side_effect = ValueError("original startup error")
        with pytest.raises(ValueError, match="original startup error"):
            launch(tmp_path)
    worker.return_value.close.assert_called_once_with(raise_failure=False)


def test_startup_timeout_cancels_and_joins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stopped = Event()

    async def never_ready(self: Worker) -> None:
        self.loop = asyncio.get_running_loop()
        self.task = asyncio.current_task()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(worker_module, "START_TIMEOUT", 0.05)
    monkeypatch.setattr(Worker, "_serve", never_ready)
    worker = Worker(tmp_path, "token", "http://localhost")
    try:
        with pytest.raises(RuntimeError, match="startup timed out"):
            worker.start()
    finally:
        worker.close()
    assert stopped.is_set()
    assert not worker.thread.is_alive()


def test_original_startup_timeout_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail(self: Worker) -> None:
        raise TimeoutError("original backend timeout")

    monkeypatch.setattr(Worker, "_serve", fail)
    worker = Worker(tmp_path, "token", "http://localhost")
    try:
        with pytest.raises(TimeoutError, match="original backend timeout"):
            worker.start()
    finally:
        worker.close(raise_failure=False)


def test_unresponsive_worker_has_bounded_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = Event()

    async def blocked(self: Worker) -> None:
        self.ready.set_result("ws://localhost:1234")
        release.wait()

    monkeypatch.setattr(worker_module, "STOP_TIMEOUT", 0.01)
    monkeypatch.setattr(Worker, "_serve", blocked)
    worker = Worker(tmp_path, "token", "http://localhost")
    worker.start()
    try:
        with pytest.raises(RuntimeError, match="shutdown timed out"):
            worker.close()
        assert worker.thread.daemon
    finally:
        release.set()
        worker.thread.join(timeout=2)
    assert not worker.thread.is_alive()


def test_shutdown_cancels_cooperative_worker_after_grace_period(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cancelled = Event()

    async def waiting(self: Worker) -> None:
        self.loop = asyncio.get_running_loop()
        self.task = asyncio.current_task()
        self.ready.set_result("ws://localhost:1234")
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(worker_module, "STOP_TIMEOUT", 0.05)
    monkeypatch.setattr(Worker, "_serve", waiting)
    worker = Worker(tmp_path, "token", "http://localhost")
    worker.start()
    worker.close()
    assert cancelled.is_set()
    assert not worker.thread.is_alive()


def test_project_ownership_error_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOPS_RUNNER_CONFIG_DIR", str(tmp_path / "config"))
    first = Worker(tmp_path, secrets.token_urlsafe(32), "http://localhost")
    second = Worker(tmp_path, secrets.token_urlsafe(32), "http://localhost")
    first.start()
    try:
        with pytest.raises(RuntimeError, match="another runner owns this project"):
            second.start()
    finally:
        second.close(raise_failure=False)
        first.close()
