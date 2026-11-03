import asyncio
import json
from pathlib import Path
import secrets

import pydantic
import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from vectorops.runner.control import protocol
from runner_desktop import settings
from runner_desktop.server import websocket_server
from runner_desktop.service import DesktopService


async def test_defaults_persistence_and_concurrent_field_updates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config" / "desktop-settings.json"
    assert (
        await settings.Store(path).dispatch("desktop.settings.get", {})
        == settings.Settings().model_dump()
    )
    await asyncio.gather(
        settings.Store(path).dispatch(
            "desktop.settings.update", {"appearance": "dark"}
        ),
        settings.Store(path).dispatch(
            "desktop.settings.update", {"default_workflow": "custom:review"}
        ),
    )
    result = settings.Settings.model_validate(
        await settings.Store(path).dispatch("desktop.settings.get", {})
    )
    assert result.appearance == "dark"
    assert result.default_workflow == "custom:review"
    assert result.history_limits == settings.HistoryLimits()
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "patch",
    [
        {"appearance": "unknown"},
        {"default_workflow": ""},
        {"token": "not-a-setting"},
        {"history_limits": {"initial": True}},
        {"history_limits": {"older": 501}},
        {"history_limits": {"maximum": 10}},
        {"history_limits": {"target": 1.5}},
        {"appearance": None},
    ],
)
async def test_invalid_updates_do_not_overwrite_settings(
    tmp_path: Path, patch: dict[str, pydantic.JsonValue]
) -> None:
    store = settings.Store(tmp_path / "desktop-settings.json")
    await store.dispatch("desktop.settings.update", {"appearance": "dark"})
    previous = store.path.read_bytes()
    with pytest.raises(protocol.RemoteError) as error:
        await store.dispatch("desktop.settings.update", patch)
    assert error.value.code == "invalid_argument"
    assert store.path.read_bytes() == previous


async def test_corrupt_storage_is_reported_and_preserved(tmp_path: Path) -> None:
    path = tmp_path / "desktop-settings.json"
    path.write_text("not json")
    with pytest.raises(protocol.RemoteError) as error:
        await settings.Store(path).dispatch(
            "desktop.settings.update", {"appearance": "dark"}
        )
    assert error.value.code == "settings_storage_error"
    assert path.read_text() == "not json"


async def test_settings_require_authentication_and_survive_changed_origins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOPS_RUNNER_CONFIG_DIR", str(tmp_path / "config"))
    path = tmp_path / "config" / "desktop-settings.json"
    token = secrets.token_urlsafe(32)
    for index, origin in enumerate(
        ["http://127.0.0.1:12001", "http://127.0.0.1:12002"]
    ):
        project = tmp_path / f"project-{index}"
        project.mkdir()
        async with DesktopService(project) as service:
            async with websocket_server(service, token, [origin]) as listener:
                port = next(iter(listener.sockets)).getsockname()[1]
                endpoint = f"ws://127.0.0.1:{port}"
                async with connect(endpoint, origin=origin) as socket:
                    await socket.send(
                        json.dumps(
                            {
                                "kind": "request",
                                "id": "unauthenticated",
                                "method": "desktop.settings.update",
                                "params": {"appearance": "light"},
                            }
                        )
                    )
                    with pytest.raises(ConnectionClosed):
                        await socket.recv()
                    if index == 0:
                        assert not path.exists()
                async with connect(endpoint, origin=origin) as socket:
                    await socket.send(
                        json.dumps(
                            {"kind": "authenticate", "version": 2, "token": token}
                        )
                    )
                    assert json.loads(await socket.recv())["kind"] == "authenticated"
                    await socket.send(
                        json.dumps(
                            {
                                "kind": "request",
                                "id": "settings",
                                "method": (
                                    "desktop.settings.update"
                                    if index == 0
                                    else "desktop.settings.get"
                                ),
                                "params": {"appearance": "dark"} if index == 0 else {},
                            }
                        )
                    )
                    reply = json.loads(await socket.recv())
                    assert reply["kind"] == "response"
                    assert reply["result"]["appearance"] == "dark"
