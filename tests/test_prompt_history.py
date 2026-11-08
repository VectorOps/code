import asyncio
import json
from pathlib import Path
import secrets
import sqlite3

import pytest
import pydantic
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed
from websockets.typing import Origin

from vectorops.runner.control import protocol
from vectorops_code.prompt_history import Entry, Page, PromptHistory
from vectorops_code.server import websocket_server
from vectorops_code.service import DesktopService


async def append(
    history: PromptHistory,
    text: str,
    project: str = "project-a",
    execution: str = "chat-a",
    key: str = "",
) -> Entry:
    return Entry.model_validate(
        await history.dispatch(
            "prompt_history.append",
            {
                "text": text,
                "execution_id": execution,
                "idempotency_key": key or secrets.token_hex(16),
            },
            project,
        )
    )


async def test_persists_across_clients_projects_and_restarts(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite3"
    first = await append(PromptHistory(path), "First\nmultiline prompt")
    second = await append(PromptHistory(path), "Other project", "project-b", "chat-b")
    history = PromptHistory(path)
    global_page = Page.model_validate(
        await history.dispatch("prompt_history.list", {}, "project-c")
    )
    assert global_page.entries == [second, first]
    project_page = Page.model_validate(
        await history.dispatch("prompt_history.list", {"scope": "project"}, "project-a")
    )
    assert project_page.entries == [first]
    session_page = Page.model_validate(
        await history.dispatch(
            "prompt_history.list",
            {"scope": "session", "execution_id": "chat-b"},
            "project-a",
        )
    )
    assert session_page.entries == []
    assert path.stat().st_mode & 0o777 == 0o600
    path.chmod(0o644)
    await history.dispatch("prompt_history.list", {}, "project-c")
    assert path.stat().st_mode & 0o777 == 0o600


async def test_default_location_preserves_existing_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert PromptHistory().path == tmp_path / "vectorops-code/prompt-history.sqlite3"
    legacy = tmp_path / "runner-desktop/prompt-history.sqlite3"
    entry = await append(PromptHistory(legacy), "Saved before the rename")
    history = PromptHistory()
    assert history.path == legacy
    page = Page.model_validate(
        await history.dispatch("prompt_history.list", {}, "project-a")
    )
    assert page.entries == [entry]


async def test_search_is_literal_unicode_aware_deduplicated_and_paginated(
    tmp_path: Path,
) -> None:
    history = PromptHistory(tmp_path / "history.sqlite3")
    await append(history, "Straße 100%_done")
    other = await append(history, "Another prompt")
    latest = await append(history, "Straße 100%_done", execution="other-chat")
    result = Page.model_validate(
        await history.dispatch("prompt_history.list", {"query": "STRASSE 100%_"}, "p")
    )
    assert result.entries == [latest]
    first = Page.model_validate(
        await history.dispatch("prompt_history.list", {"limit": 1}, "p")
    )
    second = Page.model_validate(
        await history.dispatch(
            "prompt_history.list", {"limit": 1, "before": first.next_before}, "p"
        )
    )
    assert first.entries == [latest]
    assert second.entries == [other]
    assert second.next_before is None


async def test_concurrent_appends_are_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite3"
    await append(PromptHistory(path), "seed")
    entries = await asyncio.gather(
        *(append(PromptHistory(path), "Saved once", key="same-key") for _ in range(8))
    )
    assert all(entry == entries[0] for entry in entries)
    with pytest.raises(protocol.RemoteError, match="invalid prompt history"):
        await append(PromptHistory(path), "Changed text", key="same-key")


@pytest.mark.parametrize(
    ("method", "params"),
    [
        (
            "prompt_history.append",
            {"text": "   ", "execution_id": "e", "idempotency_key": "k"},
        ),
        (
            "prompt_history.append",
            {"text": "x" * 65537, "execution_id": "e", "idempotency_key": "k"},
        ),
        ("prompt_history.list", {"scope": "session"}),
        ("prompt_history.list", {"scope": "unknown"}),
        ("prompt_history.list", {"limit": 21}),
        ("prompt_history.list", {"before": -1}),
        ("prompt_history.list", {"path": "/tmp/arbitrary-file"}),
    ],
)
async def test_rejects_invalid_requests(
    tmp_path: Path, method: str, params: dict[str, pydantic.JsonValue]
) -> None:
    with pytest.raises(protocol.RemoteError) as error:
        await PromptHistory(tmp_path / "history.sqlite3").dispatch(method, params, "p")
    assert error.value.code == "invalid_argument"


async def test_retention_is_bounded(tmp_path: Path) -> None:
    history = PromptHistory(tmp_path / "history.sqlite3")
    await append(history, "seed")
    with sqlite3.connect(history.path) as database:
        database.executemany(
            "INSERT INTO prompts(project_id, execution_id, timestamp, text, idempotency_key) "
            "VALUES ('p', 'e', 'now', ?, ?)",
            [(f"prompt-{index}", f"key-{index}") for index in range(1001)],
        )
    await append(history, "latest")
    with sqlite3.connect(history.path) as database:
        assert database.execute("SELECT COUNT(*) FROM prompts").fetchone()[0] == 1000


async def test_storage_errors_are_reported_without_paths_or_prompt_contents(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "not-a-directory"
    parent.write_text("file")
    with pytest.raises(protocol.RemoteError) as error:
        await PromptHistory(parent / "history.sqlite3").dispatch(
            "prompt_history.list", {}, "p"
        )
    assert error.value.code == "history_storage_error"
    assert str(parent) not in str(error.value)


async def test_history_uses_authenticated_websocket_and_does_not_dispatch_workflows(
    tmp_path: Path,
) -> None:
    token = secrets.token_urlsafe(32)
    origin = Origin("http://127.0.0.1:5173")
    async with DesktopService(
        tmp_path, history_path=tmp_path / "prompts.sqlite3"
    ) as service:
        async with websocket_server(service, token, [origin]) as listener:
            port = next(iter(listener.sockets)).getsockname()[1]
            async with connect(
                f"ws://127.0.0.1:{port}", origin=origin
            ) as unauthenticated:
                await unauthenticated.send(
                    json.dumps(
                        {
                            "kind": "request",
                            "id": "no-auth",
                            "method": "prompt_history.list",
                            "params": {},
                        }
                    )
                )
                with pytest.raises(ConnectionClosed):
                    await unauthenticated.recv()
                assert not service.prompt_history.path.exists()
            async with connect(f"ws://127.0.0.1:{port}", origin=origin) as socket:
                await socket.send(
                    json.dumps({"kind": "authenticate", "version": 2, "token": token})
                )
                assert isinstance(
                    protocol.PACKET.validate_json(await socket.recv()),
                    protocol.Authenticated,
                )

                async def request(
                    method: str, params: dict[str, pydantic.JsonValue]
                ) -> pydantic.JsonValue:
                    await socket.send(
                        json.dumps(
                            {
                                "kind": "request",
                                "id": "history",
                                "method": method,
                                "params": params,
                            }
                        )
                    )
                    packet = protocol.PACKET.validate_json(await socket.recv())
                    assert isinstance(packet, protocol.Response)
                    return packet.result

                await request(
                    "prompt_history.append",
                    {
                        "text": "A host-side prompt",
                        "execution_id": "chat",
                        "idempotency_key": "history-only",
                    },
                )
                page = await request("prompt_history.list", {})
                assert Page.model_validate(page).entries[0].text == "A host-side prompt"
                result = await request("executions.list", {})
                assert isinstance(result, dict)
                assert result["execution_ids"] == []
