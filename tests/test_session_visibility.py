import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pydantic
import pytest
from vectorops.runner.control import protocol
from vectorops.runner.control.service import RunnerService

from vectorops_code.service import DesktopService
from vectorops_code.session_visibility import SessionVisibility, Snapshot, Update


async def test_visibility_is_durable_idempotent_and_workspace_scoped(
    tmp_path: Path,
) -> None:
    path = tmp_path / "one" / "desktop-sessions.sqlite3"
    store = SessionVisibility(path)
    assert await store.access() == Snapshot()
    hidden = await store.access(Update(execution_id="one", hidden=True))
    assert hidden == Snapshot(revision=1, hidden_execution_ids=["one"])
    assert await store.access(Update(execution_id="one", hidden=True)) == hidden
    assert await SessionVisibility(path).access() == hidden
    assert await SessionVisibility(tmp_path / "two" / path.name).access() == Snapshot()
    shown = await store.access(Update(execution_id="one", hidden=False))
    assert shown == Snapshot(revision=2)
    assert await store.access(Update(execution_id="one", hidden=False)) == shown
    assert path.stat().st_mode & 0o777 == 0o600


async def test_concurrent_updates_do_not_overwrite_other_sessions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "visibility.sqlite3"
    await asyncio.gather(
        *(
            SessionVisibility(path).access(Update(execution_id=str(index), hidden=True))
            for index in range(10)
        )
    )
    assert await SessionVisibility(path).access() == Snapshot(
        revision=10, hidden_execution_ids=[str(index) for index in range(10)]
    )


@pytest.mark.parametrize(
    "params",
    [
        {"execution_id": "", "hidden": True},
        {"execution_id": "one", "hidden": "true"},
        {"execution_id": "one", "hidden": 1},
        {"execution_id": "one"},
        {"execution_id": "one", "hidden": True, "path": "/tmp/other"},
    ],
)
def test_update_validation(params: dict[str, pydantic.JsonValue]) -> None:
    with pytest.raises(pydantic.ValidationError):
        Update.model_validate(params)


async def test_storage_errors_are_safe(tmp_path: Path) -> None:
    parent = tmp_path / "file"
    parent.write_text("not a directory")
    with pytest.raises(protocol.RemoteError) as error:
        await SessionVisibility(parent / "visibility.sqlite3").access()
    assert error.value.code == "visibility_storage_error"
    assert str(parent) not in str(error.value)


async def test_desktop_delegates_summaries_and_broadcasts_visibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary: dict[str, pydantic.JsonValue] = {
        "execution_id": "one",
        "root_workflow_id": "root",
        "reference": "test:workflow",
        "status": "running",
        "active": True,
        "pending_inputs": 1,
    }
    delegate = AsyncMock(return_value={"executions": [summary], "next_after": "one"})
    monkeypatch.setattr(RunnerService, "dispatch", delegate)
    async with DesktopService(tmp_path) as service:
        first: list[dict[str, pydantic.JsonValue]] = []
        second: list[dict[str, pydantic.JsonValue]] = []
        service.listen(first.append)
        service.listen(second.append)
        page = await service.dispatch("desktop.sessions.list", {"limit": 1})
        assert isinstance(page, dict)
        assert page["visibility"] == {"revision": 0, "hidden_execution_ids": []}
        assert page["next_after"] == "one"
        delegate.assert_awaited_once_with(
            "executions.summaries", {"limit": 1}, session_id="embedded"
        )
        assert service.session_visibility is not None
        assert (
            service.session_visibility.path.parent
            == (await service.runner.project()).data_path
        )
        delegate.return_value = summary
        result = await service.dispatch(
            "desktop.sessions.set_hidden", {"execution_id": "one", "hidden": True}
        )
        delegate.assert_awaited_with(
            "executions.summary", {"execution_id": "one"}, session_id="embedded"
        )
        event = {"kind": "desktop.sessions.visibility", "visibility": result}
        assert first == second == [event]
        service.unlisten(second.append)
        await service.dispatch(
            "desktop.sessions.set_hidden", {"execution_id": "one", "hidden": False}
        )
        assert len(first) == 2
        assert len(second) == 1
        delegate.side_effect = KeyError("missing")
        with pytest.raises(KeyError):
            await service.dispatch(
                "desktop.sessions.set_hidden",
                {"execution_id": "missing", "hidden": True},
            )
        assert await service.session_visibility.access() == Snapshot(revision=2)
