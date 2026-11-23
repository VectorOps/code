import asyncio
import datetime as dt
from pathlib import Path
from threading import Event
from typing import Optional
from unittest.mock import AsyncMock

import pydantic
import pytest
from vectorops.runner.control import protocol
from vectorops.runner.control.service import EventSink, RunnerService
from vectorops.runner.runner.execution_state import ExecutionRecord
from vectorops.runner.runner.workflow import WorkflowRunner

from vectorops_code import session_visibility as visibility, settings
from vectorops_code.service import DesktopService


async def test_strict_inactivity_boundary_and_idempotent_batches(
    tmp_path: Path,
) -> None:
    store = visibility.SessionVisibility(tmp_path / "visibility.sqlite3")
    before = dt.datetime.now(dt.UTC) - dt.timedelta(days=7)
    activity = {
        "old": before - dt.timedelta(microseconds=1),
        "boundary": before,
        "recent": before + dt.timedelta(days=1),
        "future": before + dt.timedelta(days=10),
        "offset": (before - dt.timedelta(seconds=1)).astimezone(
            dt.timezone(dt.timedelta(hours=5))
        ),
    }
    snapshot = await store.hide_inactive(activity, before)
    assert snapshot == visibility.Snapshot(
        revision=1, hidden_execution_ids=["offset", "old"]
    )
    assert await store.hide_inactive(activity, before) is None
    assert await visibility.SessionVisibility(store.path).access() == snapshot


async def test_manual_show_survives_concurrent_cleanup_and_grace_expires(
    tmp_path: Path,
) -> None:
    store = visibility.SessionVisibility(tmp_path / "visibility.sqlite3")
    now = dt.datetime.now(dt.UTC)
    activity = {"old": now - dt.timedelta(days=30)}
    await store.access(visibility.Update(execution_id="old", hidden=True))
    await asyncio.gather(
        store.hide_inactive(activity, now - dt.timedelta(days=7)),
        visibility.SessionVisibility(store.path).access(
            visibility.Update(execution_id="old", hidden=False)
        ),
    )
    assert (await store.access()).hidden_execution_ids == []
    assert await store.hide_inactive(activity, now - dt.timedelta(days=7)) is None
    expired = await store.hide_inactive(activity, now + dt.timedelta(days=1))
    assert expired is not None
    assert expired.hidden_execution_ids == ["old"]


async def test_cleanup_storage_does_not_block_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = Event()
    release = Event()
    store = visibility.SessionVisibility(tmp_path / "visibility.sqlite3")
    access = store._access

    def slow_access(
        update: Optional[visibility.Update],
        inactive: tuple[str, ...] = (),
        before: float = 0,
        restore: tuple[str, ...] = (),
        revision: int = 0,
    ) -> tuple[visibility.Snapshot, bool]:
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release storage")
        return access(update, inactive, before, restore, revision)

    monkeypatch.setattr(store, "_access", slow_access)
    now = dt.datetime.now(dt.UTC)
    task = asyncio.create_task(
        store.hide_inactive(
            {"old": now - dt.timedelta(days=8)}, now - dt.timedelta(days=7)
        )
    )
    try:
        assert await asyncio.wait_for(asyncio.to_thread(started.wait, 1), 2)
        await asyncio.wait_for(asyncio.sleep(0), 1)
        assert not task.done()
    finally:
        release.set()
        await task


async def test_activity_repair_preserves_concurrent_manual_hide(tmp_path: Path) -> None:
    store = visibility.SessionVisibility(tmp_path / "visibility.sqlite3")
    before = dt.datetime.now(dt.UTC) - dt.timedelta(days=7)
    snapshot = await store.hide_inactive(
        {
            "resumed": before - dt.timedelta(days=1),
            "manual": before - dt.timedelta(days=1),
        },
        before,
    )
    assert snapshot is not None
    await store.access(visibility.Update(execution_id="manual", hidden=True))
    restored = await store.restore_active(("resumed", "manual"), snapshot.revision)
    assert restored == visibility.Snapshot(revision=2, hidden_execution_ids=["manual"])
    assert await store.restore_active(("resumed", "manual"), snapshot.revision) is None


@pytest.mark.parametrize("days", [0, 1, 7, 36500])
async def test_inactivity_setting_defaults_and_persists(
    tmp_path: Path, days: int
) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"appearance":"dark"}')
    store = settings.Store(path)
    initial = settings.Snapshot.model_validate(
        await store.dispatch("desktop.settings.get", {})
    )
    assert initial.auto_hide_after_days == 7
    await store.dispatch("desktop.settings.update", {"auto_hide_after_days": days})
    saved = settings.Snapshot.model_validate(
        await settings.Store(path).dispatch("desktop.settings.get", {})
    )
    assert saved.auto_hide_after_days == days
    assert saved.appearance == "dark"


@pytest.mark.parametrize("days", [-1, 36501, True, 1.5, "7", None])
async def test_invalid_inactivity_settings_are_not_saved(
    tmp_path: Path, days: pydantic.JsonValue
) -> None:
    store = settings.Store(tmp_path / "settings.json")
    with pytest.raises(protocol.RemoteError) as error:
        await store.dispatch("desktop.settings.update", {"auto_hide_after_days": days})
    assert error.value.code == "invalid_argument"
    assert not store.path.exists()


async def test_startup_cleanup_pages_records_and_publishes_only_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identifiers = [f"execution-{index}" for index in range(101)]
    pages = AsyncMock(side_effect=[identifiers[:100], identifiers[100:]])
    now = dt.datetime.now(dt.UTC)

    async def state(identifier: str) -> ExecutionRecord:
        if identifier == "execution-1":
            raise FileNotFoundError
        updated_at = (now - dt.timedelta(days=8)).isoformat()
        if identifier == "execution-0":
            updated_at = now.isoformat()
        elif identifier == "execution-2":
            updated_at = "invalid"
        elif identifier == "execution-3":
            updated_at = (now - dt.timedelta(days=8)).replace(tzinfo=None).isoformat()
        return ExecutionRecord(
            id=identifier,
            root_workflow_id="root",
            status="paused" if identifier == "execution-100" else "running",
            updated_at=updated_at,
        )

    reads = AsyncMock(side_effect=state)
    monkeypatch.setattr(WorkflowRunner, "execution_ids", pages)
    monkeypatch.setattr(WorkflowRunner, "execution_state", reads)
    async with DesktopService(
        tmp_path, settings_path=tmp_path / "settings.json"
    ) as service:
        events: list[dict[str, pydantic.JsonValue]] = []
        service.listen(events.append)
        assert service._session_cleanup_task is not None
        await service._session_cleanup_task
        assert service.session_visibility is not None
        result = await service.session_visibility.access()
        assert result.hidden_execution_ids == sorted(identifiers[4:])
        assert result.revision == 2
        assert pages.await_args_list[0].kwargs == {"after": None, "limit": 100}
        assert pages.await_args_list[1].kwargs == {
            "after": "execution-99",
            "limit": 100,
        }
        assert len(events) == 2
        assert events[-1] == {
            "kind": "desktop.sessions.visibility",
            "visibility": result.model_dump(mode="json"),
        }
        pages.side_effect = [identifiers[:100], identifiers[100:]]
        reads.reset_mock()
        service.schedule_session_cleanup()
        await service._session_cleanup_task
        assert len(events) == 2
        assert reads.await_count == 4


@pytest.mark.parametrize(("days", "hidden"), [(0, False), (10, False), (1, True)])
async def test_cleanup_uses_configured_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, days: int, hidden: bool
) -> None:
    path = tmp_path / "settings.json"
    await settings.Store(path).dispatch(
        "desktop.settings.update", {"auto_hide_after_days": days}
    )
    pages = AsyncMock(return_value=["old"])
    reads = AsyncMock(
        return_value=ExecutionRecord(
            id="old",
            root_workflow_id="root",
            status="paused",
            updated_at=(dt.datetime.now(dt.UTC) - dt.timedelta(days=8)).isoformat(),
        )
    )
    monkeypatch.setattr(WorkflowRunner, "execution_ids", pages)
    monkeypatch.setattr(WorkflowRunner, "execution_state", reads)
    async with DesktopService(tmp_path, settings_path=path) as service:
        assert service._session_cleanup_task is not None
        await service._session_cleanup_task
        if days == 0:
            pages.assert_not_awaited()
            reads.assert_not_awaited()
            assert service.session_visibility is None
        else:
            assert service.session_visibility is not None
            assert (await service.session_visibility.access()).hidden_execution_ids == (
                ["old"] if hidden else []
            )


async def test_startup_and_start_return_without_waiting_and_coalesce_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def cleanup() -> None:
        started.set()
        await release.wait()

    background = AsyncMock(side_effect=cleanup)
    monkeypatch.setattr(DesktopService, "_cleanup_sessions", background)
    delegate = AsyncMock(return_value={"execution_id": "new"})
    monkeypatch.setattr(RunnerService, "dispatch", delegate)
    service = DesktopService(tmp_path, settings_path=tmp_path / "settings.json")
    async with service:
        await asyncio.wait_for(started.wait(), 1)
        task = service._session_cleanup_task
        assert task is not None and not task.done()
        result = await asyncio.wait_for(service.dispatch("executions.start", {}), 1)
        assert result == {"execution_id": "new"}
        assert service._session_cleanup_task is task
        background.assert_awaited_once()
        release.set()
        await task
        delegate.side_effect = protocol.RemoteError("invalid_argument", "cannot start")
        with pytest.raises(protocol.RemoteError):
            await service.dispatch("executions.start", {})
        assert service._session_cleanup_task is task
        delegate.side_effect = None
        release.clear()
        started.clear()
        await service.dispatch("executions.start", {})
        assert service._session_cleanup_task is not task
        await asyncio.wait_for(started.wait(), 1)
        assert background.await_count == 2
    assert service._session_cleanup_task is not None
    assert service._session_cleanup_task.cancelled()
    service.schedule_session_cleanup()
    assert background.await_count == 2


@pytest.mark.parametrize("phase", ["scan", "commit"])
async def test_activity_during_cleanup_is_not_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    listeners: list[EventSink] = []
    listen = RunnerService.listen

    def track(service: RunnerService, listener: EventSink) -> None:
        listeners.append(listener)
        listen(service, listener)

    async def state(identifier: str) -> ExecutionRecord:
        if identifier == "other" and phase == "scan":
            listeners[0]({"kind": "execution.updated", "execution_id": "resumed"})
        return ExecutionRecord(
            id=identifier,
            root_workflow_id="root",
            status="paused",
            updated_at=(dt.datetime.now(dt.UTC) - dt.timedelta(days=30)).isoformat(),
        )

    hide = visibility.SessionVisibility.hide_inactive

    async def concurrent_hide(
        store: visibility.SessionVisibility,
        activity_times: dict[str, dt.datetime],
        before: dt.datetime,
    ) -> Optional[visibility.Snapshot]:
        result = await hide(store, activity_times, before)
        if phase == "commit":
            listeners[0]({"kind": "execution.updated", "execution_id": "resumed"})
        return result

    monkeypatch.setattr(visibility.SessionVisibility, "hide_inactive", concurrent_hide)
    monkeypatch.setattr(RunnerService, "listen", track)
    monkeypatch.setattr(
        WorkflowRunner, "execution_ids", AsyncMock(return_value=["resumed", "other"])
    )
    monkeypatch.setattr(WorkflowRunner, "execution_state", AsyncMock(side_effect=state))
    async with DesktopService(
        tmp_path, settings_path=tmp_path / "settings.json"
    ) as service:
        assert service._session_cleanup_task is not None
        await service._session_cleanup_task
        assert service.session_visibility is not None
        assert (await service.session_visibility.access()).hidden_execution_ids == [
            "other"
        ]


async def test_cleanup_errors_are_logged_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pages = AsyncMock(side_effect=OSError("cannot read index"))
    monkeypatch.setattr(WorkflowRunner, "execution_ids", pages)
    async with DesktopService(
        tmp_path, settings_path=tmp_path / "settings.json"
    ) as service:
        assert service._session_cleanup_task is not None
        await service._session_cleanup_task
        assert "Automatic session cleanup failed" in caplog.text
        pages.side_effect = None
        pages.return_value = []
        service.schedule_session_cleanup()
        await service._session_cleanup_task
        assert pages.await_count == 2
