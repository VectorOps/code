"""Desktop capabilities behind the runner's authenticated control transport."""

import asyncio
import datetime as dt
import logging
from pathlib import Path
from typing import Optional, Self

import pydantic
from vectorops.runner.control import protocol
from vectorops.runner.control.service import EventSink, RunnerService

from vectorops_code.prompt_history import PromptHistory
from vectorops_code.file_search import FileSearch
from vectorops_code import session_visibility, settings


class DesktopService(RunnerService):
    def __init__(
        self,
        path: Path,
        *,
        history_path: Optional[Path] = None,
        settings_path: Optional[Path] = None,
        development: bool = False,
    ) -> None:
        super().__init__(path)
        self.prompt_history = PromptHistory(history_path)
        self.desktop_settings = settings.Store(
            settings_path or self.configuration.path / "desktop-settings.json",
            development=development,
        )
        self.file_search: Optional[FileSearch] = None
        self.session_visibility: Optional[session_visibility.SessionVisibility] = None
        self.visibility_listeners: set[EventSink] = set()
        self._session_cleanup_task: Optional[asyncio.Task[None]] = None
        self._closing = False

    async def __aenter__(self) -> Self:
        await super().__aenter__()
        self.schedule_session_cleanup()
        return self

    def schedule_session_cleanup(self) -> None:
        if self._closing or (
            self._session_cleanup_task is not None
            and not self._session_cleanup_task.done()
        ):
            return
        self._session_cleanup_task = asyncio.create_task(
            self._cleanup_sessions(), name="desktop-session-cleanup"
        )

    async def _visibility(self) -> session_visibility.SessionVisibility:
        if self.session_visibility is None:
            project = await self.runner.project()
            self.session_visibility = session_visibility.SessionVisibility(
                project.data_path / "desktop-sessions.sqlite3"
            )
        return self.session_visibility

    def _publish_visibility(self, snapshot: session_visibility.Snapshot) -> None:
        event: dict[str, pydantic.JsonValue] = {
            "kind": "desktop.sessions.visibility",
            "visibility": snapshot.model_dump(mode="json"),
        }
        for listener in tuple(self.visibility_listeners):
            listener(event)

    async def _cleanup_sessions(self) -> None:
        """Scan in small yielding pages without holding execution or request locks."""
        touched: set[str] = set()

        def activity(event: dict[str, pydantic.JsonValue]) -> None:
            identifier = event.get("execution_id")
            if isinstance(identifier, str):
                touched.add(identifier)

        super().listen(activity)
        try:
            preferences = settings.Snapshot.model_validate(
                await self.desktop_settings.dispatch("desktop.settings.get", {})
            )
            if preferences.auto_hide_after_days == 0:
                return
            before = dt.datetime.now(dt.UTC) - dt.timedelta(
                days=preferences.auto_hide_after_days
            )
            visibility = await self._visibility()
            hidden = set((await visibility.access()).hidden_execution_ids)
            after = None
            seen: set[str] = set()
            while not self._closing:
                identifiers = await self.runner.execution_ids(after=after, limit=100)
                activity_times: dict[str, dt.datetime] = {}
                fresh = set(identifiers) - seen
                if not fresh:
                    return
                seen.update(fresh)
                for identifier in identifiers:
                    if identifier not in fresh or identifier in hidden:
                        continue
                    await asyncio.sleep(0)
                    try:
                        record = await self.runner.execution_state(identifier)
                        updated_at = dt.datetime.fromisoformat(record.updated_at)
                        if updated_at.tzinfo is not None:
                            activity_times[identifier] = updated_at
                    except (FileNotFoundError, KeyError, ValueError):
                        continue
                snapshot = await visibility.hide_inactive(
                    {
                        key: value
                        for key, value in activity_times.items()
                        if key not in touched
                    },
                    before,
                )
                if snapshot is not None:
                    restored = await visibility.restore_active(
                        tuple(touched.intersection(activity_times)), snapshot.revision
                    )
                    if restored is not None:
                        snapshot = restored
                    hidden.update(snapshot.hidden_execution_ids)
                    self._publish_visibility(snapshot)
                if len(identifiers) < 100:
                    return
                after = identifiers[-1]
        except Exception:
            logging.getLogger(__name__).exception("Automatic session cleanup failed")
        finally:
            super().unlisten(activity)

    def listen(self, listener: EventSink) -> None:
        super().listen(listener)
        self.visibility_listeners.add(listener)

    def unlisten(self, listener: EventSink) -> None:
        super().unlisten(listener)
        self.visibility_listeners.discard(listener)

    async def sessions(
        self, method: str, params: dict[str, pydantic.JsonValue], session_id: str
    ) -> pydantic.JsonValue:
        """Delegate execution state while attaching desktop visibility metadata."""
        store = await self._visibility()
        if method == "desktop.sessions.list":
            page = protocol.SummaryPage.model_validate(
                await super().dispatch(
                    "executions.summaries", params, session_id=session_id
                )
            )
            visibility = await store.access()
            return {
                **page.model_dump(mode="json"),
                "visibility": visibility.model_dump(mode="json"),
            }
        if method == "desktop.sessions.set_hidden":
            update = session_visibility.Update.model_validate(params)
            await super().dispatch(
                "executions.summary",
                {"execution_id": update.execution_id},
                session_id=session_id,
            )
            visibility = await store.access(update)
            self._publish_visibility(visibility)
            return visibility.model_dump(mode="json")
        raise protocol.RemoteError(
            "unsupported_method", "unknown desktop sessions method"
        )

    async def aclose(self) -> None:
        self._closing = True
        try:
            if self._session_cleanup_task is not None:
                self._session_cleanup_task.cancel()
                await asyncio.gather(self._session_cleanup_task, return_exceptions=True)
            if self.file_search is not None:
                await self.file_search.close()
        finally:
            await super().aclose()

    async def dispatch(
        self,
        method: str,
        params: dict[str, pydantic.JsonValue],
        *,
        session_id: str = "embedded",
    ) -> pydantic.JsonValue:
        if method.startswith("desktop.sessions."):
            return await self.sessions(method, params, session_id)
        if method.startswith("desktop.settings."):
            return await self.desktop_settings.dispatch(method, params)
        if method.startswith("prompt_history."):
            return await self.prompt_history.dispatch(method, params, str(self.path))
        if method.startswith("file_search."):
            if self.file_search is None:
                self.file_search = FileSearch(self.path)
            return await self.file_search.dispatch(method, params)
        result = await super().dispatch(method, params, session_id=session_id)
        if method == "executions.start":
            self.schedule_session_cleanup()
        return result
