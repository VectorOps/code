"""Desktop capabilities behind the runner's authenticated control transport."""

from pathlib import Path
from typing import Optional

import pydantic
from vectorops.runner.control import protocol
from vectorops.runner.control.service import EventSink, RunnerService

from vectorops_code.prompt_history import PromptHistory
from vectorops_code.file_search import FileSearch
from vectorops_code.settings import Store
from vectorops_code import session_visibility


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
        self.desktop_settings = Store(
            settings_path or self.configuration.path / "desktop-settings.json",
            development=development,
        )
        self.file_search: Optional[FileSearch] = None
        self.session_visibility: Optional[session_visibility.SessionVisibility] = None
        self.visibility_listeners: set[EventSink] = set()

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
        if self.session_visibility is None:
            project = await self.runner.project()
            self.session_visibility = session_visibility.SessionVisibility(
                project.data_path / "desktop-sessions.sqlite3"
            )
        if method == "desktop.sessions.list":
            page = protocol.SummaryPage.model_validate(
                await super().dispatch(
                    "executions.summaries", params, session_id=session_id
                )
            )
            visibility = await self.session_visibility.access()
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
            visibility = await self.session_visibility.access(update)
            event: dict[str, pydantic.JsonValue] = {
                "kind": "desktop.sessions.visibility",
                "visibility": visibility.model_dump(mode="json"),
            }
            for listener in tuple(self.visibility_listeners):
                listener(event)
            return visibility.model_dump(mode="json")
        raise protocol.RemoteError(
            "unsupported_method", "unknown desktop sessions method"
        )

    async def aclose(self) -> None:
        try:
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
        return await super().dispatch(method, params, session_id=session_id)
