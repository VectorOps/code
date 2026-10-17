"""Desktop capabilities behind the runner's authenticated control transport."""

from pathlib import Path
from typing import Optional

import pydantic
from vectorops.runner.control.service import RunnerService

from runner_desktop.prompt_history import PromptHistory
from runner_desktop.file_search import FileSearch


class DesktopService(RunnerService):
    def __init__(self, path: Path, *, history_path: Optional[Path] = None) -> None:
        super().__init__(path)
        self.prompt_history = PromptHistory(history_path)
        self.file_search: Optional[FileSearch] = None

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
        if method.startswith("prompt_history."):
            return await self.prompt_history.dispatch(method, params, str(self.path))
        if method.startswith("file_search."):
            if self.file_search is None:
                self.file_search = FileSearch(self.path)
            return await self.file_search.dispatch(method, params)
        return await super().dispatch(method, params, session_id=session_id)
