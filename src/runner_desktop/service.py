"""Desktop capabilities behind the runner's authenticated control transport."""

from pathlib import Path
from typing import Optional

import pydantic
from vectorops.runner.control.service import RunnerService

from runner_desktop.prompt_history import PromptHistory


class DesktopService(RunnerService):
    def __init__(self, path: Path, *, history_path: Optional[Path] = None) -> None:
        super().__init__(path)
        self.prompt_history = PromptHistory(history_path)

    async def dispatch(
        self,
        method: str,
        params: dict[str, pydantic.JsonValue],
        *,
        session_id: str = "embedded",
    ) -> pydantic.JsonValue:
        if method.startswith("prompt_history."):
            return await self.prompt_history.dispatch(method, params, str(self.path))
        return await super().dispatch(method, params, session_id=session_id)
