"""Bounded request snapshots from the runner's workspace file search service."""

from pathlib import Path

import pydantic
from vectorops.runner import file_search
from vectorops.runner.control import protocol


class Query(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", strict=True)

    query: str = pydantic.Field(default="", max_length=256)
    limit: int = pydantic.Field(default=20, ge=1, le=20)
    refresh: bool = False


class FileSearch:
    def __init__(self, root: Path) -> None:
        self.service = file_search.FileSearchService(root, file_search.SearchConfig())
        self.started = False

    async def dispatch(
        self, method: str, params: dict[str, pydantic.JsonValue]
    ) -> pydantic.JsonValue:
        """Return one latest snapshot; clients poll only while discovery is active."""
        if method != "file_search.search":
            raise protocol.RemoteError(
                "unsupported_method", "unknown file search method"
            )
        try:
            query = Query.model_validate(params)
        except pydantic.ValidationError:
            raise protocol.RemoteError(
                "invalid_argument", "invalid file search parameters"
            ) from None
        if query.refresh or not self.started:
            await self.service.refresh()
            self.started = True
        updates = self.service.search(query.query, limit=query.limit)
        try:
            update = await anext(updates)
            return update.model_dump(mode="json")
        finally:
            await updates.aclose()

    async def close(self) -> None:
        await self.service.close()
