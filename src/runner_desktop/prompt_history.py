"""Durable, bounded prompt recall shared by desktop clients on this host."""

import asyncio
from contextlib import contextmanager
import datetime as dt
import os
from pathlib import Path
import sqlite3
from typing import Iterator, Literal, Optional

import pydantic

from vectorops.runner.control import protocol


class Append(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", hide_input_in_errors=True)
    text: str = pydantic.Field(min_length=1, max_length=65536)
    execution_id: str = pydantic.Field(min_length=1, max_length=128)
    idempotency_key: str = pydantic.Field(min_length=1, max_length=256)

    @pydantic.field_validator("text", mode="after")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must contain text")
        return value


class Query(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    scope: Literal["global", "project", "session"] = "global"
    execution_id: Optional[str] = pydantic.Field(default=None, max_length=128)
    query: str = pydantic.Field(default="", max_length=4096)
    before: Optional[int] = pydantic.Field(default=None, ge=1, strict=True)
    limit: int = pydantic.Field(default=20, ge=1, le=20, strict=True)

    @pydantic.model_validator(mode="after")
    def session_required(self) -> "Query":
        if self.scope == "session" and not self.execution_id:
            raise ValueError("session scope requires an execution ID")
        return self


class Entry(pydantic.BaseModel):
    id: int
    project_id: str
    execution_id: str
    timestamp: str
    text: str


class Page(pydantic.BaseModel):
    entries: list[Entry]
    next_before: Optional[int] = None


class PromptHistory:
    def __init__(self, path: Optional[Path] = None) -> None:
        state = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
        if not state.is_absolute():
            state = Path.home() / ".local/state"
        self.path = path or state / "runner-desktop/prompt-history.sqlite3"

    @contextmanager
    def _database(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if os.name == "posix":
                os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        database = sqlite3.connect(self.path, timeout=5)
        database.row_factory = sqlite3.Row
        database.create_function("casefold", 1, str.casefold, deterministic=True)
        try:
            with database:
                database.execute(
                    "CREATE TABLE IF NOT EXISTS prompts ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "project_id TEXT NOT NULL, execution_id TEXT NOT NULL, "
                    "timestamp TEXT NOT NULL, text TEXT NOT NULL, "
                    "idempotency_key TEXT NOT NULL UNIQUE)"
                )
                yield database
        finally:
            database.close()

    def _append(self, project: str, request: Append) -> Entry:
        with self._database() as database:
            database.execute(
                "INSERT INTO prompts (project_id, execution_id, timestamp, text, "
                "idempotency_key) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(idempotency_key) DO NOTHING",
                (
                    project,
                    request.execution_id,
                    dt.datetime.now(dt.UTC).isoformat(),
                    request.text,
                    request.idempotency_key,
                ),
            )
            row = database.execute(
                "SELECT * FROM prompts WHERE idempotency_key = ?",
                (request.idempotency_key,),
            ).fetchone()
            entry = Entry.model_validate(dict(row))
            if (entry.project_id, entry.execution_id, entry.text) != (
                project,
                request.execution_id,
                request.text,
            ):
                raise ValueError("history idempotency key was reused")
            database.execute(
                "DELETE FROM prompts WHERE id NOT IN "
                "(SELECT id FROM prompts ORDER BY id DESC LIMIT 1000)"
            )
            return entry

    def _list(self, project: str, request: Query) -> Page:
        with self._database() as database:
            rows = database.execute(
                "WITH matches AS (SELECT * FROM prompts WHERE "
                "(? = 'global' OR project_id = ?) AND "
                "(? != 'session' OR execution_id = ?) AND "
                "instr(casefold(text), casefold(?)) > 0), "
                "latest AS (SELECT MAX(id) AS id FROM matches GROUP BY text) "
                "SELECT * FROM matches WHERE id IN (SELECT id FROM latest) "
                "AND (? IS NULL OR id < ?) ORDER BY id DESC LIMIT ?",
                (
                    request.scope,
                    project,
                    request.scope,
                    request.execution_id,
                    request.query,
                    request.before,
                    request.before,
                    request.limit + 1,
                ),
            ).fetchall()
            entries = [Entry.model_validate(dict(row)) for row in rows[: request.limit]]
            return Page(
                entries=entries,
                next_before=entries[-1].id if len(rows) > request.limit else None,
            )

    async def dispatch(
        self, method: str, params: dict[str, pydantic.JsonValue], project: str
    ) -> pydantic.JsonValue:
        """Validate requests and move all SQLite I/O off the event loop."""
        try:
            if method == "prompt_history.append":
                entry = await asyncio.to_thread(
                    self._append, project, Append.model_validate(params)
                )
                return entry.model_dump(mode="json")
            if method == "prompt_history.list":
                page = await asyncio.to_thread(
                    self._list, project, Query.model_validate(params)
                )
                return page.model_dump(mode="json")
        except (pydantic.ValidationError, ValueError):
            raise protocol.RemoteError(
                "invalid_argument", "invalid prompt history parameters"
            ) from None
        except (OSError, sqlite3.Error):
            raise protocol.RemoteError(
                "history_storage_error", "cannot access prompt history on host"
            ) from None
        raise protocol.RemoteError(
            "unsupported_method", "unknown prompt history method"
        )
