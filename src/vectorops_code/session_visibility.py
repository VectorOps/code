"""Workspace-owned desktop visibility, independent of execution state."""

import asyncio
import os
from pathlib import Path
import sqlite3
from typing import Optional

import pydantic
from vectorops.runner.control import protocol


class Update(protocol.Execution):
    hidden: bool = pydantic.Field(strict=True)


class Snapshot(pydantic.BaseModel):
    revision: int = 0
    hidden_execution_ids: list[str] = pydantic.Field(default_factory=list)


class SessionVisibility:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _access(self, update: Optional[Update]) -> Snapshot:
        """Read or atomically update visibility and its revision."""
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if os.name == "posix":
                os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        database = sqlite3.connect(self.path, timeout=5)
        try:
            with database:
                database.execute("BEGIN IMMEDIATE")
                database.execute(
                    "CREATE TABLE IF NOT EXISTS hidden_sessions "
                    "(execution_id TEXT PRIMARY KEY)"
                )
                database.execute(
                    "CREATE TABLE IF NOT EXISTS visibility_revision "
                    "(id INTEGER PRIMARY KEY CHECK (id = 1), revision INTEGER NOT NULL)"
                )
                database.execute(
                    "INSERT OR IGNORE INTO visibility_revision VALUES (1, 0)"
                )
                if update is not None:
                    changed = database.execute(
                        (
                            "INSERT OR IGNORE INTO hidden_sessions VALUES (?)"
                            if update.hidden
                            else "DELETE FROM hidden_sessions WHERE execution_id = ?"
                        ),
                        (update.execution_id,),
                    ).rowcount
                    if changed:
                        database.execute(
                            "UPDATE visibility_revision SET revision = revision + 1"
                        )
                return Snapshot(
                    revision=database.execute(
                        "SELECT revision FROM visibility_revision WHERE id = 1"
                    ).fetchone()[0],
                    hidden_execution_ids=[
                        row[0]
                        for row in database.execute(
                            "SELECT execution_id FROM hidden_sessions ORDER BY execution_id"
                        )
                    ],
                )
        finally:
            database.close()

    async def access(self, update: Optional[Update] = None) -> Snapshot:
        try:
            return await asyncio.to_thread(self._access, update)
        except (OSError, sqlite3.Error):
            raise protocol.RemoteError(
                "visibility_storage_error", "cannot access workspace session visibility"
            ) from None
