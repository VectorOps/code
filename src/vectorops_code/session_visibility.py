"""Workspace-owned desktop visibility, independent of execution state."""

import asyncio
from datetime import datetime
import os
from pathlib import Path
import sqlite3
import time
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

    def _access(
        self,
        update: Optional[Update],
        inactive: tuple[str, ...] = (),
        before: float = 0,
        restore: tuple[str, ...] = (),
        revision: int = 0,
    ) -> tuple[Snapshot, bool]:
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
                database.execute(
                    "CREATE TABLE IF NOT EXISTS visibility_metadata "
                    "(execution_id TEXT PRIMARY KEY, shown_at REAL, automatic_revision INTEGER)"
                )
                current_revision = database.execute(
                    "SELECT revision FROM visibility_revision WHERE id = 1"
                ).fetchone()[0]
                changed = 0
                if update is not None:
                    changed = database.execute(
                        (
                            "INSERT OR IGNORE INTO hidden_sessions VALUES (?)"
                            if update.hidden
                            else "DELETE FROM hidden_sessions WHERE execution_id = ?"
                        ),
                        (update.execution_id,),
                    ).rowcount
                    database.execute(
                        "INSERT INTO visibility_metadata VALUES (?, ?, NULL) "
                        "ON CONFLICT(execution_id) DO UPDATE "
                        "SET shown_at = excluded.shown_at, automatic_revision = NULL",
                        (update.execution_id, None if update.hidden else time.time()),
                    )
                elif inactive:
                    for identifier in inactive:
                        inserted = database.execute(
                            "INSERT OR IGNORE INTO hidden_sessions "
                            "SELECT ? WHERE NOT EXISTS (SELECT 1 FROM visibility_metadata "
                            "WHERE execution_id = ? AND shown_at >= ?)",
                            (identifier, identifier, before),
                        ).rowcount
                        if inserted:
                            database.execute(
                                "INSERT INTO visibility_metadata(execution_id, automatic_revision) "
                                "VALUES (?, ?) ON CONFLICT(execution_id) DO UPDATE "
                                "SET automatic_revision = excluded.automatic_revision",
                                (identifier, current_revision + 1),
                            )
                            changed += inserted
                elif restore:
                    changed = database.executemany(
                        "DELETE FROM hidden_sessions WHERE execution_id IN "
                        "(SELECT execution_id FROM visibility_metadata "
                        "WHERE execution_id = ? AND automatic_revision = ?)",
                        [(identifier, revision) for identifier in restore],
                    ).rowcount
                if changed:
                    database.execute(
                        "UPDATE visibility_revision SET revision = revision + 1"
                    )
                snapshot = Snapshot(
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
                return snapshot, changed > 0
        finally:
            database.close()

    async def access(self, update: Optional[Update] = None) -> Snapshot:
        snapshot, _ = await self._apply(update)
        return snapshot

    async def hide_inactive(
        self, activity: dict[str, datetime], before: datetime
    ) -> Optional[Snapshot]:
        """Hide stale executions unless a recent manual show grants a grace period."""
        inactive = tuple(
            identifier
            for identifier, updated_at in activity.items()
            if updated_at < before
        )
        if not inactive:
            return None
        snapshot, changed = await self._apply(None, inactive, before.timestamp())
        return snapshot if changed else None

    async def _apply(
        self,
        update: Optional[Update],
        inactive: tuple[str, ...] = (),
        before: float = 0,
        restore: tuple[str, ...] = (),
        revision: int = 0,
    ) -> tuple[Snapshot, bool]:
        try:
            return await asyncio.to_thread(
                self._access, update, inactive, before, restore, revision
            )
        except (OSError, sqlite3.Error):
            raise protocol.RemoteError(
                "visibility_storage_error", "cannot access workspace session visibility"
            ) from None

    async def restore_active(
        self, identifiers: tuple[str, ...], revision: int
    ) -> Optional[Snapshot]:
        """Undo this cleanup's hides after concurrent activity, preserving manual choices."""
        if not identifiers:
            return None
        snapshot, changed = await self._apply(
            None, restore=identifiers, revision=revision
        )
        return snapshot if changed else None
