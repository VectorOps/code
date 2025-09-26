from __future__ import annotations
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING
import asyncio
import contextlib

if TYPE_CHECKING:
    from vocode.project import Project
    from vocode.ui.base import UIState  # for typing only


@dataclass
class CommandContext:
    project: "Project"
    ui: "UIState"  # type: ignore[name-defined]


@dataclass
class CommandDef:
    name: str
    help: str
    usage: Optional[str]
    handler: Callable[[CommandContext, List[str]], Awaitable[Optional[str]]]


@dataclass
class CommandDelta:
    added: List[CommandDef]
    removed: List[str]


class CommandManager:
    def __init__(self) -> None:
        self._registry: Dict[str, CommandDef] = {}
        self._listeners: List[asyncio.Queue[CommandDelta]] = []

    def register(
        self,
        name: str,
        help: str,
        handler: Callable[[CommandContext, List[str]], Awaitable[Optional[str]]],
        usage: Optional[str] = None,
    ) -> None:
        # If replacing, emit as remove + add
        if name in self._registry:
            old = self._registry[name]
            self._registry[name] = CommandDef(
                name=name, help=help, usage=usage, handler=handler
            )
            self._notify(
                CommandDelta(added=[self._registry[name]], removed=[old.name])
            )
            return
        cmd = CommandDef(name=name, help=help, usage=usage, handler=handler)
        self._registry[name] = cmd
        self._notify(CommandDelta(added=[cmd], removed=[]))

    def unregister(self, name: str) -> None:
        if name in self._registry:
            del self._registry[name]
            self._notify(CommandDelta(added=[], removed=[name]))

    def clear(self) -> None:
        if not self._registry:
            return
        removed = list(self._registry.keys())
        self._registry.clear()
        self._notify(CommandDelta(added=[], removed=removed))

    def get(self, name: str) -> Optional[CommandDef]:
        return self._registry.get(name)

    def list(self) -> List[CommandDef]:
        return sorted(self._registry.values(), key=lambda c: c.name)

    def subscribe(self) -> asyncio.Queue[CommandDelta]:
        q: asyncio.Queue[CommandDelta] = asyncio.Queue()
        self._listeners.append(q)
        # Emit current state initially
        if self._registry:
            q.put_nowait(CommandDelta(added=self.list(), removed=[]))
        return q

    def _notify(self, delta: CommandDelta) -> None:
        for q in list(self._listeners):
            try:
                q.put_nowait(delta)
            except Exception:
                with contextlib.suppress(ValueError):
                    self._listeners.remove(q)

    async def execute(
        self, name: str, ctx: CommandContext, args: List[str]
    ) -> Optional[str]:
        cmd = self._registry.get(name)
        if cmd is None:
            raise KeyError(f"Unknown command '{name}'")
        return await cmd.handler(ctx, args)
