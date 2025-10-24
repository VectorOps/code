from __future__ import annotations

import asyncio
import logging
import sys
from typing import Optional

from vocode.ui.terminal.helpers import out, out_fmt


class TerminalUILogHandler(logging.Handler):
    """
    Logging handler that marshals log records onto an asyncio event loop
    and emits them using the terminal helpers (out/out_fmt).

    - Thread-safe: safe to call from non-async contexts and background threads.
    - Non-blocking: scheduling is done via loop.create_task or run_coroutine_threadsafe.
    - Fallback: if no running loop is available, falls back to a plain print.
    """

    def __init__(
        self,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        *,
        use_fmt: bool = False,
        level_prefix: bool = True,
        name_prefix: bool = True,
    ) -> None:
        super().__init__()
        self._loop = loop
        self._use_fmt = use_fmt
        self._level_prefix = level_prefix
        self._name_prefix = name_prefix

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Optionally set or update the target loop after construction."""
        self._loop = loop

    def _prefix(self, record: logging.LogRecord) -> str:
        parts = []
        if self._level_prefix:
            parts.append(f"[{record.levelname}]")
        if self._name_prefix and record.name:
            parts.append(f"{record.name}:")
        return " ".join(parts)

    async def _emit_async(self, text: str) -> None:
        # Choose helper based on configuration; out() is plain text,
        # out_fmt() can accept formatted/ANSI text if desired.
        if self._use_fmt:
            # For now, we just pass through text; styling can be added by callers
            # via a custom Formatter that injects ANSI if desired.
            await out_fmt(text)
        else:
            await out(text)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg_body = self.format(record)
        except Exception:
            # If formatting fails, delegate to logging's error handling.
            self.handleError(record)
            return

        prefix = self._prefix(record)
        text = f"{prefix} {msg_body}" if prefix else msg_body

        loop = self._loop
        # If no loop provided, try to pick up the currently running loop.
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

        if loop and loop.is_running():
            try:
                # If called on the same loop thread, schedule directly.
                try:
                    current = asyncio.get_running_loop()
                except RuntimeError:
                    current = None

                if current is loop:
                    asyncio.create_task(self._emit_async(text))
                else:
                    asyncio.run_coroutine_threadsafe(self._emit_async(text), loop)
            except Exception:
                # Preserve logging's default error handling path.
                self.handleError(record)
        else:
            # Fallback to synchronous printing if we can't reach a running loop.
            try:
                stream = sys.stderr if record.levelno >= logging.WARNING else sys.stdout
                print(text, file=stream, flush=True)
            except Exception:
                self.handleError(record)


def install(
    loop: asyncio.AbstractEventLoop,
    *,
    level: int = logging.INFO,
    logger: Optional[logging.Logger] = None,
    use_fmt: bool = False,
) -> TerminalUILogHandler:
    """
    Convenience installer.
    - loop: target asyncio loop to schedule terminal emissions.
    - level: minimum level for the handler.
    - logger: which logger to attach to (defaults to root logger).
    - use_fmt: if True, emits via out_fmt() instead of out().
    Returns the created handler for further customization.
    """
    handler = TerminalUILogHandler(loop=loop, use_fmt=use_fmt)
    handler.setLevel(level)
    # Default formatter includes level, name, and message; tracebacks are handled via exc_info.
    handler.setFormatter(
        logging.Formatter(fmt="%(message)s")
    )
    target_logger = logger or logging.getLogger()
    target_logger.addHandler(handler)
    return handler
