import asyncio
import logging
from typing import Any, Dict, Optional

from vocode.state import LogLevel


def map_std_level(levelno: int) -> LogLevel:
    if levelno >= logging.ERROR:
        return LogLevel.error
    if levelno >= logging.WARNING:
        return LogLevel.warning
    if levelno >= logging.INFO:
        return LogLevel.info
    return LogLevel.debug


class UILoggingHandler(logging.Handler):
    """
    Logging handler that forwards log records to a thread-safe asyncio queue.
    UIState consumes this queue and forwards logs to the UI protocol.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: "asyncio.Queue[Dict[str, Any]]",
        level: int = logging.NOTSET,
    ) -> None:
        super().__init__(level=level)
        self._loop = loop
        self._queue = queue
        # Use a basic formatter only for exception formatting if needed
        self._fmt = logging.Formatter()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)

        exc_text: Optional[str] = None
        if record.exc_info:
            try:
                exc_text = self._fmt.formatException(record.exc_info)
            except Exception:
                # Fallback: repr
                exc_text = repr(record.exc_info)

        payload = {
            "level": map_std_level(record.levelno),
            "message": msg,
            "logger": getattr(record, "name", None),
            "pathname": getattr(record, "pathname", None),
            "lineno": getattr(record, "lineno", None),
            "exc_text": exc_text,
        }
        # Thread-safe enqueue
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, payload)
        except Exception:
            # Best-effort only; never raise from emit
            pass
