import asyncio
import logging
from typing import Dict, Any, Optional

from vocode.settings import Settings
from vocode.state import LogLevel
from .logging_interceptor import UILoggingHandler


_LEVEL_MAP: Dict[LogLevel, int] = {
    LogLevel.debug: logging.DEBUG,
    LogLevel.info: logging.INFO,
    LogLevel.warning: logging.WARNING,
    LogLevel.error: logging.ERROR,
}


def _to_std_level(level: LogLevel) -> int:
    return _LEVEL_MAP.get(level, logging.INFO)


def _get_default_level(settings: Settings) -> int:
    # Prefer new logging.default_level; fallback to UI log level; default INFO
    if settings.logging and settings.logging.default_level:
        return _to_std_level(settings.logging.default_level)
    if settings.ui and settings.ui.log_level:
        return _to_std_level(settings.ui.log_level)
    return logging.INFO


def configure_stdlib_logging(settings: Settings) -> None:
    """
    Configure stdlib logging levels:
    - Root at INFO to suppress debug from non-configured loggers.
    - vocode and knowlt set to default level from settings (or UI level).
    - Apply per-logger overrides from settings.logging.enabled_loggers.
    """
    root = logging.getLogger()
    # Suppress debug globally; individual loggers will be adjusted below
    root.setLevel(logging.INFO)

    default_level = _get_default_level(settings)

    # Always configure our primary loggers
    for name in ("vocode", "knowlt"):
        lg = logging.getLogger(name)
        lg.setLevel(default_level)
        lg.propagate = True

    # Optional per-logger overrides
    overrides: Dict[str, LogLevel] = {}
    if settings.logging and settings.logging.enabled_loggers:
        overrides = settings.logging.enabled_loggers

    for name, lvl in overrides.items():
        try:
            lno = _to_std_level(lvl)
        except Exception:
            lno = default_level
        lg = logging.getLogger(name)
        lg.setLevel(lno)
        lg.propagate = True


def attach_ui_interceptor(
    loop: asyncio.AbstractEventLoop, queue: "asyncio.Queue[Dict[str, Any]]"
) -> logging.Handler:
    """
    Create and attach UILoggingHandler to the root logger, returning the handler.
    """
    handler = UILoggingHandler(loop, queue)
    logging.getLogger().addHandler(handler)
    return handler
