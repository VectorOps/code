import asyncio
import logging
import contextlib
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
    return logging.INFO


def _remove_all_handlers(lg: logging.Logger) -> None:
    # Remove and close all handlers to eliminate existing log destinations
    for h in list(lg.handlers):
        try:
            lg.removeHandler(h)
        finally:
            with contextlib.suppress(Exception):
                h.close()


def configure_stdlib_logging(settings: Settings) -> None:
    """
    Configure stdlib logging levels:
    - Root at INFO to suppress debug from non-configured loggers.
    - vocode and knowlt set to default level from settings (or UI level).
    - Apply per-logger overrides from settings.logging.enabled_loggers.
    - Remove any existing log destinations (handlers) so only the UI interceptor
      will be active after attach_ui_interceptor is called.
    """
    root = logging.getLogger()

    # Remove existing log destinations on root; suppress debug globally.
    _remove_all_handlers(root)
    root.setLevel(logging.INFO)

    default_level = _get_default_level(settings)

    # Optional per-logger overrides
    overrides: Dict[str, LogLevel] = {}
    if settings.logging and settings.logging.enabled_loggers:
        overrides = settings.logging.enabled_loggers
    # Determine all loggers we configure levels for and clear their handlers
    names_to_configure = {"vocode", "knowlt"}
    names_to_configure.update(overrides.keys())

    # Configure primary and overridden loggers
    for name in names_to_configure:
        lg = logging.getLogger(name)
        # Ensure records propagate to root (which will hold the UI handler)
        lg.propagate = True
        # Remove any direct handlers to avoid duplicate emissions
        _remove_all_handlers(lg)
        # Set levels
        if name in overrides:
            try:
                lno = _to_std_level(overrides[name])
            except Exception:
                lno = default_level
            lg.setLevel(lno)
        else:
            lg.setLevel(default_level)


def attach_ui_interceptor(
    loop: asyncio.AbstractEventLoop, queue: "asyncio.Queue[Dict[str, Any]]"
) -> logging.Handler:
    """
    Create and attach UILoggingHandler to the root logger, returning the handler.
    """
    handler = UILoggingHandler(loop, queue)
    logging.getLogger().addHandler(handler)
    return handler
