from __future__ import annotations

import asyncio
import faulthandler
import signal
import sys
import warnings
from collections.abc import Awaitable, Callable
from typing import Any, Mapping, Optional


def setup_fault_handlers() -> None:
    """
    Enable faulthandler and install default warning filters for UI entrypoints.
    Shared between terminal and other UIs.
    """
    faulthandler.enable(sys.stderr)
    faulthandler.register(signal.SIGUSR1)

    # Fix litellm / Pydantic deprecation noise.
    from pydantic.warnings import (
        PydanticDeprecatedSince211,
        PydanticDeprecatedSince20,
    )

    warnings.filterwarnings(action="ignore", category=PydanticDeprecatedSince211)
    warnings.filterwarnings(action="ignore", category=PydanticDeprecatedSince20)
    warnings.filterwarnings(action="ignore", category=PydanticDeprecatedSince20)
    warnings.filterwarnings(action="ignore", category=DeprecationWarning)


def install_unhandled_exception_logging(
    loop: asyncio.AbstractEventLoop,
    *,
    should_exit_cb: Optional[Callable[[], bool]] = None,
    log_coro: Callable[[str], Awaitable[None]],
) -> None:
    """
    Register an asyncio event-loop exception handler that logs unhandled
    exceptions via the provided async logging coroutine.

    `should_exit_cb`, if provided, is checked before logging so shutdown
    can suppress extra noise.
    """

    def _handler(loop_: asyncio.AbstractEventLoop, context: Mapping[str, Any]) -> None:
        if should_exit_cb is not None and should_exit_cb():
            return

        msg = context.get("exception", context.get("message"))
        if msg is None:
            msg = "Unknown event loop error"

        text = f"\n--- Unhandled exception in event loop ---\n{msg}\n"
        coro = log_coro(text)
        asyncio.run_coroutine_threadsafe(coro, loop_)

    loop.set_exception_handler(_handler)
