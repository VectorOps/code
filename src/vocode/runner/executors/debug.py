from __future__ import annotations

from typing import AsyncIterator, List
import asyncio

from vocode.runner.runner import Executor
from vocode.state import Message
from vocode.runner.models import (
    ReqPacket,
    ReqFinalMessage,
    ReqLogMessage,
    LogLevel,
    RespMessage,
)


class DebugExecutor(Executor):
    # Node type this executor handles
    type = "debug"

    def __init__(self, config, project):
        super().__init__(config=config, project=project)
        # Intentionally do not enforce a specific Node subclass to keep this executor usable
        # without requiring a DebugNode model.

    async def run(self, messages: List[Message]) -> AsyncIterator[ReqPacket]:
        # Log all incoming messages at debug level
        for m in messages:
            try:
                text = f"{m.role}: {m.text}"
            except Exception:
                text = str(m)
            yield ReqLogMessage(level=LogLevel.debug, text=text)

        # Yield an empty final message and loop like noop: ignore user messages, otherwise pause.
        while True:
            final = Message(role="agent", text="")
            resp = (yield ReqFinalMessage(message=final))
            if isinstance(resp, RespMessage):
                # Ignore content, loop to yield another final
                continue
            # Pause indefinitely until stopped/canceled
            await asyncio.Event().wait()
