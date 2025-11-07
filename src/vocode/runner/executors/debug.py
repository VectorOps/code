from __future__ import annotations

from typing import AsyncIterator, List, Optional, Any
import logging

from vocode.runner.runner import Executor
from vocode.state import Message
from vocode.runner.models import (
    ReqPacket,
    ReqFinalMessage,
    ExecRunInput,
)
from vocode.logger import logger


class DebugExecutor(Executor):
    # Node type this executor handles
    type = "debug"

    def __init__(self, config, project):
        super().__init__(config=config, project=project)
        # Intentionally do not enforce a specific Node subclass to keep this executor usable
        # without requiring a DebugNode model.

    async def run(
        self, inp: ExecRunInput
    ) -> AsyncIterator[tuple[ReqPacket, Optional[Any]]]:
        # Log all incoming messages at debug level using stdlib logging (forwarded to UI by interceptor)
        for m in inp.messages:
            try:
                text = f"{m.role}: {m.text}"
            except Exception:
                text = str(m)
            logger.debug(text)

        # Yield an empty final message for this cycle
        final = Message(role="agent", text="")
        yield (ReqFinalMessage(message=final), inp.state)
        return
