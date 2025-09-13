from __future__ import annotations

from typing import AsyncIterator, Optional, Any
import asyncio

from vocode.runner.runner import Executor
from vocode.graph.models import NoopNode
from vocode.state import Message
from vocode.runner.models import ReqPacket, ReqFinalMessage, ExecRunInput

class NoopExecutor(Executor):
    # Must match NoopNode.type
    type = "noop"

    def __init__(self, config, project):
        super().__init__(config=config, project=project)
        if not isinstance(config, NoopNode):
            raise TypeError("NoopExecutor requires config to be a NoopNode")

    async def run(self, inp: ExecRunInput) -> AsyncIterator[tuple[ReqPacket, Optional[Any]]]:
        cfg: NoopNode = self.config  # type: ignore[assignment]
        # Optional delay before emitting the final message
        delay = cfg.sleep_seconds
        if delay is not None and delay > 0:
            await asyncio.sleep(delay)
        outcome_name: Optional[str] = None

        outs = cfg.outcomes or []
        if len(outs) == 1:
            outcome_name = outs[0].name
        elif len(outs) > 1:
            names = [s.name for s in outs]
            for pref in ("next", "success"):
                if pref in names:
                    outcome_name = pref
                    break
            if outcome_name is None:
                outcome_name = outs[0].name

        final = Message(role="agent", text="")
        yield (ReqFinalMessage(message=final, outcome_name=outcome_name), None)
        return
