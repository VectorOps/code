from __future__ import annotations

from typing import AsyncIterator, List, Optional
import asyncio

from vocode.runner.runner import Executor
from vocode.graph.models import NoopNode
from vocode.state import Message
from vocode.runner.models import ReqPacket, ReqFinalMessage, RespMessage

class NoopExecutor(Executor):
    # Must match NoopNode.type
    type = "noop"

    def __init__(self, config, project):
        super().__init__(config=config, project=project)
        if not isinstance(config, NoopNode):
            raise TypeError("NoopExecutor requires config to be a NoopNode")

    async def run(self, messages: List[Message]) -> AsyncIterator[ReqPacket]:
        cfg: NoopNode = self.config  # type: ignore[assignment]
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

        while True:
            final = Message(role="agent", text="")
            resp = (yield ReqFinalMessage(message=final, outcome_name=outcome_name))
            # If runner sends a message back (additional requirements), just loop and immediately
            # provide another final; otherwise executor remains paused on approval.
            if isinstance(resp, RespMessage):
                # ignore content, loop to yield another final
                continue
            await asyncio.Event().wait()
