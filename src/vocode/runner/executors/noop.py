from __future__ import annotations

from typing import AsyncIterator, List, Optional

from vocode.runner.runner import Executor
from vocode.graph.models import NoopNode
from vocode.state import Message
from vocode.runner.models import ReqPacket, ReqFinalMessage

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
            # Prefer a conventional outcome name if present, otherwise first
            names = [s.name for s in outs]
            for pref in ("next", "success"):
                if pref in names:
                    outcome_name = pref
                    break
            if outcome_name is None:
                outcome_name = outs[0].name

        # Emit a final message and let Runner transition (or stop if no outcomes)
        final = Message(role="agent", text="")
        _ = (yield ReqFinalMessage(message=final, outcome_name=outcome_name))
        return
