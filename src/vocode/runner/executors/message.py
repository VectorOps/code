from typing import AsyncIterator, List
import asyncio

from vocode.runner.runner import Executor
from vocode.graph.models import MessageNode
from vocode.state import Message
from vocode.runner.models import ReqPacket, ReqFinalMessage, RespMessage

class MessageExecutor(Executor):
    # Must match MessageNode.type
    type = "message"

    def __init__(self, config, project):
        super().__init__(config=config, project=project)
        if not isinstance(config, MessageNode):
            raise TypeError("MessageExecutor requires config to be a MessageNode")

    async def run(self, messages: List[Message]) -> AsyncIterator[ReqPacket]:
        cfg: MessageNode = self.config  # type: ignore[assignment]
        while True:
            # Emit configured text as the final message and then behave like noop:
            # - if the runner sends a user message back, ignore and yield another final
            # - otherwise pause indefinitely (until stopped/canceled)
            final = Message(role="agent", text=cfg.message)
            resp = (yield ReqFinalMessage(message=final))
            if isinstance(resp, RespMessage):
                continue
            await asyncio.Event().wait()
