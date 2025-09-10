from typing import AsyncIterator, List

from vocode.runner.runner import Executor
from vocode.graph.models import MessageNode
from vocode.state import Message
from vocode.runner.models import ReqPacket, ReqFinalMessage

class MessageExecutor(Executor):
    # Must match MessageNode.type
    type = "message"

    def __init__(self, config, project):
        super().__init__(config=config, project=project)
        if not isinstance(config, MessageNode):
            raise TypeError("MessageExecutor requires config to be a MessageNode")

    async def run(self, messages: List[Message]) -> AsyncIterator[ReqPacket]:
        cfg: MessageNode = self.config  # type: ignore[assignment]
        # Emit configured text as the final message and finish.
        yield ReqFinalMessage(message=Message(role="agent", text=cfg.message))
