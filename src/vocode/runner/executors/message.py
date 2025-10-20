from typing import AsyncIterator, List, Optional, Any
import asyncio

from vocode.runner.runner import Executor
from vocode.models import Node
from vocode.state import Message
from vocode.runner.models import ReqPacket, ReqFinalMessage, ExecRunInput, ReqLogMessage


class MessageNode(Node):
    type: str = "message"
    message: str


class MessageExecutor(Executor):
    # Must match MessageNode.type
    type = "message"

    def __init__(self, config, project):
        super().__init__(config=config, project=project)
        if not isinstance(config, MessageNode):
            raise TypeError("MessageExecutor requires config to be a MessageNode")

    async def run(
        self, inp: ExecRunInput
    ) -> AsyncIterator[tuple[ReqPacket, Optional[Any]]]:
        cfg: MessageNode = self.config  # type: ignore[assignment]
        final = Message(role="agent", text=cfg.message)
        # Emit the final response first
        yield ReqFinalMessage(message=final), None
