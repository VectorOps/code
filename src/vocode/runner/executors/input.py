from __future__ import annotations

from typing import AsyncIterator, List

from vocode.runner.runner import Executor
from vocode.graph.models import InputNode
from vocode.state import Message
from vocode.runner.models import (
    ReqPacket,
    ReqMessageRequest,
    ReqInterimMessage,
    ReqFinalMessage,
    RespPacket,
    RespMessage,
)


class InputExecutor(Executor):
    # Must match InputNode.type
    type = "input"

    def __init__(self, config, project):
        super().__init__(config=config, project=project)
        if not isinstance(config, InputNode):
            raise TypeError("InputExecutor requires config to be an InputNode")

    async def run(self, messages: List[Message]) -> AsyncIterator[ReqPacket]:
        cfg: InputNode = self.config  # type: ignore[assignment]

        # 1) Send configured prompt message as an interim agent message
        _ = yield ReqInterimMessage(message=Message(role="agent", text=cfg.message))

        # 2) Request input from the user
        resp: RespPacket = (yield ReqMessageRequest())
        print(resp)

        # Runner guarantees a RespMessage response here
        user_text = resp.message.text if isinstance(resp, RespMessage) else ""

        # 3) Send the user input back as the final agent message
        final_msg = Message(role="agent", text=user_text)
        _ = (yield ReqFinalMessage(message=final_msg))
        return
