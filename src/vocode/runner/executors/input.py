from __future__ import annotations

from typing import AsyncIterator, List
import asyncio

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

        next_user_text: str | None = None

        while True:
            # 1) Send configured prompt message as an interim agent message
            _ = yield ReqInterimMessage(message=Message(role="agent", text=cfg.message))

            # 2) Request input from the user (unless we already have it from a previous final response)
            if next_user_text is None:
                resp: RespPacket = (yield ReqMessageRequest())
                # Runner guarantees a RespMessage here
                user_text = resp.message.text if isinstance(resp, RespMessage) else ""
            else:
                user_text = next_user_text
                next_user_text = None

            # 3) Send the user input back as the final agent message
            final_msg = Message(role="agent", text=user_text)
            post_final = (yield ReqFinalMessage(message=final_msg))

            # If runner sent a message back (additional requirements), use it as the next user_text
            if isinstance(post_final, RespMessage):
                next_user_text = post_final.message.text
                continue

            # Otherwise, pause here on approval; remain suspended until the runner closes/recreates us.
            await asyncio.Event().wait()
