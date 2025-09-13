from __future__ import annotations

from typing import AsyncIterator, Optional, Any

from vocode.runner.runner import Executor
from vocode.graph.models import InputNode
from vocode.state import Message
from vocode.runner.models import (
    ReqPacket,
    ReqMessageRequest,
    ReqInterimMessage,
    ReqFinalMessage,
    ExecRunInput,
    PACKET_MESSAGE,
)


class InputExecutor(Executor):
    # Must match InputNode.type
    type = "input"

    def __init__(self, config, project):
        super().__init__(config=config, project=project)
        if not isinstance(config, InputNode):
            raise TypeError("InputExecutor requires config to be an InputNode")

    async def run(self, inp: ExecRunInput) -> AsyncIterator[tuple[ReqPacket, Optional[Any]]]:
        cfg: InputNode = self.config  # type: ignore[assignment]
        # If runner already provided a user message, finalize immediately.
        if inp.response is not None and inp.response.kind == PACKET_MESSAGE and inp.response.message is not None:
            user_text = inp.response.message.text or ""
            final_msg = Message(role="agent", text=user_text)
            yield ReqFinalMessage(message=final_msg), inp.state
            return
        # Otherwise, prompt and request a user message.
        if cfg.message:
            yield ReqInterimMessage(message=Message(role="agent", text=cfg.message)), inp.state
        yield ReqMessageRequest(), inp.state
