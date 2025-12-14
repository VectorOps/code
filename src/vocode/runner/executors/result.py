from __future__ import annotations

from typing import AsyncIterator, Optional, Any
from pydantic import BaseModel, Field

from vocode.runner.runner import Executor
from vocode.models import Node, Confirmation, MessageMode
from vocode.state import Message
from vocode.runner.models import ReqPacket, ReqFinalMessage, ExecRunInput


class ResultNode(Node):
    type: str = "result"
    # Auto-skip without prompting for approval
    confirmation: Confirmation = Field(
        default=Confirmation.auto, description="No-op auto confirmation"
    )


class ResultExecutor(Executor):
    # Must match ResultNode.type
    type = "result"

    def __init__(self, config, project):
        super().__init__(config=config, project=project)
        if not isinstance(config, ResultNode):
            raise TypeError("ResultExecutor requires config to be a ResultNode")

    async def run(
        self, inp: ExecRunInput
    ) -> AsyncIterator[tuple[ReqPacket, Optional[Any]]]:
        texts = [m.text for m in inp.messages if m.text is not None]
        combined = "\n".join(texts)
        final = Message(role="agent", text=combined)
        yield ReqFinalMessage(message=final), None
