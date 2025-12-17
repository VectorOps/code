from typing import AsyncIterator, Optional, Any

from pydantic import Field

from vocode.models import Node
from vocode.runner.runner import Executor
from vocode.runner.models import (
    ReqStartWorkflow,
    ReqFinalMessage,
    ExecRunInput,
    ReqPacket,
    PACKET_MESSAGE,
)
from vocode.state import Message


class RunAgentNode(Node):
    """Node that starts a nested workflow/agent in a new runner frame.

    This replaces the former StartWorkflowNode. The underlying runner packet
    remains ReqStartWorkflow(kind="start_workflow") for wire compatibility.
    """

    type: str = Field(default="run_agent")
    workflow: str = Field(
        ..., description="Name of the workflow/agent to start in a new runner frame"
    )
    initial_text: Optional[str] = Field(
        default=None,
        description="Optional initial user message text for the child agent",
    )


class RunAgentExecutor(Executor):
    """Executor for RunAgentNode.

    It uses the existing ReqStartWorkflow packet so the UI/runner stack logic
    does not change on the wire.
    """

    type = "run_agent"

    async def run(
        self, inp: ExecRunInput
    ) -> AsyncIterator[tuple[ReqPacket, Optional[Any]]]:
        if inp.response is not None and inp.response.kind == PACKET_MESSAGE:
            yield ReqFinalMessage(message=inp.response.message), inp.state
            return

        if not inp.state:
            cfg: RunAgentNode = self.config
            initial_msg: Optional[Message] = None
            if cfg.initial_text is not None:
                initial_msg = Message(role="user", text=cfg.initial_text)
            yield ReqStartWorkflow(
                workflow=cfg.workflow, initial_message=initial_msg
            ), True
            return

        return


Executor.register("run_agent", RunAgentExecutor)
