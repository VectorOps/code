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


# New node to start a stacked workflow
class StartWorkflowNode(Node):
    # Enforce node type for registry/dispatch
    type: str = Field(default="start_workflow")
    # Name of the child workflow to start
    workflow: str = Field(
        ..., description="Name of the workflow to start in a new runner frame"
    )
    # Optional initial text for the child workflow (sent as a user message)
    initial_text: Optional[str] = Field(
        default=None,
        description="Optional initial user message text for the child workflow",
    )


class StartWorkflowExecutor(Executor):
    # Handles nodes of type "start_workflow"
    type = "start_workflow"

    async def run(
        self, inp: ExecRunInput
    ) -> AsyncIterator[tuple[ReqPacket, Optional[Any]]]:
        # Child finished: UIState resumes us with a RespMessage; finalize with that message.
        if inp.response is not None and inp.response.kind == PACKET_MESSAGE:
            yield ReqFinalMessage(message=inp.response.message), inp.state
            return

        # First cycle: request starting the child workflow.
        if not inp.state:
            cfg: StartWorkflowNode = self.config  # node config is strongly-typed
            initial_msg: Optional[Message] = None
            if cfg.initial_text is not None:
                initial_msg = Message(role="user", text=cfg.initial_text)
            yield ReqStartWorkflow(
                workflow=cfg.workflow, initial_message=initial_msg
            ), True
            return

        # Already requested; wait for child final to resume.
        return


# Explicitly register to be robust even if automatic subclass registration changes.
Executor.register("start_workflow", StartWorkflowExecutor)
