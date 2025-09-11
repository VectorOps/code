from typing import Annotated, Optional, Union, List, Literal, Any
from pydantic import BaseModel, Field
from enum import Enum

from vocode.state import Message, ToolCall, Activity

PACKET_MESSAGE_REQUEST = "message_request"
PACKET_TOOL_CALL = "tool_call"
PACKET_MESSAGE = "message"
PACKET_FINAL_MESSAGE = "final_message"
PACKET_LOG = "log"
PACKET_APPROVAL = "approval"

class LogLevel(str, Enum):
    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"

# Executor -> Runner events (discriminated by 'kind')
class ReqMessageRequest(BaseModel):
    """
    Request a message from a user.
    """
    kind: Literal["message_request"] = PACKET_MESSAGE_REQUEST

class ReqToolCall(BaseModel):
    """
    Tool call requests to be executed by the runner.
    """
    kind: Literal["tool_call"] = PACKET_TOOL_CALL
    tool_calls: List[ToolCall]

class ReqInterimMessage(BaseModel):
    """
    Intermediate message provided by a tool or a user
    """
    kind: Literal["message"] = PACKET_MESSAGE
    message: Message

class ReqFinalMessage(BaseModel):
    """
    Final agent message (optional if node finishes without emitting a message)
    """
    kind: Literal["final_message"] = PACKET_FINAL_MESSAGE
    message: Optional[Message] = None
    outcome_name: Optional[str] = None


class ReqLogMessage(BaseModel):
    """
    Debug/log message emitted by an executor. Never requests input.
    """
    kind: Literal["log"] = PACKET_LOG
    text: str
    level: Optional[LogLevel] = None


ReqPacket = Annotated[
    Union[
        ReqMessageRequest,
        ReqToolCall,
        ReqInterimMessage,
        ReqFinalMessage,
        ReqLogMessage,
    ],
    Field(discriminator="kind"),
]

# Runner -> Executor responses
class RespMessage(BaseModel):
    """
    Message provided by a user.
    """
    kind: Literal["message"] = PACKET_MESSAGE
    message: Message

class RespToolCall(BaseModel):
    """
    Tool call responses.
    """
    kind: Literal["tool_call"] = PACKET_TOOL_CALL
    tool_calls: List[ToolCall]

class RespApproval(BaseModel):
    """
    Approval response
    """
    kind: Literal["approval"] = PACKET_APPROVAL
    approved: bool


RespPacket = Annotated[
    Union[
        RespMessage,
        RespToolCall,
        RespApproval,
    ],
    Field(discriminator="kind"),
]


class ExecRunInput(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    state: Optional[Any] = None
    response: Optional["RespPacket"] = None


# Runner to consumer events
class RunEvent(BaseModel):
    node: str = Field(..., description="Node name this event pertains to")
    execution: Activity = Field(
        description="Execution result for this node, when available"
    )
    event: ReqPacket
    input_requested: bool


class RunInput(BaseModel):
    response: Optional[RespPacket] = Field(
        None, description="Optional response packet"
    )
