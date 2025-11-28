from typing import Annotated, Optional, Union, List, Literal, Any
from pydantic import BaseModel, Field
from enum import Enum

from vocode.state import Message, ToolCall, Activity, RunnerStatus

PACKET_MESSAGE_REQUEST = "message_request"
PACKET_TOOL_CALL = "tool_call"
PACKET_TOOL_RESULT = "tool_result"
PACKET_MESSAGE = "message"
PACKET_FINAL_MESSAGE = "final_message"
PACKET_APPROVAL = "approval"
PACKET_TOKEN_USAGE = "token_usage"
PACKET_STATUS_CHANGE = "status_change"
PACKET_STOP = "stop"
PACKET_START_WORKFLOW = "start_workflow"

# Packet kinds that are considered "interim" (do not end an executor cycle)
INTERIM_PACKETS: tuple[str, ...] = (PACKET_MESSAGE, PACKET_TOKEN_USAGE)
PACKETS_FOR_HISTORY = (PACKET_FINAL_MESSAGE, PACKET_MESSAGE_REQUEST)


# Executor -> Runner events (discriminated by 'kind')
class TokenUsageTotals(BaseModel):
    """
    Aggregate LLM usage totals within the current process.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_dollars: float = 0.0
    current_prompt_tokens: Optional[int] = None
    current_completion_tokens: Optional[int] = None
    token_limit: Optional[int] = None


class ReqMessageRequest(BaseModel):
    """
    Request a message from a user.
    """

    kind: Literal["message_request"] = PACKET_MESSAGE_REQUEST
    message: Optional[str] = None


class ReqToolCall(BaseModel):
    """
    Tool call requests to be executed by the runner.
    """

    kind: Literal["tool_call"] = PACKET_TOOL_CALL
    tool_calls: List[ToolCall]


class ReqToolResult(BaseModel):
    """
    Tool call result notifications emitted by the runner for UI consumption.
    """

    kind: Literal["tool_result"] = PACKET_TOOL_RESULT
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


class ReqTokenUsage(BaseModel):
    """
    Accumulated token usage/cost report for the project.
    Sent as an interim packet prior to final response.
    """

    kind: Literal["token_usage"] = PACKET_TOKEN_USAGE
    acc_prompt_tokens: int
    acc_completion_tokens: int
    acc_cost_dollars: float
    current_prompt_tokens: Optional[int] = None
    current_completion_tokens: Optional[int] = None
    token_limit: Optional[int] = None
    # Indicates usage was generated in this process (hint for UIs)
    local: Optional[bool] = True
class ReqStartWorkflow(BaseModel):
    """
    Runner-level request to start another workflow and return its final message
    back to the requesting executor as a RespMessage. Handled internally by UIState.
    """
    kind: Literal["start_workflow"] = PACKET_START_WORKFLOW
    workflow: str
    initial_message: Optional[Message] = None

class ReqStatusChange(BaseModel):
    """
    Runner-emitted packet indicating a transition between nodes (and optional status change).
    """
    kind: Literal["status_change"] = PACKET_STATUS_CHANGE
    old_status: RunnerStatus
    new_status: RunnerStatus
    old_node: Optional[str] = None
    new_node: Optional[str] = None

class ReqStop(BaseModel):
    """
    Executor-emitted packet requesting the runner to stop gracefully.
    """
    kind: Literal["stop"] = PACKET_STOP
    reason: Optional[str] = None


ReqPacket = Annotated[
    Union[
        ReqMessageRequest,
        ReqToolCall,
        ReqToolResult,
        ReqInterimMessage,
        ReqFinalMessage,
        ReqTokenUsage,
        ReqStatusChange,
        ReqStop,
        ReqStartWorkflow,
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


class RunnerState(BaseModel):
    state: Optional[Any] = None
    req: Optional[ReqPacket] = None
    response: Optional[RespPacket] = None
    messages: Optional[List[Message]] = None


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
    response: Optional[RespPacket] = Field(None, description="Optional response packet")

class RunStats(BaseModel):
    """
    Per-node runtime statistics for a single Runner instance.
    Currently tracks the number of completed executions for the node.
    """
    run_count: int = 0
