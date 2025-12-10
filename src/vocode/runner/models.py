from typing import Annotated, Optional, Union, List, Literal, Any
from pydantic import BaseModel, Field
from enum import Enum

import vocode.state as vstate

PACKET_MESSAGE_REQUEST = "message_request"
PACKET_TOOL_CALL = "tool_call"
PACKET_TOOL_RESULT = "tool_result"
PACKET_MESSAGE = "message"
PACKET_FINAL_MESSAGE = "final_message"
PACKET_APPROVAL = "approval"
PACKET_TOKEN_USAGE = "token_usage"
PACKET_LOCAL_TOKEN_USAGE = "local_token_usage"
PACKET_STATUS_CHANGE = "status_change"
PACKET_STOP = "stop"
PACKET_START_WORKFLOW = "start_workflow"

# Packet kinds that are considered "interim" (do not end an executor cycle)
INTERIM_PACKETS: tuple[str, ...] = (
    PACKET_MESSAGE,
    PACKET_TOKEN_USAGE,
    PACKET_LOCAL_TOKEN_USAGE,
)
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
    tool_calls: List[vstate.ToolCall]


class ReqToolResult(BaseModel):
    """
    Tool call result notifications emitted by the runner for UI consumption.
    """

    kind: Literal["tool_result"] = PACKET_TOOL_RESULT
    tool_calls: List[vstate.ToolCall]


class ReqInterimMessage(BaseModel):
    """
    Intermediate message provided by a tool or a user
    """

    kind: Literal["message"] = PACKET_MESSAGE
    message: vstate.Message


class ReqFinalMessage(BaseModel):
    """
    Final agent message (optional if node finishes without emitting a message)
    """

    kind: Literal["final_message"] = PACKET_FINAL_MESSAGE
    message: Optional[vstate.Message] = None
    outcome_name: Optional[str] = None


class ReqTokenUsage(BaseModel):
    """
    Accumulated token usage/cost report for the project.
    Sent as an interim packet prior to final response.
    """

    kind: Literal["token_usage"] = PACKET_TOKEN_USAGE
    global_usage: vstate.LLMUsageStats
    session_usage: vstate.LLMUsageStats
    node_usage: vstate.LLMUsageStats
    # Per-call context-window usage (e.g. latest prompt/response) for the current node.
    context_usage: vstate.LLMUsageStats = Field(
        default_factory=vstate.LLMUsageStats,
        description="Per-call context-window usage for the current node.",
    )
    # Indicates usage was generated in this process (hint for UIs)
    local: bool = True


class ReqLocalTokenUsage(BaseModel):
    """
    Per-node local token usage emitted by executors.
    `usage` is the absolute per-node totals at the time of emission.
    `delta` is the incremental usage since the previous emission, or None when
    no new tokens are being reported.
    Runner aggregates only non-zero `delta` into session/project totals and
    forwards `usage` as node_usage inside ReqTokenUsage for UI consumption.
    """

    kind: Literal["local_token_usage"] = PACKET_LOCAL_TOKEN_USAGE
    # Per-call context-window usage for the last completed LLM round.
    context_usage: vstate.LLMUsageStats = Field(
        default_factory=vstate.LLMUsageStats,
        description="Per-call context-window usage for the last completed LLM round.",
    )
    # Accumulated per-node totals at the time of emission.
    usage: vstate.LLMUsageStats
    delta: Optional[vstate.LLMUsageStats] = Field(
        default=None,
        description="Incremental usage since previous emission; None when no new tokens.",
    )


class ReqStartWorkflow(BaseModel):
    """
    Runner-level request to start another workflow and return its final message
    back to the requesting executor as a RespMessage. Handled internally by UIState.
    """

    kind: Literal["start_workflow"] = PACKET_START_WORKFLOW
    workflow: str
    initial_message: Optional[vstate.Message] = None


class ReqStatusChange(BaseModel):
    """
    Runner-emitted packet indicating a transition between nodes (and optional status change).
    """

    kind: Literal["status_change"] = PACKET_STATUS_CHANGE
    old_status: vstate.RunnerStatus
    new_status: vstate.RunnerStatus
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
        ReqLocalTokenUsage,
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
    message: vstate.Message


class RespToolCall(BaseModel):
    """
    Tool call responses.
    """

    kind: Literal["tool_call"] = PACKET_TOOL_CALL
    tool_calls: List[vstate.ToolCall]


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
    messages: Optional[List[vstate.Message]] = None


class ExecRunInput(BaseModel):
    messages: List[vstate.Message] = Field(default_factory=list)
    state: Optional[Any] = None
    response: Optional["RespPacket"] = None


# Runner to consumer events
class RunEvent(BaseModel):
    node: str = Field(..., description="Node name this event pertains to")
    execution: vstate.Activity = Field(
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
