from enum import Enum
from typing import List, Optional, Annotated
from pydantic import BaseModel, Field, StringConstraints
from uuid import UUID, uuid4


# Roles like agent, user, system, etc. Keep flexible to allow custom roles.
Role = Annotated[str, StringConstraints(min_length=1)]


class RunnerStatus(str, Enum):
    idle = "idle"
    running = "running"
    waiting_input = "waiting_input"
    canceled = "canceled"
    stopped = "stopped"
    finished = "finished"

class StepStatus(str, Enum):
    running = "running"
    finished = "finished"
    canceled = "canceled"
    stopped = "stopped"


class ToolCallStatus(str, Enum):
    created = "created"
    running = "running"
    completed = "completed"
    failed = "failed"
    rejected = "rejected"


class ToolCall(BaseModel):
    id: Optional[str] = Field(
        default=None, description="Provider-issued id for this tool call (e.g., 'call_...')"
    )
    type: str = Field(
        default="function",
        description="Tool call type (currently 'function' per OpenAI schema)",
    )
    status: ToolCallStatus = Field(default=ToolCallStatus.created, description="Tool call status")
    name: str = Field(..., description="Function name to call")
    arguments: str = Field(..., description="JSON-encoded arguments passed to the function")
    result: Optional[str] = Field(
        default=None,
        description="JSON-encoded result of the function call; None until completed",
    )


class Message(BaseModel):
    """
    A single human-readable message that's produced by a human or a tool.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this step")
    role: Role = Field(..., description="Sender role: agent, user, system, tool, etc.")
    text: str = Field(..., description="Original message as received/emitted")

    tool_calls: List[ToolCall] = Field(default_factory=list, description="List of recorded tool calls for this message")


class NodeExecution(BaseModel):
    """
    A single Node execution state.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this node execution")
    input_messages: List[Message] = Field(default_factory=list, description="Input messages given to this execution")
    output_message: Optional[Message] = None
    outcome_name: Optional[str] = None
    is_canceled: bool = Field(
        False,
        description="True when this execution was explicitly canceled by the runner",
    )
    is_complete: bool = Field(
        False,
        description="True only when the executor's async generator finished and this is the final aggregated result",
    )

    def clone(self, **overrides) -> "NodeExecution":
        data = {
            "id": self.id,
            "input_messages": list(self.input_messages),
            "output_messages": list(self.output_messages),
            "outcome_name": self.outcome_name,
            "is_canceled": self.is_canceled,
            "is_complete": self.is_complete,
        }
        data.update(overrides)
        return NodeExecution(**data)


class Step(BaseModel):
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this step")
    node: str = Field(..., description="Node name this step pertains to")
    executions: List[NodeExecution] = Field(default_factory=list)
    status: StepStatus = Field(
        default=StepStatus.running,
        description="Current status of this step: running until finalized as finished, canceled, or stopped",
    )


class Task(BaseModel):
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this task")
    steps: List[Step] = Field(default_factory=list)
