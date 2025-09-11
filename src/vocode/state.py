from enum import Enum
from typing import List, Optional, Annotated, Dict, Any, Union
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

class ActivityType(str, Enum):
    executor = "executor"
    user = "user"
    created = "created"
    running = "running"
    completed = "completed"
    failed = "failed"
    rejected = "rejected"

class ToolCallStatus(str, Enum):
    created = "created"
    completed = "completed"
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
    arguments: Dict[str, Any] = Field(
        ..., description="Decoded JSON arguments passed to the function"
    )
    result: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(
        default=None,
        description="Decoded JSON result of the function call; may be a dict or a list of dicts; None until completed",
    )


class Message(BaseModel):
    """
    A single human-readable message that's produced by a human or a tool.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this step")
    role: Role = Field(..., description="Sender role: agent, user, system, tool, etc.")
    text: str = Field(..., description="Original message as received/emitted")

    tool_calls: List[ToolCall] = Field(default_factory=list, description="List of recorded tool calls for this message")


class Activity(BaseModel):
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this node execution")
    type: ActivityType = Field(..., description="Origin of this activity: executor or user")
    message: Optional[Message] = Field(default=None, description="Message carried by this activity, if any")
    outcome_name: Optional[str] = None
    is_canceled: bool = Field(False, description="True when this execution was explicitly canceled by the runner")
    is_complete: bool = Field(False, description="True when this activity represents a completed final result")
    state: Optional[Any] = Field(default=None, description="Opaque executor state to carry between run cycles")

    def clone(self, **overrides) -> "Activity":
        data = {
            "id": self.id,
            "type": self.type,
            "message": self.message,
            "outcome_name": self.outcome_name,
            "is_canceled": self.is_canceled,
            "is_complete": self.is_complete,
            "state": self.state,
        }
        data.update(overrides)
        return Activity(**data)


class Step(BaseModel):
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this step")
    node: str = Field(..., description="Node name this step pertains to")
    executions: List[Activity] = Field(default_factory=list)
    status: StepStatus = Field(
        default=StepStatus.running,
        description="Current status of this step: running until finalized as finished, canceled, or stopped",
    )


class Assignment(BaseModel):
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this assignment")
    steps: List[Step] = Field(default_factory=list)
