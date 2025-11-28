from enum import Enum
from typing import List, Optional, Annotated, Dict, Any, Union
from pydantic import BaseModel, Field, StringConstraints
from pydantic import model_validator
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


class RunStatus(str, Enum):
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


class LogLevel(str, Enum):
    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"


class ToolCallStatus(str, Enum):
    created = "created"
    completed = "completed"
    rejected = "rejected"


class ToolCall(BaseModel):
    id: Optional[str] = Field(
        default=None,
        description="Provider-issued id for this tool call (e.g., 'call_...')",
    )
    type: str = Field(
        default="function",
        description="Tool call type (currently 'function' per OpenAI schema)",
    )
    status: ToolCallStatus = Field(
        default=ToolCallStatus.created, description="Tool call status"
    )
    name: str = Field(..., description="Function name to call")
    arguments: Dict[str, Any] = Field(
        ..., description="Decoded JSON arguments passed to the function"
    )
    result: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(
        default=None,
        description="Decoded JSON result of the function call; may be a dict or a list of dicts; None until completed",
    )
    tool_spec: Optional[Any] = Field(
        default=None,
        description="Effective tool spec for this call (vocode.settings.ToolSpec). Must be resolved by the executor; None is invalid at runtime.",
    )


class Message(BaseModel):
    """
    A single human-readable message that's produced by a human or a tool.
    """

    id: UUID = Field(
        default_factory=uuid4, description="Unique identifier for this step"
    )
    role: Role = Field(..., description="Sender role: agent, user, system, tool, etc.")
    text: str = Field(..., description="Original message as received/emitted")
    node: Optional[str] = Field(
        default=None, description="Name of the node that produced this message"
    )

    tool_calls: List[ToolCall] = Field(
        default_factory=list, description="List of recorded tool calls for this message"
    )


class Activity(BaseModel):
    id: UUID = Field(
        default_factory=uuid4, description="Unique identifier for this node execution"
    )
    type: ActivityType = Field(
        ..., description="Origin of this activity: executor or user"
    )
    message: Optional[Message] = Field(
        default=None, description="Message carried by this activity, if any"
    )
    outcome_name: Optional[str] = None
    is_canceled: bool = Field(
        False,
        description="True when this execution was explicitly canceled by the runner",
    )
    is_complete: bool = Field(
        False, description="True when this activity represents a completed final result"
    )
    runner_state: Optional[Any] = Field(
        default=None,
        description="Partial runner state (BaseModel instance) with optional fields: 'state', 'req', 'response'",
    )
    ephemeral: bool = Field(
        default=True,
        description="True when this activity is temporary (not persisted to history/step)",
    )


    def clone(self, **overrides) -> "Activity":
        data = {
            "id": self.id,
            "type": self.type,
            "message": self.message,
            "outcome_name": self.outcome_name,
            "is_canceled": self.is_canceled,
            "is_complete": self.is_complete,
            "runner_state": self.runner_state,
            "ephemeral": self.ephemeral,
        }
        data.update(overrides)
        return Activity(**data)


class Step(BaseModel):
    id: UUID = Field(
        default_factory=uuid4, description="Unique identifier for this step"
    )
    node: str = Field(..., description="Node name this step pertains to")
    executions: List[Activity] = Field(default_factory=list)
    status: RunStatus = Field(
        default=RunStatus.running,
        description="Current status of this step: running until finalized as finished, canceled, or stopped",
    )


class Assignment(BaseModel):
    id: UUID = Field(
        default_factory=uuid4, description="Unique identifier for this assignment"
    )
    status: RunStatus = Field(
        default=RunStatus.running,
        description="Current status of this assignment",
    )
    steps: List[Step] = Field(default_factory=list)


class TaskStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"


class Task(BaseModel):
    id: str = Field(
        ...,
        description="Stable identifier for this task (e.g. 'step-1').",
    )
    title: str = Field(
        ...,
        description="Short human-readable description of this task.",
    )
    status: TaskStatus = Field(
        default=TaskStatus.pending,
        description="Current status of this task.",
    )


class TaskList(BaseModel):
    todos: List[Task] = Field(
        default_factory=list,
        description="Current ordered list of tasks for this run.",
    )

    @model_validator(mode="after")
    def _validate_single_in_progress(self) -> "TaskList":
        in_progress_count = sum(
            1 for task in self.todos if task.status == TaskStatus.in_progress
        )
        if in_progress_count > 1:
            raise ValueError(
                "At most one task may have status 'in_progress' in the task list."
            )
        return self
