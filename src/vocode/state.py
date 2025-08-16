from __future__ import annotations

from enum import Enum
from typing import List, Optional, Annotated
from pydantic import BaseModel, Field, StringConstraints
from uuid import UUID, uuid4


# Roles like agent, user, system, tool, etc. Keep flexible to allow custom roles.
Role = Annotated[str, StringConstraints(min_length=1)]


class BlockType(str, Enum):
    plain = "plain"
    code = "code"
    diff = "diff"

class RunnerStatus(str, Enum):
    idle = "idle"
    running = "running"
    waiting_input = "waiting_input"
    canceled = "canceled"
    stopped = "stopped"
    finished = "finished"


class MessageBlock(BaseModel):
    type: BlockType = Field(..., description="Type of parsed block: plain, code, diff, etc.")
    text: str = Field(..., description="Block contents as text")
    language: Optional[str] = Field(
        None,
        description="Programming/markup language for code-like blocks (e.g., python, diff, md).",
    )


class Message(BaseModel):
    role: Role = Field(..., description="Sender role: agent, user, system, etc.")
    raw: str = Field(..., description="Original raw message as received/emitted (unparsed)")
    blocks: List[MessageBlock] = Field(
        default_factory=list,
        description="Parsed blocks extracted from raw",
    )


class NodeExecution(BaseModel):
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this node execution")
    input_messages: List[Message] = Field(default_factory=list, description="Input messages given to this execution")
    messages: List[Message] = Field(default_factory=list)
    output_name: Optional[str] = None


class Step(BaseModel):
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this step")
    node: str = Field(..., description="Node name this step pertains to")
    executions: List[NodeExecution] = Field(default_factory=list)


class Task(BaseModel):
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this task")
    steps: List[Step] = Field(default_factory=list)


class RunEvent(BaseModel):
    node: str = Field(..., description="Node name this event pertains to")
    need_input: bool = Field(
        False, description="True when runner requests additional input for the node"
    )
    execution: Optional[NodeExecution] = Field(
        None, description="Execution result for this node, when available"
    )

class RunInput(BaseModel):
    loop: bool = Field(
        False, description="When True, re-run the current node with any provided messages"
    )
    messages: List[Message] = Field(
        default_factory=list,
        description="Additional messages to provide to the node on re-run",
    )
