from __future__ import annotations

from enum import Enum
from typing import List, Optional, Annotated
from pydantic import BaseModel, Field, StringConstraints


# Roles like agent, user, system, tool, etc. Keep flexible to allow custom roles.
Role = Annotated[str, StringConstraints(min_length=1)]


class BlockType(str, Enum):
    plain = "plain"
    code = "code"
    diff = "diff"


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
    messages: List[Message] = Field(default_factory=list)
    output_name: Optional[str] = None


class Step(BaseModel):
    executions: List[NodeExecution] = Field(default_factory=list)


class Task(BaseModel):
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
