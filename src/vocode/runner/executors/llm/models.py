from __future__ import annotations

from typing import AsyncIterator, List, Optional, Dict, Any, Final
import json
import re
import asyncio
from vocode.runner.executors.llm.preprocessors.base import apply_preprocessors
from litellm import acompletion, completion_cost
import litellm
from enum import Enum
from pydantic import BaseModel, Field
from vocode.runner.runner import Executor
from vocode.models import Node, PreprocessorSpec, OutcomeStrategy, Mode
from vocode.state import Message, ToolCall, LogLevel
from vocode.runner.models import (
    ReqPacket,
    ReqToolCall,
    ReqFinalMessage,
    ReqInterimMessage,
    ReqTokenUsage,
    RespToolCall,
    RespMessage,
    ExecRunInput,
    ReqLogMessage,
)

from vocode.settings import ToolSpec


class LLMNode(Node):
    type: str = "llm"
    model: str
    system: Optional[str] = None
    system_append: Optional[str] = Field(
        default=None,
        description="Optional content appended to the main system prompt before preprocessors are applied.",
    )
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    outcome_strategy: OutcomeStrategy = Field(default=OutcomeStrategy.tag)
    # Structured tool specs with short-hand coercion from strings
    tools: List[ToolSpec] = Field(
        default_factory=list,
        description="Enabled tools (supports string or object spec)",
    )
    extra: Dict[str, Any] = Field(default_factory=dict)
    preprocessors: List[PreprocessorSpec] = Field(
        default_factory=list,
        description="Pre-execution preprocessors applied to the LLM system prompt",
    )
    function_tokens_pct: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description=(
            "Optional percent (1-100) of the model's total token limit to use as max_tokens "
            "when tools are enabled (function-answer mode). "
            "Requires extra['model_max_tokens'] to be set."
        ),
    )


# Internal LLM state
class LLMExpect(str, Enum):
    none = "none"
    tools = "tools"
    post_final_user = "post_final_user"


class LLMState(BaseModel):
    conv: List[Dict[str, Any]] = Field(default_factory=list)
    expect: LLMExpect = LLMExpect.none
    pending_outcome_name: Optional[str] = None
    tool_rounds: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_dollars: float = 0.0
