from __future__ import annotations

try:
    from litellm import acompletion
except Exception as e:
    raise RuntimeError("litellm is required for LLMExecutor. Install with `pip install litellm`.") from e

from typing import List, AsyncIterator, Dict, Any, Optional
import json

from ...state import Message, NodeExecution
from ...graph.models import Node, LLMNode
from ...runner.runner import Executor  # import directly from module to avoid circulars

def _map_role_to_llm(role: str) -> str:
    # Map internal "agent" to OpenAI-style "assistant"
    return "assistant" if role == "agent" else role


def _build_output_tool(allowed: List[str]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "select_outcome",
            "description": "Select exactly one outcome.",
            "parameters": {
                "type": "object",
                "properties": {
                    "outcome_name": {
                        "type": "string",
                        "enum": allowed,
                        "description": "Name of the chosen outcome",
                    }
                },
                "required": ["outcome_name"],
                "additionalProperties": False,
            },
        },
    }

def _compose_system_with_rules(cfg: LLMNode) -> Optional[str]:
    base = cfg.system or ""
    outputs = [o.name for o in cfg.outputs]
    if len(outputs) <= 1:
        return base if base else None
    lines = []
    if base:
        lines.append(base.strip())
    lines.append(
        "You must always choose the next step by calling the function select_outcome with a single parameter 'outcome_name'. "
        "Provide your assistant response content, and also make the function call in the same turn."
    )
    lines.append("Available outputs and when to use them:")
    for s in cfg.outputs:
        desc = s.description or ""
        lines.append(f"- {s.name}: {desc}")
    return "\n".join(lines)

def _extract_tool_selected_output(resp: Any) -> Optional[str]:
    data = _normalize_completion(resp)
    choices = data.get("choices") or []
    if not choices:
        return None
    msg = (choices[0] or {}).get("message") or {}
    # OpenAI tool_calls
    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        func = (tc or {}).get("function") or {}
        if func.get("name") == "select_outcome":
            args = func.get("arguments") or ""
            try:
                parsed = json.loads(args) if isinstance(args, str) else args
            except Exception:
                return None
            out = parsed.get("outcome_name")
            if isinstance(out, str):
                return out
    return None


def _to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj

    for attr in ("to_dict", "model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    mdj = getattr(obj, "model_dump_json", None)
    if callable(mdj):
        try:
            return json.loads(mdj())
        except Exception:
            pass
    try:
        return dict(obj)  # type: ignore[arg-type]
    except Exception:
        return {}

def _normalize_completion(resp: Any) -> Dict[str, Any]:
    return _to_dict(resp)

def _normalize_stream_chunk(chunk: Any) -> Dict[str, Any]:
    return _to_dict(chunk)


def _extract_chunk_text(chunk: Any) -> Optional[str]:
    data = _normalize_stream_chunk(chunk)
    choices = data.get("choices") or []
    if not choices:
        return None
    first = choices[0] or {}
    delta = first.get("delta") or first.get("message") or {}
    return delta.get("content")


class LLMExecutor(Executor):
    type = "llm"

    def __init__(self, config: Node):
        super().__init__(config)
        if not isinstance(config, LLMNode):
            raise TypeError(f"LLMExecutor requires LLMNode config, got {type(config).__name__}")
        self.node: LLMNode = config

    async def run(self, messages: List[Message]) -> AsyncIterator[NodeExecution]:
        cfg: LLMNode = self.node
        model: str = cfg.model
        temperature = cfg.temperature
        max_tokens = cfg.max_tokens
        extra: Dict[str, Any] = dict(cfg.extra or {})

        params: Dict[str, Any] = {}
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        params.update(extra)

        outputs = [s.name for s in (cfg.outputs or [])]
        default_output = outputs[0] if outputs else "done"
        multi_output = len(outputs) > 1
        tools = [_build_output_tool(outputs)] if multi_output else None

        # Build messages with enhanced system text
        sys_text = _compose_system_with_rules(cfg)
        llm_messages: List[Dict[str, str]] = []
        if sys_text:
            llm_messages.append({"role": "system", "content": sys_text})
        for m in messages:
            llm_messages.append({"role": _map_role_to_llm(m.role), "content": m.raw})

        assembled = ""
        stream_gen = await acompletion(
            model=model, messages=llm_messages, stream=True, **({**params, "tools": tools} if tools else params)
        )
        async for chunk in stream_gen:
            text = _extract_chunk_text(chunk)
            if not text:
                continue
            assembled += text
            msgs_out = [Message(role="agent", raw=assembled)]
            yield NodeExecution(
                messages=msgs_out,
                output_name=None,
            )

        # Decide output (may require an explicit forced tool call)
        chosen_output = default_output
        if multi_output:
            followup_messages = list(llm_messages)
            if assembled:
                followup_messages.append({"role": "assistant", "content": assembled})
            followup_messages.append({
                "role": "user",
                "content": "Now, select the appropriate output by calling the function select_outcome with the 'outcome_name' only.",
            })
            resp2 = await acompletion(
                model=model,
                messages=followup_messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "select_outcome"}},
                **params,
            )
            picked = _extract_tool_selected_output(resp2)
            if isinstance(picked, str) and picked in outputs:
                chosen_output = picked

        msgs_out = [Message(role="agent", raw=assembled)]
        yield NodeExecution(
            messages=msgs_out,
            output_name=chosen_output,
        )

        return
