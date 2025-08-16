from __future__ import annotations

from typing import List, AsyncIterator, Dict, Any, Optional
import json

from ...state import Message, NodeExecution
from ...graph.models import Node, LLMNode
from ...runner.runner import Executor  # import directly from module to avoid circulars


def _map_role_to_llm(role: str) -> str:
    # Map internal "agent" to OpenAI-style "assistant"
    return "assistant" if role == "agent" else role


def _to_llm_messages(cfg: LLMNode, messages: List[Message]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if cfg.system:
        out.append({"role": "system", "content": cfg.system})
    for m in messages:
        out.append({"role": _map_role_to_llm(m.role), "content": m.raw})
    return out

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
        try:
            from litellm import acompletion
        except Exception as e:
            raise RuntimeError("litellm is required for LLMExecutor. Install with `pip install litellm`.") from e

        cfg: LLMNode = self.node
        model: str = cfg.model
        streaming: bool = cfg.streaming
        temperature = cfg.temperature
        max_tokens = cfg.max_tokens
        extra: Dict[str, Any] = dict(cfg.extra or {})

        params: Dict[str, Any] = {}
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        params.update(extra)

        llm_messages = _to_llm_messages(cfg, messages)

        if streaming:
            assembled = ""
            stream_gen = await acompletion(model=model, messages=llm_messages, stream=True, **params)
            async for chunk in stream_gen:
                text = _extract_chunk_text(chunk)
                if not text:
                    continue
                assembled += text
                # Stream partials; keep output_name None to indicate more to come
                yield NodeExecution(
                    messages=list(messages) + [Message(role="agent", raw=assembled)],
                    output_name=None,
                )
            # Final result to drive graph forward
            yield NodeExecution(
                messages=list(messages) + [Message(role="agent", raw=assembled)],
                output_name="done",
            )
            return

        # Non-streaming single completion
        resp = await acompletion(model=model, messages=llm_messages, **params)
        data = _normalize_completion(resp)
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""
        else:
            content = ""
        yield NodeExecution(
            messages=list(messages) + [Message(role="agent", raw=content)],
            output_name="done",
        )
