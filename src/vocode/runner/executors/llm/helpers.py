from __future__ import annotations

from typing import Any, Dict, List, Optional, Final, Tuple
import re
import contextlib
import json

import litellm
from litellm import completion_cost, token_counter

from vocode.state import Message
from vocode.settings import ToolSpec  # type: ignore
from vocode.runner.executors.llm.preprocessors.base import apply_preprocessors

from .models import LLMNode

# Constants and regexes
CHOOSE_OUTCOME_TOOL_NAME: Final[str] = "__choose_outcome__"
OUTCOME_TAG_RE = re.compile(r"^\s*OUTCOME\s*:\s*([A-Za-z0-9_\-]+)\s*$")
OUTCOME_LINE_PREFIX_RE = re.compile(r"^\s*OUTCOME\s*:\s*")
MAX_ROUNDS: Final[int] = 32


# Message mapping and prompt building
def map_message_to_llm_dict(m: Message, cfg: LLMNode) -> Dict[str, Any]:
    role = m.role or "user"
    is_external = m.node is None or m.node != cfg.name
    if is_external:
        role = "user"
    else:
        if role == "agent":
            role = "assistant"
        elif role not in ("user", "system", "tool", "assistant"):
            role = "user"
    return {"role": role, "content": m.text}


def build_base_messages(
    cfg: LLMNode, history: List[Message], project: Any
) -> List[Dict[str, Any]]:
    system_prompt_parts: List[str] = []
    if cfg.system:
        system_prompt_parts.append(cfg.system)
    if cfg.system_append:
        system_prompt_parts.append(cfg.system_append)
    system_prompt = "".join(system_prompt_parts)

    base_messages = list(history)
    if system_prompt:
        base_messages.insert(
            0, Message(role="system", text=system_prompt, node=cfg.name)
        )

    if cfg.preprocessors:
        base_messages = apply_preprocessors(cfg.preprocessors, project, base_messages)

    return [map_message_to_llm_dict(m, cfg) for m in base_messages]


# Outcome helpers
def parse_outcome_from_text(text: str, valid_outcomes: List[str]) -> Optional[str]:
    for line in text.splitlines()[::-1]:
        m = OUTCOME_TAG_RE.match(line.strip())
        if m:
            cand = m.group(1)
            if cand in valid_outcomes:
                return cand
    return None


def strip_outcome_line(text: str) -> str:
    return "\n".join(
        [ln for ln in text.splitlines() if not OUTCOME_LINE_PREFIX_RE.match(ln.strip())]
    ).rstrip()


def build_choose_outcome_tool(
    outcomes: List[str],
    outcome_desc_bullets: str,
    outcome_choice_desc: str,
) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": CHOOSE_OUTCOME_TOOL_NAME,
            "description": "Selects the conversation outcome to take next. Available outcomes:\n"
            + outcome_desc_bullets,
            "parameters": {
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "enum": outcomes,
                        "description": outcome_choice_desc,
                    }
                },
                "required": ["outcome"],
            },
        },
    }


def build_tag_system_instruction(
    outcomes: List[str],
    outcome_desc_bullets: str,
) -> str:
    return (
        "Consider the available outcomes and pick the best fit based on the conversation:\n"
        f"{outcome_desc_bullets}\n\n"
        "After producing your final answer, append a last line exactly as:\n"
        f"OUTCOME: <one of {outcomes}>\n"
        "Only output the outcome name on that line and nothing else."
    )


def get_outcome_names(cfg: LLMNode) -> List[str]:
    return [s.name for s in (cfg.outcomes or [])]


def get_outcome_desc_bullets(cfg: LLMNode) -> str:
    lines: List[str] = []
    for s in cfg.outcomes or []:
        desc = s.description or ""
        lines.append(f"- {s.name}: {desc}".rstrip())
    return "\n".join(lines)


def get_outcome_choice_desc(cfg: LLMNode, outcome_desc_bullets: str) -> str:
    if outcome_desc_bullets.strip():
        return "Choose exactly one of the following outcomes:\n" + outcome_desc_bullets
    return "Choose the appropriate outcome."


# Pricing/model helpers
def get_input_cost_per_1k(cfg: LLMNode) -> float:
    extra = cfg.extra or {}
    return float(
        extra.get("input_cost_per_1k") or extra.get("prompt_cost_per_1k") or 0.0
    )


def get_output_cost_per_1k(cfg: LLMNode) -> float:
    extra = cfg.extra or {}
    return float(
        extra.get("output_cost_per_1k") or extra.get("completion_cost_per_1k") or 0.0
    )


def _safe_get(obj: Any, key: str) -> Any:
    try:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)
    except Exception:
        return None


def _get_model_info(model: str) -> Optional[Any]:
    with contextlib.suppress(Exception):
        return litellm.get_model_info(model)
    return None


def _get_model_info_value(model: str, key: str) -> Optional[Any]:
    mi = _get_model_info(model)
    if mi is None:
        return None
    return _safe_get(mi, key)


def calc_cost_from_model_info(
    model: str, prompt_tokens: int, completion_tokens: int
) -> Optional[float]:
    try:
        mi = _get_model_info(model)
        if mi is None:
            return None
        in_per_tok = float(_safe_get(mi, "input_cost_per_token") or 0.0)
        out_per_tok = float(_safe_get(mi, "output_cost_per_token") or 0.0)
        if (in_per_tok or out_per_tok) and (prompt_tokens or completion_tokens):
            return (prompt_tokens * in_per_tok) + (completion_tokens * out_per_tok)
    except Exception:
        return None
    return None


def get_round_cost(
    stream: Any, model: str, cfg: LLMNode, prompt_tokens: int, completion_tokens: int
) -> float:
    round_cost = 0.0
    try:
        cost = completion_cost(completion_response=stream, model=model)
        if cost is not None:
            round_cost = float(cost)
    except Exception:
        pass

    if round_cost == 0.0:
        model_info_cost = calc_cost_from_model_info(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        if model_info_cost is not None:
            round_cost = float(model_info_cost)

    if round_cost == 0.0:
        in_per_1k = get_input_cost_per_1k(cfg)
        out_per_1k = get_output_cost_per_1k(cfg)
        if in_per_1k > 0.0 or out_per_1k > 0.0:
            round_cost = (prompt_tokens / 1000.0) * in_per_1k + (
                completion_tokens / 1000.0
            ) * out_per_1k
    return round_cost


def build_effective_tool_specs(project: Any, cfg: LLMNode) -> Dict[str, ToolSpec]:
    """Merge node-level ToolSpec with project-level (global) ToolSpec by name.

    Precedence rules:
    - Global .enabled overrides node .enabled when provided.
    - Global .auto_approve overrides node .auto_approve when non-None.
    - Global .auto_approve_rules extend node .auto_approve_rules.
    - .config is merged shallowly: node.config first, then global.config.

    This helper constructs the effective ToolSpec via model_copy/update so new
    fields added to ToolSpec are preserved by default and do not need special
    handling here.
    Only returns specs for tools listed on this node.
    """
    global_specs: Dict[str, ToolSpec] = {}
    try:
        settings_tools = (
            (project.settings.tools or []) if project and project.settings else []
        )
        for ts in settings_tools:
            global_specs[ts.name] = ts
    except Exception:
        global_specs = {}

    effective: Dict[str, ToolSpec] = {}
    for node_spec in cfg.tools or []:
        gspec = global_specs.get(node_spec.name)

        # Start from a shallow copy of the node spec so any future fields on
        # ToolSpec are preserved automatically.
        base = node_spec.model_copy(deep=True)

        if gspec is not None:
            # enabled: global overrides node when provided
            if isinstance(gspec.enabled, bool):
                base.enabled = gspec.enabled

            # auto_approve: global wins when explicitly set
            if gspec.auto_approve is not None:
                base.auto_approve = gspec.auto_approve

            # auto_approve_rules: concatenate node + global so that both
            # scopes can contribute matchers.
            if getattr(gspec, "auto_approve_rules", None):
                # Ensure list exists on base
                base.auto_approve_rules = list(base.auto_approve_rules or [])
                base.auto_approve_rules.extend(gspec.auto_approve_rules)

            # config: node first, then global (global overrides on conflicts)
            merged_cfg: Dict[str, Any] = {}
            merged_cfg.update(node_spec.config or {})
            merged_cfg.update(gspec.config or {})
            base.config = merged_cfg

        effective[node_spec.name] = base
    return effective


def resolve_model_token_limit(cfg: LLMNode) -> Optional[int]:
    """
    Resolve the model input context window (prompt token limit) with fallbacks:
    1) litellm.get_model_info(cfg.model).max_input_tokens
    2) cfg.extra['model_max_tokens']
    3) litellm.get_model_info(cfg.model).max_tokens
    """
    with contextlib.suppress(Exception):
        v = _get_model_info_value(cfg.model, "max_input_tokens")
        if v:
            v = int(v or 0)
            if v > 0:
                return v
    with contextlib.suppress(Exception):
        v = int((cfg.extra or {}).get("model_max_tokens") or 0)
        if v > 0:
            return v
    with contextlib.suppress(Exception):
        v = _get_model_info_value(cfg.model, "max_tokens")
        if v:
            v = int(v or 0)
            if v > 0:
                return v
    return None


# Usage/token helpers
def extract_usage_tokens(
    stream: Any, last_chunk_usage: Optional[Any]
) -> Tuple[int, int]:
    prompt_tokens = 0
    completion_tokens = 0
    try:
        if last_chunk_usage is not None:
            if isinstance(last_chunk_usage, dict):
                pt = last_chunk_usage.get("prompt_tokens")
                ct = last_chunk_usage.get("completion_tokens")
            else:
                pt = getattr(last_chunk_usage, "prompt_tokens", None)
                ct = getattr(last_chunk_usage, "completion_tokens", None)
            if isinstance(pt, int):
                prompt_tokens = pt
            if isinstance(ct, int):
                completion_tokens = ct
    except Exception:
        pass

    if prompt_tokens == 0 and completion_tokens == 0:
        try:
            usage_obj = getattr(stream, "usage", None)
            if usage_obj:
                if isinstance(usage_obj, dict):
                    prompt_tokens = int(usage_obj.get("prompt_tokens") or 0)
                    completion_tokens = int(usage_obj.get("completion_tokens") or 0)
                else:
                    prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
                    completion_tokens = int(
                        getattr(usage_obj, "completion_tokens", 0) or 0
                    )
        except Exception:
            pass

    if prompt_tokens == 0 and completion_tokens == 0:
        try:
            resp = getattr(stream, "response", None)
            if resp is not None:
                usage_obj = (
                    resp.get("usage")
                    if isinstance(resp, dict)
                    else getattr(resp, "usage", None)
                )
                if usage_obj:
                    if isinstance(usage_obj, dict):
                        prompt_tokens = int(usage_obj.get("prompt_tokens") or 0)
                        completion_tokens = int(usage_obj.get("completion_tokens") or 0)
                    else:
                        prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
                        completion_tokens = int(
                            getattr(usage_obj, "completion_tokens", 0) or 0
                        )
        except Exception:
            pass
    return prompt_tokens, completion_tokens


def estimate_usage_tokens(
    model: str,
    prompt_messages: List[Dict[str, Any]],
    assistant_text: Optional[str],
) -> Tuple[int, int]:
    est_prompt = 0
    est_completion = 0
    try:
        est_prompt = int(token_counter(model=model, messages=prompt_messages) or 0)
    except Exception:
        pass
    if assistant_text:
        try:
            est_completion = int(
                token_counter(
                    model=model,
                    messages=[{"role": "assistant", "content": assistant_text}],
                )
                or 0
            )
        except Exception:
            pass
    return est_prompt, est_completion
