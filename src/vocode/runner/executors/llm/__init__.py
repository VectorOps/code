from __future__ import annotations

from typing import AsyncIterator, List, Optional, Dict, Any, Final
import json
import re
import asyncio
from vocode.runner.executors.llm.preprocessors.base import apply_preprocessors
from litellm import acompletion, completion_cost
import litellm
from enum import Enum
from pydantic import BaseModel, Field, model_validator
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

from vocode.settings import ToolSettings  # type: ignore
from .models import LLMToolSpec, LLMNode, LLMExpect, LLMState


CHOOSE_OUTCOME_TOOL_NAME: Final[str] = "__choose_outcome__"
OUTCOME_TAG_RE = re.compile(r"^\s*OUTCOME\s*:\s*([A-Za-z0-9_\-]+)\s*$")
OUTCOME_LINE_PREFIX_RE = re.compile(r"^\s*OUTCOME\s*:\s*")
MAX_ROUNDS = 32


# Executor
class LLMExecutor(Executor):
    # Must match LLMNode.type
    type = "llm"

    def __init__(self, config, project):
        super().__init__(config=config, project=project)
        if not isinstance(config, LLMNode):
            # Allow base Node but prefer LLMNode
            raise TypeError("LLMExecutor requires config to be an LLMNode")

    def _map_message_to_llm_dict(self, m: Message, cfg: LLMNode) -> Dict[str, Any]:
        """Maps a vocode.state.Message to an OpenAI-style dictionary for the LLM."""
        role = m.role or "user"
        is_external = m.node is None or m.node != cfg.name

        if is_external:
            # Any message from another node is treated as user input to this node.
            role = "user"
        else:
            # For messages from this node or from the user, map 'agent' to 'assistant'.
            if role == "agent":
                role = "assistant"
            elif role not in ("user", "system", "tool", "assistant"):
                role = "user"

        return {"role": role, "content": m.text}

    def _build_base_messages(
        self, cfg: LLMNode, history: List[Message]
    ) -> List[Dict[str, Any]]:
        system_prompt_parts = []
        if cfg.system:
            system_prompt_parts.append(cfg.system)
        if cfg.system_append:
            system_prompt_parts.append(cfg.system_append)
        system_prompt = "".join(system_prompt_parts)

        base_messages = list(history)
        if system_prompt:
            base_messages.insert(0, Message(role="system", text=system_prompt, node=cfg.name))

        if cfg.preprocessors:
            base_messages = apply_preprocessors(
                cfg.preprocessors, self.project, base_messages
            )

        return [self._map_message_to_llm_dict(m, cfg) for m in base_messages]

    def _parse_outcome_from_text(
        self, text: str, valid_outcomes: List[str]
    ) -> Optional[str]:
        # Look for a line like "OUTCOME: name"
        for line in text.splitlines()[::-1]:
            m = OUTCOME_TAG_RE.match(line.strip())
            if m:
                cand = m.group(1)
                if cand in valid_outcomes:
                    return cand
        return None

    def _strip_outcome_line(self, text: str) -> str:
        return "\n".join(
            [
                ln
                for ln in text.splitlines()
                if not OUTCOME_LINE_PREFIX_RE.match(ln.strip())
            ]
        ).rstrip()

    def _build_choose_outcome_tool(
        self,
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

    def _build_tag_system_instruction(
        self,
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

    def _get_outcome_names(self, cfg: LLMNode) -> List[str]:
        return [s.name for s in (cfg.outcomes or [])]

    def _get_outcome_desc_bullets(self, cfg: LLMNode) -> str:
        lines: List[str] = []
        for s in cfg.outcomes or []:
            desc = s.description or ""
            lines.append(f"- {s.name}: {desc}".rstrip())
        return "\n".join(lines)

    def _get_outcome_choice_desc(self, cfg: LLMNode, outcome_desc_bullets: str) -> str:
        if outcome_desc_bullets.strip():
            return (
                "Choose exactly one of the following outcomes:\n" + outcome_desc_bullets
            )
        return "Choose the appropriate outcome."

    def _estimate_text_tokens(self, text: str, model: str) -> int:
        try:
            import tiktoken  # type: ignore

            try:
                enc = tiktoken.encoding_for_model(model)
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text or ""))
        except Exception:
            # Fallback heuristic: ~4 chars per token
            return max(1, int(len(text or "") / 4))

    def _estimate_messages_tokens(self, msgs: List[Dict[str, Any]], model: str) -> int:
        total = 0
        for m in msgs:
            content = m.get("content")
            if isinstance(content, str):
                total += self._estimate_text_tokens(content, model)
        return total

    def _get_input_cost_per_1k(self, cfg: LLMNode) -> float:
        extra = cfg.extra or {}
        return float(
            extra.get("input_cost_per_1k") or extra.get("prompt_cost_per_1k") or 0.0
        )

    def _get_output_cost_per_1k(self, cfg: LLMNode) -> float:
        extra = cfg.extra or {}
        return float(
            extra.get("output_cost_per_1k")
            or extra.get("completion_cost_per_1k")
            or 0.0
        )

    def _get_round_cost(
        self,
        stream: Any,
        model: str,
        cfg: "LLMNode",
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Get cost from litellm completion response, with fallback to node.extra per-1k pricing."""
        round_cost = 0.0
        try:
            # Primary method: use litellm.completion_cost()
            cost = completion_cost(completion_response=stream, model=model)
            if cost is not None:
                round_cost = float(cost)
        except Exception:
            pass  # Continue to fallbacks

        if round_cost == 0.0:
            # Fallback to hidden params for some models/older litellm versions
            try:
                if (
                    hasattr(stream, "_hidden_params")
                    and stream._hidden_params
                    and "response_cost" in stream._hidden_params
                ):
                    cost = stream._hidden_params["response_cost"]
                    if cost is not None:
                        round_cost = float(cost)
            except Exception:
                pass

        if round_cost == 0.0:
            # Final fallback: compute using per-1k pricing configured on the node.
            in_per_1k = self._get_input_cost_per_1k(cfg)
            out_per_1k = self._get_output_cost_per_1k(cfg)
            if in_per_1k > 0.0 or out_per_1k > 0.0:
                round_cost = (prompt_tokens / 1000.0) * in_per_1k + (
                    completion_tokens / 1000.0
                ) * out_per_1k
        return round_cost

    async def run(
        self, inp: ExecRunInput
    ) -> AsyncIterator[tuple[ReqPacket, Optional[Any]]]:
        cfg: LLMNode = self.config  # type: ignore[assignment]

        # Restore or initialize state
        if inp.state is not None:
            state: LLMState = inp.state
        else:
            # Build conversation from node system + provided history messages
            base_conv = self._build_base_messages(cfg, inp.messages or [])
            state = LLMState(conv=base_conv)

        # If messages are provided together with existing state, treat them as additional inputs
        # (typical when re-entering node with keep_state).
        if inp.state is not None and inp.messages:
            for m in inp.messages:
                state.conv.append(self._map_message_to_llm_dict(m, cfg))
            state.expect = LLMExpect.none

        # Integrate incoming response into conversation
        if inp.response is not None and inp.response.kind == "tool_call":
            # Append tool results into the conversation for each completed tool call
            for rcall in inp.response.tool_calls:
                state.conv.append(
                    {
                        "role": "tool",
                        "tool_call_id": rcall.id or "",
                        "name": rcall.name,
                        "content": (rcall.result if rcall.result is not None else ""),
                    }
                )
            state.expect = LLMExpect.none
            # Count only rounds that include external tool calls
            state.tool_rounds = max(0, state.tool_rounds)
        elif inp.response is not None and inp.response.kind == "message":
            # Append user reply to conversation (post-final continuation).
            # Always treat a follow-up message as a user message.
            state.conv.append({"role": "user", "content": inp.response.message.text})
            state.expect = LLMExpect.none

        conv: List[Dict[str, Any]] = state.conv

        # Outcome handling strategy
        outcomes: List[str] = self._get_outcome_names(cfg)
        outcome_desc_bullets: str = self._get_outcome_desc_bullets(cfg)
        outcome_choice_desc: str = self._get_outcome_choice_desc(
            cfg, outcome_desc_bullets
        )
        outcome_strategy: OutcomeStrategy = cfg.outcome_strategy
        outcome_name: Optional[str] = state.pending_outcome_name or None
        # Build global tool auto-approve map (name -> Optional[bool])
        global_auto: Dict[str, Optional[bool]] = {}
        try:
            settings_tools = (
                (self.project.settings.tools or [])
                if self.project and self.project.settings
                else []
            )
            for ts in settings_tools:
                # ts is ToolSettings; may not include auto_approve when omitted
                global_auto[ts.name] = getattr(ts, "auto_approve", None)
        except Exception:
            global_auto = {}

        # Get tool specs from node config and compute effective auto-approve per tool:
        # precedence: global setting (if provided) -> node-level -> None
        external_tools: List[Dict[str, Any]] = []
        # Map tool name -> Optional[bool]
        tool_auto_approve: Dict[str, Optional[bool]] = {}
        for spec in cfg.tools or []:
            tool_name = spec.name
            node_auto = getattr(spec, "auto_approve", None)
            eff_auto: Optional[bool] = global_auto.get(tool_name, None)
            if eff_auto is None:
                eff_auto = node_auto if isinstance(node_auto, bool) else None
            tool_auto_approve[tool_name] = eff_auto
            tool = self.project.tools.get(tool_name)
            if tool:
                external_tools.append(
                    {"type": "function", "function": tool.openapi_spec()}
                )

        tools: Optional[List[Dict[str, Any]]] = None
        if len(outcomes) > 1 and outcome_strategy == OutcomeStrategy.function_call:
            tools = external_tools + [
                self._build_choose_outcome_tool(
                    outcomes=outcomes,
                    outcome_desc_bullets=outcome_desc_bullets,
                    outcome_choice_desc=outcome_choice_desc,
                )
            ]
        else:
            tools = external_tools or None
            if len(outcomes) > 1 and outcome_strategy == OutcomeStrategy.tag:
                # Add clear instruction to output the tag line
                tag_instr = self._build_tag_system_instruction(
                    outcomes=outcomes,
                    outcome_desc_bullets=outcome_desc_bullets,
                )
                conv.insert(0, {"role": "system", "content": tag_instr})

        # Drive the LLM loop with tool-calls as needed
        assistant_text: str = ""

        while True:
            assistant_text_parts: List[str] = []
            tool_calls_by_idx: Dict[int, Dict[str, Any]] = {}
            # Filter extra to avoid overriding explicit kwargs (e.g., 'tools')
            extra_args = dict(cfg.extra or {})
            for k in (
                "tools",
                "tool_choice",
                "messages",
                "model",
                "stream",
                "temperature",
                "max_tokens",
            ):
                extra_args.pop(k, None)

            # Compute effective max_tokens
            effective_max_tokens = cfg.max_tokens
            if tools and cfg.function_tokens_pct is not None:
                try:
                    model_limit = int((cfg.extra or {}).get("model_max_tokens") or 0)
                except Exception:
                    model_limit = 0
                if model_limit > 0:
                    pct = cfg.function_tokens_pct
                    effective_max_tokens = max(1, int(model_limit * pct / 100))

            # Log the full prompt at debug level before sending
            _ = yield (
                ReqLogMessage(
                    level=LogLevel.debug,
                    text=f"LLM request:\n{json.dumps(conv, indent=2)}",
                ),
                None,
            )

            # Start streaming with retries and error handling
            max_retries = 3
            attempt = 0
            while True:
                try:
                    completion_coro = acompletion(
                        model=cfg.model,
                        messages=conv,
                        temperature=cfg.temperature,
                        max_tokens=effective_max_tokens,
                        tools=tools,
                        tool_choice="auto" if tools else None,
                        stream=True,
                        stream_options={"include_usage": True},
                        **extra_args,
                    )
                    # Wrap in a named task for better diagnostics
                    task_name = f"llm.acompletion:{cfg.name}"
                    stream_task = asyncio.create_task(completion_coro, name=task_name)
                    stream = await stream_task

                    # Collect streamed content and tool_calls; emit interim messages for content chunks
                    async for chunk in stream:
                        try:
                            ch = (chunk.get("choices") or [])[0]
                            delta = ch.get("delta") or {}
                        except Exception:
                            delta = {}

                        # Stream content
                        content_piece = delta.get("content")
                        if content_piece:
                            assistant_text_parts.append(content_piece)
                            _ = yield (
                                ReqInterimMessage(
                                    message=Message(
                                        role="agent", text=content_piece, node=cfg.name
                                    )
                                ),
                                None,
                            )

                        # Accumulate streamed tool_calls (OpenAI-style deltas)
                        tc_deltas = delta.get("tool_calls") or []
                        for dtc in tc_deltas:
                            idx = dtc.get("index", 0)
                            _id = dtc.get("id")
                            _type = dtc.get("type")
                            fn = dtc.get("function") or {}
                            name = fn.get("name")
                            args_part = fn.get("arguments") or ""

                            entry = tool_calls_by_idx.get(idx)
                            if entry is None:
                                entry = {
                                    "id": None,
                                    "type": "function",
                                    "function": {"name": None, "arguments": ""},
                                }
                                tool_calls_by_idx[idx] = entry

                            if _id:
                                entry["id"] = _id
                            if _type:
                                entry["type"] = _type
                            if name:
                                entry["function"]["name"] = name
                            if args_part:
                                entry["function"]["arguments"] += args_part
                    break  # success
                except Exception as e:
                    status_code = getattr(e, "status_code", None)
                    try:
                        should_retry = bool(litellm._should_retry(status_code))
                    except Exception:
                        should_retry = False

                    if should_retry and attempt < max_retries:
                        attempt += 1
                        if isinstance(e, litellm.RateLimitError):
                            await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
                        else:
                            await asyncio.sleep(0)
                        _ = yield (
                            ReqLogMessage(
                                level=LogLevel.warning,
                                text=f"LLM retry {attempt}/{max_retries} after error (status={status_code}): {str(e)}",
                            ),
                            None,
                        )
                        continue

                    _ = yield (
                        ReqLogMessage(
                            level=LogLevel.error,
                            text=f"LLM error (status={status_code}): {str(e)}",
                        ),
                        None,
                    )
                    selected_outcome = None
                    if len(outcomes) > 1:
                        selected_outcome = outcomes[0]
                    error_msg = Message(
                        role="agent", text=f"Error: {str(e)}", node=cfg.name
                    )
                    yield (
                        ReqFinalMessage(
                            message=error_msg, outcome_name=selected_outcome
                        ),
                        state,
                    )
                    return

            assistant_text = "".join(assistant_text_parts)

            # Get token usage from response, with fallback to estimation
            prompt_tokens = 0
            completion_tokens = 0
            try:
                # After stream is exhausted, litellm provides usage.
                if hasattr(stream, "usage") and stream.usage:
                    prompt_tokens = stream.usage.prompt_tokens or 0
                    completion_tokens = stream.usage.completion_tokens or 0
            except Exception:
                pass  # Fallback logic below will handle it

            # Fallback to estimation if litellm data is missing
            if prompt_tokens == 0 and completion_tokens == 0:
                prompt_tokens = self._estimate_messages_tokens(conv, cfg.model)
                completion_tokens = self._estimate_text_tokens(
                    assistant_text, cfg.model
                )

            round_cost = self._get_round_cost(
                stream, cfg.model, cfg, prompt_tokens, completion_tokens
            )

            self.project.add_llm_usage(
                prompt_delta=prompt_tokens,
                completion_delta=completion_tokens,
                cost_delta=round_cost,
            )

            state.total_prompt_tokens += prompt_tokens
            state.total_completion_tokens += completion_tokens
            state.total_cost_dollars += round_cost

            # Build final assistant message dict and append to conversation
            streamed_tool_calls = []
            # Deterministic order by index to preserve original sequence
            for idx in sorted(tool_calls_by_idx.keys()):
                tc_obj = tool_calls_by_idx[idx] or {}
                fn = tc_obj.get("function") or {}
                streamed_tool_calls.append(
                    {
                        "id": (tc_obj.get("id") or ""),
                        "type": tc_obj.get("type", "function"),
                        "function": {
                            "name": (fn.get("name") or ""),
                            "arguments": (fn.get("arguments") or ""),
                        },
                    }
                )

            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": assistant_text or None,
            }
            if streamed_tool_calls:
                assistant_msg["tool_calls"] = streamed_tool_calls
            conv.append(assistant_msg)

            # Separate outcome selection vs. external tool calls
            external_calls: List[ToolCall] = []
            for tc in streamed_tool_calls:
                name = (tc.get("function") or {}).get("name") or ""
                arguments = (tc.get("function") or {}).get("arguments") or ""
                call_id = tc.get("id")
                if name == CHOOSE_OUTCOME_TOOL_NAME:
                    try:
                        parsed = json.loads(arguments) if arguments else {}
                    except Exception:
                        parsed = {}
                    sel = parsed.get("outcome")
                    if isinstance(sel, str) and sel in outcomes:
                        # record selected outcome for next cycle
                        state.pending_outcome_name = sel
                    # Do not forward this special call to the runner
                    continue
                # Forward all other calls to the runner using our protocol
                # Parse JSON arguments to a dict as ToolCall.arguments expects a mapping
                try:
                    args_obj = (
                        json.loads(arguments)
                        if isinstance(arguments, str) and arguments
                        else {}
                    )
                except Exception:
                    args_obj = {}
                external_calls.append(
                    ToolCall(
                        id=call_id,
                        name=name,
                        arguments=args_obj,
                        type="function",
                        auto_approve=tool_auto_approve.get(name, None),
                    )
                )

            if external_calls:
                # Expect tool results on the next cycle
                state.expect = LLMExpect.tools
                # Limit tool call loops
                state.tool_rounds += 1
                if state.tool_rounds > MAX_ROUNDS:
                    raise RuntimeError(
                        f"LLMExecutor exceeded maximum function-call rounds ({MAX_ROUNDS}); possible tool loop"
                    )
                # Report token usage for this tool-using round before requesting tool execution
                _ = yield (
                    ReqTokenUsage(
                        acc_prompt_tokens=self.project.llm_usage.prompt_tokens,
                        acc_completion_tokens=self.project.llm_usage.completion_tokens,
                        acc_cost_dollars=self.project.llm_usage.cost_dollars,
                        current_prompt_tokens=state.total_prompt_tokens,
                        current_completion_tokens=state.total_completion_tokens,
                        token_limit=effective_max_tokens,
                        local=True,
                    ),
                    state,
                )
                yield (ReqToolCall(tool_calls=external_calls), state)
                return

            # No tool calls -> proceed to finalize content and outcome
            # assistant_text already set above

            outcome_name: Optional[str] = state.pending_outcome_name or None

            if len(outcomes) > 1:
                if outcome_strategy == OutcomeStrategy.tag:
                    selected = self._parse_outcome_from_text(assistant_text, outcomes)
                    if selected:
                        outcome_name = selected
                        assistant_text = self._strip_outcome_line(assistant_text)
                        state.pending_outcome_name = selected
                    else:
                        # Fallback if tag not found
                        outcome_name = outcome_name or outcomes[0]
                elif outcome_strategy == OutcomeStrategy.function_call:
                    # If model never called choose_outcome, try to infer from tag, else fallback
                    if outcome_name is None:
                        selected = self._parse_outcome_from_text(
                            assistant_text, outcomes
                        )
                        if selected:
                            outcome_name = selected
                            assistant_text = self._strip_outcome_line(assistant_text)
                            state.pending_outcome_name = selected
                        else:
                            outcome_name = outcomes[0]

            # Report token usage/cost before finalizing
            _ = yield (
                ReqTokenUsage(
                    acc_prompt_tokens=self.project.llm_usage.prompt_tokens,
                    acc_completion_tokens=self.project.llm_usage.completion_tokens,
                    acc_cost_dollars=self.project.llm_usage.cost_dollars,
                    current_prompt_tokens=state.total_prompt_tokens,
                    current_completion_tokens=state.total_completion_tokens,
                    token_limit=effective_max_tokens,
                    local=True,
                ),
                state,
            )
            # Finalize: prepare final message and persist state. Allow post-final user reply on next cycle.
            final_msg = Message(role="agent", text=assistant_text, node=cfg.name)
            selected_outcome = outcome_name
            state.pending_outcome_name = None
            state.expect = LLMExpect.post_final_user
            yield (
                ReqFinalMessage(message=final_msg, outcome_name=selected_outcome),
                state,
            )
            return
