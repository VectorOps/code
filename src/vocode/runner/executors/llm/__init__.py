from __future__ import annotations

from typing import AsyncIterator, List, Optional, Dict, Any
import json
import asyncio
from litellm import acompletion
import litellm
from vocode.runner.runner import Executor
from vocode.models import OutcomeStrategy
from vocode.state import Message, ToolCall
from vocode.runner.models import (
    ReqPacket,
    ReqToolCall,
    ReqFinalMessage,
    ReqInterimMessage,
    ReqTokenUsage,
    RespToolCall,
    RespMessage,
    ExecRunInput,
)

from vocode.settings import ToolSpec  # type: ignore
from vocode.logger import logger
from .models import LLMNode, LLMExpect, LLMState
from .helpers import (
    CHOOSE_OUTCOME_TOOL_NAME,
    map_message_to_llm_dict as h_map_message_to_llm_dict,
    build_base_messages as h_build_base_messages,
    parse_outcome_from_text as h_parse_outcome_from_text,
    strip_outcome_line as h_strip_outcome_line,
    build_choose_outcome_tool as h_build_choose_outcome_tool,
    build_tag_system_instruction as h_build_tag_system_instruction,
    get_outcome_names as h_get_outcome_names,
    get_outcome_desc_bullets as h_get_outcome_desc_bullets,
    get_outcome_choice_desc as h_get_outcome_choice_desc,
    get_round_cost as h_get_round_cost,
    build_effective_tool_specs as h_build_effective_tool_specs,
    resolve_model_token_limit as h_resolve_model_token_limit,
    extract_usage_tokens as h_extract_usage_tokens,
    estimate_usage_tokens as h_estimate_usage_tokens,
    MAX_ROUNDS,
)


# Executor
class LLMExecutor(Executor):
    # Must match LLMNode.type
    type = "llm"

    def __init__(self, config, project):
        super().__init__(config=config, project=project)
        if not isinstance(config, LLMNode):
            # Allow base Node but prefer LLMNode
            raise TypeError("LLMExecutor requires config to be an LLMNode")
    async def run(
        self, inp: ExecRunInput
    ) -> AsyncIterator[tuple[ReqPacket, Optional[Any]]]:
        cfg: LLMNode = self.config  # type: ignore[assignment]

        # Restore or initialize state
        if inp.state is not None:
            state: LLMState = inp.state
        else:
            # Build conversation from node system + provided history messages
            base_conv = h_build_base_messages(cfg, inp.messages or [], self.project)
            state = LLMState(conv=base_conv)

        # If messages are provided together with existing state, treat them as additional inputs
        # (typical when re-entering node with keep_state).
        if inp.state is not None and inp.messages:
            for m in inp.messages:
                state.conv.append(h_map_message_to_llm_dict(m, cfg))
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
        outcomes: List[str] = h_get_outcome_names(cfg)
        outcome_desc_bullets: str = h_get_outcome_desc_bullets(cfg)
        outcome_choice_desc: str = h_get_outcome_choice_desc(
            cfg, outcome_desc_bullets
        )
        outcome_strategy: OutcomeStrategy = cfg.outcome_strategy
        outcome_name: Optional[str] = state.pending_outcome_name or None
        # Compute effective tool specs (node merged with global settings)
        eff_specs: Dict[str, ToolSpec] = h_build_effective_tool_specs(self.project, cfg)
        external_tools: List[Dict[str, Any]] = []

        for spec in cfg.tools or []:
            tool_name = spec.name
            tool = self.project.tools.get(tool_name)
            if tool:
                external_tools.append(
                    {
                        "type": "function",
                        "function": await tool.openapi_spec(
                            self.project, eff_specs[tool_name]
                        ),
                    }
                )

        tools: Optional[List[Dict[str, Any]]] = None
        if len(outcomes) > 1 and outcome_strategy == OutcomeStrategy.function_call:
            tools = external_tools + [
                h_build_choose_outcome_tool(
                    outcomes=outcomes,
                    outcome_desc_bullets=outcome_desc_bullets,
                    outcome_choice_desc=outcome_choice_desc,
                )
            ]
        else:
            tools = external_tools or None
            if len(outcomes) > 1 and outcome_strategy == OutcomeStrategy.tag:
                # Add clear instruction to output the tag line
                tag_instr = h_build_tag_system_instruction(
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
                # Use resolved model context window (get_max_tokens or node extra)
                model_limit = int(h_resolve_model_token_limit(cfg) or 0)
                if model_limit > 0:
                    pct = cfg.function_tokens_pct
                    effective_max_tokens = max(1, int(model_limit * pct / 100))

            # Log the full prompt at debug level before sending
            logger.debug("LLM request:", req=conv)

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
                    last_usage: Optional[Dict[str, Any]] = None
                    async for chunk in stream:
                        try:
                            ch = (chunk.get("choices") or [])[0]
                            delta = ch.get("delta") or {}
                        except Exception:
                            delta = {}

                        # Capture usage if provider includes it per chunk (include_usage=True)
                        try:
                            usage_obj = chunk.get(
                                "usage"
                            )  # may be litellm Usage object
                            if usage_obj is not None:
                                # Accept both dict and object; _extract_usage_tokens handles details
                                last_usage = usage_obj
                        except Exception:
                            pass

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
                        logger.warning(
                            "LLM retry",
                            attempt=attempt,
                            max_retries=max_retries,
                            status_code=status_code,
                            err=str(e),
                        )
                        continue

                    logger.error("LLM error", status_code=status_code, err=e)
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
            # Get token usage from stream and last chunk (robust across litellm/provider changes)
            prompt_tokens, completion_tokens = h_extract_usage_tokens(
                stream, last_usage
            )
            if prompt_tokens == 0 and completion_tokens == 0:
                est_prompt, est_completion = h_estimate_usage_tokens(
                    cfg.model, conv, assistant_text
                )
                prompt_tokens = est_prompt
                completion_tokens = est_completion
            round_cost = h_get_round_cost(
                stream, cfg.model, cfg, prompt_tokens, completion_tokens
            )
            # Per-cycle usage and model context window for reporting
            current_prompt_tokens = prompt_tokens
            current_completion_tokens = completion_tokens
            model_input_token_limit = h_resolve_model_token_limit(cfg)

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

                eff = eff_specs.get(name)
                external_calls.append(
                    ToolCall(
                        id=call_id,
                        name=name,
                        arguments=args_obj,
                        type="function",
                        tool_spec=eff,
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
                        current_prompt_tokens=current_prompt_tokens,
                        current_completion_tokens=current_completion_tokens,
                        token_limit=model_input_token_limit,
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
                    selected = h_parse_outcome_from_text(assistant_text, outcomes)
                    if selected:
                        outcome_name = selected
                        assistant_text = h_strip_outcome_line(assistant_text)
                        state.pending_outcome_name = selected
                    else:
                        # Fallback if tag not found
                        outcome_name = outcome_name or outcomes[0]
                elif outcome_strategy == OutcomeStrategy.function_call:
                    # If model never called choose_outcome, try to infer from tag, else fallback
                    if outcome_name is None:
                        selected = h_parse_outcome_from_text(
                            assistant_text, outcomes
                        )
                        if selected:
                            outcome_name = selected
                            assistant_text = h_strip_outcome_line(assistant_text)
                            state.pending_outcome_name = selected
                        else:
                            outcome_name = outcomes[0]

            # Report token usage/cost before finalizing
            _ = yield (
                ReqTokenUsage(
                    acc_prompt_tokens=self.project.llm_usage.prompt_tokens,
                    acc_completion_tokens=self.project.llm_usage.completion_tokens,
                    acc_cost_dollars=self.project.llm_usage.cost_dollars,
                    current_prompt_tokens=current_prompt_tokens,
                    current_completion_tokens=current_completion_tokens,
                    token_limit=model_input_token_limit,
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
