from __future__ import annotations

from typing import AsyncIterator, List, Optional, Dict, Any
import json
import asyncio
from litellm import acompletion, completion_cost
import litellm
from vocode.runner.runner import Executor
from vocode.models import OutcomeStrategy
from vocode.state import Message, ToolCall, LLMUsageStats
from vocode.runner.models import (
    ReqPacket,
    ReqToolCall,
    ReqFinalMessage,
    ReqInterimMessage,
    ReqLocalTokenUsage,
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
    build_effective_tool_specs as h_build_effective_tool_specs,
    resolve_model_token_limit as h_resolve_model_token_limit,
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
        conv: List[Dict[str, Any]] = state.conv

        # Resolve model token limit once for this run.
        model_input_token_limit = h_resolve_model_token_limit(cfg)
        # Attach model input token limit to usage snapshots for UI and aggregation.
        if model_input_token_limit is not None:
            if state.total_usage.input_token_limit is None:
                state.total_usage.input_token_limit = model_input_token_limit
            if state.last_usage.input_token_limit is None:
                state.last_usage.input_token_limit = model_input_token_limit

        # Emit per-node usage snapshot at the start of this run using values stored
        # in state:
        # - context_usage: last round's per-call usage (zero-initialized for new nodes),
        # - usage: accumulated totals for this node.
        _ = yield (
            ReqLocalTokenUsage(
                context_usage=state.last_usage.model_copy(),
                usage=state.total_usage.model_copy(),
            ),
            state,
        )

        # Outcome handling strategy
        outcomes: List[str] = h_get_outcome_names(cfg)
        outcome_desc_bullets: str = h_get_outcome_desc_bullets(cfg)
        outcome_choice_desc: str = h_get_outcome_choice_desc(cfg, outcome_desc_bullets)
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
                        "function": await tool.openapi_spec(eff_specs[tool_name]),
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
        while True:
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
                "reasoning_effort",
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
                    args = {
                        "model": cfg.model,
                        "messages": conv,
                        "temperature": cfg.temperature,
                        "max_tokens": effective_max_tokens,
                        "tools": tools,
                        "tool_choice": "auto" if tools else None,
                        "reasoning_effort": cfg.reasoning_effort,
                        "stream": True,
                        "stream_options": {
                            "include_usage": True,
                        },
                        **extra_args,
                    }

                    completion_coro = acompletion(**args)

                    # Wrap in a named task for better diagnostics
                    task_name = f"llm.acompletion:{cfg.name}"
                    stream_task = asyncio.create_task(completion_coro, name=task_name)
                    stream = await stream_task

                    chunks: List[Any] = []

                    # Collect streamed chunks; emit interim messages for content deltas.
                    async for chunk in stream:
                        chunks.append(chunk)

                        content_piece: Optional[str] = None
                        try:
                            choices_obj = chunk.choices
                        except Exception:
                            choices_obj = None
                        if choices_obj:
                            delta_obj = choices_obj[0].delta
                            if delta_obj is not None:
                                try:
                                    value = delta_obj.content
                                except Exception:
                                    value = None
                                if isinstance(value, str):
                                    content_piece = value

                        if content_piece:
                            _ = yield (
                                ReqInterimMessage(
                                    message=Message(
                                        role="agent",
                                        text=content_piece,
                                        node=cfg.name,
                                    )
                                ),
                                None,
                            )
                    break  # success
                except Exception as e:
                    # Avoid getattr/hasattr; access status_code directly when present.
                    try:
                        status_code = e.status_code  # type: ignore[attr-defined]
                    except Exception:
                        status_code = None
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

            # Build final response object from streamed chunks.
            response = litellm.stream_chunk_builder(
                chunks,
                messages=conv,
            )

            # Extract assistant message and full content from final response.
            if not response.choices:
                raise RuntimeError("LLM response missing choices")

            choice = response.choices[0]
            message_obj = choice.message
            assistant_text = message_obj.content or ""

            # Get token usage from response.usage (official litellm usage object).
            prompt_tokens = 0
            completion_tokens = 0
            usage_obj = response.usage
            if usage_obj is not None:
                prompt_tokens = int(usage_obj.prompt_tokens or 0)
                completion_tokens = int(usage_obj.completion_tokens or 0)

            # Compute round cost using litellm.completion_cost on the full response.
            round_cost = 0.0
            try:
                cost_val = completion_cost(
                    completion_response=response, model=cfg.model
                )
                if cost_val is not None:
                    round_cost = float(cost_val)
            except Exception:
                round_cost = 0.0

            # Per-call usage for this round (also used as delta for aggregation).
            delta = LLMUsageStats(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_dollars=round_cost,
                input_token_limit=model_input_token_limit,
            )

            # Update per-node usage in state.
            state.last_usage = delta
            state.total_usage.prompt_tokens += delta.prompt_tokens
            state.total_usage.completion_tokens += delta.completion_tokens
            state.total_usage.cost_dollars += delta.cost_dollars
            if model_input_token_limit is not None:
                state.total_usage.input_token_limit = model_input_token_limit

            # Absolute per-node accumulated usage and per-call context usage.
            usage = state.total_usage.model_copy()
            context_usage = state.last_usage.model_copy()

            # Build final assistant message dict and append to conversation
            raw_tool_calls = message_obj.tool_calls or []
            assistant_tool_calls_for_conv: List[Dict[str, Any]] = []
            for tc in raw_tool_calls:
                function = tc.function
                function_name = (
                    function.name
                    if function is not None and function.name is not None
                    else ""
                )
                function_arguments = (
                    function.arguments
                    if function is not None and function.arguments is not None
                    else ""
                )
                assistant_tool_calls_for_conv.append(
                    {
                        "id": tc.id or "",
                        "type": tc.type or "function",
                        "function": {
                            "name": function_name,
                            "arguments": function_arguments,
                        },
                    }
                )

            assistant_msg: Dict[str, Any] = {
                "role": message_obj.role or "assistant",
                "content": assistant_text or None,
            }
            if assistant_tool_calls_for_conv:
                assistant_msg["tool_calls"] = assistant_tool_calls_for_conv
            conv.append(assistant_msg)

            # Separate outcome selection vs. external tool calls
            external_calls: List[ToolCall] = []
            for tc in raw_tool_calls:
                function = tc.function
                name = (
                    function.name
                    if function is not None and function.name is not None
                    else ""
                )
                arguments_str = (
                    function.arguments
                    if function is not None and function.arguments is not None
                    else ""
                )
                call_id = tc.id

                if name == CHOOSE_OUTCOME_TOOL_NAME:
                    try:
                        parsed = json.loads(arguments_str) if arguments_str else {}
                    except Exception:
                        parsed = {}
                    sel = parsed.get("outcome")
                    if isinstance(sel, str) and sel in outcomes:
                        # record selected outcome for next cycle
                        state.pending_outcome_name = sel
                    # Do not forward this special call to the runner
                    continue

                try:
                    args_obj = json.loads(arguments_str) if arguments_str else {}
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
                max_rounds = cfg.max_rounds
                # max_rounds == 0 => unlimited
                if max_rounds and state.tool_rounds > max_rounds:
                    raise RuntimeError(
                        f"LLMExecutor exceeded maximum function-call rounds ({max_rounds}); possible tool loop"
                    )
                # Report per-call local usage before requesting tool execution
                _ = yield (
                    ReqLocalTokenUsage(
                        context_usage=context_usage,
                        usage=usage,
                        delta=delta,
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
                        selected = h_parse_outcome_from_text(assistant_text, outcomes)
                        if selected:
                            outcome_name = selected
                            assistant_text = h_strip_outcome_line(assistant_text)
                            state.pending_outcome_name = selected
                        else:
                            outcome_name = outcomes[0]

            # Report per-call local usage before finalizing
            _ = yield (
                ReqLocalTokenUsage(
                    context_usage=context_usage,
                    usage=usage,
                    delta=delta,
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
