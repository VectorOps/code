from __future__ import annotations

from typing import AsyncIterator, List, Optional, Dict, Any, Final
import json
import re
import asyncio
from vocode.runner.preprocessors.base import apply_preprocessors

CHOOSE_OUTCOME_TOOL_NAME: Final[str] = "__choose_outcome__"
OUTCOME_TAG_RE = re.compile(r"^\s*OUTCOME\s*:\s*([A-Za-z0-9_\-]+)\s*$")
OUTCOME_LINE_PREFIX_RE = re.compile(r"^\s*OUTCOME\s*:\s*")
MAX_ROUNDS = 32

from litellm import acompletion

from vocode.runner.runner import Executor
from vocode.graph.models import LLMNode, OutcomeStrategy
from vocode.state import Message, ToolCall
from vocode.runner.models import (
    ReqPacket,
    ReqToolCall,
    ReqFinalMessage,
    ReqInterimMessage,
    RespPacket,
    RespMessage,          # add this
    PACKET_TOOL_CALL,
)


class LLMExecutor(Executor):
    # Must match LLMNode.type
    type = "llm"

    def __init__(self, config, project):
        super().__init__(config=config, project=project)
        if not isinstance(config, LLMNode):
            # Allow base Node but prefer LLMNode
            raise TypeError("LLMExecutor requires config to be an LLMNode")

    def _map_role(self, role: str) -> str:
        # Normalize internal "agent" to OpenAI "assistant"
        return "assistant" if role == "agent" else role

    def _build_base_messages(self, cfg: LLMNode, history: List[Message]) -> List[Dict[str, Any]]:
        msgs: List[Dict[str, Any]] = []
        if cfg.system:
            sys_text = cfg.system
            # Apply LLMNode-specific preprocessors to the system prompt only
            preprocs = cfg.preprocessors or []
            if preprocs:
                preproc_names = [p.name for p in preprocs]
                sys_text = apply_preprocessors(preproc_names, sys_text)
            msgs.append({"role": "system", "content": sys_text})
        for m in history:
            msgs.append({"role": self._map_role(m.role), "content": m.text})
        return msgs

    def _parse_outcome_from_text(self, text: str, valid_outcomes: List[str]) -> Optional[str]:
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
            [ln for ln in text.splitlines() if not OUTCOME_LINE_PREFIX_RE.match(ln.strip())]
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
        for s in (cfg.outcomes or []):
            desc = s.description or ""
            lines.append(f"- {s.name}: {desc}".rstrip())
        return "\n".join(lines)

    def _get_outcome_choice_desc(self, cfg: LLMNode, outcome_desc_bullets: str) -> str:
        if outcome_desc_bullets.strip():
            return "Choose exactly one of the following outcomes:\n" + outcome_desc_bullets
        return "Choose the appropriate outcome."

    async def run(self, messages: List[Message]) -> AsyncIterator[ReqPacket]:
        cfg: LLMNode = self.config  # type: ignore[assignment]
        conv: List[Dict[str, Any]] = self._build_base_messages(cfg, messages)

        # Outcome handling strategy
        outcomes: List[str] = self._get_outcome_names(cfg)
        outcome_desc_bullets: str = self._get_outcome_desc_bullets(cfg)
        outcome_choice_desc: str = self._get_outcome_choice_desc(cfg, outcome_desc_bullets)
        outcome_strategy: OutcomeStrategy = cfg.outcome_strategy
        outcome_name: Optional[str] = None

        # Get tool specs from node config
        external_tools: List[Dict[str, Any]] = []
        for tool_name in cfg.tools:
            tool = self.project.tools.get(tool_name)
            if tool:
                external_tools.append({
                    "type": "function",
                    "function": tool.openapi_spec(),
                })

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
        tool_rounds = 0

        while True:

            assistant_text_parts: List[str] = []
            tool_call_builders: Dict[int, Dict[str, Any]] = {}
            # Filter extra to avoid overriding explicit kwargs (e.g., 'tools')
            extra_args = dict(cfg.extra or {})
            for k in ("tools", "tool_choice", "messages", "model", "stream", "temperature", "max_tokens"):
                extra_args.pop(k, None)

            # Start streaming
            stream = await acompletion(
                model=cfg.model,
                messages=conv,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                tools=tools,
                tool_choice="auto" if tools else None,
                stream=True,
                **extra_args,
            )

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
                    # Emit an interim message with just this piece
                    _ = yield ReqInterimMessage(message=Message(role="agent", text=content_piece))

                # Accumulate streamed tool_calls (OpenAI-style deltas)
                tc_deltas = delta.get("tool_calls") or []
                for dtc in tc_deltas:
                    idx = dtc.get("index", 0)
                    b = tool_call_builders.setdefault(
                        idx,
                        {"id": dtc.get("id"), "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if dtc.get("id") and not b.get("id"):
                        b["id"] = dtc.get("id")
                    fn = dtc.get("function") or {}
                    if "name" in fn and fn["name"]:
                        b["function"]["name"] = fn["name"]
                    if "arguments" in fn and fn["arguments"]:
                        b["function"]["arguments"] += fn["arguments"]

            assistant_text = "".join(assistant_text_parts)

            # Build final assistant message dict and append to conversation
            streamed_tool_calls = []
            for idx in sorted(tool_call_builders.keys()):
                b = tool_call_builders[idx]
                streamed_tool_calls.append(
                    {
                        "id": b.get("id"),
                        "type": b.get("type", "function"),
                        "function": {
                            "name": (b.get("function") or {}).get("name", ""),
                            "arguments": (b.get("function") or {}).get("arguments", ""),
                        },
                    }
                )

            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": assistant_text or None}
            if streamed_tool_calls:
                assistant_msg["tool_calls"] = streamed_tool_calls
            conv.append(assistant_msg)

            # Separate outcome selection vs. external tool calls
            external_calls: List[ToolCall] = []
            for tc in streamed_tool_calls:
                name = ((tc.get("function") or {}).get("name") or "")
                arguments = ((tc.get("function") or {}).get("arguments") or "")
                call_id = tc.get("id")
                if name == CHOOSE_OUTCOME_TOOL_NAME:
                    try:
                        parsed = json.loads(arguments) if arguments else {}
                    except Exception:
                        parsed = {}
                    sel = parsed.get("outcome")
                    if isinstance(sel, str) and sel in outcomes:
                        outcome_name = sel
                    # Do not forward this special call to the runner
                    continue
                # Forward all other calls to the runner using our protocol
                # Parse JSON arguments to a dict as ToolCall.arguments expects a mapping
                try:
                    args_obj = json.loads(arguments) if isinstance(arguments, str) and arguments else {}
                except Exception:
                    args_obj = {}
                external_calls.append(
                    ToolCall(id=call_id, name=name, arguments=args_obj, type="function")
                )

            if external_calls:
                # Ask runner to execute tools and send back results
                resp_packet: RespPacket = (yield ReqToolCall(tool_calls=external_calls))
                returned_calls: List[ToolCall]
                if resp_packet.kind == PACKET_TOOL_CALL:
                    returned_calls = resp_packet.tool_calls
                else:
                    returned_calls = external_calls

                for rcall in returned_calls:
                    conv.append(
                        {
                            "role": "tool",
                            "tool_call_id": rcall.id or "",
                            "name": rcall.name,
                            # LLM expects a string content; serialize dict result to JSON
                            "content": json.dumps(rcall.result) if rcall.result is not None else "",
                        }
                    )

                    #print(conv[-1])
                # Count only rounds that include external tool calls
                tool_rounds += 1
                if tool_rounds > MAX_ROUNDS:
                    raise RuntimeError(
                        f"LLMExecutor exceeded maximum function-call rounds ({MAX_ROUNDS}); possible tool loop"
                    )

                # Continue loop for model to incorporate tool results
                continue

            # No tool calls -> proceed to finalize content and outcome
            # assistant_text already set above

            if len(outcomes) > 1:
                if outcome_strategy == OutcomeStrategy.tag:
                    selected = self._parse_outcome_from_text(assistant_text, outcomes)
                    if selected:
                        outcome_name = selected
                        assistant_text = self._strip_outcome_line(assistant_text)
                    else:
                        # Fallback if tag not found
                        outcome_name = outcome_name or outcomes[0]
                elif outcome_strategy == OutcomeStrategy.function_call:
                    # If model never called choose_outcome, try to infer from tag, else fallback
                    if outcome_name is None:
                        selected = self._parse_outcome_from_text(assistant_text, outcomes)
                        if selected:
                            outcome_name = selected
                            assistant_text = self._strip_outcome_line(assistant_text)
                        else:
                            outcome_name = outcomes[0]

            # Instead of exiting after one round, present the final and possibly accept a post-final user message
            final_msg = Message(role="agent", text=assistant_text)
            resp = (yield ReqFinalMessage(message=final_msg, outcome_name=outcome_name))

            # If runner sent a user message in response to the final (additional requirements),
            # continue the same executor by appending it to the conversation; otherwise
            # just pause here (runner will not send anything on approval).
            if isinstance(resp, RespMessage):
                # Append new user message and continue the loop without resetting conv
                conv.append({"role": self._map_role(resp.message.role), "content": resp.message.text})
                # Reset outcome_name for next turn
                outcome_name = None
                assistant_text = ""
                # Reset function-call round counter for a new user-provided turn
                tool_rounds = 0
                # Continue outer while loop to produce a next assistant response
                continue

            # Otherwise, stay paused on this executor until the runner closes or resets it.
            # Await indefinitely to remain suspended; runner will aclose() / recreate as needed.
            await asyncio.Event().wait()
