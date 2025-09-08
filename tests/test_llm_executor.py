import asyncio
import json
import pytest

from pathlib import Path
from vocode.testing import ProjectSandbox

from vocode.runner.executors import llm as llm_mod
from vocode.runner.executors.llm import LLMExecutor, CHOOSE_OUTCOME_TOOL_NAME
from vocode.graph.models import LLMNode, OutcomeStrategy, OutcomeSlot
from vocode.state import Message, ToolCall
from vocode.runner.models import (
    ReqInterimMessage,
    ReqToolCall,
    ReqFinalMessage,
    RespToolCall,
)


def chunk_content(text: str):
    return {"choices": [{"delta": {"content": text}}]}


def chunk_tool_call(index: int, call_id: str, name: str, args_part: str):
    return {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call_id,
                            "function": {"name": name, "arguments": args_part},
                        }
                    ]
                }
            }
        ]
    }


class ACompletionStub:
    def __init__(self, sequences):
        # sequences: list[list[chunk dict]]
        self._sequences = [list(seq) for seq in sequences]
        self.calls = []

    async def __call__(self, *_, **kwargs):
        # Record selected arguments for inspection
        self.calls.append(
            {
                "model": kwargs.get("model"),
                "messages": kwargs.get("messages"),
                "tools": kwargs.get("tools"),
                "tool_choice": kwargs.get("tool_choice"),
                "temperature": kwargs.get("temperature"),
                "max_tokens": kwargs.get("max_tokens"),
            }
        )
        seq = self._sequences.pop(0)

        async def agen():
            for ch in seq:
                await asyncio.sleep(0)
                yield ch

        return agen()


async def drain_until_non_interim(agen):
    interim_texts = []
    while True:
        pkt = await anext(agen)
        if isinstance(pkt, ReqInterimMessage):
            interim_texts.append(pkt.message.text)
            continue
        return interim_texts, pkt


@pytest.mark.asyncio
async def test_llm_executor_function_call_and_outcome_selection(monkeypatch, tmp_path):
    # Two outcomes, function_call strategy, one external tool
    cfg = LLMNode(
        name="LLM",
        model="gpt-x",
        outcomes=[OutcomeSlot(name="accept"), OutcomeSlot(name="reject")],
        outcome_strategy=OutcomeStrategy.function_call,
        tools=["weather"],
    )

    # First model streaming: emits content and an external tool call (weather)
    # Provide full JSON in a single delta; current executor doesn’t merge partial tool_call argument streams.
    seq1 = [
        chunk_content("Hi "),
        chunk_content("there. "),
        chunk_tool_call(0, "call_1", "weather", '{"city":"NYC"}'),
    ]
    # Second model streaming: emits content and chooses outcome via special tool
    seq2 = [
        chunk_content("It is "),
        chunk_content("sunny."),
        chunk_tool_call(0, "call_co", CHOOSE_OUTCOME_TOOL_NAME, '{"outcome":"accept"}'),
    ]

    stub = ACompletionStub([seq1, seq2])
    monkeypatch.setattr(llm_mod, "acompletion", stub)

    async with ProjectSandbox.create(tmp_path) as project:
        class _DummyWeatherTool:
            def openapi_spec(self):
                return {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                }

        project.tools["weather"] = _DummyWeatherTool()

        execu = LLMExecutor(cfg, project)
        agen = execu.run(messages=[Message(role="user", text="Weather?")])

        # Drain interim messages until tool call request
        interim_msgs, pkt = await drain_until_non_interim(agen)
        assert interim_msgs == ["Hi ", "there. "]
        assert isinstance(pkt, ReqToolCall)
        assert len(pkt.tool_calls) == 1
        tc = pkt.tool_calls[0]
        assert tc.name == "weather"
        assert tc.id == "call_1"
        assert tc.arguments == {"city": "NYC"}  # parsed from streamed parts

        # Ensure tools passed include both external tool and choose_outcome function
        tools_sent = stub.calls[0]["tools"]
        assert isinstance(tools_sent, list)
        tool_names = [t.get("function", {}).get("name") for t in tools_sent]
        assert "weather" in tool_names
        assert CHOOSE_OUTCOME_TOOL_NAME in tool_names
        assert stub.calls[0]["tool_choice"] == "auto"

        # Send tool result back and capture immediate yield (may be an interim chunk)
        tool_result = ToolCall(
            id="call_1",
            name="weather",
            type="function",
            arguments={"city": "NYC"},
            result={"temp": 72},
        )
        first_pkt = await agen.asend(RespToolCall(tool_calls=[tool_result]))

        # Drain interim messages until final
        interim_msgs2 = []
        if isinstance(first_pkt, ReqInterimMessage):
            interim_msgs2.append(first_pkt.message.text)
            # continue draining remaining packets until non-interim
            while True:
                pkt2 = await anext(agen)
                if isinstance(pkt2, ReqInterimMessage):
                    interim_msgs2.append(pkt2.message.text)
                    continue
                break
        else:
            pkt2 = first_pkt
        assert interim_msgs2 == ["It is ", "sunny."]
        assert isinstance(pkt2, ReqFinalMessage)
        assert pkt2.message is not None
        assert pkt2.message.text == "It is sunny."
        assert pkt2.outcome_name == "accept"

        # Finalize generator
        await agen.aclose()


@pytest.mark.asyncio
async def test_llm_executor_tag_strategy_streaming_and_strip(monkeypatch, tmp_path):
    cfg = LLMNode(
        name="LLM",
        model="gpt-x",
        system="You are helpful.",
        outcomes=[OutcomeSlot(name="approve"), OutcomeSlot(name="reject")],
        outcome_strategy=OutcomeStrategy.tag,
    )

    seq = [
        chunk_content("Answer body"),
        chunk_content("\nOUTCOME: reject"),
    ]
    stub = ACompletionStub([seq])
    monkeypatch.setattr(llm_mod, "acompletion", stub)

    async with ProjectSandbox.create(tmp_path) as project:
        agen = LLMExecutor(cfg, project).run(messages=[Message(role="user", text="Question")])

        interim, pkt = await drain_until_non_interim(agen)
        assert interim == ["Answer body", "\nOUTCOME: reject"]
        assert isinstance(pkt, ReqFinalMessage)
        assert pkt.message is not None
        # Outcome line stripped from final text
        assert pkt.message.text == "Answer body"
        assert pkt.outcome_name == "reject"

        # Verify tag strategy injected a system instruction at the start
        sent_msgs = stub.calls[0]["messages"]
        assert sent_msgs[0]["role"] == "system"
        assert "OUTCOME:" in sent_msgs[0]["content"]
        # No tools provided for tag strategy with no external tools
        assert stub.calls[0]["tools"] is None

        await agen.aclose()


@pytest.mark.asyncio
async def test_llm_executor_tag_strategy_fallback_to_first_outcome(monkeypatch, tmp_path):
    cfg = LLMNode(
        name="LLM",
        model="gpt-x",
        outcomes=[OutcomeSlot(name="first"), OutcomeSlot(name="second")],
        outcome_strategy=OutcomeStrategy.tag,
    )
    seq = [chunk_content("No explicit outcome provided.")]
    stub = ACompletionStub([seq])
    monkeypatch.setattr(llm_mod, "acompletion", stub)

    async with ProjectSandbox.create(tmp_path) as project:
        agen = LLMExecutor(cfg, project).run(messages=[Message(role="user", text="Go")])

        _, pkt = await drain_until_non_interim(agen)
        assert isinstance(pkt, ReqFinalMessage)
        assert pkt.message is not None
        assert pkt.message.text == "No explicit outcome provided."
        # Fallback to first outcome when tag missing
        assert pkt.outcome_name == "first"

        await agen.aclose()


@pytest.mark.asyncio
async def test_llm_executor_single_outcome_no_choose_tool_and_role_mapping(monkeypatch, tmp_path):
    cfg = LLMNode(
        name="LLM",
        model="gpt-x",
        system="sys",
        outcomes=[OutcomeSlot(name="done")],
        outcome_strategy=OutcomeStrategy.function_call,
        tools=["weather"],
    )
    seq = [chunk_content("Fin.")]
    stub = ACompletionStub([seq])
    monkeypatch.setattr(llm_mod, "acompletion", stub)

    async with ProjectSandbox.create(tmp_path) as project:
        class _DummyWeatherTool:
            def openapi_spec(self):
                return {
                    "name": "weather",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                }

        project.tools["weather"] = _DummyWeatherTool()

        history = [Message(role="user", text="Hi"), Message(role="agent", text="Prev assistant")]
        agen = LLMExecutor(cfg, project).run(messages=history)

        _, pkt = await drain_until_non_interim(agen)
        assert isinstance(pkt, ReqFinalMessage)
        assert pkt.message is not None
        assert pkt.message.text == "Fin."
        # No explicit outcome expected for single outcome
        assert pkt.outcome_name is None

        # Tools should be only external ones (no choose_outcome injection)
        tools_sent = stub.calls[0]["tools"]
        assert isinstance(tools_sent, list)
        tool_names = [t.get("function", {}).get("name") for t in tools_sent]
        assert tool_names == ["weather"]

        # Verify base messages and role mapping ("agent" -> "assistant")
        sent_msgs = stub.calls[0]["messages"]
        assert sent_msgs[0] == {"role": "system", "content": "sys"}
        assert {"role": "user", "content": "Hi"} in sent_msgs
        assert {"role": "assistant", "content": "Prev assistant"} in sent_msgs

        await agen.aclose()
