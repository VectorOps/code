import asyncio
import json
import pytest

from pathlib import Path
from vocode.testing import ProjectSandbox

from vocode.runner.executors import llm as llm_mod
from vocode.runner.executors.llm import LLMExecutor, CHOOSE_OUTCOME_TOOL_NAME
from vocode.graph.models import LLMNode, OutcomeStrategy, OutcomeSlot, LLMToolSpec
from vocode.state import Message, ToolCall
from vocode.runner.models import (
    ReqInterimMessage,
    ReqToolCall,
    ReqFinalMessage,
    RespToolCall,
    ExecRunInput,
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
    st_last = None
    while True:
        pkt, st = await anext(agen)
        st_last = st
        if pkt.kind == "message":
            interim_texts.append(pkt.message.text)
            continue
        # Skip over interim token usage packets
        if pkt.kind == "token_usage":
            continue
        return interim_texts, pkt, st_last


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
        state = None
        agen = execu.run(ExecRunInput(messages=[Message(role="user", text="Weather?")], state=state))

        # Drain interim messages until tool call request
        interim_msgs, pkt, state = await drain_until_non_interim(agen)
        assert interim_msgs == ["Hi ", "there. "]
        assert pkt.kind == "tool_call"
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

        # Send tool result back by starting a new cycle with the returned state
        tool_result = ToolCall(
            id="call_1",
            name="weather",
            type="function",
            arguments={"city": "NYC"},
            result={"temp": 72},
        )
        agen2 = execu.run(ExecRunInput(messages=[], state=state, response=RespToolCall(tool_calls=[tool_result])))
        interim_msgs2, pkt2, state2 = await drain_until_non_interim(agen2)

        assert interim_msgs2 == ["It is ", "sunny."]
        assert pkt2.kind == "final_message"
        assert pkt2.message is not None
        assert pkt2.message.text == "It is sunny."
        assert pkt2.outcome_name == "accept"


@pytest.mark.asyncio
async def test_llm_executor_function_tokens_pct_applied(monkeypatch, tmp_path):
    # Two outcomes force choose_outcome tool injection -> tools enabled
    cfg = LLMNode(
        name="LLM",
        model="gpt-x",
        outcomes=[OutcomeSlot(name="yes"), OutcomeSlot(name="no")],
        outcome_strategy=OutcomeStrategy.function_call,
        function_tokens_pct=20,
        extra={"model_max_tokens": 1000},
    )
    seq = [
        chunk_content("Short reply."),
        chunk_tool_call(0, "co_1", CHOOSE_OUTCOME_TOOL_NAME, '{"outcome":"yes"}'),
    ]
    stub = ACompletionStub([seq])
    monkeypatch.setattr(llm_mod, "acompletion", stub)

    async with ProjectSandbox.create(tmp_path) as project:
        agen = LLMExecutor(cfg, project).run(ExecRunInput(messages=[Message(role="user", text="Q?")], state=None))

        # Drain until final (no external tools to execute)
        _, pkt, _ = await drain_until_non_interim(agen)
        assert pkt.kind == "final_message"
        assert pkt.message is not None
        assert pkt.message.text == "Short reply."
        assert pkt.outcome_name == "yes"

        # Verify that max_tokens was set to 20% of model_max_tokens
        call = stub.calls[0]
        assert call["tools"] is not None  # choose_outcome tool injected
        assert call["max_tokens"] == 200


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
        agen = LLMExecutor(cfg, project).run(ExecRunInput(messages=[Message(role="user", text="Question")], state=None))

        interim, pkt, _ = await drain_until_non_interim(agen)
        assert interim == ["Answer body", "\nOUTCOME: reject"]
        assert pkt.kind == "final_message"
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
        agen = LLMExecutor(cfg, project).run(ExecRunInput(messages=[Message(role="user", text="Go")], state=None))

        _, pkt, _ = await drain_until_non_interim(agen)
        assert pkt.kind == "final_message"
        assert pkt.message is not None
        assert pkt.message.text == "No explicit outcome provided."
        # Fallback to first outcome when tag missing
        assert pkt.outcome_name == "first"


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
        agen = LLMExecutor(cfg, project).run(ExecRunInput(messages=history, state=None))

        _, pkt, _ = await drain_until_non_interim(agen)
        assert pkt.kind == "final_message"
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

@pytest.mark.asyncio
async def test_llm_executor_additional_messages_with_existing_state(monkeypatch, tmp_path):
    cfg = LLMNode(
        name="LLM",
        model="gpt-x",
        outcomes=[OutcomeSlot(name="done")],
        outcome_strategy=OutcomeStrategy.function_call,
    )
    seq1 = [chunk_content("First.")]
    seq2 = [chunk_content("Second.")]
    stub = ACompletionStub([seq1, seq2])
    monkeypatch.setattr(llm_mod, "acompletion", stub)

    async with ProjectSandbox.create(tmp_path) as project:
        execu = LLMExecutor(cfg, project)

        # First run with initial user history (no state yet)
        agen1 = execu.run(ExecRunInput(messages=[Message(role="user", text="Hi")], state=None))
        _, pkt1, state1 = await drain_until_non_interim(agen1)
        assert pkt1.kind == "final_message"
        assert pkt1.message is not None and pkt1.message.text == "First."

        # Second run: existing state plus additional input messages should be appended to state
        agen2 = execu.run(ExecRunInput(messages=[Message(role="user", text="More")], state=state1))
        _, pkt2, _ = await drain_until_non_interim(agen2)
        assert pkt2.kind == "final_message"
        assert pkt2.message is not None and pkt2.message.text == "Second."

        # Verify that the second call included the previous assistant text and the new user message
        sent_msgs_2 = stub.calls[1]["messages"]
        assert {"role": "assistant", "content": "First."} in sent_msgs_2
        # The executor appends the assistant reply after the call, so the last item is the assistant ("Second.").
        # Ensure the new user message is present and precedes the final assistant reply.
        assert {"role": "user", "content": "More"} in sent_msgs_2
        assert sent_msgs_2[-1] == {"role": "assistant", "content": "Second."}
        user_idx = next(i for i, m in enumerate(sent_msgs_2) if m == {"role": "user", "content": "More"})
        assert user_idx < len(sent_msgs_2) - 1


@pytest.mark.asyncio
async def test_llm_executor_tool_call_auto_approve_passthrough(monkeypatch, tmp_path):
    # Configure a tool with auto_approve=True and verify it is passed through on emitted tool call.
    cfg = LLMNode(
        name="LLM",
        model="gpt-x",
        outcomes=[OutcomeSlot(name="done")],
        outcome_strategy=OutcomeStrategy.function_call,
        tools=[LLMToolSpec(name="weather", auto_approve=True)],
    )

    # Stream emits a single external tool call
    seq = [
        chunk_content("Fetching "),
        chunk_tool_call(0, "call_1", "weather", '{"city":"NYC"}'),
    ]
    stub = ACompletionStub([seq])
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
        agen = execu.run(ExecRunInput(messages=[Message(role="user", text="Weather?")], state=None))

        _, pkt, _ = await drain_until_non_interim(agen)
        assert pkt.kind == "tool_call"
        assert len(pkt.tool_calls) == 1
        assert pkt.tool_calls[0].auto_approve is True
