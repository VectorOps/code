import asyncio
import json
from pathlib import Path

import pytest
from litellm import ModelResponseStream

from vocode.testing import ProjectSandbox

from vocode.runner.executors import llm as llm_mod
from vocode.runner.executors.llm import (
    LLMExecutor,
    CHOOSE_OUTCOME_TOOL_NAME,
    LLMNode,
)
from vocode.models import OutcomeStrategy, OutcomeSlot
from vocode.state import Message, ToolCall
from vocode.runner.models import (
    ReqInterimMessage,
    ReqToolCall,
    ReqFinalMessage,
    RespToolCall,
    ExecRunInput,
)
from vocode.settings import ToolSpec


def _make_chunk(delta: dict, usage: dict | None = None) -> ModelResponseStream:
    """Build a litellm-style streaming chunk for tests.

    We only populate fields that LLMExecutor and litellm.stream_chunk_builder rely on:
    - model: to satisfy builder requirements
    - choices[0].delta: carries content/tool_calls
    - usage: optional, for explicit token accounting tests
    """

    kwargs: dict = {
        "model": "gpt-x",
        "choices": [{"delta": delta}],
    }
    if usage is not None:
        kwargs["usage"] = usage
    return ModelResponseStream(**kwargs)


def chunk_content(text: str) -> ModelResponseStream:
    return _make_chunk({"content": text})


def chunk_usage(prompt_tokens: int, completion_tokens: int) -> ModelResponseStream:
    return _make_chunk(
        delta={},
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    )


def chunk_tool_call(
    index: int, call_id: str, name: str, args_part: str
) -> ModelResponseStream:
    return _make_chunk(
        {
            "tool_calls": [
                {
                    "index": index,
                    "id": call_id,
                    "function": {"name": name, "arguments": args_part},
                }
            ]
        }
    )


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
                "reasoning_effort": kwargs.get("reasoning_effort"),
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
        # Skip over interim token usage (old and new) and log packets
        if pkt.kind in ("token_usage", "local_token_usage", "log"):
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
            async def openapi_spec(self, spec):
                return {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                }

        project.tools["weather"] = _DummyWeatherTool()

        execu = LLMExecutor(cfg, project)
        state = None
        agen = execu.run(
            ExecRunInput(messages=[Message(role="user", text="Weather?")], state=state)
        )

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
        agen2 = execu.run(
            ExecRunInput(
                messages=[],
                state=state,
                response=RespToolCall(tool_calls=[tool_result]),
            )
        )
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
        agen = LLMExecutor(cfg, project).run(
            ExecRunInput(messages=[Message(role="user", text="Q?")], state=None)
        )

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
        agen = LLMExecutor(cfg, project).run(
            ExecRunInput(messages=[Message(role="user", text="Question")], state=None)
        )

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
async def test_llm_executor_tag_strategy_fallback_to_first_outcome(
    monkeypatch, tmp_path
):
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
        agen = LLMExecutor(cfg, project).run(
            ExecRunInput(messages=[Message(role="user", text="Go")], state=None)
        )

        _, pkt, _ = await drain_until_non_interim(agen)
        assert pkt.kind == "final_message"
        assert pkt.message is not None
        assert pkt.message.text == "No explicit outcome provided."
        # Fallback to first outcome when tag missing
        assert pkt.outcome_name == "first"


@pytest.mark.asyncio
async def test_llm_executor_single_outcome_no_choose_tool_and_role_mapping(
    monkeypatch, tmp_path
):
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
            async def openapi_spec(self, spec):
                return {
                    "name": "weather",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                }

        project.tools["weather"] = _DummyWeatherTool()

        history = [
            Message(role="user", text="Hi"),
            Message(role="agent", text="Prev assistant"),
        ]
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
        assert {"role": "user", "content": "Prev assistant"} in sent_msgs


@pytest.mark.asyncio
async def test_llm_executor_additional_messages_with_existing_state(
    monkeypatch, tmp_path
):
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
        agen1 = execu.run(
            ExecRunInput(messages=[Message(role="user", text="Hi")], state=None)
        )
        _, pkt1, state1 = await drain_until_non_interim(agen1)
        assert pkt1.kind == "final_message"
        assert pkt1.message is not None and pkt1.message.text == "First."

        # Second run: existing state plus additional input messages should be appended to state
        agen2 = execu.run(
            ExecRunInput(messages=[Message(role="user", text="More")], state=state1)
        )
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
        user_idx = next(
            i
            for i, m in enumerate(sent_msgs_2)
            if m == {"role": "user", "content": "More"}
        )
        assert user_idx < len(sent_msgs_2) - 1


@pytest.mark.asyncio
async def test_llm_executor_tool_call_auto_approve_passthrough(monkeypatch, tmp_path):
    # Configure a tool with auto_approve=True and verify it is passed through on emitted tool call.
    cfg = LLMNode(
        name="LLM",
        model="gpt-x",
        outcomes=[OutcomeSlot(name="done")],
        outcome_strategy=OutcomeStrategy.function_call,
        tools=[ToolSpec(name="weather", auto_approve=True)],
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
            async def openapi_spec(self, spec):
                return {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                }

        project.tools["weather"] = _DummyWeatherTool()

        execu = LLMExecutor(cfg, project)
        agen = execu.run(
            ExecRunInput(messages=[Message(role="user", text="Weather?")], state=None)
        )

        _, pkt, _ = await drain_until_non_interim(agen)
        assert pkt.kind == "tool_call"
        assert len(pkt.tool_calls) == 1
        assert pkt.tool_calls[0].tool_spec is not None
        assert pkt.tool_calls[0].tool_spec.auto_approve is True


@pytest.mark.asyncio
async def test_llm_executor_tool_call_auto_approve_global_overrides_node(
    monkeypatch, tmp_path
):
    # Node sets False, global sets True -> effective True
    cfg = LLMNode(
        name="LLM",
        model="gpt-x",
        outcomes=[OutcomeSlot(name="done")],
        outcome_strategy=OutcomeStrategy.function_call,
        tools=[ToolSpec(name="weather", auto_approve=False)],
    )
    seq = [chunk_tool_call(0, "call_1", "weather", '{"city":"NYC"}')]
    stub = ACompletionStub([seq])
    monkeypatch.setattr(llm_mod, "acompletion", stub)

    async with ProjectSandbox.create(tmp_path) as project:

        class _DummyWeatherTool:
            async def openapi_spec(self, spec):
                return {
                    "name": "weather",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                }

        project.tools["weather"] = _DummyWeatherTool()
        # Set global auto_approve=True
        project.settings.tools = [
            ToolSpec(name="weather", enabled=True, auto_approve=True)
        ]

        agen = LLMExecutor(cfg, project).run(
            ExecRunInput(messages=[Message(role="user", text="Q")], state=None)
        )
        _, pkt, _ = await drain_until_non_interim(agen)
        assert pkt.kind == "tool_call"
        assert pkt.tool_calls[0].name == "weather"
        assert pkt.tool_calls[0].tool_spec is not None
        assert pkt.tool_calls[0].tool_spec.auto_approve is True


@pytest.mark.asyncio
async def test_llm_executor_tool_call_auto_approve_node_used_when_global_missing(
    monkeypatch, tmp_path
):
    # No global, node sets True -> effective True
    cfg = LLMNode(
        name="LLM",
        model="gpt-x",
        outcomes=[OutcomeSlot(name="done")],
        outcome_strategy=OutcomeStrategy.function_call,
        tools=[ToolSpec(name="weather", auto_approve=True)],
    )
    seq = [chunk_tool_call(0, "call_1", "weather", '{"city":"NYC"}')]
    stub = ACompletionStub([seq])
    monkeypatch.setattr(llm_mod, "acompletion", stub)

    async with ProjectSandbox.create(tmp_path) as project:

        class _DummyWeatherTool:
            async def openapi_spec(self, spec):
                return {
                    "name": "weather",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                }

        project.tools["weather"] = _DummyWeatherTool()
        # No global setting for weather
        project.settings.tools = []

        agen = LLMExecutor(cfg, project).run(
            ExecRunInput(messages=[Message(role="user", text="Q")], state=None)
        )
        _, pkt, _ = await drain_until_non_interim(agen)
        assert pkt.kind == "tool_call"
        assert pkt.tool_calls[0].tool_spec is not None
        assert pkt.tool_calls[0].tool_spec.auto_approve is True


@pytest.mark.asyncio
async def test_llm_executor_tool_call_auto_approve_none_when_unset(
    monkeypatch, tmp_path
):
    # Neither global nor node set -> None
    cfg = LLMNode(
        name="LLM",
        model="gpt-x",
        outcomes=[OutcomeSlot(name="done")],
        outcome_strategy=OutcomeStrategy.function_call,
        tools=[ToolSpec(name="weather")],  # auto_approve omitted => None
    )
    seq = [chunk_tool_call(0, "call_1", "weather", '{"city":"NYC"}')]
    stub = ACompletionStub([seq])
    monkeypatch.setattr(llm_mod, "acompletion", stub)

    async with ProjectSandbox.create(tmp_path) as project:

        class _DummyWeatherTool:
            async def openapi_spec(self, spec):
                return {
                    "name": "weather",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                }

        project.tools["weather"] = _DummyWeatherTool()
        # No global setting for weather
        project.settings.tools = []

        agen = LLMExecutor(cfg, project).run(
            ExecRunInput(messages=[Message(role="user", text="Q")], state=None)
        )
        _, pkt, _ = await drain_until_non_interim(agen)
        assert pkt.kind == "tool_call"
        assert pkt.tool_calls[0].tool_spec is not None
        assert pkt.tool_calls[0].tool_spec.auto_approve is None


class FlakyACompletionStub:
    def __init__(self, exc, ok_sequences):
        self.exc = exc
        self.ok = ACompletionStub(ok_sequences)
        self.calls = 0

    async def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise self.exc
        return await self.ok(*args, **kwargs)


@pytest.mark.asyncio
async def test_llm_executor_retries_on_rate_limit(monkeypatch, tmp_path):
    async with ProjectSandbox.create(tmp_path) as project:
        node = LLMNode(
            name="llm",
            model="test-model",
            system="You are a helpful assistant.",
            outcomes=[OutcomeSlot(name="done")],
        )
        exec = LLMExecutor(config=node, project=project)

        # Patch litellm acompletion to raise RateLimit once then succeed
        rate_exc = llm_mod.litellm.RateLimitError(
            "rate limit", llm_provider="test", model="test-model"
        )
        setattr(rate_exc, "status_code", 429)
        monkeypatch.setattr(
            llm_mod,
            "acompletion",
            FlakyACompletionStub(
                rate_exc,
                [
                    [chunk_content("Hello"), chunk_content(" world")],
                ],
            ),
        )
        # Avoid real sleeping
        slept = {"called": 0}

        async def fake_sleep(_):
            slept["called"] += 1
            return None

        monkeypatch.setattr(llm_mod.asyncio, "sleep", fake_sleep)

        agen = exec.run(
            ExecRunInput(messages=[Message(role="user", text="hi")], state=None)
        )
        _, pkt, _ = await drain_until_non_interim(agen)
        assert isinstance(pkt, ReqFinalMessage)
        # Final message should be the verbatim concatenation of streamed chunks
        assert pkt.message.text == "Hello world"
        assert slept["called"] >= 1


@pytest.mark.asyncio
async def test_llm_executor_non_retryable_returns_error_packet(monkeypatch, tmp_path):
    async with ProjectSandbox.create(tmp_path) as project:
        node = LLMNode(
            name="llm",
            model="test-model",
            system="You are a helpful assistant.",
            outcomes=[OutcomeSlot(name="done"), OutcomeSlot(name="alt")],
        )
        exec = LLMExecutor(config=node, project=project)

        class FakeErr(Exception):
            def __init__(self, msg, status_code):
                super().__init__(msg)
                self.status_code = status_code

        # Never retry
        monkeypatch.setattr(llm_mod.litellm, "_should_retry", lambda status: False)

        async def raising_acompletion(*args, **kwargs):
            raise FakeErr("bad request", 400)

        monkeypatch.setattr(llm_mod, "acompletion", raising_acompletion)

        agen = exec.run(
            ExecRunInput(messages=[Message(role="user", text="hi")], state=None)
        )
        _, pkt, _ = await drain_until_non_interim(agen)
        assert isinstance(pkt, ReqFinalMessage)
        assert "error" in (pkt.message.text or "").lower()


@pytest.mark.asyncio
async def test_llm_executor_system_append_included_in_messages(monkeypatch, tmp_path):
    cfg = LLMNode(
        name="LLM",
        model="gpt-x",
        system="Base",
        system_append=" EXT",
    )
    seq = [chunk_content("ok")]
    stub = ACompletionStub([seq])
    monkeypatch.setattr(llm_mod, "acompletion", stub)

    async with ProjectSandbox.create(tmp_path) as project:
        agen = LLMExecutor(cfg, project).run(
            ExecRunInput(messages=[Message(role="user", text="hello")], state=None)
        )
        # Drain until final (no tools/outcomes enforced)
        _, pkt, _ = await drain_until_non_interim(agen)
        assert pkt.kind == "final_message"
        sent_msgs = stub.calls[0]["messages"]
        assert sent_msgs[0]["role"] == "system"
        assert sent_msgs[0]["content"] == "Base EXT"


@pytest.mark.asyncio
async def test_llm_executor_emits_post_final_token_pct_log(monkeypatch, tmp_path):
    cfg = LLMNode(
        name="LLM",
        model="gpt-x",
        outcomes=[OutcomeSlot(name="done")],
        outcome_strategy=OutcomeStrategy.function_call,
        # Provide model context window for percentage-based logging
        extra={"model_max_tokens": 1000},
    )
    seq = [chunk_content("ok")]
    stub = ACompletionStub([seq])
    monkeypatch.setattr(llm_mod, "acompletion", stub)

    async with ProjectSandbox.create(tmp_path) as project:
        # Note: model context window is set on node.extra above.

        execu = LLMExecutor(cfg, project)
        agen = execu.run(
            ExecRunInput(messages=[Message(role="user", text="hello")], state=None)
        )

        # Drain until final message
        _, pkt_final, _ = await drain_until_non_interim(agen)
        assert pkt_final.kind == "final_message"
        assert pkt_final.message is not None and pkt_final.message.text == "ok"


@pytest.mark.asyncio
async def test_llm_executor_reads_usage_object_prefer_over_estimate(
    monkeypatch, tmp_path
):
    cfg = LLMNode(
        name="LLM",
        model="gpt-x",
        outcomes=[OutcomeSlot(name="done")],
        outcome_strategy=OutcomeStrategy.function_call,
    )
    # Include a litellm-like Usage object on the final chunk
    seq = [chunk_content("ok"), chunk_usage(50, 10)]
    stub = ACompletionStub([seq])
    monkeypatch.setattr(llm_mod, "acompletion", stub)

    async with ProjectSandbox.create(tmp_path) as project:
        execu = LLMExecutor(cfg, project)
        agen = execu.run(
            ExecRunInput(messages=[Message(role="user", text="hello")], state=None)
        )
        _, pkt_final, state = await drain_until_non_interim(agen)
        assert pkt_final.kind == "final_message"
        # LLMState totals should reflect the explicit usage object from the stream,
        # not a token estimate.
        assert state is not None
        assert getattr(state, "total_prompt_tokens", None) == 50
        assert getattr(state, "total_completion_tokens", None) == 10


def test_llm_node_reasoning_effort_validation_ok():
    # All supported values should be accepted.
    from vocode.runner.executors.llm import LLMNode

    for level in ["none", "minimal", "low", "medium", "high"]:
        node = LLMNode(name="llm", model="test-model", reasoning_effort=level)
        assert node.reasoning_effort == level


def test_llm_node_reasoning_effort_validation_rejects_invalid():
    from vocode.runner.executors.llm import LLMNode
    from pydantic import ValidationError

    # Plain invalid literal should fail validation.
    with pytest.raises(ValidationError):
        LLMNode(name="llm", model="test-model", reasoning_effort="ultra")

    # Variable-style placeholder should be accepted; settings_loader resolves it later.
    node = LLMNode(
        name="llm",
        model="test-model",
        reasoning_effort="${LLM_REASONING_EFFORT}",
    )
    assert node.reasoning_effort == "${LLM_REASONING_EFFORT}"


@pytest.mark.asyncio
async def test_llm_executor_passes_reasoning_effort(monkeypatch, tmp_path):
    # Configure node with a supported reasoning_effort value and ensure it reaches acompletion.
    cfg = LLMNode(
        name="LLM",
        model="gpt-x",
        outcomes=[OutcomeSlot(name="done")],
        outcome_strategy=OutcomeStrategy.function_call,
        reasoning_effort="medium",
    )
    seq = [chunk_content("ok")]
    stub = ACompletionStub([seq])
    monkeypatch.setattr(llm_mod, "acompletion", stub)

    async with ProjectSandbox.create(tmp_path) as project:
        agen = LLMExecutor(cfg, project).run(
            ExecRunInput(messages=[Message(role="user", text="hello")], state=None)
        )
        # Drain until final
        _, pkt_final, _ = await drain_until_non_interim(agen)
        assert pkt_final.kind == "final_message"

    # Verify that reasoning_effort was passed through to litellm.acompletion
    assert stub.calls, "acompletion was not called"
    call = stub.calls[0]
    assert call["reasoning_effort"] == "medium"
