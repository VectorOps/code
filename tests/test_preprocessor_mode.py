import pytest
from typing import List
from unittest.mock import patch
from vocode.runner.executors.llm.preprocessors.base import register_preprocessor
from vocode.runner.executors.llm import LLMExecutor, LLMNode
from vocode.models import PreprocessorSpec, Mode
from vocode.state import Message


# Faking this import, since it's only used for test setup
from vocode.runner.executors.llm.preprocessors import diff # noqa



def mark_preprocessor(project, spec: PreprocessorSpec, messages: List[Message]) -> List[Message]:
    """A test preprocessor that finds a message and adds a suffix."""
    suffix = (spec.options or {}).get("suffix", "")
    target_message = None

    if spec.mode == Mode.System:
        for msg in messages:
            if msg.role == "system":
                target_message = msg
                break
    elif spec.mode == Mode.User:
        for msg in reversed(messages):
            if msg.role == "user":
                target_message = msg
                break

    if target_message:
        if spec.prepend:
            target_message.text = f"{suffix}{target_message.text}"
        else:
            target_message.text = f"{target_message.text}{suffix}"

    return messages


register_preprocessor(name="mark", description="append marker", func=mark_preprocessor)


class DummyProject:
    def __init__(self):
        pass

    def add_llm_usage(self, **kwargs):
        pass


@pytest.fixture
def base_messages():
    return [
        Message(role="system", text="system prompt"),
        Message(role="user", text="hello"),
        Message(role="user", text="world"),
    ]


def test_preprocessor_mode_system(base_messages):
    """Test system-mode preprocessor appends to system message."""
    node = LLMNode(
        name="test",
        model="dummy-model",
        system="system prompt",
        preprocessors=[
            PreprocessorSpec(name="mark", options={"suffix": " S"}, mode=Mode.System)
        ],
    )
    executor = LLMExecutor(config=node, project=DummyProject())
    # The system message is added by _build_base_messages from the node config,
    # so we only need to pass the user messages.
    result_messages = executor._build_base_messages(node, base_messages[1:])
    assert result_messages[0]["content"] == "system prompt S"
    assert result_messages[1]["content"] == "hello"
    assert result_messages[2]["content"] == "world"


def test_preprocessor_mode_user(base_messages):
    """Test user-mode preprocessor appends to LAST user message."""
    node = LLMNode(
        name="test",
        model="dummy-model",
        system="system prompt",
        preprocessors=[
            PreprocessorSpec(name="mark", options={"suffix": " U"}, mode=Mode.User)
        ],
    )
    executor = LLMExecutor(config=node, project=DummyProject())
    result_messages = executor._build_base_messages(node, base_messages[1:])
    assert result_messages[0]["content"] == "system prompt"
    assert result_messages[1]["content"] == "hello"
    assert result_messages[2]["content"] == "world U"


def test_preprocessor_mode_system_prepend(base_messages):
    """Test system-mode preprocessor prepends to system message."""
    node = LLMNode(
        name="test",
        model="dummy-model",
        system="system prompt",
        preprocessors=[
            PreprocessorSpec(
                name="mark", options={"suffix": "S "}, mode=Mode.System, prepend=True
            )
        ],
    )
    executor = LLMExecutor(config=node, project=DummyProject())
    result_messages = executor._build_base_messages(node, base_messages[1:])
    assert result_messages[0]["content"] == "S system prompt"
    assert result_messages[1]["content"] == "hello"
    assert result_messages[2]["content"] == "world"


def test_preprocessor_multiple_are_applied(base_messages):
    """Test multiple preprocessors are applied in order."""
    node = LLMNode(
        name="test",
        model="dummy-model",
        system="system prompt",
        preprocessors=[
            PreprocessorSpec(name="mark", options={"suffix": " S1"}),
            PreprocessorSpec(name="mark", options={"suffix": " S2"}),
        ],
    )
    executor = LLMExecutor(config=node, project=DummyProject())
    result_messages = executor._build_base_messages(node, base_messages[1:])
    assert result_messages[0]["content"] == "system prompt S1 S2"
    assert result_messages[1]["content"] == "hello"


def test_diff_preprocessor(base_messages):
    """Test diff preprocessor modifies system message."""
    node = LLMNode(
        name="test",
        model="dummy-model",
        system="system prompt",
        preprocessors=[
            PreprocessorSpec(name="diff", options={"format": "test_format"})
        ],
    )
    executor = LLMExecutor(config=node, project=DummyProject())

    with patch(
        "vocode.runner.executors.llm.preprocessors.diff.SUPPORTED_PATCH_FORMATS",
        {"test_format": type("obj", (object,), {"system_prompt": " DIFF"})}
    ):
        result_messages = executor._build_base_messages(node, base_messages[1:])

    assert result_messages[0]["content"] == "system prompt DIFF"
    assert result_messages[1]["content"] == "hello"
    assert result_messages[2]["content"] == "world"
