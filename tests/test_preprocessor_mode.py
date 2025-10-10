import pytest

from vocode.runner.preprocessors.base import register_preprocessor
from vocode.runner.executors.llm import LLMExecutor, LLMNode
from vocode.models import PreprocessorSpec, Mode
from vocode.state import Message

# Register a simple test preprocessor
register_preprocessor(
    name="mark",
    description="append marker",
    func=lambda project, spec, text: (
        f"{(spec.options or {}).get('suffix', '')}{text}"
        if spec.prepend
        else f"{text}{(spec.options or {}).get('suffix', '')}"
    ),
)


class DummyProject:
    # Minimal stub; executor._build_base_messages does not use project.
    def __init__(self):
        self.settings = None
        self.tools = {}
        class _Usage:
            prompt_tokens = 0
            completion_tokens = 0
            cost_dollars = 0.0
        self.llm_usage = _Usage()
    def add_llm_usage(self, **kwargs):
        pass


def test_preprocessor_mode_system():
    cfg = LLMNode(
        name="node1",
        model="dummy-model",
        system="SYS",
        preprocessors=[
            PreprocessorSpec(name="mark", mode=Mode.System, options={"suffix": " [S]"})
        ],
    )
    execu = LLMExecutor(config=cfg, project=DummyProject())
    history = [Message(role="user", text="hello", node=None)]
    msgs = execu._build_base_messages(cfg, history)
    # System mutated
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "SYS [S]"
    # User unchanged
    assert any(m["role"] == "user" and m["content"] == "hello" for m in msgs)


def test_preprocessor_mode_user():
    cfg = LLMNode(
        name="node1",
        model="dummy-model",
        system="SYS",
        preprocessors=[
            PreprocessorSpec(name="mark", mode=Mode.User, options={"suffix": " [U]"})
        ],
    )
    execu = LLMExecutor(config=cfg, project=DummyProject())
    history = [
        Message(role="user", text="first", node=None),
        Message(role="agent", text="assistant reply", node="other"),  # will map to user? ensure role mapping to not user
        Message(role="user", text="second", node=None),
    ]
    msgs = execu._build_base_messages(cfg, history)
    # System unchanged
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "SYS"
    # Last user mutated, earlier user not mutated
    contents = [m["content"] for m in msgs if m["role"] == "user"]
    assert contents[0] == "first"
    assert contents[-1] == "second [U]"


def test_preprocessor_mode_system_prepend():
    cfg = LLMNode(
        name="node1",
        model="dummy-model",
        system="SYS",
        preprocessors=[
            PreprocessorSpec(name="mark", mode=Mode.System, options={"suffix": " [S]"}, prepend=True)
        ],
    )
    execu = LLMExecutor(config=cfg, project=DummyProject())
    history = [Message(role="user", text="hello", node=None)]
    msgs = execu._build_base_messages(cfg, history)
    # System mutated with prefix
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == " [S]SYS"
    # User unchanged
    assert any(m["role"] == "user" and m["content"] == "hello" for m in msgs)


def test_preprocessor_mode_user_prepend():
    cfg = LLMNode(
        name="node1",
        model="dummy-model",
        system="SYS",
        preprocessors=[
            PreprocessorSpec(name="mark", mode=Mode.User, options={"suffix": " [U]"}, prepend=True)
        ],
    )
    execu = LLMExecutor(config=cfg, project=DummyProject())
    history = [
        Message(role="user", text="first", node=None),
        Message(role="agent", text="assistant reply", node="other"),
        Message(role="user", text="second", node=None),
    ]
    msgs = execu._build_base_messages(cfg, history)
    # System unchanged
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "SYS"
    # Last user mutated with prefix, earlier user not mutated
    contents = [m["content"] for m in msgs if m["role"] == "user"]
    assert contents[0] == "first"
    assert contents[-1] == " [U]second"
