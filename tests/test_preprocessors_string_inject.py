import vocode.runner.executors.llm.preprocessors  # noqa: F401

from vocode.models import Mode, PreprocessorSpec
from vocode.runner.executors.llm.preprocessors.base import apply_preprocessors
from vocode.state import Message


def test_string_inject_creates_system_message_when_empty():
    spec = PreprocessorSpec(
        name="string_inject",
        options={"text": "Hello world"},
        mode=Mode.System,
        prepend=False,
    )

    result = apply_preprocessors([spec], project=None, messages=[])

    assert len(result) == 1
    msg = result[0]
    assert msg.role == "system"
    assert msg.text == "Hello world"


def test_string_inject_appends_to_last_user_message():
    initial_messages = [
        Message(role="system", text="System base"),
        Message(role="user", text="User base"),
    ]

    spec = PreprocessorSpec(
        name="string_inject",
        options={"text": "extra context"},
        mode=Mode.User,
        prepend=False,
    )

    result = apply_preprocessors([spec], project=None, messages=initial_messages)

    assert len(result) == 2
    system_msg, user_msg = result
    assert system_msg.text == "System base"
    assert user_msg.role == "user"
    assert user_msg.text == "User base\n\nextra context"


def test_string_inject_prepends_and_deduplicates():
    messages = [Message(role="system", text="Base system")]

    spec = PreprocessorSpec(
        name="string_inject",
        options={"text": "Injected"},
        mode=Mode.System,
        prepend=True,
    )

    result1 = apply_preprocessors([spec], project=None, messages=messages)
    assert result1[0].text == "Injected\n\nBase system"

    # Applying again should not duplicate the injected text
    result2 = apply_preprocessors([spec], project=None, messages=result1)
    assert result2[0].text == "Injected\n\nBase system"


def test_string_inject_uses_custom_separator():
    messages = [Message(role="user", text="base")]

    spec = PreprocessorSpec(
        name="string_inject",
        options={"text": "ctx", "separator": " | "},
        mode=Mode.User,
        prepend=False,
    )

    result = apply_preprocessors([spec], project=None, messages=messages)
    assert result[0].text == "base | ctx"