import vocode.runner.executors.llm.preprocessors  # noqa: F401

from vocode.models import Mode, PreprocessorSpec
from vocode.runner.executors.llm.preprocessors.base import apply_preprocessors
from vocode.state import Message

def test_system_details_injects_os_shell(monkeypatch):
    # Ensure module is imported so we can patch internals.
    import vocode.runner.executors.llm.preprocessors.system_details  # noqa: F401
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.delenv("COMSPEC", raising=False)
    monkeypatch.setattr(
        "vocode.runner.executors.llm.preprocessors.system_details.platform.system",
        lambda: "TestOS",
    )
    monkeypatch.setattr(
        "vocode.runner.executors.llm.preprocessors.system_details.platform.release",
        lambda: "1.0",
    )

    spec = PreprocessorSpec(
        name="system_details",
        options={},
        mode=Mode.System,
        prepend=False,
    )

    result = apply_preprocessors([spec], project=None, messages=[])

    assert len(result) == 1
    msg = result[0]
    assert msg.role == "system"
    text = msg.text

    assert "System details:" in text
    assert "- OS: TestOS 1.0" in text
    assert "- Shell: /bin/zsh" in text


def test_system_details_user_mode_prepends_and_deduplicates():
    messages = [
        Message(role="system", text="System base"),
        Message(role="user", text="First"),
        Message(role="user", text="Second"),
    ]

    spec = PreprocessorSpec(
        name="system_details",
        options={"header": "Env info"},
        mode=Mode.User,
        prepend=True,
    )

    result1 = apply_preprocessors([spec], project=None, messages=messages)

    assert len(result1) == 3
    system_msg, first_user, last_user = result1

    assert system_msg.text == "System base"
    assert first_user.text == "First"

    # Should prepend details before existing last user message
    assert "Env info" in last_user.text
    assert "Second" in last_user.text
    assert last_user.text.startswith("Env info")

    # Applying again should not duplicate the injected block
    result2 = apply_preprocessors([spec], project=None, messages=result1)
    assert result2[2].text == last_user.text
