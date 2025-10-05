import pytest
from typing import List, Tuple

from vocode.ui.terminal.buf import MessageBuffer
from vocode.ui.terminal.colors import render_markdown
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.formatted_text.utils import split_lines


def _ft_to_lines_str(lines: List[List[Tuple[str, str]]]) -> str:
    """Helper to convert prompt_toolkit formatted text lines back to a simple string."""
    return "\n".join("".join(text for _, text in line) for line in lines)


def test_simple_append_and_diff():
    """Test basic appending and that the returned diff is correct."""
    buf = MessageBuffer(speaker="Agent")

    # First chunk includes prefix
    new1, old1 = buf.append("Hello")
    assert buf.full_text == "Hello"
    assert _ft_to_lines_str(new1) == "Agent: Hello\n"
    assert not old1

    # When appending to the same line, the whole line is re-rendered and returned
    # for overwriting in the terminal UI.
    new2, old2 = buf.append(" world")
    assert buf.full_text == "Hello world"
    assert _ft_to_lines_str(new2) == "Agent: Hello world\n"
    assert _ft_to_lines_str(old2) == "Agent: Hello\n"


def test_streaming_accumulation():
    """Ensures that streaming chunks accumulates the full_text correctly."""
    buf = MessageBuffer(speaker="Test")
    chunks = ["This", " is", " a", " test", "\n", "with", " multiple", " parts."]
    expected_text = "".join(chunks)

    for chunk in chunks:
        buf.append(chunk)

    assert buf.full_text == expected_text


def test_streaming_markdown_code_block():
    """
    Simulates streaming a full markdown code block, character by character,
    ensuring the final text is perfectly reconstructed.
    """
    buf = MessageBuffer(speaker="Coder")

    full_text = (
        "Here is some code:\n```python\ndef main():\n    print('Hello')\n```\nDone."
    )

    for char in full_text:
        buf.append(char)

    assert buf.full_text == full_text

    # Also verify final rendered output as a sanity check
    final_render = render_markdown(buf.full_text, prefix=buf._prefix)
    final_render_str = _ft_to_lines_str(list(split_lines(to_formatted_text(final_render))))

    # Check that key parts are present
    assert "Coder: Here is some code:" in final_render_str
    assert "def main():" in final_render_str
    assert "print('Hello')" in final_render_str
    assert "Done." in final_render_str
    # The markdown parser will strip the final newline from a code block if it's the last thing
    assert final_render_str.count("```") == 2


def test_empty_and_whitespace_chunks():
    """Test that empty or whitespace-only chunks don't break the buffer."""
    buf = MessageBuffer(speaker="Agent")

    # Empty chunk
    new1, old1 = buf.append("")
    assert buf.full_text == ""
    assert not new1  # Expect empty list
    assert not old1

    # Whitespace chunk
    buf.append("   ")
    assert buf.full_text == "   "

    # Leading content followed by empty
    buf.append("text")
    new2, old2 = buf.append("")
    assert buf.full_text == "   text"
    assert not new2
    assert not old2
