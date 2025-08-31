import pytest
from typing import List, Tuple

from vocode.ui.terminal.buf import MessageBuffer
from vocode.ui.terminal.colors import render_markdown
from prompt_toolkit.formatted_text import to_formatted_text


def _ft_to_str(ft: List[Tuple[str, str]]) -> str:
    """Helper to convert prompt_toolkit formatted text list back to a simple string."""
    return "".join(text for _, text in ft)


def test_simple_append_and_diff():
    """Test basic appending and that the returned diff is correct."""
    buf = MessageBuffer(speaker="Agent")

    # First chunk includes prefix
    diff1 = buf.append("Hello")
    assert buf.full_text == "Hello"
    assert _ft_to_str(list(to_formatted_text(diff1))) == "Agent: Hello"

    # When appending to the same line, the whole line is re-rendered and returned
    # for overwriting in the terminal UI.
    diff2 = buf.append(" world")
    assert buf.full_text == "Hello world"
    assert _ft_to_str(list(to_formatted_text(diff2))) == "Agent: Hello world"


@pytest.mark.parametrize(
    "initial_text, chunk, expected_full_text, expected_diff_suffix",
    [
        # --- Case 1: Chunk has NO trailing newline ---
        # Pygments might add one, so our logic should remove it.
        ("", "Hello", "Hello", "Agent: Hello"),
        ("Line1\n", "Line2", "Line1\nLine2", "Line2"),

        # --- Case 2: Chunk has ONE trailing newline ---
        # This is the standard case, should be preserved.
        ("", "Hello\n", "Hello\n", "Agent: Hello\n"),
        ("Line1\n", "Line2\n", "Line1\nLine2\n", "Line2\n"),

        # --- Case 3: Chunk has MULTIPLE trailing newlines ---
        # Pygments will only add one, so our logic must add the rest back.
        ("", "Hello\n\n", "Hello\n\n", "Agent: Hello\n\n"),
        ("Line1\n", "Line2\n\n\n", "Line1\nLine2\n\n\n", "Line2\n\n\n"),

        # --- Case 4: Chunk is ONLY newlines ---
        ("", "\n", "\n", "Agent: \n"),
        ("Hi", "\n\n", "Hi\n\n", "\n\n"),

        # --- Case 5: Chunk with internal and multiple trailing newlines ---
        ("Start.", " Mid\nEnd\n\n", "Start. Mid\nEnd\n\n", " Mid\nEnd\n\n"),
    ],
)
def test_trailing_newline_handling(initial_text, chunk, expected_full_text, expected_diff_suffix):
    """
    Tests the core logic of adjusting trailing newlines.
    This simulates a scenario where the underlying `render_markdown` (with Pygments)
    might incorrectly add a single newline, and our buffer logic must correct it
    to match the number of trailing newlines in the raw source `chunk`.
    """
    buf = MessageBuffer(speaker="Agent")
    if initial_text:
        buf.append(initial_text) # Pre-populate buffer

    diff = buf.append(chunk)

    assert buf.full_text == expected_full_text

    # The returned diff should have the correct number of trailing newlines.
    # We check the suffix because the diffing logic might return more than just the chunk.
    assert _ft_to_str(list(to_formatted_text(diff))).endswith(expected_diff_suffix)


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
    
    full_text = "Here is some code:\n```python\ndef main():\n    print('Hello')\n```\nDone."
    
    for char in full_text:
        buf.append(char)

    assert buf.full_text == full_text

    # Also verify final rendered output as a sanity check
    final_render = render_markdown(buf.full_text, prefix=buf._prefix)
    final_render_str = _ft_to_str(list(to_formatted_text(final_render)))
    
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
    diff1 = buf.append("")
    assert buf.full_text == ""
    assert not diff1 # Expect empty list

    # Whitespace chunk
    buf.append("   ")
    assert buf.full_text == "   "
    
    # Leading content followed by empty
    buf.append("text")
    diff2 = buf.append("")
    assert buf.full_text == "   text"
    assert not diff2
