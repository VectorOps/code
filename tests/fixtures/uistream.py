import asyncio
import sys
from pathlib import Path
import shutil
import contextlib

# Add project root to sys.path to allow imports from src
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text.utils import split_lines, fragment_list_width, to_formatted_text

from vocode.state import RunnerStatus
from vocode.ui.terminal.toolbar import build_prompt, build_toolbar
from vocode.ui.terminal.buf import MessageBuffer
from vocode.ui.terminal.helpers import out_fmt_stream, ANSI_CURSOR_DOWN, out
from vocode.ui.terminal import colors

class MockUIState:
    def __init__(self):
        self.selected_workflow_name = "demo-workflow"
        self.current_node_name = "stream_output"
        self.status = RunnerStatus.running
        self.acc_prompt_tokens = 12345
        self.acc_completion_tokens = 6789
        self.acc_cost_dollars = 0.04567
        self.workflow = None

# A long text with markdown and very long lines to test wrapping.
LONG_TEXT = """This is a demonstration of streaming text output with wrapping. The following paragraph is a single long line designed to be over 300 characters to properly test how the terminal handles wrapping for very long, unbroken strings of text. It's important that this line wraps correctly without breaking words and maintains proper formatting across multiple visual lines in the terminal window. Let's see how it performs, this should be more than enough text to trigger multiple wraps on a standard 80-column terminal, and even on much wider displays. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.

Here are some other features:
* A list item.
* Another list item.

And a `code block` within a sentence.

```python
def hello():
    print("Hello, world!")
```

This is the end of the text.
"""

def _finish_stream(stream_buffer: MessageBuffer) -> None:
    if not stream_buffer:
        return
    ft = colors.render_markdown(
        stream_buffer.full_text, prefix=f"{stream_buffer.speaker}: "
    )
    parts = list(to_formatted_text(ft))
    lines = list(split_lines(parts))
    last_line = colors.get_last_non_empty_line(lines) or []
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    last_width = fragment_list_width(last_line)
    wraps = (last_width - 1) // width if (width > 0 and last_width > 0) else 0
    if wraps > 0:
        print(ANSI_CURSOR_DOWN * wraps, end="")
    print()


async def main():
    """
    Shows a prompt and simulates streaming text above it concurrently.
    """
    mock_ui_state = MockUIState()
    session = PromptSession()

    async def stream_and_finish():
        """Simulates streaming text to the terminal."""
        out("--- Starting text stream simulation ---")

        stream_buffer = MessageBuffer(speaker="assistant")
        words = LONG_TEXT.split(" ")

        for word in words:
            # Append word and a space, get the formatted text for the update
            diff = stream_buffer.append(word + " ")
            if diff:
                out_fmt_stream(diff)
            await asyncio.sleep(0.02)  # Simulate network delay/word-by-word generation

        # Finalize the stream output by moving cursor and printing a newline
        _finish_stream(stream_buffer)
        out("\n--- Stream finished. You can now type your response. ---")

    # Run the streaming in the background
    stream_task = asyncio.create_task(stream_and_finish())

    try:
        line = await session.prompt_async(
            lambda: build_prompt(mock_ui_state, None),
            bottom_toolbar=lambda: build_toolbar(mock_ui_state, None),
        )
        out(f"\nYou entered: {line}")
    except (EOFError, KeyboardInterrupt):
        out("\nExiting.")
    finally:
        stream_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stream_task


if __name__ == "__main__":
    asyncio.run(main())
