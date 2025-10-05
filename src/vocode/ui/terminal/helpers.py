from __future__ import annotations

import shutil
from typing import Optional

from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.formatted_text.utils import split_lines, fragment_list_width
from prompt_toolkit.shortcuts import print_formatted_text

from vocode.ui.terminal import colors
from vocode.ui.base import UIState
from vocode.ui.proto import UIPacketEnvelope, UIPacketRunInput
from vocode.runner.models import RunInput, RespPacket, RespMessage, RespApproval
from vocode.state import Message

# ANSI escape sequences used by streaming output and finalization
ANSI_CARRIAGE_RETURN_AND_CLEAR_TO_EOL = "\r\x1b[K"
ANSI_CURSOR_UP = "\x1b[1A"
ANSI_CURSOR_DOWN = "\x1b[1B"


def out(*args, **kwargs) -> None:
    def _p():
        print(*args, **kwargs, flush=True)

    try:
        run_in_terminal(_p)
    except Exception:
        _p()


def out_fmt(ft) -> None:
    # Print prompt_toolkit AnyFormattedText with our console style.
    def _p():
        print_formatted_text(to_formatted_text(ft), style=colors.get_console_style())

    try:
        run_in_terminal(_p)
    except Exception:
        # Fallback: degrade to plain text
        print(to_formatted_text(ft).text, flush=True)


def out_fmt_stream(ft) -> None:
    """
    Print prompt_toolkit AnyFormattedText without a trailing newline,
    using a carriage return to overwrite the current line.
    Handles wrapped lines by splitting on newlines and tracking visual wraps.
    """
    parts = list(ft)

    # Clear the current line first (we're overwriting the streamed line).
    print(ANSI_CARRIAGE_RETURN_AND_CLEAR_TO_EOL, end="")

    # Split into lines using prompt_toolkit helper.
    lines = list(split_lines(parts))

    # Print all full lines (all but last) with a newline.
    for line in lines[:-1]:
        print_formatted_text(to_formatted_text(line), style=colors.get_console_style())

    # Prepare last line.
    last_line = lines[-1] if lines else []

    # Compute how many wraps the last line will take.
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    last_width = fragment_list_width(last_line)
    wraps = (last_width - 1) // width if (width > 0 and last_width > 0) else 0

    # Print the last line without a trailing newline (to keep streaming).
    if last_line:
        print_formatted_text(
            to_formatted_text(last_line), style=colors.get_console_style(), end=""
        )

    # If the last line wrapped, move the cursor back up so the next update
    # will overwrite from the first visual line of this wrapped block.
    if wraps > 0:
        print(ANSI_CURSOR_UP * wraps, end="")


async def respond_packet(
    ui: UIState, source_msg_id: int, packet: Optional[RespPacket]
) -> None:
    inp = RunInput(response=packet) if packet is not None else RunInput(response=None)
    await ui.send(
        UIPacketEnvelope(
            msg_id=ui.next_client_msg_id(),
            source_msg_id=source_msg_id,
            payload=UIPacketRunInput(input=inp),
        )
    )


async def respond_message(ui: UIState, source_msg_id: int, message: Message) -> None:
    await respond_packet(ui, source_msg_id, RespMessage(message=message))


async def respond_approval(ui: UIState, source_msg_id: int, approved: bool) -> None:
    await respond_packet(ui, source_msg_id, RespApproval(approved=approved))
