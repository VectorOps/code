from __future__ import annotations

import shutil
from typing import Optional

from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.formatted_text.utils import split_lines, fragment_list_width
from prompt_toolkit.shortcuts import print_formatted_text

from vocode.ui.terminal import colors
from vocode.ui.base import UIState
from vocode.ui.proto import UIPacketEnvelope, UIPacketRunInput
from vocode.runner.models import RunInput, RespPacket, RespMessage, RespApproval
from vocode.state import Message


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


async def print_updated_lines(session, new_lines: list, old_lines: list) -> None:
    """
    Overwrites previous output with new lines.
    Calculates visual lines for old content to move cursor up, then prints new content.
    """
    width = shutil.get_terminal_size(fallback=(80, 24)).columns

    visual_lines_up = 0

    app = session.app

    for line in old_lines:
        line_width = fragment_list_width(line)
        wraps = (line_width - 1) // width if (width > 0 and line_width > 0) else 0
        visual_lines_up += 1 + wraps

    def _printer():
        print_formatted_text("\r", end="", flush=True)
        if visual_lines_up > 0:
            app.output.cursor_up(visual_lines_up)

        for line in new_lines:
            print_formatted_text(
                to_formatted_text(line),
                style=colors.get_console_style(),
                flush=True,
            )

    await run_in_terminal(_printer)


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
