from __future__ import annotations

import contextlib
import shutil
from typing import Optional, TYPE_CHECKING
import asyncio

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
from vocode.ui.terminal.buf import MessageBuffer

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession


async def out(*args, **kwargs) -> None:
    def _p():
        print(*args, **kwargs, flush=True)

    try:
        await run_in_terminal(_p)
    except Exception:
        _p()


async def out_fmt(ft) -> None:
    # Print prompt_toolkit AnyFormattedText with our console style.
    def _p():
        print_formatted_text(to_formatted_text(ft), style=colors.get_console_style())

    try:
        # TODO: Do we need run_in_terminal?
        await run_in_terminal(_p)
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


class StreamThrottler:
    """
    Helper to throttle streaming updates to the terminal to avoid flickering.
    It accumulates text chunks and flushes them to a MessageBuffer at a controlled rate
    using a background worker task.
    """

    def __init__(
        self,
        session: "PromptSession",
        speaker: str,
        *,
        interval_s: float = 1.0 / 20.0,
    ) -> None:
        self._session = session
        self._stream_buffer = MessageBuffer(speaker=speaker)
        self._interval_s = interval_s
        self._buffer = ""
        self._buffer_lock = asyncio.Lock()
        self._new_text_event = asyncio.Event()
        self._closed = False
        self._worker_task: asyncio.Task[None] = asyncio.create_task(self._worker())

    @property
    def full_text(self) -> str:
        """Returns the full accumulated text, including unflushed buffer."""
        return self._stream_buffer.full_text + self._buffer

    async def append(self, text: str) -> None:
        """Append text. It will be flushed by the background worker."""
        if not text:
            return
        async with self._buffer_lock:
            self._buffer += text
        self._new_text_event.set()

    async def _worker(self) -> None:
        try:
            while not self._closed:
                await self._new_text_event.wait()
                if self._closed:
                    break
                self._new_text_event.clear()

                await self._flush_buffer()
                await asyncio.sleep(self._interval_s)
        except asyncio.CancelledError:
            # Final flush is handled in close()
            pass

    async def close(self) -> None:
        """Stops the background worker and performs a final flush."""
        if self._closed:
            return
        self._closed = True
        self._new_text_event.set()  # Wake up worker to exit loop
        self._worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker_task
        await self._flush_buffer()  # Final flush

    async def _flush_buffer(self) -> None:
        """Flushes the current buffer to the terminal."""
        text_to_flush = ""
        async with self._buffer_lock:
            if self._buffer:
                text_to_flush = self._buffer
                self._buffer = ""

        if not text_to_flush:
            return

        new_changed_lines, old_changed_lines = self._stream_buffer.append(text_to_flush)
        do_print = bool(new_changed_lines or old_changed_lines)

        if do_print:
            # Shield to prevent cancellation during terminal writes
            await asyncio.shield(
                print_updated_lines(self._session, new_changed_lines, old_changed_lines)
            )

    async def flush(self) -> None:
        """Force flush any buffered text."""
        await self._flush_buffer()
