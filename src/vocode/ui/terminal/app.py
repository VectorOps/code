import asyncio
import contextlib
from typing import Optional

import click

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.patch_stdout import patch_stdout

from vocode.project import Project
from vocode.ui.base import UIState
from vocode.ui.proto import UIReqRunEvent, UIReqStatus, UI_PACKET_RUN_EVENT, UI_PACKET_STATUS
from vocode.runner.models import (
    ReqMessageRequest,
    ReqToolCall,
    ReqInterimMessage,
    ReqFinalMessage,
    PACKET_MESSAGE_REQUEST,
    PACKET_TOOL_CALL,
    PACKET_MESSAGE,
    PACKET_FINAL_MESSAGE,
)
from vocode.state import Message, RunnerStatus
from vocode.ui.terminal.commands import CommandContext, run as run_command

def _prompt_text(ui: UIState) -> HTML:
    wf = ui.selected_workflow_name or "-"
    node = ui.current_node_name or "-"
    status = getattr(ui.status, "value", str(ui.status))
    return HTML(f"<b>{wf}</b>@<ansicyan>{node}</ansicyan> [{status}]> ")

async def run_terminal(project: Project) -> None:
    ui = UIState(project)
    session = PromptSession()
    kb = KeyBindings()
    should_exit = False

    change_event = asyncio.Event()

    interrupt_count = {"n": 0}
    def reset_interrupt():
        interrupt_count["n"] = 0

    async def stop_toggle():
        n = interrupt_count["n"]
        interrupt_count["n"] = n + 1
        await (ui.stop() if n == 0 else ui.cancel())

    @kb.add("c-c")
    def _(event):
        asyncio.create_task(stop_toggle())

    def out(*args, **kwargs):
        def _p():
            print(*args, **kwargs, flush=True)
        try:
            run_in_terminal(_p)
        except Exception:
            _p()

    def request_exit():
        nonlocal should_exit
        should_exit = True
        with contextlib.suppress(Exception):
            # End the current prompt gracefully
            session.app.exit(result="")

    ctx = CommandContext(ui=ui, out=lambda s: out(s), stop_toggle=stop_toggle, request_exit=request_exit)

    pending_req: Optional[UIReqRunEvent] = None

    stream_speaker: Optional[str] = None
    stream_buf: str = ""

    def _finish_stream():
        nonlocal stream_buf, stream_speaker
        if stream_buf:
            # End the in-progress line cleanly.
            out()
            stream_buf = ""
            stream_speaker = None

    def _toolbar():
        nonlocal pending_req
        hint = ""
        if pending_req is not None:
            ev = pending_req.event.event
            if ev.kind == PACKET_MESSAGE_REQUEST:
                hint = "(your message)"
            elif ev.kind == PACKET_TOOL_CALL:
                hint = "(approve? [Y/n])"
            elif ev.kind == PACKET_FINAL_MESSAGE:
                hint = "(Enter to continue, or type a reply)"
        return HTML(hint) if hint else None

    async def handle_run_event(req: UIReqRunEvent) -> None:
        nonlocal pending_req, stream_speaker, stream_buf
        ev = req.event.event

        if ev.kind == PACKET_MESSAGE and ev.message:
            speaker = ev.message.role or "assistant"
            text = ev.message.text or ""
            # Start a new stream line if speaker changed or no active stream.
            if stream_speaker != speaker:
                _finish_stream()
                out(f"{speaker}: ", end="")
                stream_speaker = speaker
            # Append chunk and print without newline.
            stream_buf += text
            out(text, end="")
            return

        if ev.kind == PACKET_MESSAGE_REQUEST:
            _finish_stream()
            pending_req = req
            return

        if ev.kind == PACKET_TOOL_CALL:
            _finish_stream()
            out(f"Tool calls requested: {len(ev.tool_calls)}")
            for i, tc in enumerate(ev.tool_calls, start=1):
                out(f"  {i}. {tc.name}({tc.arguments})")
            pending_req = req
            return

        if ev.kind == PACKET_FINAL_MESSAGE:
            if ev.message:
                speaker = ev.message.role or "assistant"
                text = ev.message.text or ""
                if stream_speaker == speaker and stream_buf and stream_buf == text:
                    # Final equals accumulated interim; just terminate the line.
                    _finish_stream()
                else:
                    # End any partial stream and print the final message.
                    _finish_stream()
                    out(f"{speaker}: {text}")
            pending_req = req
            return

    async def event_consumer():
        nonlocal pending_req
        while True:
            msg = await ui.recv()
            if msg.kind == UI_PACKET_STATUS:
                reset_interrupt()
                change_event.set()
                continue
            if msg.kind == UI_PACKET_RUN_EVENT:
                await handle_run_event(msg)  # type: ignore[arg-type]
                change_event.set()
                continue

    consumer_task = asyncio.create_task(event_consumer())

    out("Type /help for commands.")
    try:
        with patch_stdout():
            while True:
                # Wait until we should show a prompt:
                while ui.is_active() and pending_req is None:
                    await change_event.wait()
                    change_event.clear()

                # Drop any queued keystrokes (e.g., stray Enters) and clear buffer before showing prompt
                with contextlib.suppress(Exception):
                    # Clear prompt buffer text/cursor/history state
                    session.default_buffer.reset()
                    # Flush any pending keys in the input queue
                    if hasattr(session, "app") and session.app is not None:
                        session.app.input.flush_keys()

                line = await session.prompt_async(
                    lambda: _prompt_text(ui),
                    key_bindings=kb,
                    default="",
                    bottom_toolbar=_toolbar,
                )
                text = line.rstrip("\n")

                if text.startswith("/"):
                    handled = await run_command(text, ctx)
                    if should_exit:
                        break
                    if not handled:
                        out("Unknown command. Type /help")
                    continue

                if pending_req is not None:
                    ev = pending_req.event.event
                    req_id = pending_req.req_id

                    if ev.kind == PACKET_MESSAGE_REQUEST:
                        if text == "":
                            await ui.respond_packet(req_id, None)
                        else:
                            await ui.respond_message(req_id, Message(role="user", text=text))
                        pending_req = None
                        continue

                    if ev.kind == PACKET_TOOL_CALL:
                        if text.strip().lower() in ("", "y", "yes"):
                            await ui.respond_approval(req_id, True)
                        else:
                            await ui.respond_approval(req_id, False)
                        pending_req = None
                        continue

                    if ev.kind == PACKET_FINAL_MESSAGE:
                        if text.strip() == "":
                            await ui.respond_approval(req_id, True)
                        else:
                            await ui.respond_message(req_id, Message(role="user", text=text))
                        pending_req = None
                        continue

                # No pending request here: runner is active, so hide prompt until a new input request or status/event arrives.
                continue

    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        if ui.is_active():
            await ui.stop()
        consumer_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer_task

@click.command()
@click.argument("project_path", type=click.Path(exists=True, file_okay=False, path_type=str))
def main(project_path: str) -> None:
    project = Project.from_base_path(project_path)
    asyncio.run(run_terminal(project))

if __name__ == "__main__":
    main()
