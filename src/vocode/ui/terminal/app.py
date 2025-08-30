import asyncio
import contextlib
from typing import Optional

import click

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML

from vocode.project import Project
from vocode.ui.base import UIState
from vocode.ui.proto import UIReqRunEvent, UIReqStatus
from vocode.runner.models import (
    ReqMessageRequest,
    ReqToolCall,
    ReqInterimMessage,
    ReqFinalMessage,
)
from vocode.state import Message
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
        print(*args, **kwargs, flush=True)

    def request_exit():
        nonlocal should_exit
        should_exit = True
        with contextlib.suppress(Exception):
            # End the current prompt gracefully
            session.app.exit(result="")

    ctx = CommandContext(ui=ui, out=lambda s: out(s), stop_toggle=stop_toggle, request_exit=request_exit)

    pending_req: Optional[UIReqRunEvent] = None

    async def handle_run_event(req: UIReqRunEvent) -> None:
        nonlocal pending_req
        ev = req.event.event
        if isinstance(ev, ReqInterimMessage) and ev.message:
            speaker = ev.message.role or "assistant"
            out(f"{speaker}: {ev.message.text or ''}")
            return

        if isinstance(ev, ReqMessageRequest):
            pending_req = req
            return

        if isinstance(ev, ReqToolCall):
            out(f"Tool calls requested: {len(ev.tool_calls)}")
            for i, tc in enumerate(ev.tool_calls, start=1):
                out(f"  {i}. {tc.name}({tc.args})")
            pending_req = req
            return

        if isinstance(ev, ReqFinalMessage):
            if ev.message:
                speaker = ev.message.role or "assistant"
                out(f"{speaker}: {ev.message.text or ''}")
            pending_req = req
            return

    async def event_consumer():
        nonlocal pending_req
        while True:
            msg = await ui.recv()
            if isinstance(msg, UIReqStatus):
                reset_interrupt()
                continue
            if isinstance(msg, UIReqRunEvent):
                await handle_run_event(msg)
                continue

    consumer_task = asyncio.create_task(event_consumer())

    out("Type /help for commands.")
    try:
        while True:
            prompt_text = _prompt_text(ui)

            hint = ""
            if pending_req is not None:
                ev = pending_req.event.event
                if isinstance(ev, ReqMessageRequest):
                    hint = "(your message) "
                elif isinstance(ev, ReqToolCall):
                    hint = "(approve? [Y/n]) "
                elif isinstance(ev, ReqFinalMessage):
                    hint = "(Enter to continue, or type a reply) "

            line = await session.prompt_async(prompt_text, key_bindings=kb, default="")
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

                if isinstance(ev, ReqMessageRequest):
                    if text == "":
                        # No response -> runner will reprompt
                        await ui.respond_packet(req_id, None)
                    else:
                        await ui.respond_message(req_id, Message(role="user", text=text))
                    pending_req = None
                    continue

                if isinstance(ev, ReqToolCall):
                    if text.strip().lower() in ("", "y", "yes"):
                        await ui.respond_approval(req_id, True)
                    else:
                        await ui.respond_approval(req_id, False)
                    pending_req = None
                    continue

                if isinstance(ev, ReqFinalMessage):
                    if text.strip() == "":
                        await ui.respond_approval(req_id, True)
                    else:
                        await ui.respond_message(req_id, Message(role="user", text=text))
                    pending_req = None
                    continue

                pending_req = None
                continue

            if text.strip():
                out("Not expecting input. Use /help for commands.")

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
