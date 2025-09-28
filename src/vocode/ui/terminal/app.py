import asyncio
import contextlib
import signal
from typing import Optional, Union
import shutil
from pathlib import Path

import click

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.formatted_text.utils import split_lines, fragment_list_width
from prompt_toolkit.history import FileHistory
from prompt_toolkit.enums import EditingMode
from vocode.ui.terminal import colors
from vocode.ui.terminal.completer import TerminalCompleter
from vocode.ui.terminal.ac_client import make_canned_provider

from vocode.project import Project
from vocode.ui.terminal.buf import MessageBuffer
from vocode.ui.base import UIState
from vocode.ui.proto import (
    UIPacketEnvelope,
    UIPacketRunEvent,
    UIPacketCustomCommands,
    UIPacketCommandResult,
    UIPacketRunCommand,
    UIPacketRunInput,
    PACKET_RUN_EVENT,
    PACKET_STATUS,
    PACKET_CUSTOM_COMMANDS,
    PACKET_COMMAND_RESULT,
    PACKET_RUN_COMMAND,
)
from vocode.ui.rpc import RpcHelper
from vocode.runner.models import (
    ReqMessageRequest,
    ReqToolCall,
    ReqInterimMessage,
    ReqFinalMessage,
    ReqMessageRequest,
    ReqToolCall,
    ReqInterimMessage,
    ReqFinalMessage,
    RespMessage,
    RespApproval,
    PACKET_MESSAGE_REQUEST,
    PACKET_TOOL_CALL,
    PACKET_MESSAGE,
    PACKET_FINAL_MESSAGE,
    PACKET_LOG,
    RunInput,
    RespPacket,
)
from vocode.state import Message, RunnerStatus, LogLevel
from vocode.ui.terminal.toolbar import (
    build_prompt,
    build_toolbar,
    _current_node_confirmation,
)
from vocode.models import Confirmation
from vocode.ui.terminal.commands import (
    CommandContext,
    Commands,
    register_default_commands,
)

# ANSI escape sequence for carriage return and clearing from cursor to the end of the line.
# This is used for overwriting the current line during streaming output.
ANSI_CARRIAGE_RETURN_AND_CLEAR_TO_EOL = "\r\x1b[K"

# ANSI escape sequence for moving the cursor up one line.
ANSI_CURSOR_UP = "\x1b[1A"


LOG_LEVEL_ORDER = {
    LogLevel.debug: 0,
    LogLevel.info: 1,
    LogLevel.warning: 2,
    LogLevel.error: 3,
}


def out(*args, **kwargs):
    def _p():
        print(*args, **kwargs, flush=True)

    try:
        run_in_terminal(_p)
    except Exception:
        _p()


def out_fmt(ft):
    """
    Print prompt_toolkit AnyFormattedText with our console style.
    """

    def _p():
        print_formatted_text(to_formatted_text(ft), style=colors.get_console_style())

    try:
        run_in_terminal(_p)
    except Exception:
        # Fallback: degrade to plain text
        print(to_formatted_text(ft).text, flush=True)


def out_fmt_stream(ft):
    """
    Print prompt_toolkit AnyFormattedText without a trailing newline,
    using a carriage return to overwrite the current line.
    Handles wrapped lines correctly by splitting on newlines, printing
    full lines, and keeping the cursor positioned at the start of the
    last (potentially wrapped) line.
    """
    # Normalize to a list of (style, text) tuples.
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


async def run_terminal(project: Project) -> None:
    ui = UIState(project)

    rpc = RpcHelper(ui.send, "TerminalApp", id_generator=ui.next_client_msg_id)
    ac_factory = lambda name: make_canned_provider(rpc, name)

    ui_cfg = project.settings.ui if project.settings else None
    multiline = True if ui_cfg is None else bool(ui_cfg.multiline)
    editing_mode = None
    if ui_cfg and ui_cfg.edit_mode:
        mode = str(ui_cfg.edit_mode).lower()
        if mode in ("vi", "vim"):
            editing_mode = EditingMode.VI
        elif mode == "emacs":
            editing_mode = EditingMode.EMACS

    # Initialize commands early so the completer can reference them.
    commands = register_default_commands(Commands(), ui, ac_factory=ac_factory)
    completer = TerminalCompleter(ui, commands)

    try:
        hist_dir = project.base_path / ".vocode"
        hist_dir.mkdir(parents=True, exist_ok=True)
        hist_path = hist_dir / "data" / "terminal_history"
        kwargs = {
            "history": FileHistory(str(hist_path)),
            "multiline": multiline,
            "completer": completer,
        }
        if editing_mode is not None:
            kwargs["editing_mode"] = editing_mode
        session = PromptSession(**kwargs)
    except Exception:
        # Fall back to in-memory history if anything goes wrong
        kwargs = {"multiline": multiline, "completer": completer}
        if editing_mode is not None:
            kwargs["editing_mode"] = editing_mode
        session = PromptSession(**kwargs)
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

    # Register a SIGINT handler that triggers stop_toggle so Ctrl-C works
    # even when not focused on prompt_toolkit. Save the previous handler so
    # it can be restored on exit.
    old_sigint = None
    try:
        old_sigint = signal.getsignal(signal.SIGINT)

        def _sigint_handler(signum, frame):
            import asyncio, traceback

            for t in asyncio.all_tasks():
                print(
                    t,
                    t.get_coro(),
                    "".join(
                        traceback.format_stack(
                            sys._current_frames()[t.get_loop()._thread_id]
                        )
                    ),
                )

            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(lambda: asyncio.create_task(stop_toggle()))
            except RuntimeError:
                # No running loop; attempt to create the task directly (best-effort).
                try:
                    asyncio.create_task(stop_toggle())
                except Exception:
                    pass

        signal.signal(signal.SIGINT, _sigint_handler)
    except Exception:
        old_sigint = None

    def request_exit():
        nonlocal should_exit
        should_exit = True
        with contextlib.suppress(Exception):
            # End the current prompt gracefully
            session.app.exit(result="")

    ctx = CommandContext(
        ui=ui,
        out=lambda s: out(s),
        stop_toggle=stop_toggle,
        request_exit=request_exit,
        rpc=rpc,
    )

    pending_req_env: Optional[UIPacketEnvelope] = None
    pending_cmd: Optional[str] = None
    # Track dynamic custom CLI commands to avoid affecting built-ins
    dynamic_cli_commands: set[str] = set()
    queued_resp: Optional[Union[RespMessage, RespApproval]] = None

    stream_buffer: Optional[MessageBuffer] = None

    def _finish_stream():
        nonlocal stream_buffer
        if stream_buffer:
            out("")  # Newline to finish the streamed line
            stream_buffer = None

    # ----------------------------
    # Packet handlers (extracted)
    # ----------------------------
    async def handle_custom_commands_packet(cd: UIPacketCustomCommands) -> None:
        # Register added commands as proxies without overriding built-ins
        existing_names = {c.name for c in commands.list_commands()}
        for c in cd.added:
            cli_name = f"/{c.name}"
            # Skip if a non-dynamic command already exists (do not override globals)
            if cli_name in existing_names and cli_name not in dynamic_cli_commands:
                continue
            # If previously registered dynamically, remove first
            if cli_name in dynamic_cli_commands:
                commands.unregister(cli_name)
                dynamic_cli_commands.discard(cli_name)

            # Create proxy handler
            async def _proxy(ctx: CommandContext, args: list[str], *, _cname=c.name):
                nonlocal pending_cmd
                pending_cmd = _cname
                change_event.set()
                try:
                    res = await rpc.call(UIPacketRunCommand(name=_cname, input=args))
                    if res is None:
                        return
                    if res.kind != PACKET_COMMAND_RESULT:
                        ctx.out(
                            f"Command '{_cname}' failed: unexpected response {res.kind}"
                        )
                        return

                    if res.ok:
                        if res.output:
                            ctx.out(res.output)
                    else:
                        ctx.out(
                            f"Command '{_cname}' failed: {res.error or 'unknown error'}"
                        )
                except Exception as e:
                    ctx.out(f"Error executing command '{_cname}': {e}")
                finally:
                    pending_cmd = None
                    change_event.set()

            commands.register(
                cli_name,
                c.help or "",
                c.usage,
                (ac_factory(c.autocompleter) if c.autocompleter else None),
            )(
                _proxy
            )  # type: ignore[arg-type]
            dynamic_cli_commands.add(cli_name)
            existing_names.add(cli_name)
        # Unregister removed commands only if they were dynamically added
        for name in cd.removed:
            cli_name = f"/{name}"
            if cli_name in dynamic_cli_commands:
                commands.unregister(cli_name)
                dynamic_cli_commands.discard(cli_name)
        change_event.set()

    async def handle_run_event(envelope: UIPacketEnvelope) -> None:
        nonlocal pending_req_env, stream_buffer, queued_resp
        req_payload = envelope.payload
        assert req_payload.kind == PACKET_RUN_EVENT
        ev = req_payload.event.event

        # Debug/log messages from executors: print immediately without requesting input
        if ev.kind == PACKET_LOG:
            _finish_stream()
            # Determine configured log level
            cfg_log_level = LogLevel.info
            if ui.project.settings and ui.project.settings.ui:
                cfg_log_level = ui.project.settings.ui.log_level

            # Determine message log level, default to info
            msg_level = ev.level or LogLevel.info

            # Filter messages below configured level
            if LOG_LEVEL_ORDER[msg_level] < LOG_LEVEL_ORDER[cfg_log_level]:
                return

            level_str = msg_level.value
            prefix = f"[{level_str}] "
            out(prefix + ev.text)
            return

        if ev.kind == PACKET_MESSAGE and ev.message:
            speaker = ev.message.role or "assistant"
            text = ev.message.text or ""
            if not stream_buffer or stream_buffer.speaker != speaker:
                _finish_stream()
                stream_buffer = MessageBuffer(speaker=speaker)
            diff = stream_buffer.append(text)
            if diff:
                out_fmt_stream(diff)
            return

        if ev.kind == PACKET_MESSAGE_REQUEST:
            _finish_stream()
            pending_req_env = envelope if req_payload.event.input_requested else None
            # Auto-respond if we have a queued response
            if pending_req_env is not None and queued_resp is not None:
                await respond_packet(ui, pending_req_env.msg_id, queued_resp)
                queued_resp = None
                pending_req_env = None
            return

        if ev.kind == PACKET_TOOL_CALL:
            _finish_stream()
            out(f"Tool calls requested: {len(ev.tool_calls)}")
            for i, tc in enumerate(ev.tool_calls, start=1):
                out(f"  {i}. {tc.name} arguments:")
                out_fmt(colors.render_json(tc.arguments))
            pending_req_env = envelope if req_payload.event.input_requested else None
            # Auto-respond if we have a queued response (typically approval)
            if pending_req_env is not None and queued_resp is not None:
                await respond_packet(ui, pending_req_env.msg_id, queued_resp)
                queued_resp = None
                pending_req_env = None
            return

        if ev.kind == PACKET_FINAL_MESSAGE:
            if stream_buffer:
                # We have been streaming, so the message is already on screen.
                # Just add a newline and clear.
                _finish_stream()
            elif ev.message:
                # No streaming occurred, print the final message at once.
                speaker = ev.message.role or "assistant"
                text = ev.message.text or ""
                out_fmt(colors.render_markdown(text, prefix=f"{speaker}: "))

            pending_req_env = envelope if req_payload.event.input_requested else None

            # Auto-respond if we have a queued response (approval or user message)
            if pending_req_env is not None and queued_resp is not None:
                await respond_packet(ui, pending_req_env.msg_id, queued_resp)
                queued_resp = None
                pending_req_env = None
            return

    async def event_consumer():
        nonlocal pending_req_env, pending_cmd
        while True:
            try:
                envelope = await ui.recv()
                if rpc.handle_response(envelope):
                    change_event.set()
                    continue

                msg = envelope.payload
                if msg.kind == PACKET_STATUS:
                    reset_interrupt()
                    change_event.set()
                    continue
                if msg.kind == PACKET_RUN_EVENT:
                    await handle_run_event(envelope)
                    change_event.set()
                    continue
                if msg.kind == PACKET_CUSTOM_COMMANDS:
                    await handle_custom_commands_packet(msg)
                    continue
            except Exception as ex:
                import traceback

                # TODO: Propagate up
                print("Exception", traceback.format_exc())
    # Show startup banner (configurable)
    show_banner = True
    if ui_cfg is not None:
        show_banner = ui_cfg.show_banner

    if show_banner:
        banner_lines = [
            r" _      ____  __   _____  ___   ___   ___   ___   __       __    ___   ___   ____ ",
            r"\ \  / | |_  / /`   | |  / / \ | |_) / / \ | |_) ( (`     / /`  / / \ | | \ | |_  ",
            r" \_\/  |_|__ \_\_,  |_|  \_\_/ |_| \ \_\_/ |_|   _)_)     \_\_, \_\_/ |_|_/ |_|__ ",
        ]

        colors_fg = [
            "ansimagenta",
            "ansiblue",
            "ansicyan",
            "ansigreen",
            "ansiyellow",
            "ansired",
        ]
        # Build prompt_toolkit fragments to preserve spacing/alignment.
        fragments = []
        for line, fg in zip(banner_lines, colors_fg):
            fragments.append((f"fg:{fg}", line + "\n"))
        # Add a blank line and the help hint as part of the same terminal write to avoid reordering.
        fragments.append(("", "\n"))
        fragments.append(("", "Type /help for commands.\n"))
        out_fmt(fragments)
    else:
        out("Type /help for commands.")

    # Start event consumer after initial banner/help to avoid interleaved output on startup.
    consumer_task = asyncio.create_task(event_consumer())

    try:
        while True:
            # Wait until we should show a prompt:
            while (pending_cmd is not None) or (
                ui.is_active() and pending_req_env is None
            ):
                await change_event.wait()
                change_event.clear()

            prompt_payload = (
                pending_req_env.payload
                if pending_req_env and pending_req_env.payload.kind == PACKET_RUN_EVENT
                else None
            )
            line = await session.prompt_async(
                lambda: build_prompt(ui, prompt_payload),
                key_bindings=kb,
                default="",
                bottom_toolbar=lambda: build_toolbar(ui, prompt_payload),
            )
            text = line.rstrip("\n")

            if text.startswith("/"):
                handled = await commands.run(text, ctx)
                if should_exit:
                    break
                if not handled:
                    out("Unknown command. Type /help")
                continue

            # If stopped and no pending request, treat input as a replacement for the last input boundary.
            if pending_req_env is None and ui.status == RunnerStatus.stopped:
                t = text.strip()
                # Build a queued response:
                # - empty => implicit approval (continue)
                # - y/n   => explicit approval
                # - other => user message
                if t == "":
                    queued_resp = RespApproval(approved=True)
                elif t.lower() in ("y", "yes", "n", "no"):
                    approved = t.lower() in ("y", "yes")
                    queued_resp = RespApproval(approved=approved)
                else:
                    queued_resp = RespMessage(message=Message(role="user", text=text))

                # Rewind one retriable history step and restart; next input boundary auto-receives queued_resp.
                try:
                    await ui.replace_last_user_input(queued_resp)
                except Exception as e:
                    out(f"Failed to prepare replacement: {e}")
                    queued_resp = None
                    continue

                try:
                    await ui.restart()
                except Exception as e:
                    out(f"Failed to restart: {e}")
                    queued_resp = None
                # Do not prompt further here; event_consumer will deliver the next request and auto-reply.
                continue

            if pending_req_env is not None:
                assert pending_req_env.payload.kind == PACKET_RUN_EVENT
                ev = pending_req_env.payload.event.event
                msg_id = pending_req_env.msg_id

                if ev.kind == PACKET_MESSAGE_REQUEST:
                    if text == "":
                        await respond_packet(ui, msg_id, None)
                    else:
                        await respond_message(
                            ui, msg_id, Message(role="user", text=text)
                        )
                    pending_req_env = None
                    continue

                if ev.kind == PACKET_TOOL_CALL:
                    if text.strip().lower() in ("", "y", "yes"):
                        await respond_approval(ui, msg_id, True)
                    else:
                        await respond_approval(ui, msg_id, False)
                    pending_req_env = None
                    continue

                if ev.kind == PACKET_FINAL_MESSAGE:
                    conf = _current_node_confirmation(ui)
                    if conf == Confirmation.confirm:
                        ans = text.strip().lower()
                        if ans in ("y", "yes"):
                            await respond_approval(ui, msg_id, True)
                            pending_req_env = None
                            continue
                        if ans in ("n", "no"):
                            await respond_approval(ui, msg_id, False)
                            pending_req_env = None
                            continue
                        out("Please answer Y or N.")
                        # Do not clear pending_req; re-prompt
                        continue
                    # prompt mode behavior
                    if text.strip() == "":
                        await respond_approval(ui, msg_id, True)
                    else:
                        await respond_message(
                            ui, msg_id, Message(role="user", text=text)
                        )
                    pending_req_env = None
                    continue

            # No pending request here: runner is active; ignore input and re-prompt.
            continue

    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        # Restore previous SIGINT handler
        with contextlib.suppress(Exception):
            signal.signal(signal.SIGINT, old_sigint)
        if ui.is_active():
            await ui.stop()
        consumer_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer_task


@click.command()
@click.argument(
    "project_path", type=click.Path(exists=True, file_okay=False, path_type=str)
)
def main(project_path: str) -> None:
    project = Project.from_base_path(project_path)
    asyncio.run(run_terminal(project))


if __name__ == "__main__":
    import faulthandler, signal, sys

    faulthandler.enable(sys.stderr)  # or just faulthandler.enable()
    faulthandler.register(signal.SIGUSR1)

    main()
