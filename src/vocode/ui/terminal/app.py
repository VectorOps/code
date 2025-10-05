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
from vocode.ui.terminal.ac_client import (
    make_canned_provider,
    make_general_filelist_provider,
)

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
    UIPacketUIReset,
    PACKET_UI_RESET,
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
from vocode.ui.terminal.helpers import (
    out,
    out_fmt,
    out_fmt_stream,
    respond_packet,
    respond_message,
    respond_approval,
    ANSI_CURSOR_DOWN,
)


LOG_LEVEL_ORDER = {
    LogLevel.debug: 0,
    LogLevel.info: 1,
    LogLevel.warning: 2,
    LogLevel.error: 3,
}


class TerminalApp:
    def __init__(self, project: Project) -> None:
        self.project = project
        self.ui: Optional[UIState] = None
        self.rpc: Optional[RpcHelper] = None
        self.commands: Optional[Commands] = None
        self.session: Optional[PromptSession] = None
        self.kb: Optional[KeyBindings] = None
        self.should_exit: bool = False
        self.change_event: Optional[asyncio.Event] = None
        self.interrupt_count: int = 0
        self.pending_req_env: Optional[UIPacketEnvelope] = None
        self.pending_cmd: Optional[str] = None
        self.dynamic_cli_commands: set[str] = set()
        self.queued_resp: Optional[Union[RespMessage, RespApproval]] = None
        self.stream_buffer: Optional[MessageBuffer] = None

    def reset_interrupt(self) -> None:
        self.interrupt_count = 0

    async def stop_toggle(self) -> None:
        n = self.interrupt_count
        self.interrupt_count = n + 1
        assert self.ui is not None
        await (self.ui.stop() if n == 0 else self.ui.cancel())

    def request_exit(self) -> None:
        self.should_exit = True
        if self.session is not None:
            with contextlib.suppress(Exception):
                self.session.app.exit(result="")

    def _finish_stream(self) -> None:
        if not self.stream_buffer:
            return
        ft = colors.render_markdown(
            self.stream_buffer.full_text, prefix=f"{self.stream_buffer.speaker}: "
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
        self.stream_buffer = None

    async def handle_custom_commands_packet(self, cd: UIPacketCustomCommands) -> None:
        assert (
            self.commands is not None and self.rpc is not None and self.ui is not None
        )
        commands = self.commands
        rpc = self.rpc
        ac_factory = lambda name: make_canned_provider(rpc, name)
        existing_names = {c.name for c in commands.list_commands()}
        for c in cd.added:
            cli_name = f"/{c.name}"
            if cli_name in existing_names and cli_name not in self.dynamic_cli_commands:
                continue
            if cli_name in self.dynamic_cli_commands:
                commands.unregister(cli_name)
                self.dynamic_cli_commands.discard(cli_name)

            async def _proxy(ctx: CommandContext, args: list[str], *, _cname=c.name):
                self.pending_cmd = _cname
                assert self.change_event is not None
                self.change_event.set()
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
                    self.pending_cmd = None
                    assert self.change_event is not None
                    self.change_event.set()

            commands.register(
                cli_name,
                c.help or "",
                c.usage,
                (ac_factory(c.autocompleter) if c.autocompleter else None),
            )(
                _proxy
            )  # type: ignore[arg-type]
            self.dynamic_cli_commands.add(cli_name)
            existing_names.add(cli_name)
        for name in cd.removed:
            cli_name = f"/{name}"
            if cli_name in self.dynamic_cli_commands:
                commands.unregister(cli_name)
                self.dynamic_cli_commands.discard(cli_name)
        assert self.change_event is not None
        self.change_event.set()

    async def handle_run_event(self, envelope: UIPacketEnvelope) -> None:
        assert self.ui is not None
        req_payload = envelope.payload
        assert req_payload.kind == PACKET_RUN_EVENT
        ev = req_payload.event.event
        if ev.kind == PACKET_LOG:
            self._finish_stream()
            cfg_log_level = LogLevel.info
            if self.ui.project.settings and self.ui.project.settings.ui:
                cfg_log_level = self.ui.project.settings.ui.log_level
            msg_level = ev.level or LogLevel.info
            if LOG_LEVEL_ORDER[msg_level] < LOG_LEVEL_ORDER[cfg_log_level]:
                return
            out(f"[{msg_level.value}] " + ev.text)
            return
        if ev.kind == PACKET_MESSAGE and ev.message:
            speaker = ev.message.role or "assistant"
            text = ev.message.text or ""
            if not self.stream_buffer or self.stream_buffer.speaker != speaker:
                self._finish_stream()
                self.stream_buffer = MessageBuffer(speaker=speaker)
            diff = self.stream_buffer.append(text)
            if diff:
                out_fmt_stream(diff)
            return
        if ev.kind == PACKET_MESSAGE_REQUEST:
            self._finish_stream()
            if ev.message:
                out_fmt(colors.render_markdown(ev.message))
            self.pending_req_env = (
                envelope if req_payload.event.input_requested else None
            )
            if self.pending_req_env is not None and self.queued_resp is not None:
                await respond_packet(
                    self.ui, self.pending_req_env.msg_id, self.queued_resp
                )
                self.queued_resp = None
                self.pending_req_env = None
            return
        if ev.kind == PACKET_TOOL_CALL:
            self._finish_stream()
            out(f"Tool calls requested: {len(ev.tool_calls)}")
            for i, tc in enumerate(ev.tool_calls, start=1):
                out(f"  {i}. {tc.name} arguments:")
                out_fmt(colors.render_json(tc.arguments))
            self.pending_req_env = (
                envelope if req_payload.event.input_requested else None
            )
            if self.pending_req_env is not None and self.queued_resp is not None:
                await respond_packet(
                    self.ui, self.pending_req_env.msg_id, self.queued_resp
                )
                self.queued_resp = None
                self.pending_req_env = None
            return
        if ev.kind == PACKET_FINAL_MESSAGE:
            if self.stream_buffer:
                self._finish_stream()
            elif ev.message:
                speaker = ev.message.role or "assistant"
                text = ev.message.text or ""
                out_fmt(colors.render_markdown(text, prefix=f"{speaker}: "))
            self.pending_req_env = (
                envelope if req_payload.event.input_requested else None
            )
            if self.pending_req_env is not None and self.queued_resp is not None:
                await respond_packet(
                    self.ui, self.pending_req_env.msg_id, self.queued_resp
                )
                self.queued_resp = None
                self.pending_req_env = None
            return

    async def event_consumer(self) -> None:
        assert self.ui is not None
        while True:
            try:
                envelope = await self.ui.recv()
                assert self.rpc is not None and self.change_event is not None
                if self.rpc.handle_response(envelope):
                    self.change_event.set()
                    continue
                msg = envelope.payload
                if msg.kind == PACKET_STATUS:
                    self.reset_interrupt()
                    self.change_event.set()
                    continue
                if msg.kind == PACKET_RUN_EVENT:
                    await self.handle_run_event(envelope)
                    self.change_event.set()
                    continue
                if msg.kind == PACKET_UI_RESET:
                    self.pending_req_env = None
                    self.queued_resp = None
                    self.change_event.set()
                    continue
                if msg.kind == PACKET_CUSTOM_COMMANDS:
                    await self.handle_custom_commands_packet(msg)
                    continue
            except Exception:
                import traceback

                print("Exception", traceback.format_exc())

    async def run(self) -> None:
        await self.project.start()
        self.ui = UIState(self.project)
        self.rpc = RpcHelper(
            self.ui.send, "TerminalApp", id_generator=self.ui.next_client_msg_id
        )
        ac_factory = lambda name: make_canned_provider(self.rpc, name)
        ui_cfg = self.project.settings.ui if self.project.settings else None
        multiline = True if ui_cfg is None else bool(ui_cfg.multiline)
        editing_mode = None
        if ui_cfg and ui_cfg.edit_mode:
            mode = str(ui_cfg.edit_mode).lower()
            if mode in ("vi", "vim"):
                editing_mode = EditingMode.VI
            elif mode == "emacs":
                editing_mode = EditingMode.EMACS
        self.commands = register_default_commands(
            Commands(), self.ui, ac_factory=ac_factory
        )
        completer = TerminalCompleter(
            self.ui,
            self.commands,
            general_provider=make_general_filelist_provider(self.rpc),
        )
        try:
            hist_dir = self.project.base_path / ".vocode"
            hist_dir.mkdir(parents=True, exist_ok=True)
            hist_path = hist_dir / "data" / "terminal_history"
            kwargs = {
                "history": FileHistory(str(hist_path)),
                "multiline": multiline,
                "completer": completer,
                "complete_while_typing": False,
            }
            if editing_mode is not None:
                kwargs["editing_mode"] = editing_mode
            self.session = PromptSession(**kwargs)
        except Exception:
            kwargs = {
                "multiline": multiline,
                "completer": completer,
                "complete_while_typing": False,
            }
            if editing_mode is not None:
                kwargs["editing_mode"] = editing_mode
            self.session = PromptSession(**kwargs)
        self.kb = KeyBindings()
        self.should_exit = False
        self.change_event = asyncio.Event()
        # Key bindings
        kb = self.kb

        @kb.add("c-c")
        def _kb_stop(event):
            asyncio.create_task(self.stop_toggle())

        @kb.add("c-g", eager=True)
        def _kb_reset(event):
            event.app.current_buffer.reset()

        # Signal handlers
        old_sigint = None
        old_sigterm = None
        try:
            old_sigint = signal.getsignal(signal.SIGINT)
            old_sigterm = signal.getsignal(signal.SIGTERM)

            def _sigint_handler(signum, frame):
                # Suppress stacktrace dumps on Ctrl+C; just trigger a graceful stop/cancel.
                try:
                    loop = asyncio.get_running_loop()
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self.stop_toggle())
                    )
                except RuntimeError:
                    try:
                        asyncio.create_task(self.stop_toggle())
                    except Exception:
                        pass

            signal.signal(signal.SIGINT, _sigint_handler)

            def _sigterm_handler(signum, frame):
                try:
                    self.request_exit()
                finally:
                    try:
                        loop = asyncio.get_running_loop()
                        loop.call_soon_threadsafe(
                            lambda: asyncio.create_task(
                                self.ui.cancel() if self.ui else asyncio.sleep(0)
                            )
                        )
                    except RuntimeError:
                        try:
                            asyncio.create_task(
                                self.ui.cancel() if self.ui else asyncio.sleep(0)
                            )
                        except Exception:
                            pass

            signal.signal(signal.SIGTERM, _sigterm_handler)
        except Exception:
            old_sigint = None
            old_sigterm = None
        ctx = CommandContext(
            ui=self.ui,
            out=lambda s: out(s),
            stop_toggle=self.stop_toggle,
            request_exit=self.request_exit,
            rpc=self.rpc,
        )
        # Banner
        show_banner = True if ui_cfg is None else ui_cfg.show_banner
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
            fragments = []
            for line, fg in zip(banner_lines, colors_fg):
                fragments.append((f"fg:{fg}", line + "\n"))
            fragments.append(("", "\n"))
            fragments.append(("", "Type /help for commands.\n"))
            out_fmt(fragments)
        else:
            out("Type /help for commands.")
        consumer_task = asyncio.create_task(self.event_consumer())
        try:
            while True:
                prompt_payload = (
                    self.pending_req_env.payload
                    if self.pending_req_env
                    and self.pending_req_env.payload.kind == PACKET_RUN_EVENT
                    else None
                )
                line = await self.session.prompt_async(
                    lambda: build_prompt(self.ui, prompt_payload),
                    key_bindings=self.kb,
                    default="",
                    bottom_toolbar=(lambda: build_toolbar(self.ui, prompt_payload)),
                )
                if self.should_exit:
                    break
                text = line.rstrip("\n")
                if text.startswith("/"):
                    handled = await self.commands.run(text, ctx)
                    if self.should_exit:
                        break
                    if not handled:
                        out("Unknown command. Type /help")
                    continue
                if self.pending_req_env is not None:
                    assert self.pending_req_env.payload.kind == PACKET_RUN_EVENT
                    ev = self.pending_req_env.payload.event.event
                    msg_id = self.pending_req_env.msg_id
                    if ev.kind == PACKET_MESSAGE_REQUEST:
                        if text == "":
                            await respond_packet(self.ui, msg_id, None)
                        else:
                            await respond_message(
                                self.ui, msg_id, Message(role="user", text=text)
                            )
                        self.pending_req_env = None
                        continue
                    if ev.kind == PACKET_TOOL_CALL:
                        if text.strip().lower() in ("", "y", "yes"):
                            await respond_approval(self.ui, msg_id, True)
                        else:
                            await respond_approval(self.ui, msg_id, False)
                        self.pending_req_env = None
                        continue
                    if ev.kind == PACKET_FINAL_MESSAGE:
                        conf = _current_node_confirmation(self.ui)
                        if conf == Confirmation.confirm:
                            ans = text.strip().lower()
                            if ans in ("y", "yes"):
                                await respond_approval(self.ui, msg_id, True)
                                self.pending_req_env = None
                                continue
                            if ans in ("n", "no"):
                                await respond_approval(self.ui, msg_id, False)
                                self.pending_req_env = None
                                continue
                            out("Please answer Y or N.")
                            continue
                        if text.strip() == "":
                            await respond_approval(self.ui, msg_id, True)
                        else:
                            await respond_message(
                                self.ui, msg_id, Message(role="user", text=text)
                            )
                        self.pending_req_env = None
                        continue
                # No pending input and not a command: show contextual hints.
                if self.ui.is_active():
                    out(
                        "No input is currently requested. Press Ctrl+C to stop the run, then respond when prompted."
                    )
                else:
                    out(
                        "No active run. Start a workflow with /run <workflow> or /use <workflow>. Type /workflows to list available workflows."
                    )
                continue
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            with contextlib.suppress(Exception):
                signal.signal(signal.SIGINT, old_sigint)
            with contextlib.suppress(Exception):
                signal.signal(signal.SIGTERM, old_sigterm)
            if self.ui.is_active():
                await self.ui.stop()
            consumer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer_task
            with contextlib.suppress(Exception):
                await self.project.shutdown()


async def run_terminal(project: Project) -> None:
    # Thin wrapper: defer to TerminalApp for all terminal behavior.
    app = TerminalApp(project)
    await app.run()


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
