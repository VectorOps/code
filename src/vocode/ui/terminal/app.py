import asyncio
import contextlib
import difflib
import signal
import sys
import time
from typing import Optional, Union
import shutil
from pathlib import Path

import click

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.formatted_text.utils import split_lines, fragment_list_width
from prompt_toolkit.history import FileHistory
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.output import ColorDepth
from vocode.ui.terminal import colors, styles
from vocode.ui.terminal.toolcall_format import render_tool_call
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
    PACKET_PROJECT_OP_START,
    PACKET_PROJECT_OP_PROGRESS,
    PACKET_PROJECT_OP_FINISH,
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
    print_updated_lines,
    respond_packet,
    respond_message,
    respond_approval,
    StreamThrottler,
)
from vocode.ui.terminal.logger import install as install_terminal_logger
from vocode import diagnostics

LOG_LEVEL_ORDER = {
    LogLevel.debug: 0,
    LogLevel.info: 1,
    LogLevel.warning: 2,
    LogLevel.error: 3,
}


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


class TerminalApp:
    def __init__(self, project: Project) -> None:
        self.project = project
        self.ui: Optional[UIState] = None
        self.rpc: Optional[RpcHelper] = None
        self.commands: Optional[Commands] = None
        self.session: Optional[PromptSession] = None
        self.kb: Optional[KeyBindings] = None
        self.should_exit: bool = False
        self.interrupt_count: int = 0
        self.pending_req_env: Optional[UIPacketEnvelope] = None
        self.pending_cmd: Optional[str] = None
        self.dynamic_cli_commands: set[str] = set()
        self.queued_resp: Optional[Union[RespMessage, RespApproval]] = None
        self.stream_throttler: Optional[StreamThrottler] = None
        self.last_streamed_text: Optional[str] = None
        self._toolbar_ticker_task: Optional[asyncio.Task] = None
        # Project op progress state for toolbar
        self._op_message = None
        self._op_progress = None
        self._op_total = None
        self.old_sigint = None
        self.old_sigterm = None
        self.old_sigusr2 = None
        self.consumer_task: Optional[asyncio.Task] = None
        self._log_handler = None

    def _toolbar_should_animate(self) -> bool:
        """
        Animate toolbar when runner is running and not waiting for input.
        """
        if self.session is None or self.ui is None:
            return False
        if self.ui.status != RunnerStatus.running:
            return False
        # If there's a pending request that requires input, do not animate.
        if (
            self.pending_req_env
            and self.pending_req_env.payload.kind == PACKET_RUN_EVENT
        ):
            if self.pending_req_env.payload.event.input_requested:
                return False
        return True

    def _start_toolbar_ticker(self) -> None:
        if self._toolbar_ticker_task is None or self._toolbar_ticker_task.done():
            self._toolbar_ticker_task = asyncio.create_task(self._toolbar_ticker())

    async def _stop_toolbar_ticker(self) -> None:
        task = self._toolbar_ticker_task
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._toolbar_ticker_task = None

    async def _update_toolbar_ticker(self) -> None:
        if self._toolbar_should_animate():
            self._start_toolbar_ticker()
        else:
            await self._stop_toolbar_ticker()

    async def _toolbar_ticker(self) -> None:
        """
        Periodically invalidate prompt_toolkit app to refresh toolbar frames.
        """
        try:
            while self._toolbar_should_animate():
                if self.session is None:
                    break
                try:
                    # Cause prompt_toolkit to re-render, which will call our bottom_toolbar renderer.
                    self.session.app.invalidate()
                except Exception:
                    pass
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("c-c", eager=True)
        def _kb_stop(event):
            asyncio.create_task(self.stop_toggle())

        @kb.add("c-g", eager=True)
        def _kb_reset(event):
            event.app.current_buffer.reset()

        @kb.add("c-p", eager=True)
        def _kb_dump(event):
            # Run diagnostics in the terminal context to avoid corrupting the UI.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            def _dump():
                diagnostics.dump_all(loop=loop)

            try:
                run_in_terminal(_dump)
            except Exception:
                _dump()

        return kb

    def _unhandled_exception_handler(
        self, loop: asyncio.AbstractEventLoop, context: dict
    ) -> None:
        if self.should_exit:
            return
        msg = context.get("exception", context["message"])
        coro = out(f"\n--- Unhandled exception in event loop ---\n{msg}\n")
        asyncio.run_coroutine_threadsafe(coro, loop)

    def _setup_signal_handlers(self) -> None:
        try:
            self.old_sigint = signal.getsignal(signal.SIGINT)
            self.old_sigterm = signal.getsignal(signal.SIGTERM)
            try:
                self.old_sigusr2 = signal.getsignal(signal.SIGUSR2)  # type: ignore[attr-defined]
            except Exception:
                self.old_sigusr2 = None

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

            try:
                SIGUSR2 = signal.SIGUSR2  # type: ignore[attr-defined]

                def _sigusr2_handler(signum, frame):
                    diagnostics.dump_all(loop=asyncio.get_running_loop())

                signal.signal(SIGUSR2, _sigusr2_handler)
            except Exception:
                pass
        except Exception:
            self.old_sigint, self.old_sigterm, self.old_sigusr2 = None, None, None

    async def _cleanup(self) -> None:
        """Gracefully shut down the application."""
        with contextlib.suppress(Exception):
            if self.old_sigint:
                signal.signal(signal.SIGINT, self.old_sigint)
        with contextlib.suppress(Exception):
            if self.old_sigterm:
                signal.signal(signal.SIGTERM, self.old_sigterm)
        # Remove terminal logger handler if installed
        with contextlib.suppress(Exception):
            if self._log_handler is not None:
                import logging
                logging.getLogger().removeHandler(self._log_handler)
                self._log_handler = None
        await self._stop_toolbar_ticker()
        if self.ui and self.ui.is_active():
            await self.ui.stop()
        if self.consumer_task:
            self.consumer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.consumer_task
        with contextlib.suppress(Exception):
            await self.project.shutdown()

    def _pending_run_event_payload(self) -> Optional[UIPacketRunEvent]:
        """
        Return the current UIPacketRunEvent payload if a request is pending; otherwise None.
        Computed on demand to ensure dynamic prompt/toolbar reflect live state.
        """
        env = self.pending_req_env
        if env and env.payload.kind == PACKET_RUN_EVENT:
            return env.payload  # type: ignore[return-value]
        return None

    def reset_interrupt(self) -> None:
        self.interrupt_count = 0

    async def stop_toggle(self) -> None:
        n = self.interrupt_count
        self.interrupt_count = n + 1
        assert self.ui is not None
        # Flush any active streaming before issuing stop to avoid lingering output.
        await self._flush_and_clear_stream()
        if n == 0:
            await self.ui.stop()
        else:
            # Subsequent Ctrl+C presses are ignored (no cancel).
            pass
        # Clear any pending input locally and refresh toolbar state.
        self.pending_req_env = None
        await self._update_toolbar_ticker()

    def request_exit(self) -> None:
        self.should_exit = True
        if self.session is not None:
            with contextlib.suppress(Exception):
                self.session.app.exit(result="")

    async def _flush_and_clear_stream(self) -> None:
        if self.stream_throttler:
            self.last_streamed_text = self.stream_throttler.full_text
            await self.stream_throttler.close()
        self.stream_throttler = None

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
                try:
                    res = await rpc.call(UIPacketRunCommand(name=_cname, input=args))
                    if res is None:
                        return
                    if res.kind != PACKET_COMMAND_RESULT:
                        await ctx.out(
                            f"Command '{_cname}' failed: unexpected response {res.kind}"
                        )
                        return
                    if res.ok:
                        if res.output:
                            await ctx.out(res.output)
                    else:
                        await ctx.out(
                            f"Command '{_cname}' failed: {res.error or 'unknown error'}"
                        )
                except Exception as e:
                    await ctx.out(f"Error executing command '{_cname}': {e}")
                finally:
                    self.pending_cmd = None

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

    async def handle_run_event(self, envelope: UIPacketEnvelope) -> None:
        assert self.ui is not None
        req_payload = envelope.payload
        assert req_payload.kind == PACKET_RUN_EVENT
        ev = req_payload.event.event

        if ev.kind == PACKET_LOG:
            cfg_log_level = LogLevel.info
            if self.ui.project.settings and self.ui.project.settings.ui:
                cfg_log_level = self.ui.project.settings.ui.log_level
            msg_level = ev.level or LogLevel.info
            if LOG_LEVEL_ORDER[msg_level] < LOG_LEVEL_ORDER[cfg_log_level]:
                return
            if self.stream_throttler:
                await self._flush_and_clear_stream()
            await out(f"[{msg_level.value}] " + ev.text)
            return

        if ev.kind == PACKET_MESSAGE and ev.message:
            speaker = ev.message.role or "assistant"
            text = ev.message.text or ""
            if not self.stream_throttler:
                assert self.session is not None
                self.last_streamed_text = None
                self.stream_throttler = StreamThrottler(
                    session=self.session, speaker=speaker
                )

            assert self.stream_throttler is not None
            await self.stream_throttler.append(text)

            return

        if self.stream_throttler:
            await self._flush_and_clear_stream()

        if ev.kind == PACKET_MESSAGE_REQUEST:
            self.last_streamed_text = None  # New stream context, clear old state
            if ev.message:
                await out_fmt(colors.render_markdown(ev.message))
            self.pending_req_env = (
                envelope if req_payload.event.input_requested else None
            )
            if self.pending_req_env is not None and self.queued_resp is not None:
                await respond_packet(
                    self.ui, self.pending_req_env.msg_id, self.queued_resp
                )
                self.queued_resp = None
                self.pending_req_env = None
            if self.session:
                self.session.app.invalidate()
            await self._update_toolbar_ticker()
            return

        if ev.kind == PACKET_TOOL_CALL:
            self.last_streamed_text = None  # New stream context, clear old state
            # Render each tool call as a formatted function-call preview.
            fmt_map = (
                self.ui.project.settings.tool_call_formatters
                if (self.ui and self.ui.project.settings)
                else None
            )
            term_width = shutil.get_terminal_size(fallback=(80, 24)).columns
            # If input is requested, a confirmation is being asked for the tool call.
            print_source = bool(req_payload.event.input_requested)
            # Header
            header_fragments = [
                ("class:system.star", "* "),
                ("class:system.text", "Tool call"),
            ]
            print_formatted_text(
                to_formatted_text(header_fragments), style=styles.get_pt_style()
            )
            for tc in ev.tool_calls:
                fragments = render_tool_call(
                    tc.name,
                    tc.arguments,
                    fmt_map,
                    terminal_width=term_width,
                    print_source=print_source,
                )
                # Print a prefixed line: "| " + <formatted tool call>
                # Use end="" for the prefix so the tool call prints on the same line.
                print_formatted_text(
                    to_formatted_text([("class:system.text", "| ")]),
                    style=styles.get_pt_style(),
                    end="",
                )
                print_formatted_text(
                    to_formatted_text(fragments),
                    style=styles.get_pt_style(),
                )
            # Empty line after the tool-call block
            print_formatted_text("")
            self.pending_req_env = (
                envelope if req_payload.event.input_requested else None
            )
            if self.pending_req_env is not None and self.queued_resp is not None:
                await respond_packet(
                    self.ui, self.pending_req_env.msg_id, self.queued_resp
                )
                self.queued_resp = None
                self.pending_req_env = None
            if self.session:
                self.session.app.invalidate()
            await self._update_toolbar_ticker()
            return

        if ev.kind == PACKET_FINAL_MESSAGE:
            streamed_text = self.last_streamed_text
            self.last_streamed_text = None  # Consume it
            if ev.message:
                # If the final message is the same as what we just streamed, don't print it.
                if streamed_text is None or ev.message.text != streamed_text:
                    text = ev.message.text or ""
                    await out_fmt(colors.render_markdown(text))
            self.pending_req_env = (
                envelope if req_payload.event.input_requested else None
            )
            if self.pending_req_env is not None and self.queued_resp is not None:
                await respond_packet(
                    self.ui, self.pending_req_env.msg_id, self.queued_resp
                )
                self.queued_resp = None
                self.pending_req_env = None
            if self.session:
                self.session.app.invalidate()
            await self._update_toolbar_ticker()
            return

    async def event_consumer(self) -> None:
        assert self.ui is not None
        while True:
            try:
                envelope = await self.ui.recv()
                assert self.rpc is not None
                if self.rpc.handle_response(envelope):
                    continue
                msg = envelope.payload
                if msg.kind == PACKET_STATUS:
                    self.reset_interrupt()
                    await self._update_toolbar_ticker()
                    # If node changed, print a styled "Running <node>" line
                    curr_node = msg.curr_node
                    prev_node = msg.prev_node
                    if curr_node and curr_node != prev_node:
                        node_display = msg.curr_node_description or curr_node
                        fragments = [
                            ("class:system.star", "* "),
                            ("class:system.text", f"Running {node_display}"),
                        ]
                        print_formatted_text(
                            to_formatted_text(fragments),
                            style=styles.get_pt_style(),
                        )
                    # When runner is no longer active, clear pending input and stop any streaming.
                    if msg.curr in (
                        RunnerStatus.stopped,
                        RunnerStatus.canceled,
                        RunnerStatus.finished,
                        RunnerStatus.idle,
                    ):
                        self.pending_req_env = None
                        self.queued_resp = None
                        await self._flush_and_clear_stream()
                        await self._update_toolbar_ticker()
                    continue
                if msg.kind == PACKET_RUN_EVENT:
                    await self.handle_run_event(envelope)
                    continue
                if msg.kind == PACKET_UI_RESET:
                    self.pending_req_env = None
                    self.queued_resp = None
                    await self._update_toolbar_ticker()
                    continue
                if msg.kind == PACKET_CUSTOM_COMMANDS:
                    await self.handle_custom_commands_packet(msg)
                    continue
                if msg.kind == PACKET_PROJECT_OP_START:
                    # await out(f"Starting: {msg.message}...")
                    self._op_message = msg.message
                    self._op_progress = 0
                    self._op_total = None
                    if self.session:
                        self.session.app.invalidate()
                    continue
                if msg.kind == PACKET_PROJECT_OP_PROGRESS:
                    self._op_progress = msg.progress
                    self._op_total = msg.total
                    # TODO: Fix me
                    await out(f"Progress: {msg.progress} / {msg.total}...\r", end="")
                    if self.session:
                        self.session.app.invalidate()
                    continue
                if msg.kind == PACKET_PROJECT_OP_FINISH:
                    # await out(f"Completed: {self._op_message}.")

                    self._op_message = None
                    self._op_progress = None
                    self._op_total = None
                    if self.session:
                        self.session.app.invalidate()
                    continue
            except Exception:
                import traceback

                print("Exception", traceback.format_exc())

    async def run(self) -> None:
        start_time = time.monotonic()

        loop = asyncio.get_running_loop()
        loop.set_exception_handler(self._unhandled_exception_handler)
        # 1) Print banner first (no dependency on UI/session)
        ui_cfg = self.project.settings.ui if self.project.settings else None
        # Auto-install terminal logger handler with UI-configured level
        try:
            import logging
            level_map = {
                LogLevel.debug: logging.DEBUG,
                LogLevel.info: logging.INFO,
                LogLevel.warning: logging.WARNING,
                LogLevel.error: logging.ERROR,
            }
            cfg_level = ui_cfg.log_level if ui_cfg else LogLevel.info
            handler_level = level_map.get(cfg_level, logging.INFO)
            self._log_handler = install_terminal_logger(
                loop, level=handler_level, use_fmt=False
            )
        except Exception:
            # Non-fatal: if logging install fails, continue without terminal handler
            self._log_handler = None
        pt_style = styles.get_pt_style()
        show_banner = True if ui_cfg is None else ui_cfg.show_banner
        if show_banner:
            banner_lines = [
                r" _      ____  __   _____  ___   ___   ___   ___   __       __    ___   ___   ____ ",
                r"\ \  / | |_  / /`   | |  / / \ | |_) / / \ | |_) ( (`     / /`  / / \ | | \ | |_  ",
                r" \_\/  |_|__ \_\_,  |_|  \_\_/ |_| \ \_\_/ |_|   _)_)     \_\_, \_\_/ |_|_/ |_|__ ",
            ]
            fragments = [
                ("class:banner.l1", banner_lines[0] + "\n"),
                ("class:banner.l2", banner_lines[1] + "\n"),
                ("class:banner.l3", banner_lines[2] + "\n"),
                ("", "\n"),
            ]
            print_formatted_text(to_formatted_text(fragments), style=pt_style)

        # 2) Initialize UI
        self.ui = UIState(self.project)
        self.rpc = RpcHelper(
            self.ui.send, "TerminalApp", id_generator=self.ui.next_client_msg_id
        )
        ac_factory = lambda name: make_canned_provider(self.rpc, name)
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
        # pt_style already computed above and reused for the session
        # (kept in variable pt_style)
        kwargs = {
            "multiline": multiline,
            "completer": completer,
            "complete_while_typing": False,
            "style": pt_style,
        }
        if editing_mode is not None:
            kwargs["editing_mode"] = editing_mode

        try:
            hist_dir = self.project.base_path / ".vocode"
            hist_dir.mkdir(parents=True, exist_ok=True)
            hist_path = hist_dir / "data" / "terminal_history"
            kwargs["history"] = FileHistory(str(hist_path))
        except Exception:
            pass  # History is optional

        self.session = PromptSession(**kwargs)
        self.kb = self._create_key_bindings()
        self.should_exit = False
        self._setup_signal_handlers()

        # Commands
        ctx = CommandContext(
            ui=self.ui,
            out=lambda s: out(s),
            stop_toggle=self.stop_toggle,
            request_exit=self.request_exit,
            rpc=self.rpc,
        )

        self.consumer_task = asyncio.create_task(self.event_consumer())

        # 3) Start project after UI is initialized.
        await out("Starting project...")

        await self.project.start()

        end_time = time.monotonic()
        await out(f"Project started in {end_time - start_time:.2f}s.")
        # Start default workflow if configured
        if self.project.settings and self.project.settings.default_workflow:
            await self.ui.start_by_name(self.project.settings.default_workflow)

        await out("Type /help for commands.")

        try:
            while True:
                # Pre-bind renderers; they compute the current pending payload at render time.
                render_prompt = lambda: build_prompt(
                    self.ui, self._pending_run_event_payload()
                )

                def _render_toolbar():
                    base = build_toolbar(self.ui, self._pending_run_event_payload())
                    fr = list(to_formatted_text(base))
                    if self._op_message is not None:
                        # Compose a simple progress display in the toolbar
                        p = int(self._op_progress or 0)
                        t = int(self._op_total or 0)
                        pct = int((p * 100 / t)) if t else 0
                        bar_width = 20
                        filled = int(bar_width * (p / t)) if t else 0
                        filled = max(0, min(bar_width, filled))
                        bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"
                        fr += [
                            ("", "  |  "),
                            ("class:system.text", f"{self._op_message} "),
                            ("class:system.text", f"{bar} {p}/{t} ({pct}%)"),
                        ]
                    return fr

                render_toolbar = _render_toolbar
                line = await self.session.prompt_async(
                    render_prompt,
                    key_bindings=self.kb,
                    default="",
                    bottom_toolbar=render_toolbar,
                )
                if self.should_exit:
                    break
                text = line.rstrip("\n")
                if text.startswith("/"):
                    handled = await self.commands.run(text, ctx)
                    if self.should_exit:
                        break
                    if not handled:
                        await out("Unknown command. Type /help")
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
                        await self._update_toolbar_ticker()
                        continue
                    if ev.kind == PACKET_TOOL_CALL:
                        if text.strip().lower() in ("", "y", "yes"):
                            await respond_approval(self.ui, msg_id, True)
                        else:
                            await respond_approval(self.ui, msg_id, False)
                        self.pending_req_env = None
                        await self._update_toolbar_ticker()
                        continue
                    if ev.kind == PACKET_FINAL_MESSAGE:
                        conf = _current_node_confirmation(self.ui)
                        if conf == Confirmation.confirm:
                            ans = text.strip().lower()
                            if ans in ("y", "yes"):
                                await respond_approval(self.ui, msg_id, True)
                                self.pending_req_env = None
                                await self._update_toolbar_ticker()
                                continue
                            if ans in ("n", "no"):
                                await respond_approval(self.ui, msg_id, False)
                                self.pending_req_env = None
                                await self._update_toolbar_ticker()
                                continue
                            await out("Please answer Y or N.")
                            continue
                        if text.strip() == "":
                            await respond_approval(self.ui, msg_id, True)
                        else:
                            await respond_message(
                                self.ui, msg_id, Message(role="user", text=text)
                            )
                        self.pending_req_env = None
                        await self._update_toolbar_ticker()
                        continue

                # No pending input and not a command.
                if self.ui.status == RunnerStatus.stopped and text:
                    # If stopped, treat this as a replacement for the last user input.
                    await self.ui.replace_user_input(
                        RespMessage(message=Message(role="user", text=text))
                    )
                    await self.ui.restart()
                    continue

                # Show contextual hints otherwise.
                if self.ui.is_active():
                    await out(
                        "No input is currently requested. Press Ctrl+C to stop the run."
                    )
                else:
                    await out(
                        "No active run. Start a workflow with /run <workflow> or /use <workflow>."
                    )
                continue
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            with contextlib.suppress(Exception):
                await self._cleanup()


async def run_terminal(project: Project) -> None:
    import logging

    logging.getLogger("asyncio").setLevel(logging.DEBUG)
    asyncio.get_event_loop().set_debug(True)
    asyncio.get_event_loop().slow_callback_duration = 0.05

    # Thin wrapper: defer to TerminalApp for all terminal behavior.
    app = TerminalApp(project)
    await app.run()

    # Suppress exceptions
    app.should_exit = True


@click.command()
@click.argument(
    "project_path", type=click.Path(exists=True, file_okay=False, path_type=str)
)
def main(project_path: str) -> None:
    import faulthandler, signal, sys, warnings, logging

    faulthandler.enable(sys.stderr)  # or just faulthandler.enable()
    faulthandler.register(signal.SIGUSR1)

    # Fix litellm warnings
    from pydantic.warnings import (
        PydanticDeprecatedSince211,
        PydanticDeprecatedSince20,
    )

    warnings.filterwarnings(action="ignore", category=PydanticDeprecatedSince211)
    warnings.filterwarnings(action="ignore", category=PydanticDeprecatedSince20)
    warnings.filterwarnings(action="ignore", category=PydanticDeprecatedSince20)
    warnings.filterwarnings(action="ignore", category=DeprecationWarning)

    # Start project
    project = Project.from_base_path(project_path)
    asyncio.run(run_terminal(project))


if __name__ == "__main__":
    main()
