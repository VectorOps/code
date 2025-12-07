from types import SimpleNamespace
import re
from prompt_toolkit.formatted_text import to_formatted_text
from vocode.ui.terminal.toolbar import build_prompt, build_toolbar
from vocode.ui.terminal.commands import (
    CommandContext,
    Commands,
    register_default_commands,
)
from vocode.ui.rpc import RpcHelper
from vocode.ui.proto import UIPacketEnvelope, UIPacketAck
from vocode.state import RunnerStatus


def _text_from_html(html_obj) -> str:
    fragments = to_formatted_text(html_obj)
    # fragments are tuples of (style, text, [mouse_handler])
    return "".join(frag[1] for frag in fragments)


def test_prompt_shows_command_when_no_pending_request():
    dummy_ui = object()  # UI not used when pending_req is None
    html = build_prompt(dummy_ui, None)
    text = _text_from_html(html)
    assert "(command)" in text


def _dummy_ui(status: RunnerStatus = RunnerStatus.running):
    # Minimal stub matching attributes used by build_toolbar via TerminalApp.ui
    return SimpleNamespace(
        selected_workflow_name="wf",
        current_node_name="node",
        status=status,
    )


def _dummy_app(status: RunnerStatus = RunnerStatus.running):
    # Minimal TerminalApp-like stub for build_toolbar(app, pending_req)
    return SimpleNamespace(
        ui=_dummy_ui(status),
        llm_usage_global=None,
        llm_usage_session=None,
        llm_usage_node=None,
    )


def test_toolbar_shows_waiting_input_when_input_requested():
    app = _dummy_app(RunnerStatus.running)
    pending_req = SimpleNamespace(event=SimpleNamespace(input_requested=True))
    html = build_toolbar(app, pending_req)
    text = _text_from_html(html)
    assert "[waiting input]" in text


def test_toolbar_running_animation_and_cancel_on_status_change(monkeypatch):
    app = _dummy_app(RunnerStatus.running)
    pending_req = None

    # Ensure deterministic frames by patching monotonic in the module under test
    monkeypatch.setattr("vocode.ui.terminal.toolbar.time.monotonic", lambda: 0)
    text1 = _text_from_html(build_toolbar(app, pending_req))
    assert re.search(r"\[running\.\s*\]", text1)

    monkeypatch.setattr("vocode.ui.terminal.toolbar.time.monotonic", lambda: 1)
    text2 = _text_from_html(build_toolbar(app, pending_req))
    assert re.search(r"\[running\.\.\s*\]", text2)

    monkeypatch.setattr("vocode.ui.terminal.toolbar.time.monotonic", lambda: 2)
    text3 = _text_from_html(build_toolbar(app, pending_req))
    assert re.search(r"\[running\.\.\.\s*\]", text3)

    # Now change status; animation should cancel and show new status
    app.ui.status = RunnerStatus.finished
    text4 = _text_from_html(build_toolbar(app, pending_req))
    assert f"[{RunnerStatus.finished.value}]" in text4
    assert "running" not in text4


def test_handoff_command_uses_last_final(monkeypatch):
    # Minimal fake UI and RPC to capture calls
    calls = {}

    class DummyUI:
        def __init__(self):
            from vocode.state import Message

            self.last_final_message = Message(role="agent", text="done")
            self.status = RunnerStatus.finished

    async def send_cb(env: UIPacketEnvelope):
        # Immediately ACK everything
        calls["last"] = env.payload

    rpc = RpcHelper(send_cb, "test")

    async def call(payload, timeout=300.0):  # type: ignore[override]
        calls["payload"] = payload
        return UIPacketAck()

    monkeypatch.setattr(rpc, "call", call)

    ui = DummyUI()
    commands = register_default_commands(Commands(), ui, ac_factory=None)

    # Find and run /handoff
    ctx = CommandContext(
        ui=ui,
        out=lambda s: None,
        stop_toggle=lambda: None,  # type: ignore[assignment]
        request_exit=lambda: None,
        rpc=rpc,
    )

    async def run_cmd():
        handled = await commands.run("/handoff wf-next", ctx)
        assert handled

    import asyncio

    asyncio.run(run_cmd())

    payload = calls.get("payload")
    assert payload is not None
    assert getattr(payload, "kind", None) == "ui_use_with_input_action"
    assert getattr(payload, "name", None) == "wf-next"
    msg = getattr(payload, "message", None)
    assert msg is not None
    assert getattr(msg, "text", None) == "done"