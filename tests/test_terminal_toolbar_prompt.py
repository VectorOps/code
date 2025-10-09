from types import SimpleNamespace
import re
from prompt_toolkit.formatted_text import to_formatted_text
from vocode.ui.terminal.toolbar import build_prompt, build_toolbar
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
    # Minimal stub matching attributes used by build_toolbar; use real RunnerStatus
    return SimpleNamespace(
        selected_workflow_name="wf",
        current_node_name="node",
        status=status,
        acc_prompt_tokens=0,
        acc_completion_tokens=0,
        acc_cost_dollars=0.0,
    )


def test_toolbar_shows_waiting_input_when_input_requested():
    ui = _dummy_ui(RunnerStatus.running)
    pending_req = SimpleNamespace(event=SimpleNamespace(input_requested=True))
    html = build_toolbar(ui, pending_req)
    text = _text_from_html(html)
    assert "[waiting input]" in text


def test_toolbar_running_animation_and_cancel_on_status_change(monkeypatch):
    ui = _dummy_ui(RunnerStatus.running)
    pending_req = None

    # Ensure deterministic frames by patching monotonic in the module under test
    monkeypatch.setattr("vocode.ui.terminal.toolbar.time.monotonic", lambda: 0)
    text1 = _text_from_html(build_toolbar(ui, pending_req))
    assert re.search(r"\[running\.\s*\]", text1)

    monkeypatch.setattr("vocode.ui.terminal.toolbar.time.monotonic", lambda: 1)
    text2 = _text_from_html(build_toolbar(ui, pending_req))
    assert re.search(r"\[running\.\.\s*\]", text2)

    monkeypatch.setattr("vocode.ui.terminal.toolbar.time.monotonic", lambda: 2)
    text3 = _text_from_html(build_toolbar(ui, pending_req))
    assert re.search(r"\[running\.\.\.\s*\]", text3)

    # Now change status; animation should cancel and show new status
    ui.status = RunnerStatus.finished
    text4 = _text_from_html(build_toolbar(ui, pending_req))
    assert f"[{RunnerStatus.finished.value}]" in text4
    assert "running" not in text4