from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import shutil
import time

from prompt_toolkit.formatted_text import FormattedText, HTML
from prompt_toolkit.formatted_text.utils import fragment_list_width

from vocode.ui.base import UIState
from vocode.ui.proto import UIPacketRunEvent
from vocode.runner.models import (
    PACKET_MESSAGE_REQUEST,
    PACKET_TOOL_CALL,
    PACKET_FINAL_MESSAGE,
)
from vocode.models import Confirmation
from vocode.state import RunnerStatus
from vocode.ui.terminal import styles

if TYPE_CHECKING:
    from .app import TerminalApp

# Tool-call rendering happens in app.py when events are printed.


def _abbr_int(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n/1000:.1f}k".rstrip("0").rstrip(".")
    return f"{n/1_000_000:.1f}M".rstrip("0").rstrip(".")


def _abbr_cost(d: float) -> str:
    x = abs(d)
    sign = "-" if d < 0 else ""
    if x < 0.01:
        s = f"{x:.3f}"
    elif x < 1:
        s = f"{x:.2f}"
    elif x < 1000:
        s = f"{x:.2f}"
    elif x < 1_000_000:
        s = f"{x/1000:.1f}k".rstrip("0").rstrip(".")
    else:
        s = f"{x/1_000_000:.1f}M".rstrip("0").rstrip(".")
    return f"{sign}{s}"


def _current_node_confirmation(ui: UIState) -> Optional[Confirmation]:
    try:
        name = ui.current_node_name
        wf = ui.workflow
        if not name or wf is None:
            return None
        for n in wf.graph.nodes:
            if n.name == name:
                return n.confirmation
        return None
    except Exception:
        return None


def _input_hint(ui: UIState, pending_req: Optional[UIPacketRunEvent]) -> str:
    if pending_req is None:
        return "(command)"
    if not pending_req.event.input_requested:
        return ""
    ev = pending_req.event.event
    if ev.kind == PACKET_MESSAGE_REQUEST:
        return "(your message)"
    if ev.kind == PACKET_TOOL_CALL:
        return "(approve? [Y/n])"
    if ev.kind == PACKET_FINAL_MESSAGE:
        conf = _current_node_confirmation(ui)
        if conf == Confirmation.confirm:
            return "(approve? [Y/N])"
        if conf == Confirmation.prompt_approve:
            return "(type '/approve' to accept, or reply to modify)"
        if conf == Confirmation.loop:
            return "(your message)"
        return "(Enter to continue, or type a reply)"
    return ""


def build_prompt(ui: UIState, pending_req: Optional[UIPacketRunEvent]) -> FormattedText:
    hint = _input_hint(ui, pending_req)
    prompt_text = f"{hint if hint else ''}> "
    return [("class:prompt", prompt_text)]


def build_toolbar(
    app: "TerminalApp",
    pending_req: Optional[UIPacketRunEvent],
) -> FormattedText:
    ui = app.ui
    wf = ui.selected_workflow_name if ui and ui.selected_workflow_name else "-"
    node = ui.current_node_name if ui and ui.current_node_name else "-"

    # Determine status display with "waiting input" and running animation.
    if ui is not None:
        raw_status = ui.status.value
        status_is_running = ui.status == RunnerStatus.running
    else:
        raw_status = RunnerStatus.idle.value
        status_is_running = False

    if pending_req is not None and pending_req.event.input_requested:
        status_display = "waiting input"
    elif status_is_running:
        dots = (int(time.monotonic()) % 3) + 1
        status_display = f"{raw_status}{'.' * dots}{' ' * (3 - dots)}"
    else:
        status_display = raw_status

    left_fragments: FormattedText = [
        ("class:toolbar.wf", wf),
        ("class:toolbar", "@"),
        ("class:toolbar.node", node),
        ("class:toolbar", f" [{status_display}]"),
    ]

    # Usage: derive from cached LLMUsageStats on TerminalApp.
    global_usage = getattr(app, "llm_usage_global", None)
    session_usage = getattr(app, "llm_usage_session", None)
    node_usage = getattr(app, "llm_usage_node", None)

    current_in = int(getattr(node_usage, "prompt_tokens", 0) or 0)
    # Best-effort input token limit: node -> session -> global.
    current_limit = None
    for stats in (node_usage, session_usage, global_usage):
        limit = getattr(stats, "input_token_limit", None) if stats is not None else None
        if limit:
            current_limit = int(limit)
            break

    total_in = int(getattr(global_usage, "prompt_tokens", 0) or 0)
    total_out = int(getattr(global_usage, "completion_tokens", 0) or 0)
    total_cost = float(getattr(global_usage, "cost_dollars", 0.0) or 0.0)

    # Percentage of local (node) prompt tokens vs input limit.
    pct_display = ""
    if current_limit and current_limit > 0:
        pct = int((current_in / current_limit) * 100)
        if pct > 999:
            pct = 999
        pct_display = f" ({pct}%)"

    if current_limit is not None:
        head = f"{_abbr_int(current_in)}/{_abbr_int(current_limit)}{pct_display}"
    else:
        head = f"{_abbr_int(current_in)}/-"

    right_text = (
        f"{head} | "
        f"ts:{_abbr_int(total_in)} "
        f"tr:{_abbr_int(total_out)} "
        f"${_abbr_cost(total_cost)}"
    )
    right_fragments: FormattedText = [("class:toolbar", right_text)]

    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    lw = fragment_list_width(left_fragments)
    rw = fragment_list_width(right_fragments)

    space = 1
    if width > lw + rw:
        space = width - lw - rw
    spaces_text = " " * max(space, 1)
    space_fragment = ("class:toolbar", spaces_text)

    return left_fragments + [space_fragment] + right_fragments
