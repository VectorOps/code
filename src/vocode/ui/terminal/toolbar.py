from __future__ import annotations
from typing import Optional
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
        return "(Enter to continue, or type a reply)"
    return ""


def build_prompt(ui: UIState, pending_req: Optional[UIPacketRunEvent]) -> FormattedText:
    hint = _input_hint(ui, pending_req)
    prompt_text = f"{hint if hint else ''}> "
    return [("class:prompt", prompt_text)]


def build_toolbar(ui: UIState, pending_req: Optional[UIPacketRunEvent]) -> FormattedText:
    wf = ui.selected_workflow_name or "-"
    node = ui.current_node_name or "-"
    # Determine status display with "waiting input" and running animation.
    raw_status = ui.status.value

    if pending_req is not None and pending_req.event.input_requested:
        status_display = "waiting input"
    elif ui.status == RunnerStatus.running:
        # Animate dots: 1..3 then loop, appended to the status text
        dots = (int(time.monotonic()) % 3) + 1
        status_display = f"{raw_status}{'.' * dots}"
    else:
        status_display = raw_status

    # Build toolbar fragments as a list of (style class, text) tuples.
    left_fragments: FormattedText = [
        ("class:toolbar.wf", wf),
        ("class:toolbar", "@"),
        ("class:toolbar.node", node),
        ("class:toolbar", f" [{status_display}]"),
    ]

    p = ui.acc_prompt_tokens
    c = ui.acc_completion_tokens
    cost = ui.acc_cost_dollars
    right_text = (
        f"p:{_abbr_int(int(p))} r:{_abbr_int(int(c))} ${_abbr_cost(float(cost))}"
    )
    right_fragments: FormattedText = [("class:toolbar", right_text)]

    # Measure widths to compute spacing.
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    lw = fragment_list_width(left_fragments)
    rw = fragment_list_width(right_fragments)

    space = 1
    if width > lw + rw:
        space = width - lw - rw
    spaces_text = " " * max(space, 1)
    space_fragment = ("class:toolbar", spaces_text)

    return left_fragments + [space_fragment] + right_fragments
