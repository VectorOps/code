from typing import Optional
import shutil

from prompt_toolkit.formatted_text import HTML, to_formatted_text
from prompt_toolkit.formatted_text.utils import fragment_list_width

from vocode.ui.base import UIState
from vocode.ui.proto import UIReqRunEvent
from vocode.runner.models import (
    PACKET_MESSAGE_REQUEST,
    PACKET_TOOL_CALL,
    PACKET_FINAL_MESSAGE,
)
from vocode.graph.models import Confirmation


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
                return getattr(n, "confirmation", None)
        return None
    except Exception:
        return None


def _input_hint(ui: UIState, pending_req: Optional[UIReqRunEvent]) -> str:
    if pending_req is None or not pending_req.event.input_requested:
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


def build_prompt(ui: UIState, pending_req: Optional[UIReqRunEvent]) -> HTML:
    hint = _input_hint(ui, pending_req)
    prompt_text = f"{hint if hint else ''}> "
    return HTML(f'<style fg="ansiyellow">{prompt_text}</style>')


def build_toolbar(ui: UIState, pending_req: Optional[UIReqRunEvent]) -> HTML:
    wf = ui.selected_workflow_name or "-"
    node = ui.current_node_name or "-"
    status = getattr(ui.status, "value", str(ui.status))

    left_markup = f"<b>{wf}</b>@{node} [{status}]"
    p = getattr(ui, "acc_prompt_tokens", 0)
    c = getattr(ui, "acc_completion_tokens", 0)
    cost = getattr(ui, "acc_cost_dollars", 0.0)
    right_markup = f"p:{_abbr_int(int(p))} r:{_abbr_int(int(c))} ${_abbr_cost(float(cost))}"

    # Measure widths to compute spacing.
    left_html = HTML(left_markup)
    right_html = HTML(right_markup)
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    lw = fragment_list_width(to_formatted_text(left_html))
    rw = fragment_list_width(to_formatted_text(right_html))
    space = 1
    if width > lw + rw:
        space = width - lw - rw
    spaces = " " * max(space, 1)

    combined = f"{left_markup}{spaces}{right_markup}"

    # Toolbar color policy:
    # - normal: white bg, black text
    # - input requested: green bg, black text
    if pending_req is not None and pending_req.event.input_requested:
        return HTML(f'<style bg="ansigreen" fg="ansiblack">{combined}</style>')
    return HTML(f'<style bg="ansiwhite" fg="ansiblack">{combined}</style>')
