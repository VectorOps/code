from prompt_toolkit.formatted_text import to_formatted_text
from vocode.ui.terminal.toolbar import build_prompt


def _text_from_html(html_obj) -> str:
    fragments = to_formatted_text(html_obj)
    # fragments are tuples of (style, text, [mouse_handler])
    return "".join(frag[1] for frag in fragments)


def test_prompt_shows_command_when_no_pending_request():
    dummy_ui = object()  # UI not used when pending_req is None
    html = build_prompt(dummy_ui, None)
    text = _text_from_html(html)
    assert "(command)" in text