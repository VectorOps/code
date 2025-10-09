import types
import pytest
from prompt_toolkit.formatted_text import to_formatted_text

from vocode.ui.terminal import toolcall_format as tcf


def test_stringify_value_quotes_strings():
    assert tcf._stringify_value("hello") == '"hello"'
    assert tcf._stringify_value(123) == "123"
    assert tcf._stringify_value(True) == "True"


def test_flatten_params_mixed_and_list():
    inp = ["a", 1, [2, "b"]]
    out = tcf._flatten_params(inp)
    assert out == ['"a"', "1", "2", '"b"']


def test_build_param_fragments_styles_and_separators():
    params = ['"a"', "1"]
    frags = tcf._build_param_fragments(params)
    assert frags == [
        ("class:toolcall.parameter", '"a"'),
        ("class:toolcall.separator", ", "),
        ("class:toolcall.parameter", "1"),
    ]


def test_truncate_params_to_width_basic_truncation():
    prefix = [
        ("class:system.star", "* "),
        ("class:toolcall.name", "t"),
        ("class:toolcall.separator", "("),
    ]
    params = [
        ("class:toolcall.parameter", "abc"),
        ("class:toolcall.separator", ", "),
        ("class:toolcall.parameter", "def"),
    ]
    suffix = [("class:toolcall.separator", ")")]
    # prefix width = 2 + 1 + 1 = 4; suffix = 1; allow only 2 for params
    max_total = 7
    frags = tcf._truncate_params_to_width(prefix, params, suffix, max_total)
    assert frags == prefix + [("class:toolcall.parameter", "...")] + suffix


def test_truncate_params_to_width_no_truncation():
    prefix = [
        ("class:system.star", "* "),
        ("class:toolcall.name", "tool"),
        ("class:toolcall.separator", "("),
    ]
    params = [
        ("class:toolcall.parameter", '"a"'),
        ("class:toolcall.separator", ", "),
        ("class:toolcall.parameter", "1"),
    ]
    suffix = [("class:toolcall.separator", ")")]
    # Large enough to fit everything
    max_total = 80
    frags = tcf._truncate_params_to_width(prefix, params, suffix, max_total)
    assert frags == prefix + params + suffix


def test_render_tool_call_fallback_no_formatter():
    frags = tcf.render_tool_call("mytool", {"x": 1}, formatter_map=None, terminal_width=100)
    assert frags == [
        ("class:toolcall.name", "mytool"),
        ("class:toolcall.separator", "("),
        ("class:toolcall.parameter", "..."),
        ("class:toolcall.separator", ")"),
    ]


def test_render_tool_call_with_formatter_and_stub_jsonpath(monkeypatch):
    # Field-name extraction: use 'values' field directly.
    class DummyFormatter:
        def __init__(self, title, rule):
            self.title = title
            self.rule = rule
    formatter_map = {"mytool": DummyFormatter(title="Run It", rule="values")}
    args = {"values": ["foo", 42]}

    frags = tcf.render_tool_call("mytool", args, formatter_map=formatter_map, terminal_width=100)
    assert frags == [
        ("class:toolcall.name", "Run It"),
        ("class:toolcall.separator", "("),
        ("class:toolcall.parameter", '"foo"'),
        ("class:toolcall.separator", ", "),
        ("class:toolcall.parameter", "42"),
        ("class:toolcall.separator", ")"),
    ]


def test_render_tool_call_truncates_params(monkeypatch):
    class DummyFormatter:
        def __init__(self, title, rule):
            self.title = title
            self.rule = rule
    formatter_map = {"mytool": DummyFormatter(title="Run It", rule="values")}
    args = {"values": ["abcdefghij", "xyz"]}
    # terminal_width=20 -> max_total = 10
    # prefix width: "Run It" (6) + "(" (1) = 7
    # suffix width: 1
    # remaining_for_params = 0 -> immediate ellipsis
    frags = tcf.render_tool_call("mytool", args, formatter_map=formatter_map, terminal_width=20)
    assert frags == [
        ("class:toolcall.name", "Run It"),
        ("class:toolcall.separator", "("),
        ("class:toolcall.parameter", "..."),
        ("class:toolcall.separator", ")"),
    ]
def test_render_tool_call_missing_field_shows_ellipsis():
    class DummyFormatter:
        def __init__(self, title, rule):
            self.title = title
            self.rule = rule
    formatter_map = {"mytool": DummyFormatter(title="Run It", rule="values")}
    # 'values' absent -> expect "..."
    args = {"other": 1}
    frags = tcf.render_tool_call("mytool", args, formatter_map=formatter_map, terminal_width=100)
    assert frags == [
        ("class:toolcall.name", "Run It"),
        ("class:toolcall.separator", "("),
        ("class:toolcall.parameter", "..."),
        ("class:toolcall.separator", ")"),
    ]

def test_render_tool_call_print_source_appends_json():
    class DummyFormatter:
        def __init__(self, title, rule):
            self.title = title
            self.rule = rule
    formatter_map = {"mytool": DummyFormatter(title="Run It", rule="values")}
    args = {"values": ["foo", 42], "extra": {"a": 1}}
    frags = tcf.render_tool_call(
        "mytool",
        args,
        formatter_map=formatter_map,
        terminal_width=100,
        print_source=True,
    )
    # Convert to plain fragments to assert rendered JSON substrings.
    flat = to_formatted_text(frags)
    # Ensure preview part exists.
    assert ("class:toolcall.name", "Run It") in flat
    # Ensure JSON content appears somewhere after the preview.
    joined = "".join(text for _, text in flat)
    assert '"values"' in joined
    assert '"extra"' in joined
