import types
import pytest
from prompt_toolkit.formatted_text import to_formatted_text

from vocode.ui.terminal import toolcall_format as tcf
from vocode.settings import ToolCallFormatter


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
    formatter_map = {
        "mytool": ToolCallFormatter(
            title="Run It",
            formatter="generic",
            options={"field": "values"},
        )
    }
    args = {"values": ["foo", 42]}

    frags = tcf.render_tool_call(
        "mytool", args, formatter_map=formatter_map, terminal_width=100
    )
    assert frags == [
        ("class:toolcall.name", "Run It"),
        ("class:toolcall.separator", "("),
        ("class:toolcall.parameter", '"foo"'),
        ("class:toolcall.separator", ", "),
        ("class:toolcall.parameter", "42"),
        ("class:toolcall.separator", ")"),
    ]


def test_render_tool_call_truncates_params(monkeypatch):
    formatter_map = {
        "mytool": ToolCallFormatter(
            title="Run It",
            formatter="generic",
            options={"field": "values"},
        )
    }
    args = {"values": ["abcdefghij", "xyz"]}
    # terminal_width=20 -> max_total = 10
    # prefix width: "Run It" (6) + "(" (1) = 7
    # suffix width: 1
    # remaining_for_params = 0 -> immediate ellipsis
    frags = tcf.render_tool_call(
        "mytool", args, formatter_map=formatter_map, terminal_width=20
    )
    assert frags == [
        ("class:toolcall.name", "Run It"),
        ("class:toolcall.separator", "("),
        ("class:toolcall.parameter", "..."),
        ("class:toolcall.separator", ")"),
    ]
def test_render_tool_call_missing_field_shows_ellipsis():
    formatter_map = {
        "mytool": ToolCallFormatter(
            title="Run It",
            formatter="generic",
            options={"field": "values"},
        )
    }
    # 'values' absent -> expect "..."
    args = {"other": 1}
    frags = tcf.render_tool_call(
        "mytool", args, formatter_map=formatter_map, terminal_width=100
    )
    assert frags == [
        ("class:toolcall.name", "Run It"),
        ("class:toolcall.separator", "("),
        ("class:toolcall.parameter", "..."),
        ("class:toolcall.separator", ")"),
    ]

def test_render_tool_call_print_source_appends_json():
    formatter_map = {
        "mytool": ToolCallFormatter(
            title="Run It",
            formatter="generic",
            options={"field": "values"},
        )
    }
    args = {"values": ["foo", 42], "extra": {"a": 1}}
    frags = tcf.render_tool_call(
        "mytool",
        args,
        formatter_map=formatter_map,
        terminal_width=100,
        print_source=True,
    )
    flat = to_formatted_text(frags)
    assert ("class:toolcall.name", "Run It") in flat
    joined = "".join(text for _, text in flat)
    assert '"values"' in joined
    assert '"extra"' in joined


def test_render_tool_result_suppressed_when_show_output_false():
    formatter_map = {
        "mytool": ToolCallFormatter(
            title="Run It",
            formatter="generic",
            show_output=False,
        )
    }
    frags = tcf.render_tool_result(
        "mytool",
        {"x": 1},
        formatter_map=formatter_map,
        terminal_width=80,
    )
    assert frags is None


def test_render_tool_result_shown_when_show_output_true():
    formatter_map = {
        "mytool": ToolCallFormatter(
            title="Run It",
            formatter="generic",
            show_output=True,
        )
    }
    frags = tcf.render_tool_result(
        "mytool",
        {"x": 1},
        formatter_map=formatter_map,
        terminal_width=80,
    )
    assert frags is not None
    flat = to_formatted_text(frags)
    joined = "".join(text for _, text in flat)
    # Generic formatter prints JSON-ish result; ensure key appears.
    assert '"x"' in joined
