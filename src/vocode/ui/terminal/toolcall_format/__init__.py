from __future__ import annotations

import shutil
from typing import Any, Dict, Mapping, Optional

from prompt_toolkit.formatted_text import AnyFormattedText

from vocode.settings import ToolCallFormatter  # type: ignore

from . import base as _base
from . import generic as _generic  # noqa: F401  # ensure registration
from . import task as _task  # noqa: F401  # ensure registration
from . import utils as _utils

# Re-export public API for callers/tests that import vocode.ui.terminal.toolcall_format
BaseToolCallFormatter = _base.BaseToolCallFormatter
DEFAULT_TOOL_CALL_FORMATTERS = _base.DEFAULT_TOOL_CALL_FORMATTERS
register_formatter = _base.register_formatter
get_registered_formatter = _base.get_registered_formatter

# Re-export helpers used by tests
_extract_field_value = _utils._extract_field_value
_stringify_value = _utils._stringify_value
_flatten_params = _utils._flatten_params
_build_param_fragments = _utils._build_param_fragments
_truncate_params_to_width = _utils._truncate_params_to_width


def _resolve_formatter_cls(
    formatter_key: str,
) -> type[BaseToolCallFormatter]:
    formatter_cls = _base.get_registered_formatter(formatter_key)
    if formatter_cls is not None:
        return formatter_cls
    return _generic.GenericToolCallFormatter


def render_tool_call(
    tool_name: str,
    arguments: Any,
    formatter_map: Optional[Mapping[str, "ToolCallFormatter"]] = None,
    *,
    terminal_width: Optional[int] = None,
    print_source: bool = False,
) -> AnyFormattedText:
    """
    Render a tool call preview by resolving a ToolCallFormatter config and
    dispatching to the configured formatter implementation.
    """
    effective_width = (
        terminal_width
        if terminal_width is not None
        else shutil.get_terminal_size(fallback=(80, 24)).columns
    )

    fmts: Dict[str, ToolCallFormatter] = {}
    fmts.update(_base.DEFAULT_TOOL_CALL_FORMATTERS)
    if formatter_map:
        fmts.update(dict(formatter_map))

    cfg = fmts.get(tool_name)

    formatter_key = cfg.formatter if cfg is not None and cfg.formatter else "generic"
    formatter_cls = _resolve_formatter_cls(formatter_key)
    formatter = formatter_cls()

    return formatter.format_input(
        tool_name=tool_name,
        arguments=arguments,
        config=cfg,
        terminal_width=effective_width,
        print_source=print_source,
    )


def render_tool_result(
    tool_name: str,
    result: Any,
    formatter_map: Optional[Mapping[str, "ToolCallFormatter"]] = None,
    *,
    terminal_width: Optional[int] = None,
) -> Optional[AnyFormattedText]:
    """
    Render a tool call result by resolving ToolCallFormatter and dispatching to
    the configured formatter implementation. Returns None when the tool is not
    configured to show output (show_output=False).
    """
    effective_width = (
        terminal_width
        if terminal_width is not None
        else shutil.get_terminal_size(fallback=(80, 24)).columns
    )

    fmts: Dict[str, ToolCallFormatter] = {}
    fmts.update(_base.DEFAULT_TOOL_CALL_FORMATTERS)
    if formatter_map:
        fmts.update(dict(formatter_map))

    cfg = fmts.get(tool_name)
    if cfg is None or not cfg.show_output:
        return None

    formatter_key = cfg.formatter if cfg.formatter else "generic"
    formatter_cls = _resolve_formatter_cls(formatter_key)
    formatter = formatter_cls()

    return formatter.format_output(
        tool_name=tool_name,
        result=result,
        config=cfg,
        terminal_width=effective_width,
    )
