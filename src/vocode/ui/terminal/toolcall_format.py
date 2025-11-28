from __future__ import annotations

import shutil
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from prompt_toolkit.formatted_text import (
    AnyFormattedText,
    FormattedText,
    merge_formatted_text,
)
from prompt_toolkit.formatted_text.utils import fragment_list_width

from vocode.ui.terminal import colors
from vocode.settings import ToolCallFormatter  # type: ignore

# Formatter base class and registry
class BaseToolCallFormatter:
    def format_input(
        self,
        tool_name: str,
        arguments: Any,
        config: Optional[ToolCallFormatter],
        *,
        terminal_width: int,
        print_source: bool,
    ) -> AnyFormattedText:
        raise NotImplementedError

    def format_output(
        self,
        tool_name: str,
        result: Any,
        config: Optional[ToolCallFormatter],
        *,
        terminal_width: int,
    ) -> AnyFormattedText:
        raise NotImplementedError


_FORMATTER_REGISTRY: Dict[str, type[BaseToolCallFormatter]] = {}


def register_formatter(name: str, formatter_cls: type[BaseToolCallFormatter]) -> None:
    _FORMATTER_REGISTRY[name] = formatter_cls


# Defaults can be extended over time. Users can override/extend via Settings.tool_call_formatters.
DEFAULT_TOOL_CALL_FORMATTERS: Dict[str, "ToolCallFormatter"] = {
    "search_project": ToolCallFormatter(
        title="SymbolSearch",
        formatter="generic",
        options={"field": "query"},
    ),
    "read_files": ToolCallFormatter(
        title="ReadFile",
        formatter="generic",
        options={"field": "path"},
    ),
    "summarize_files": ToolCallFormatter(
        title="FileSummary",
        formatter="generic",
        options={"field": "paths"},
    ),
    "list_files": ToolCallFormatter(
        title="ListFiles",
        formatter="generic",
        options={"field": "patterns"},
    ),
}


def _extract_field_value(payload: Any, field: str) -> List[Any]:
    """
    Return [payload[field]] when payload is a mapping and contains the field.
    Otherwise return an empty list, indicating 'absent'.
    """
    from collections.abc import Mapping as _Mapping

    if isinstance(payload, _Mapping) and field in payload:  # type: ignore[arg-type]
        return [payload[field]]  # type: ignore[index]
    return []


def _stringify_value(v: Any) -> str:
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)


def _flatten_params(extracted: Iterable[Any]) -> List[str]:
    """
    Flatten a mixture of scalars and lists into a flat list of displayable parameter strings.
    - Lists are expanded into individual parameters.
    - Strings are wrapped in double quotes.
    """
    out: List[str] = []
    for item in extracted:
        if isinstance(item, (list, tuple)):
            for sub in item:
                out.append(_stringify_value(sub))
        else:
            out.append(_stringify_value(item))
    return out


def _build_param_fragments(params: Sequence[str]) -> FormattedText:
    """
    Build fragments for parameters: value (toolcall.parameter) and comma+space
    separators (toolcall.separator).
    """
    fragments: FormattedText = []
    for idx, p in enumerate(params):
        fragments.append(("class:toolcall.parameter", p))
        if idx < len(params) - 1:
            fragments.append(("class:toolcall.separator", ", "))
    return fragments


def _truncate_params_to_width(
    prefix: FormattedText,
    params: FormattedText,
    suffix: FormattedText,
    max_total_width: int,
) -> FormattedText:
    """
    Truncate only the parameter fragments to ensure the whole line does not exceed
    max_total_width. Adds an ellipsis "..." styled as toolcall.parameter before the
    closing parenthesis when truncation occurs. Keeps the closing parenthesis.
    """
    prefix_w = fragment_list_width(prefix)
    suffix_w = fragment_list_width(suffix)
    remaining_for_params = max(0, max_total_width - prefix_w - suffix_w)
    if remaining_for_params <= 0:
        ellipsis = [("class:toolcall.parameter", "...")]
        return list(prefix) + ellipsis + list(suffix)

    # Accumulate parameter fragments until out of space; then append ellipsis.
    acc: FormattedText = []
    used = 0
    truncated = False
    for style, text in params:
        w = len(text)
        if used + w <= remaining_for_params:
            acc.append((style, text))
            used += w
            continue

        # No room for full fragment; add ellipsis and stop.
        acc.append(("class:toolcall.parameter", "..."))
        truncated = True
        break

    if not acc:
        acc = [("class:toolcall.parameter", "...")]
        truncated = True

    # Ensure we don't exceed overall max width; (prefix + acc + suffix) enforced by construction.
    return list(prefix) + acc + list(suffix)


class GenericToolCallFormatter(BaseToolCallFormatter):
    """
    Default formatter that renders:

        <title>(<params>)

    where params come from a single argument field configured via
    config.options["field"], matching the previous render_tool_call behavior.
    """

    def format_input(
        self,
        tool_name: str,
        arguments: Any,
        config: Optional[ToolCallFormatter],
        *,
        terminal_width: int,
        print_source: bool,
    ) -> AnyFormattedText:
        max_total = max(0, terminal_width - 10)

        title = tool_name if config is None else config.title
        field_name: Optional[str] = None
        if config is not None and "field" in config.options:
            field_name = str(config.options["field"])

        prefix: FormattedText = [
            ("class:toolcall.name", title),
            ("class:toolcall.separator", "("),
        ]

        extracted: List[Any] = []
        if field_name:
            extracted = _extract_field_value(arguments, field_name)

        if not extracted:
            params: FormattedText = [("class:toolcall.parameter", "...")]
            suffix: FormattedText = [("class:toolcall.separator", ")")]
            preview = _truncate_params_to_width(prefix, params, suffix, max_total)
            if not print_source:
                return preview
            return merge_formatted_text([preview, "\n", colors.render_json(arguments), "\n"])

        param_strings = _flatten_params(extracted)
        param_fragments = _build_param_fragments(param_strings)
        suffix: FormattedText = [("class:toolcall.separator", ")")]
        preview = _truncate_params_to_width(prefix, param_fragments, suffix, max_total)

        if not print_source:
            return preview
        return merge_formatted_text([preview, "\n", colors.render_json(arguments), "\n"])

    def format_output(
        self,
        tool_name: str,
        result: Any,
        config: Optional[ToolCallFormatter],
        *,
        terminal_width: int,
    ) -> AnyFormattedText:
        # Simple default: pretty-print JSON result.
        return colors.render_json(result)


register_formatter("generic", GenericToolCallFormatter)


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
    fmts.update(DEFAULT_TOOL_CALL_FORMATTERS)
    if formatter_map:
        fmts.update(dict(formatter_map))

    cfg = fmts.get(tool_name)

    formatter_key = cfg.formatter if cfg is not None and cfg.formatter else "generic"
    formatter_cls = _FORMATTER_REGISTRY.get(formatter_key, GenericToolCallFormatter)
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
    fmts.update(DEFAULT_TOOL_CALL_FORMATTERS)
    if formatter_map:
        fmts.update(dict(formatter_map))

    cfg = fmts.get(tool_name)
    if cfg is None or not cfg.show_output:
        return None

    formatter_key = cfg.formatter if cfg.formatter else "generic"
    formatter_cls = _FORMATTER_REGISTRY.get(formatter_key, GenericToolCallFormatter)
    formatter = formatter_cls()

    return formatter.format_output(
        tool_name=tool_name,
        result=result,
        config=cfg,
        terminal_width=effective_width,
    )
