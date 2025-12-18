from __future__ import annotations

from typing import Any, Iterable, List, Optional

from prompt_toolkit.formatted_text import (
    AnyFormattedText,
    FormattedText,
    merge_formatted_text,
)

from vocode.ui.terminal import colors
from vocode.settings import ToolCallFormatter  # type: ignore

from . import utils as _utils
from .base import BaseToolCallFormatter, register_formatter


class GenericToolCallFormatter(BaseToolCallFormatter):
    """Default formatter that renders ``<title>(<params>)``.

    Parameters are taken from one or more argument fields configured via
    ``config.options["field"]`` (string) or ``config.options["fields"]``
    (list of strings). When multiple fields are provided, their values are
    concatenated and displayed comma separated.
    """

    @staticmethod
    def _iter_field_names(config: Optional[ToolCallFormatter]) -> Iterable[str]:
        if config is None:
            return []

        opts = config.options or {}
        # Backward-compatible: single "field" key
        field = opts.get("field")
        if isinstance(field, str) and field:
            return [field]

        fields = opts.get("fields")
        if isinstance(fields, list):
            return [str(f) for f in fields if str(f)]

        return []

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

        prefix: FormattedText = [
            ("class:toolcall.name", title),
            ("class:toolcall.separator", "("),
        ]

        extracted: List[Any] = []
        for field_name in self._iter_field_names(config):
            extracted.extend(_utils._extract_field_value(arguments, field_name))

        if not extracted:
            params: FormattedText = [("class:toolcall.parameter", "...")]
            suffix: FormattedText = [("class:toolcall.separator", ")")]
            preview = _utils._truncate_params_to_width(
                prefix,
                params,
                suffix,
                max_total,
            )
            if not print_source:
                return preview
            return merge_formatted_text(
                [preview, "\n", colors.render_json(arguments), "\n"]
            )

        param_strings = _utils._flatten_params(extracted)
        param_fragments = _utils._build_param_fragments(param_strings)
        suffix: FormattedText = [("class:toolcall.separator", ")")]
        preview = _utils._truncate_params_to_width(
            prefix,
            param_fragments,
            suffix,
            max_total,
        )

        if not print_source:
            return preview
        return merge_formatted_text(
            [preview, "\n", colors.render_json(arguments), "\n"]
        )

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
