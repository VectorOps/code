from __future__ import annotations

from typing import Any, Optional

from prompt_toolkit.formatted_text import AnyFormattedText, FormattedText, merge_formatted_text

from vocode.settings import ToolCallFormatter  # type: ignore
from vocode.ui.terminal import colors

from .base import BaseToolCallFormatter, register_formatter


class ApplyPatchToolFormatter(BaseToolCallFormatter):
    """Formatter for the ``apply_patch`` tool.

    The tool has a single ``text`` parameter containing a unified diff.
    We render that diff as a fenced ``patch`` code block so it is highlighted
    using the diff lexer (see CODE_FENCE_LANG_MAP in colors.py).
    """

    def format_input(
        self,
        tool_name: str,
        arguments: Any,
        config: Optional[ToolCallFormatter],
        *,
        terminal_width: int,  # noqa: ARG002  - unused but kept for API compatibility
        print_source: bool,  # noqa: ARG002  - we always render the patch body
    ) -> AnyFormattedText:
        title = tool_name if config is None else config.title

        patch_text: str = ""
        if isinstance(arguments, dict) and "text" in arguments:
            value = arguments.get("text")
            if isinstance(value, str):
                patch_text = value
            else:
                patch_text = str(value)

        # If we cannot extract a patch body, fall back to a minimal header.
        if not patch_text:
            return [("class:toolcall.name", title)]

        header: FormattedText = [
            ("class:toolcall.name", title),
            ("", "\n"),
        ]

        fenced = f"```patch\n{patch_text}\n```"
        body = colors.render_markdown(fenced)
        return merge_formatted_text([header, body])

    def format_output(
        self,
        tool_name: str,  # noqa: ARG002
        result: Any,
        config: Optional[ToolCallFormatter],  # noqa: ARG002
        *,
        terminal_width: int,  # noqa: ARG002
    ) -> AnyFormattedText:
        # Reuse the generic JSON renderer for tool results.
        return colors.render_json(result)


register_formatter("patch", ApplyPatchToolFormatter)
