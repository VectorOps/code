from __future__ import annotations

from typing import Any, Dict, Optional

from prompt_toolkit.formatted_text import AnyFormattedText

from vocode.settings import ToolCallFormatter  # type: ignore


# Defaults can be extended over time. Users can override/extend via Settings.tool_call_formatters.
DEFAULT_TOOL_CALL_FORMATTERS: Dict[str, "ToolCallFormatter"] = {
    "search_project": ToolCallFormatter(
        title="SymbolSearch",
        formatter="generic",
        options={"fields": ["query"]},
    ),
    "read_files": ToolCallFormatter(
        title="ReadFile",
        formatter="generic",
        options={"fields": ["path"]},
    ),
    "summarize_files": ToolCallFormatter(
        title="FileSummary",
        formatter="generic",
        options={"fields": ["paths"]},
    ),
    "list_files": ToolCallFormatter(
        title="ListFiles",
        formatter="generic",
        options={"fields": ["pattern"]},
    ),
    "exec": ToolCallFormatter(
        title="Exec",
        formatter="generic",
        options={"fields": ["command"]},
    ),
    "apply_patch": ToolCallFormatter(
        title="ApplyPatch",
        formatter="patch",
        show_output=True,
    ),
    "run_agent": ToolCallFormatter(
        title="RunAgent",
        formatter="generic",
        options={"fields": ["name", "text"]},
    ),
    # Task plan tool: show a task list with a dedicated formatter.
    "update_plan": ToolCallFormatter(
        title="Task plan",
        formatter="task",
        show_output=True,
    ),
}


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


def get_registered_formatter(name: str) -> Optional[type[BaseToolCallFormatter]]:
    return _FORMATTER_REGISTRY.get(name)
