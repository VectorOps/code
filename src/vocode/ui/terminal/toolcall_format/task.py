from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from prompt_toolkit.formatted_text import AnyFormattedText, FormattedText

from vocode.settings import ToolCallFormatter  # type: ignore

from .base import BaseToolCallFormatter, register_formatter

_STATUS_PENDING = "pending"
_STATUS_IN_PROGRESS = "in_progress"
_STATUS_COMPLETED = "completed"


class TaskToolFormatter(BaseToolCallFormatter):
    """
    Formatter for the `update_plan` task tool.

    Renders a task list like:

        Task plan:
        [ ] Implement feature X (step-1)
        [>] Fix bug Y (step-2)
        [x] Write tests (step-3)
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
        title = tool_name if config is None else config.title
        tasks = self._parse_tasks_from_arguments(arguments)
        if not tasks:
            # Fall back to generic behavior when we cannot extract tasks.
            from .generic import GenericToolCallFormatter

            return GenericToolCallFormatter().format_input(
                tool_name=tool_name,
                arguments=arguments,
                config=config,
                terminal_width=terminal_width,
                print_source=print_source,
            )
        return self._format_tasks(tasks, title)

    def format_output(
        self,
        tool_name: str,
        result: Any,
        config: Optional[ToolCallFormatter],
        *,
        terminal_width: int,
    ) -> AnyFormattedText:
        title = tool_name if config is None else config.title
        tasks = self._parse_tasks_from_result(result)
        if not tasks:
            from .generic import GenericToolCallFormatter

            return GenericToolCallFormatter().format_output(
                tool_name=tool_name,
                result=result,
                config=config,
                terminal_width=terminal_width,
            )
        return self._format_tasks(tasks, title)

    @staticmethod
    def _parse_tasks_sequence(raw: Any) -> List[Dict[str, str]]:
        tasks: List[Dict[str, str]] = []
        if not isinstance(raw, list):
            return tasks

        for item in raw:
            if not isinstance(item, dict):
                continue
            raw_id = item.get("id")
            raw_title = item.get("title")
            raw_status = item.get("status")

            task_id = str(raw_id) if raw_id is not None else ""
            title = str(raw_title or raw_id or "")
            status = str(raw_status or _STATUS_PENDING)

            tasks.append(
                {
                    "id": task_id,
                    "title": title,
                    "status": status,
                }
            )
        return tasks

    def _parse_tasks_from_arguments(self, arguments: Any) -> List[Dict[str, str]]:
        if not isinstance(arguments, dict):
            return []
        todos = arguments.get("todos")
        return self._parse_tasks_sequence(todos)

    def _parse_tasks_from_result(self, result: Any) -> List[Dict[str, str]]:
        # Prefer structured dicts: {"todos": [...]}
        if isinstance(result, dict):
            if "todos" in result:
                return self._parse_tasks_sequence(result.get("todos"))
            text = result.get("text")
            if isinstance(text, str):
                try:
                    payload = json.loads(text)
                except Exception:
                    return []
                if isinstance(payload, dict) and "todos" in payload:
                    return self._parse_tasks_sequence(payload.get("todos"))
                return []

        # Raw JSON string from tool text response.
        if isinstance(result, str):
            try:
                payload = json.loads(result)
            except Exception:
                return []
            if isinstance(payload, dict) and "todos" in payload:
                return self._parse_tasks_sequence(payload.get("todos"))
        return []

    @staticmethod
    def _status_prefix(status: str) -> str:
        if status == _STATUS_COMPLETED:
            return "[x]"
        if status == _STATUS_IN_PROGRESS:
            return "[>]"
        return "[ ]"

    def _format_tasks(
        self,
        tasks: List[Dict[str, str]],
        title: str,
    ) -> AnyFormattedText:
        fragments: FormattedText = []
        fragments.append(("class:toolcall.name", f"{title}:"))
        fragments.append(("", "\n"))

        for task in tasks:
            prefix = self._status_prefix(task.get("status", ""))
            label_parts: List[str] = [prefix, " ", task.get("title", "")]
            task_id = task.get("id")
            if task_id:
                label_parts.append(f" ({task_id})")
            line = "".join(label_parts)
            fragments.append(("class:tasklist.item", line))
            fragments.append(("", "\n"))

        return fragments


register_formatter("task", TaskToolFormatter)
