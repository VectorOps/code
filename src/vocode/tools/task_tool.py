from __future__ import annotations

import json
from typing import Any, Dict, List, TYPE_CHECKING

from vocode.tools.base import BaseTool, ToolTextResponse
from vocode.settings import ToolSpec
from vocode.state import Task, TaskStatus
from vocode.tasklist import get_task_list, save_task_list, merge_tasks

if TYPE_CHECKING:
    from vocode.project import Project


class UpdatePlanTool(BaseTool):
    """
    Manage the current coding task plan for this project.

    The tool maintains an ordered list of tasks (todos) identified by stable ids.
    It can either replace the plan or merge updates into the existing plan.
    """

    name = "update_plan"

    async def run(self, spec: ToolSpec, args: Any):
        if not isinstance(spec, ToolSpec):
            raise TypeError("update_plan requires a resolved ToolSpec")
        if self.prj is None:
            raise RuntimeError("update_plan requires a project")

        if not isinstance(args, dict):
            raise TypeError("update_plan expects arguments as an object")

        merge_flag = bool(args.get("merge", True))
        raw_todos = args.get("todos")

        if not isinstance(raw_todos, list) or not raw_todos:
            raise ValueError("update_plan requires a non-empty 'todos' list")

        current = get_task_list(self.prj)

        todos: List[Task] = []
        for item in raw_todos:
            if not isinstance(item, dict):
                raise TypeError(
                    "Each todo must be an object with id, status, and optional title"
                )

            raw_id = item.get("id")
            if not isinstance(raw_id, str) or not raw_id:
                raise ValueError("Each todo must provide a non-empty 'id' string")

            raw_status = item.get("status")
            if raw_status is None:
                raise ValueError("Each todo must provide a 'status'")
            try:
                status = TaskStatus(raw_status)
            except ValueError as exc:
                raise ValueError(
                    "Invalid status; must be one of: "
                    f"{TaskStatus.pending.value}, "
                    f"{TaskStatus.in_progress.value}, "
                    f"{TaskStatus.completed.value}"
                ) from exc

            title = item.get("title")

            if merge_flag:
                # Title is optional for merge requests; if omitted, we only update status.
                if title is None:
                    existing_task = next(
                        (t for t in current.todos if t.id == raw_id),
                        None,
                    )
                    if existing_task is None:
                        raise ValueError(
                            "Title is required when adding a new task id during merge "
                            f"(missing title for id='{raw_id}')."
                        )
                    title = existing_task.title
            else:
                # For non-merge (replace) requests, a title is always required.
                if not isinstance(title, str) or not title:
                    raise ValueError(
                        "Title is required for all tasks when merge is false "
                        f"(missing or empty title for id='{raw_id}')."
                    )

            todos.append(Task(id=raw_id, title=title, status=status))

        updated = merge_tasks(current, todos, merge_flag)

        # Enforce that only one task can be in progress in the final plan.
        in_progress_count = sum(
            1 for task in updated.todos if task.status == TaskStatus.in_progress
        )
        if in_progress_count > 1:
            raise ValueError(
                "Only one task can have status 'in_progress' at a time in the task plan."
            )

        save_task_list(self.prj, updated)

        # Return structured JSON so the LLM can reason about the plan.
        payload = {
            "todos": [
                {
                    "id": task.id,
                    "title": task.title,
                    "status": task.status.value,
                }
                for task in updated.todos
            ]
        }

        return ToolTextResponse(text=json.dumps(payload))

    async def openapi_spec(self, spec: ToolSpec) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Update or replace the current task plan for this coding session. "
                "Use stable ids (e.g. 'step-1') so you can update task status over time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "merge": {
                        "type": "boolean",
                        "description": (
                            "If true, merge these todos into the existing plan "
                            "(updating tasks by id and appending new ones). "
                            "If false, replace the existing plan entirely."
                        ),
                        "default": True,
                    },
                    "todos": {
                        "type": "array",
                        "description": (
                            "Ordered list of tasks representing the plan. "
                            "Each task must have a stable id, title, and status."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": (
                                        "Stable identifier for the task "
                                        "(e.g. 'step-1')."
                                    ),
                                },
                                "title": {
                                    "type": "string",
                                    "description": (
                                        "Short description of the task. "
                                        "Optional for merge requests; when omitted, "
                                        "only the status is updated for an existing task."
                                    ),
                                },
                                "status": {
                                    "type": "string",
                                    "description": (
                                        "Current status of this task. "
                                        "Must be one of: pending, in_progress, completed."
                                    ),
                                    "enum": [
                                        TaskStatus.pending.value,
                                        TaskStatus.in_progress.value,
                                        TaskStatus.completed.value,
                                    ],
                                },
                            },
                            # Title is optional at the schema level; non-merge calls
                            # still require a title at runtime.
                            "required": ["id", "status"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["todos"],
                "additionalProperties": False,
            },
        }


# Register tool on import
try:
    from .base import register_tool

    register_tool(UpdatePlanTool.name, UpdatePlanTool)
except Exception:
    # Avoid import-time failures if the registry is not available.
    pass
