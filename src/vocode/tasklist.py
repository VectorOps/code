from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

from vocode.state import Task, TaskList, TaskStatus

if TYPE_CHECKING:
    from vocode.project import Project

# Key under Project.project_state for storing the current TaskList.
_TASK_LIST_STATE_KEY = "task_list"


def get_task_list(project: "Project") -> TaskList:
    """Return the current TaskList from project state, or an empty one."""
    value = project.project_state.get(_TASK_LIST_STATE_KEY)
    if isinstance(value, TaskList):
        return value
    if isinstance(value, dict):
        # Best-effort coercion in case a dict was stored previously.
        return TaskList(**value)
    return TaskList()


def save_task_list(project: "Project", task_list: TaskList) -> None:
    """Persist the given TaskList into project state."""
    project.project_state.set(_TASK_LIST_STATE_KEY, task_list)


def merge_tasks(existing: TaskList, new_tasks: List[Task], merge: bool) -> TaskList:
    """
    Merge new_tasks into existing according to the merge flag.

    - If merge is False, the returned TaskList contains exactly new_tasks.
    - If merge is True:
      - Tasks with ids matching existing tasks are updated (title and status).
      - New task ids are appended after existing tasks, preserving input order.
      - Existing tasks not mentioned are kept as-is.
    """
    if not merge:
        return TaskList(todos=list(new_tasks))

    by_id: Dict[str, Task] = {t.id: t for t in existing.todos}
    for task in new_tasks:
        by_id[task.id] = task

    ordered: List[Task] = []
    seen: set[str] = set()

    # Preserve order of existing tasks where possible.
    for t in existing.todos:
        if t.id in by_id:
            ordered.append(by_id[t.id])
            seen.add(t.id)

    # Append new tasks that did not exist before, in input order.
    for t in new_tasks:
        if t.id not in seen:
            ordered.append(t)
            seen.add(t.id)

    return TaskList(todos=ordered)