import json

import pytest

from vocode.state import Task, TaskStatus, TaskList
from vocode.tasklist import merge_tasks, get_task_list, save_task_list
from vocode.testing import ProjectSandbox
from vocode.tools import get_tool
from vocode.tools.base import ToolTextResponse, ToolResponseType
from vocode.settings import ToolSpec


def test_task_model_valid_status():
    task = Task(id="step-1", title="Investigate failing test", status=TaskStatus.in_progress)
    assert task.id == "step-1"
    assert task.title == "Investigate failing test"
    assert task.status is TaskStatus.in_progress


def test_task_model_invalid_status_raises():
    with pytest.raises(ValueError):
        # type: ignore[arg-type] – we intentionally pass an invalid status string
        Task(id="step-1", title="Bad status", status="unknown")


def test_merge_tasks_replace_plan():
    existing = TaskList(
        todos=[
            Task(id="step-1", title="Old", status=TaskStatus.pending),
        ]
    )
    new_tasks = [
        Task(id="step-2", title="New", status=TaskStatus.completed),
    ]

    merged = merge_tasks(existing, new_tasks, merge=False)

    assert [t.id for t in merged.todos] == ["step-2"]
    assert merged.todos[0].title == "New"
    assert merged.todos[0].status is TaskStatus.completed


def test_merge_tasks_update_and_append():
    existing = TaskList(
        todos=[
            Task(id="step-1", title="First", status=TaskStatus.pending),
            Task(id="step-2", title="Second", status=TaskStatus.pending),
        ]
    )
    new_tasks = [
        Task(id="step-1", title="First (updated)", status=TaskStatus.completed),
        Task(id="step-3", title="Third", status=TaskStatus.in_progress),
    ]

    merged = merge_tasks(existing, new_tasks, merge=True)

    assert [t.id for t in merged.todos] == ["step-1", "step-2", "step-3"]
    assert merged.todos[0].title == "First (updated)"
    assert merged.todos[0].status is TaskStatus.completed
    # Unmentioned existing task is preserved
    assert merged.todos[1].id == "step-2"


@pytest.mark.asyncio
async def test_tasklist_persistence_helpers(tmp_path):
    async with ProjectSandbox.create(tmp_path) as project:
        original = TaskList(
            todos=[
                Task(id="step-1", title="Persist me", status=TaskStatus.pending),
            ]
        )
        save_task_list(project, original)

        loaded = get_task_list(project)
        assert isinstance(loaded, TaskList)
        assert [t.id for t in loaded.todos] == ["step-1"]
        assert loaded.todos[0].title == "Persist me"


@pytest.mark.asyncio
async def test_update_plan_tool_run_and_persist(tmp_path):
    async with ProjectSandbox.create(tmp_path) as project:
        project.refresh_tools_from_registry()
        tool = get_tool("update_plan")
        assert tool is not None

        spec = ToolSpec(name="update_plan")
        args = {
            "merge": True,
            "todos": [
                {
                    "id": "step-1",
                    "title": "Investigate failing test",
                    "status": "in_progress",
                },
                {
                    "id": "step-2",
                    "title": "Apply fix and re-run tests",
                    "status": "pending",
                },
            ],
        }

        response = await tool.run(project, spec, args)
        assert isinstance(response, ToolTextResponse)
        assert response.type is ToolResponseType.text

        # Response must be structured JSON describing the current plan.
        data = json.loads(response.text or "")
        assert isinstance(data, dict)
        assert "todos" in data
        assert [t["id"] for t in data["todos"]] == ["step-1", "step-2"]
        assert data["todos"][0]["status"] == TaskStatus.in_progress.value
        assert data["todos"][1]["status"] == TaskStatus.pending.value

        # Verify tasks were persisted in project state
        task_list = get_task_list(project)
        assert [t.id for t in task_list.todos] == ["step-1", "step-2"]
        assert task_list.todos[0].status is TaskStatus.in_progress
        assert task_list.todos[1].status is TaskStatus.pending


@pytest.mark.asyncio
async def test_update_plan_openapi_spec(tmp_path):
    async with ProjectSandbox.create(tmp_path) as project:
        project.refresh_tools_from_registry()
        tool = get_tool("update_plan")
        assert tool is not None

        spec = ToolSpec(name="update_plan")
        schema = await tool.openapi_spec(project, spec)

        assert schema["name"] == "update_plan"
        params = schema["parameters"]
        assert params["type"] == "object"
        assert "todos" in params["properties"]

        todos_schema = params["properties"]["todos"]
        assert todos_schema["type"] == "array"

        item_schema = todos_schema["items"]
        props = item_schema["properties"]
        # All three properties are present, but only id and status are required.
        assert set(props.keys()) == {"id", "title", "status"}
        assert set(item_schema["required"]) == {"id", "status"}

        status_enum = props["status"]["enum"]
        assert set(status_enum) == {
            TaskStatus.pending.value,
            TaskStatus.in_progress.value,
            TaskStatus.completed.value,
        }


@pytest.mark.asyncio
async def test_single_in_progress_enforced_on_replace(tmp_path):
    async with ProjectSandbox.create(tmp_path) as project:
        project.refresh_tools_from_registry()
        tool = get_tool("update_plan")
        assert tool is not None

        spec = ToolSpec(name="update_plan")
        args = {
            "merge": False,
            "todos": [
                {"id": "step-1", "title": "First", "status": "in_progress"},
                {"id": "step-2", "title": "Second", "status": "in_progress"},
            ],
        }

        with pytest.raises(ValueError):
            await tool.run(project, spec, args)


@pytest.mark.asyncio
async def test_single_in_progress_enforced_on_merge(tmp_path):
    async with ProjectSandbox.create(tmp_path) as project:
        # Start with a valid plan that has a single in_progress task.
        initial = TaskList(
            todos=[
                Task(id="step-1", title="First", status=TaskStatus.in_progress),
                Task(id="step-2", title="Second", status=TaskStatus.pending),
            ]
        )
        save_task_list(project, initial)

        project.refresh_tools_from_registry()
        tool = get_tool("update_plan")
        assert tool is not None

        spec = ToolSpec(name="update_plan")
        # Try to mark another task as in_progress without updating the first one.
        args = {
            "merge": True,
            "todos": [
                {
                    "id": "step-2",
                    "status": "in_progress",
                    # Title omitted intentionally; should inherit from existing.
                }
            ],
        }

        with pytest.raises(ValueError):
            await tool.run(project, spec, args)


@pytest.mark.asyncio
async def test_merge_without_title_uses_existing_title(tmp_path):
    async with ProjectSandbox.create(tmp_path) as project:
        initial = TaskList(
            todos=[
                Task(id="step-1", title="Original title", status=TaskStatus.pending),
            ]
        )
        save_task_list(project, initial)

        project.refresh_tools_from_registry()
        tool = get_tool("update_plan")
        assert tool is not None

        spec = ToolSpec(name="update_plan")
        args = {
            "merge": True,
            "todos": [
                {
                    "id": "step-1",
                    "status": "completed",
                    # Title omitted: should keep "Original title".
                }
            ],
        }

        response = await tool.run(project, spec, args)
        assert isinstance(response, ToolTextResponse)

        task_list = get_task_list(project)
        assert [t.id for t in task_list.todos] == ["step-1"]
        assert task_list.todos[0].title == "Original title"
        assert task_list.todos[0].status is TaskStatus.completed


@pytest.mark.asyncio
async def test_merge_missing_title_for_new_task_errors(tmp_path):
    async with ProjectSandbox.create(tmp_path) as project:
        project.refresh_tools_from_registry()
        tool = get_tool("update_plan")
        assert tool is not None

        spec = ToolSpec(name="update_plan")
        args = {
            "merge": True,
            "todos": [
                {
                    "id": "step-1",
                    "status": "pending",
                    # No title and no existing task with this id -> error.
                }
            ],
        }

        with pytest.raises(ValueError):
            await tool.run(project, spec, args)


@pytest.mark.asyncio
async def test_replace_missing_title_errors(tmp_path):
    async with ProjectSandbox.create(tmp_path) as project:
        project.refresh_tools_from_registry()
        tool = get_tool("update_plan")
        assert tool is not None

        spec = ToolSpec(name="update_plan")
        args = {
            "merge": False,
            "todos": [
                {
                    "id": "step-1",
                    "status": "pending",
                    # Missing title is not allowed for non-merge requests.
                }
            ],
        }

        with pytest.raises(ValueError):
            await tool.run(project, spec, args)