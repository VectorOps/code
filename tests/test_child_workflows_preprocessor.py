from pathlib import Path
from typing import List

import pytest

from vocode.runner.executors.llm.preprocessors.base import get_preprocessor
from vocode.models import PreprocessorSpec, Mode
from vocode.project import Project
from vocode.settings import Settings, WorkflowConfig
from vocode.state import Message


@pytest.fixture
def base_messages() -> List[Message]:
    return [Message(role="system", text="system prompt")]


def _build_settings_with_workflows(child_allowlist: List[str] | None) -> Settings:
    parent = WorkflowConfig(
        description="Parent workflow",
        nodes=[],
        edges=[],
        agent_workflows=child_allowlist,
    )
    child_allowed = WorkflowConfig(
        description="Allowed child",
        nodes=[],
        edges=[],
    )
    child_other = WorkflowConfig(
        description="Other child",
        nodes=[],
        edges=[],
    )
    return Settings(
        workflows={
            "parent": parent,
            "child_allowed": child_allowed,
            "child_other": child_other,
        }
    )


def _make_project(settings: Settings, current_workflow: str | None) -> Project:
    prj = Project(
        base_path=Path("."),
        config_relpath=Path(".vocode/config.yaml"),
        settings=settings,
    )
    prj.current_workflow = current_workflow
    return prj


def test_child_workflows_preprocessor_uses_allowlist(
    base_messages: List[Message],
) -> None:
    settings = _build_settings_with_workflows(child_allowlist=["child_allowed"])
    project = _make_project(settings=settings, current_workflow="parent")

    pp = get_preprocessor("child_workflows")
    assert pp is not None

    spec = PreprocessorSpec(name="child_workflows", mode=Mode.System)
    out_messages = pp.func(project, spec, list(base_messages))

    text = out_messages[0].text
    assert "You have access to specialized agents." in text
    assert "- child_allowed: Allowed child" in text
    assert "- child_other: Other child" not in text


def test_child_workflows_preprocessor_lists_all_when_no_allowlist(
    base_messages: List[Message],
) -> None:
    settings = _build_settings_with_workflows(child_allowlist=None)
    project = _make_project(settings=settings, current_workflow="parent")

    pp = get_preprocessor("child_workflows")
    assert pp is not None

    spec = PreprocessorSpec(name="child_workflows", mode=Mode.System)
    out_messages = pp.func(project, spec, list(base_messages))

    text = out_messages[0].text
    # Parent itself should not be listed
    assert "- parent:" not in text
    # All non-parent workflows should be listed
    assert "- child_allowed: Allowed child" in text
    assert "- child_other: Other child" in text


def test_child_workflows_preprocessor_does_not_reinject(
    base_messages: List[Message],
) -> None:
    settings = _build_settings_with_workflows(child_allowlist=["child_allowed"])
    project = _make_project(settings=settings, current_workflow="parent")

    pp = get_preprocessor("child_workflows")
    assert pp is not None

    spec = PreprocessorSpec(name="child_workflows", mode=Mode.System)
    first = pp.func(project, spec, list(base_messages))
    second = pp.func(project, spec, first)

    assert second[0].text == first[0].text


def test_child_workflows_preprocessor_noop_in_user_mode(
    base_messages: List[Message],
) -> None:
    settings = _build_settings_with_workflows(child_allowlist=["child_allowed"])
    project = _make_project(settings=settings, current_workflow="parent")

    pp = get_preprocessor("child_workflows")
    assert pp is not None

    spec = PreprocessorSpec(name="child_workflows", mode=Mode.User)
    out_messages = pp.func(project, spec, list(base_messages))

    assert out_messages[0].text == base_messages[0].text


def test_child_workflows_preprocessor_custom_header_and_format(
    base_messages: List[Message],
) -> None:
    # No explicit allowlist -> both child workflows are listed, allowing us to
    # verify the custom separator between items.
    settings = _build_settings_with_workflows(child_allowlist=None)
    project = _make_project(settings=settings, current_workflow="parent")

    pp = get_preprocessor("child_workflows")
    assert pp is not None

    custom_header = "\n\nUse these workflows:\n"
    spec = PreprocessorSpec(
        name="child_workflows",
        mode=Mode.System,
        options={
            "header": custom_header,
            "item_format": "* {name}",
            "separator": " | ",
        },
    )

    out_messages = pp.func(project, spec, list(base_messages))
    text = out_messages[0].text

    # Custom header should be used instead of the default.
    assert custom_header in text
    assert (
        "When possible, prefer starting a custom agent over using generic tools."
        not in text
    )

    # Custom formatting and separator should be applied.
    assert "* child_allowed" in text
    assert "* child_other" in text
    assert " | " in text


def test_child_workflows_preprocessor_custom_header_dedupes(
    base_messages: List[Message],
) -> None:
    settings = _build_settings_with_workflows(child_allowlist=["child_allowed"])
    project = _make_project(settings=settings, current_workflow="parent")

    pp = get_preprocessor("child_workflows")
    assert pp is not None

    custom_header = "\n\nCustom header once:\n"
    spec = PreprocessorSpec(
        name="child_workflows",
        mode=Mode.System,
        options={"header": custom_header},
    )

    first = pp.func(project, spec, list(base_messages))
    second = pp.func(project, spec, first)

    assert second[0].text == first[0].text
