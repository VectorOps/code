from pathlib import Path
from textwrap import dedent
from typing import List

import pytest

from vocode.project import Project
from vocode.runner.executors.llm.preprocessors.base import get_preprocessor
from vocode.models import PreprocessorSpec, Mode
from vocode.settings import Settings
from vocode.state import Message


def _write_skill(tmp_path: Path, name: str, description: str) -> Path:
    skills_dir = tmp_path / ".vocode" / "skills" / name
    skills_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skills_dir / "SKILL.md"
    skill_path.write_text(
        dedent(
            f"""\
            ---
            name: {name}
            description: {description}
            ---

            # {name}
            """
        ),
        encoding="utf-8",
    )
    return skill_path


def test_discover_skills_on_project_init(tmp_path: Path) -> None:
    _write_skill(tmp_path, "skill-one", "First skill")
    _write_skill(tmp_path, "skill-two", "Second skill")

    settings = Settings(workflows={})
    project = Project(
        base_path=tmp_path,
        config_relpath=Path(".vocode/config.yaml"),
        settings=settings,
    )

    names = sorted(s.name for s in project.skills)
    assert names == ["skill-one", "skill-two"]


@pytest.fixture
def base_messages() -> List[Message]:
    return [Message(role="system", text="system prompt")]


def _make_project_with_skill(tmp_path: Path) -> Project:
    _write_skill(tmp_path, "example-skill", "Example description")
    _write_skill(tmp_path, "second-skill", "Second description")
    settings = Settings(workflows={})
    return Project(
        base_path=tmp_path,
        config_relpath=Path(".vocode/config.yaml"),
        settings=settings,
    )


def test_skills_preprocessor_injects_list(
    tmp_path: Path, base_messages: List[Message]
) -> None:
    project = _make_project_with_skill(tmp_path)

    pp = get_preprocessor("skills")
    assert pp is not None

    spec = PreprocessorSpec(name="skills", mode=Mode.System)
    out_messages = pp.func(project, spec, list(base_messages))

    text = out_messages[0].text
    assert "You have access to project skills" in text
    assert "- example-skill: Example description" in text


def test_skills_preprocessor_noop_without_system_message(tmp_path: Path) -> None:
    project = _make_project_with_skill(tmp_path)

    pp = get_preprocessor("skills")
    assert pp is not None

    spec = PreprocessorSpec(name="skills", mode=Mode.System)
    messages = [Message(role="user", text="hello")]
    out_messages = pp.func(project, spec, list(messages))

    assert out_messages[0].text == messages[0].text


def test_skills_preprocessor_does_not_reinject(
    tmp_path: Path, base_messages: List[Message]
) -> None:
    project = _make_project_with_skill(tmp_path)

    pp = get_preprocessor("skills")
    assert pp is not None

    spec = PreprocessorSpec(name="skills", mode=Mode.System)
    first = pp.func(project, spec, list(base_messages))
    second = pp.func(project, spec, first)

    assert second[0].text == first[0].text


def test_skills_preprocessor_respects_custom_header_and_format(
    tmp_path: Path, base_messages: List[Message]
) -> None:
    project = _make_project_with_skill(tmp_path)

    pp = get_preprocessor("skills")
    assert pp is not None

    custom_header = "\n\nUse these skills:\n"
    spec = PreprocessorSpec(
        name="skills",
        mode=Mode.System,
        options={
            "header": custom_header,
            "item_format": "* {name}",
            "separator": " | ",
        },
    )

    out_messages = pp.func(project, spec, list(base_messages))
    text = out_messages[0].text

    assert custom_header in text
    assert "* example-skill" in text
    assert "* second-skill" in text
    assert " | " in text
