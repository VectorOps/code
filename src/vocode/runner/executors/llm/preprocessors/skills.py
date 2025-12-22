from __future__ import annotations

from typing import Any, List

from vocode.models import PreprocessorSpec, Mode
from vocode.project import Project
from vocode.state import Message
from vocode.runner.executors.llm.preprocessors.base import register_preprocessor


_DEFAULT_HEADER = (
    "\n\n## You have access to project skills that provide reusable expertise and workflows.\n"
    "Always use an appropriate skill instead of making direct tool calls. To use the skill, \n"
    "read the provided skill file and follow instructions."
    "The list of available skills is below:\n"
)


def _skills_preprocessor(
    project: Any, spec: PreprocessorSpec, messages: List[Message]
) -> List[Message]:
    # Only operate on the system prompt.
    if spec.mode != Mode.System:
        return messages

    # Enforce concrete project type; if not a real Project, do nothing.
    if not isinstance(project, Project):
        return messages

    skills = project.skills
    if not skills:
        return messages

    target: Message | None = None
    for msg in messages:
        if msg.role == "system":
            target = msg
            break

    if target is None:
        return messages

    opts = spec.options or {}
    header = opts.get("header")
    if not isinstance(header, str) or not header:
        header = _DEFAULT_HEADER

    # Avoid duplicate injection when preprocessors are applied multiple times.
    base_text = target.text or ""
    if header in base_text:
        return messages

    item_format = opts.get("item_format")
    if not isinstance(item_format, str) or not item_format:
        item_format = "- {name}: {description} (file: {path})"

    separator = opts.get("separator")
    if not isinstance(separator, str) or not separator:
        separator = "\n"

    lines: List[str] = []
    base_path = project.base_path
    for skill in skills:
        name = skill.name
        desc = skill.description

        path_value = skill.path
        try:
            path_str = str(path_value.relative_to(base_path))
        except ValueError:
            path_str = str(path_value)

        try:
            rendered = item_format.format(
                name=name,
                description=desc,
                path=path_str,
            )
        except Exception:
            rendered = f"- {name}: {desc} (file: {path_str})"
        lines.append(rendered.rstrip())

    block = header + separator.join(lines)

    if spec.prepend:
        target.text = f"{block}{base_text}"
    else:
        target.text = f"{base_text}{block}"

    return messages


register_preprocessor(
    name="skills",
    func=_skills_preprocessor,
    description=(
        "Injects a system prompt section listing available project skills loaded "
        "from the .vocode/skills directory. Supports custom header and item "
        "formatting via PreprocessorSpec.options."
    ),
)
