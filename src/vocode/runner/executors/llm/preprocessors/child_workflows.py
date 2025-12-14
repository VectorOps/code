from __future__ import annotations

from typing import Any, List

from vocode.models import PreprocessorSpec, Mode
from vocode.project import Project
from vocode.state import Message
from vocode.runner.executors.llm.preprocessors.base import register_preprocessor


_DEFAULT_HEADER = (
    "\n\nYou have access to custom workflows (specialized agents). "
    "When possible, prefer starting a custom workflow over using generic tools. "
    'Call the "start_workflow" tool with one of the workflow names below:\n'
)


def _get_child_workflow_pairs(project: Project) -> List[tuple[str, str]]:
    """Return (name, description) pairs for workflows available as children of the current workflow."""
    settings = project.settings
    if settings is None:
        return []

    workflows = settings.workflows
    if not workflows:
        return []

    parent_name = project.current_workflow
    if parent_name is None or parent_name not in workflows:
        return []

    parent_cfg = workflows[parent_name]
    allowlist = parent_cfg.child_workflows

    if allowlist is not None:
        names: List[str] = [name for name in allowlist if name in workflows]
    else:
        # When no explicit allowlist is defined, treat all other workflows as available children.
        names = [name for name in workflows.keys() if name != parent_name]

    result: List[tuple[str, str]] = []
    for name in names:
        cfg = workflows.get(name)
        if cfg is None:
            continue
        desc = cfg.description or ""
        result.append((name, desc))
    return result


def _child_workflows_preprocessor(
    project: Any, spec: PreprocessorSpec, messages: List[Message]
) -> List[Message]:
    # Only operate on the system prompt.
    if spec.mode != Mode.System:
        return messages

    # Enforce concrete project type; if not a real Project, do nothing.
    if not isinstance(project, Project):
        return messages

    pairs = _get_child_workflow_pairs(project)
    if not pairs:
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
        item_format = "- {name}: {description}"

    separator = opts.get("separator")
    if not isinstance(separator, str) or not separator:
        separator = "\n"

    lines: List[str] = []
    for name, desc in pairs:
        try:
            rendered = item_format.format(name=name, description=desc)
        except Exception:
            # Fall back to the built-in formatting if the custom template fails.
            rendered = f"- {name}: {desc}"
        lines.append(rendered.rstrip())

    block = header + separator.join(lines)

    if spec.prepend:
        target.text = f"{block}{base_text}"
    else:
        target.text = f"{base_text}{block}"

    return messages


register_preprocessor(
    name="child_workflows",
    func=_child_workflows_preprocessor,
    description=(
        "Injects a system prompt section listing available child workflows for the "
        "current workflow, based on Settings.workflows and WorkflowConfig.child_workflows. "
        "Supports custom header and item formatting via PreprocessorSpec.options."
    ),
)
