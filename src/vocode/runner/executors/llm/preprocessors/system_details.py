from __future__ import annotations

from typing import Any, List, Dict
import os
import platform

from vocode.models import Mode, PreprocessorSpec
from vocode.runner.executors.llm.preprocessors.base import register_preprocessor
from vocode.state import Message


def _detect_shell() -> str:
    """Best-effort, cross-platform shell detection."""
    shell = os.environ.get("SHELL")
    if shell:
        return shell

    comspec = os.environ.get("COMSPEC")
    if comspec:
        return comspec

    return "unknown"


def _build_system_details_text(options: Dict[str, Any]) -> str:
    include_os = bool(options.get("include_os", True))
    include_shell = bool(options.get("include_shell", True))

    parts: list[str] = []

    if include_os:
        os_name = platform.system() or "unknown"
        os_release = platform.release() or ""
        os_display = f"{os_name} {os_release}".strip()
        parts.append(f"- OS: {os_display}")

    if include_shell:
        shell = _detect_shell()
        parts.append(f"- Shell: {shell}")

    if not parts:
        return ""

    header = options.get("header", "System details:")
    if not isinstance(header, str) or not header:
        header = "System details:"

    return "\n".join([header, *parts])


def _system_details_preprocessor(
    project: Any, spec: PreprocessorSpec, messages: List[Message]
) -> List[Message]:
    """
    Injects current OS and shell into either the system or user
    message, based on spec.mode.

    Honors:
    - spec.mode: Mode.System (default) or Mode.User (last user message).
    - spec.prepend: whether to prepend or append to the target message.
    - options.separator: separator between existing and injected text (default: "\\n\\n").
    - options.include_os/include_shell: booleans to toggle components.
    - options.header: header line for the injected block.
    """
    opts = spec.options or {}
    inject = _build_system_details_text(opts)
    if not inject:
        return messages

    target_message: Message | None = None

    if not messages:
        role = "system" if spec.mode == Mode.System else "user"
        target_message = Message(text="", role=role)
        messages.append(target_message)
    else:
        if spec.mode == Mode.System:
            for msg in messages:
                if msg.role == "system":
                    target_message = msg
                    break
        elif spec.mode == Mode.User:
            for msg in reversed(messages):
                if msg.role == "user":
                    target_message = msg
                    break

    if not target_message:
        return messages

    base_text = target_message.text or ""

    # Simple dedupe: if the injected block is already present, skip.
    if inject in base_text:
        return messages

    separator = opts.get("separator", "\n\n")
    if not isinstance(separator, str):
        separator = "\n\n"

    if spec.prepend:
        if base_text:
            target_message.text = f"{inject}{separator}{base_text}"
        else:
            target_message.text = inject
    else:
        if base_text:
            target_message.text = f"{base_text}{separator}{inject}"
        else:
            target_message.text = inject

    return messages


register_preprocessor(
    name="system_details",
    func=_system_details_preprocessor,
    description=(
        "Injects a block describing current operating system and shell "
        "into the system or last user message; supports prepend/append and basic options."
    ),
)
