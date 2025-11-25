from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

from vocode.models import Mode, PreprocessorSpec
from vocode.runner.executors.llm.preprocessors.base import (
    register_preprocessor,
)
from vocode.state import Message


def _validate_relpath(rel: str, project) -> Path | None:
    """
    Validate a project-relative path.
    Returns the resolved absolute Path if valid; otherwise None (to skip).
    """
    try:
        p = Path(rel)
        if p.is_absolute():
            return None
        base = project.base_path.resolve()
        full = (project.base_path / p).resolve()
        # Ensure within project root
        _ = full.relative_to(base)
        if not full.exists() or not full.is_file():
            return None
        return full
    except Exception:
        return None


def _read_file_text(full: Path) -> str:
    try:
        return full.read_text(encoding="utf-8", errors="replace")
    except Exception:
        try:
            return full.read_bytes().decode("utf-8", errors="replace")
        except Exception:
            return ""


def _fileread_preprocessor(
    project: Any, spec: PreprocessorSpec, messages: List[Message]
) -> List[Message]:
    """
    Reads files listed in options['paths'] (or options['files']) and concatenates their contents.
    - Silently skips absolute, escaping, non-existent, or non-file paths.
    - Injects content into a message determined by spec.mode ('system' or 'user').
    - If message list is empty, creates a new message.
    - No separators are added between files (mirrors FileReadExecutor concatenation).
    - Supports a per-file prepend template via options['prepend_template'] which defaults to
      "User provided {filename}:\n"; set to None to disable.
    """

    opts = spec.options or {}
    # Parse paths from options['paths'] or fallback to options['files']
    paths: list[str] = []
    raw = opts.get("paths")
    if isinstance(raw, str):
        paths = [raw]
    elif isinstance(raw, Sequence):
        paths = [p for p in raw if isinstance(p, str)]
    else:
        raw_files = opts.get("files")
        if isinstance(raw_files, str):
            paths = [raw_files]
        elif isinstance(raw_files, Sequence):
            paths = [p for p in raw_files if isinstance(p, str)]

    # Determine per-file prepend template (None disables)
    prepend_template: Optional[str]
    if "prepend_template" in opts and opts.get("prepend_template") is None:
        prepend_template = None
    else:
        # Use provided template if any; otherwise default
        prepend_template = opts.get("prepend_template") or "User provided {filename}:\n"

    # Collect contents in order, skipping invalid entries
    parts: List[str] = []
    for rel in paths:
        full = _validate_relpath(rel, project)
        if not full:
            continue
        if prepend_template is not None:
            try:
                parts.append(prepend_template.format(filename=full.name))
            except Exception:
                # If formatting fails, fall back to raw template string
                parts.append(str(prepend_template))
        parts.append(_read_file_text(full))

    inject = "".join(parts)
    if not inject:
        return messages

    # Find target message
    target_message: Optional[Message] = None
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

    # Update message
    if target_message:
        base_text = target_message.text or ""
        sep = opts.get("separator", "\n\n")
        # Prevent duplicate reinjection only if the combined block (with separator)
        # already exists in the base text; avoids false positives on substrings.
        if spec.prepend:
            already = f"{inject}{sep}"
        else:
            already = f"{sep}{inject}"
        if already and already in base_text:
            return messages

        if spec.prepend:
            target_message.text = f"{inject}{sep}{base_text}"
        else:
            target_message.text = f"{base_text}{sep}{inject}"

    return messages


# Register at import time
register_preprocessor(
    name="file_read",
    func=_fileread_preprocessor,
    description="Reads files from options.paths (or options.files), optionally prepends per-file template, and concatenates contents; skips invalid/missing paths.",
)
