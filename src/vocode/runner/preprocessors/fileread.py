from __future__ import annotations

from pathlib import Path
from typing import Any, List

from vocode.models import PreprocessorSpec
from vocode.runner.preprocessors.base import register_preprocessor


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


def _fileread_preprocessor(project: Any, spec: PreprocessorSpec, text: str, **_: Any) -> str:
    """
    Reads files listed in options['paths'] (or options['files']) and concatenates their contents.
    - Silently skips absolute, escaping, non-existent, or non-file paths.
    - Honors spec.prepend to place content before/after input text.
    - No separators are added (mirrors FilereadExecutor concatenation).
    """
    opts = spec.options or {}
    paths: List[str] = []
    raw = opts.get("paths")
    if isinstance(raw, list):
        paths = [p for p in raw if isinstance(p, str)]
    else:
        # Fallback to "files" for convenience
        raw_files = opts.get("files")
        if isinstance(raw_files, list):
            paths = [p for p in raw_files if isinstance(p, str)]

    # Collect contents in order, skipping invalid entries
    parts: List[str] = []
    for rel in paths:
        full = _validate_relpath(rel, project)
        if not full:
            continue
        parts.append(_read_file_text(full))

    inject = "".join(parts)
    if not inject:
        return text

    base_text = text or ""
    if spec.prepend:
        return f"{inject}{base_text}"
    return f"{base_text}{inject}"


# Register at import time
register_preprocessor(
    name="file_read",
    func=_fileread_preprocessor,
    description="Reads files from options.paths (or options.files) and concatenates contents; skips invalid/missing paths.",
)
