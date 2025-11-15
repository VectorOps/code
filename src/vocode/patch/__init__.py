from __future__ import annotations

import pathlib
from typing import Callable, Dict, List, Tuple, Optional

from .models import FileApplyStatus
from .v4a import (
    process_patch as v4a_process_patch,
    DIFF_SYSTEM_INSTRUCTION as V4A_SYSTEM_INSTRUCTION,
)
from .patch import (
    process_patch as sr_process_patch,
    DIFF_PATCH_SYSTEM_INSTRUCTION as PATCH_SYSTEM_INSTRUCTION,
)
from .v4a import DiffError

# Internal registry of supported patch formats
_REGISTRY: Dict[str, Dict[str, object]] = {
    "v4a": {
        "handler": v4a_process_patch,
        "system_prompt": V4A_SYSTEM_INSTRUCTION,
    },
    "patch": {
        "handler": sr_process_patch,
        "system_prompt": PATCH_SYSTEM_INSTRUCTION,
    },
}


def get_supported_formats() -> Tuple[str, ...]:
    return tuple(_REGISTRY.keys())


def get_system_instruction(fmt: str) -> str:
    key = (fmt or "").lower()
    entry = _REGISTRY.get(key)
    if not entry:
        raise ValueError(f"Unsupported patch format: {fmt}")
    return entry["system_prompt"]  # type: ignore[return-value]


def apply_patch(
    fmt: str,
    text: str,
    open_fn: Callable[[str], str],
    write_fn: Callable[[str, str], None],
    delete_fn: Callable[[str], None],
) -> Tuple[Dict[str, FileApplyStatus], List[object]]:
    """
    Callback-based helper; primarily for tests and custom environments.
    Returns (status_map, errors).
    """
    key = (fmt or "").lower()
    entry = _REGISTRY.get(key)
    if not entry:
        raise ValueError(f"Unsupported patch format: {fmt}")
    handler = entry["handler"]  # type: ignore[assignment]
    return handler(text, open_fn, write_fn, delete_fn)  # type: ignore[misc]


def apply_patch_to_repo(
    fmt: str,
    text: str,
    base_path: pathlib.Path,
) -> Tuple[str, str, Dict[str, str], Dict[str, FileApplyStatus], List[object]]:
    """
    Apply a patch to disk under base_path using the specified format ('v4a' or 'patch').
    Returns (summary_text, outcome_name, changes_map, status_map, errors).
    changes_map values: 'created' | 'updated' | 'deleted'
    """
    key = (fmt or "").lower()
    entry = _REGISTRY.get(key)
    if not entry:
        raise ValueError(f"Unsupported patch format: {fmt}")
    handler = entry["handler"]  # type: ignore[assignment]

    # Path helpers (safe resolution)
    def _resolve_safe_path(rel: str) -> pathlib.Path:
        if rel.startswith("/") or rel.startswith("~"):
            raise DiffError(f"Absolute paths are not allowed: {rel}")
        abs_path = (base_path / rel).resolve()
        base_resolved = base_path.resolve()
        if str(abs_path).startswith(str(base_resolved)):
            return abs_path
        raise DiffError(f"Path escapes project root: {rel}")

    # Change tracker for refresh classification
    changes_map: Dict[str, str] = {}

    def _record(rel: str, change: str) -> None:
        prev = changes_map.get(rel)
        if prev is None:
            changes_map[rel] = change
            return
        # precedence: deleted > updated > created
        if change == "deleted":
            changes_map[rel] = change
        elif change == "updated" and prev not in ("deleted",):
            changes_map[rel] = change
        # created never overrides

    # IO callbacks
    def open_fn(rel: str) -> str:
        path = _resolve_safe_path(rel)
        with path.open("rt", encoding="utf-8") as fh:
            return fh.read()

    def write_fn(rel: str, content: str) -> None:
        path = _resolve_safe_path(rel)
        existed = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wt", encoding="utf-8") as fh:
            fh.write(content)
        _record(rel, "updated" if existed else "created")

    def delete_fn(rel: str) -> None:
        path = _resolve_safe_path(rel)
        # Only record on success
        path.unlink(missing_ok=True)
        _record(rel, "deleted")

    # Process
    statuses, errs = handler(text, open_fn, write_fn, delete_fn)  # type: ignore[misc]

    # Summarize
    created = sorted([f for f, s in statuses.items() if s == FileApplyStatus.Create])
    updated_full = sorted([f for f, s in statuses.items() if s == FileApplyStatus.Update])
    updated_partial = sorted([f for f, s in statuses.items() if s == FileApplyStatus.PartialUpdate])
    deleted = sorted([f for f, s in statuses.items() if s == FileApplyStatus.Delete])

    outcome = "success"
    lines: List[str] = []

    if errs:
        applied_files = set(changes_map.keys())
        applied_any = bool(applied_files)
        applied_created = sorted([f for f in created if f in applied_files])
        applied_updated_full = sorted([f for f in updated_full if f in applied_files])
        applied_updated_partial = sorted([f for f in updated_partial if f in applied_files])
        applied_deleted = sorted([f for f in deleted if f in applied_files])
        failed_files = sorted({getattr(e, "filename", None) for e in errs if getattr(e, "filename", None)})
        not_applied_failed = sorted([f for f in failed_files if f not in applied_files])

        if not applied_any:
            lines.append("Patch application failed. No changes were applied.")
        else:
            lines.append("Patch application completed with errors. Summary:")
            if applied_created:
                lines.append("Added files (fully applied):")
                for f in applied_created:
                    lines.append(f"* {f}")
            if applied_updated_full:
                lines.append("Fully updated files:")
                for f in applied_updated_full:
                    lines.append(f"* {f}")
            if applied_updated_partial:
                lines.append("Partially updated files (some chunks failed):")
                for f in applied_updated_partial:
                    lines.append(f"* {f}")
            if applied_deleted:
                lines.append("Deleted files (fully applied):")
                for f in applied_deleted:
                    lines.append(f"* {f}")

        targets_for_fix = sorted(set(applied_updated_partial) | set(not_applied_failed))
        if targets_for_fix:
            lines.append("Please regenerate patch chunks for the failed parts in these files:")
            for f in targets_for_fix:
                lines.append(f"* {f}")
            lines.append(
                "If there were other files that were not mentioned in this response, regenerate chunks for them as well. Make sure you read the files."
            )

        lines.append("Errors:")
        for e in errs:
            msg = getattr(e, "msg", str(e))
            hint = getattr(e, "hint", None)
            filename = getattr(e, "filename", None)
            line_no = getattr(e, "line", None)
            loc = ""
            if filename and line_no is not None:
                loc = f"{filename}:{line_no}: "
            elif filename:
                loc = f"{filename}: "
            lines.append(f"* {loc}{msg}")
            if hint:
                lines.append(f"  Hint: {hint}")
        outcome = "fail"
    else:
        lines = ["Applied patch successfully."]
        if created:
            lines.append("Added files:")
            for f in created:
                lines.append(f"* {f}")
        if updated_full:
            lines.append("Fully updated files:")
            for f in updated_full:
                lines.append(f"* {f}")
        if updated_partial:
            lines.append("Partially updated files:")
            for f in updated_partial:
                lines.append(f"* {f}")
        if deleted:
            lines.append("Deleted files:")
            for f in deleted:
                lines.append(f"* {f}")

    summary = "\n".join(lines)
    return summary, outcome, changes_map, statuses, errs