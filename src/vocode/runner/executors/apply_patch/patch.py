from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple
import os
import re

from vocode.runner.executors.apply_patch.models import FileApplyStatus


class ActionType(Enum):
    ADD = auto()
    UPDATE = auto()
    DELETE = auto()


@dataclass
class PatchAction:
    type: ActionType
    search: List[str] = field(default_factory=list)
    replace: List[str] = field(default_factory=list)
    start_line: Optional[int] = None


@dataclass
class Patch:
    actions: Dict[str, PatchAction] = field(default_factory=dict)


@dataclass
class PatchError:
    msg: str
    line: Optional[int] = None
    hint: Optional[str] = None
    filename: Optional[str] = None


@dataclass
class FileChange:
    type: ActionType
    old_content: Optional[str] = None
    new_content: Optional[str] = None


@dataclass
class Commit:
    changes: Dict[str, FileChange] = field(default_factory=dict)


FENCE_RE = re.compile(r"^```")
SEARCH_MARK = "<<<<<<< SEARCH"
SPLIT_MARK = "======="
REPLACE_MARK = ">>>>>>> REPLACE"


def _is_relative_path(p: str) -> bool:
    if not p:
        return False
    if p.startswith("/") or p.startswith("\\"):
        return False
    if re.match(r"^[A-Za-z]:[\\/]", p):
        return False
    norm = os.path.normpath(p)
    return not os.path.isabs(norm)


def parse_patch(text: str) -> Tuple[Patch, List[PatchError]]:
    """
    Parse fenced SEARCH/REPLACE blocks:

    ```<lang>
    <full/path/to/file>
    <<<<<<< SEARCH
    <contiguous current content>
    =======
    <replacement content>
    >>>>>>> REPLACE
    ```
    Adds: SEARCH empty, REPLACE has full content.
    Deletes: SEARCH has full content, REPLACE empty.
    Updates: both non-empty.
    """
    errors: List[PatchError] = []
    patch = Patch()
    lines = text.splitlines()
    i = 0
    seen_paths: set[str] = set()

    def add_error(msg: str, *, line: Optional[int] = None, hint: Optional[str] = None, filename: Optional[str] = None):
        errors.append(PatchError(msg=msg, line=line, hint=hint, filename=filename))

    while i < len(lines):
        line = lines[i]
        if not FENCE_RE.match(line.strip()):
            i += 1
            continue

        fence_start_line = i + 1
        i += 1
        if i >= len(lines):
            add_error("Unterminated code fence", line=fence_start_line, hint="Add closing `` for the patch block")
            break

        # First line inside fence must be the file path
        path_line_no = i + 1
        path = lines[i].strip()
        i += 1
        if not _is_relative_path(path):
            add_error(f"Path must be relative: {path!r}", line=path_line_no, hint="Use a relative repo path", filename=path)
            # Skip to next fence end
            while i < len(lines) and not FENCE_RE.match(lines[i].strip()):
                i += 1
            if i < len(lines):
                i += 1
            continue

        if path in seen_paths:
            add_error(f"Duplicate file entry: {path}", line=path_line_no, hint="Only one block per file path", filename=path)
            # Skip rest of block
            while i < len(lines) and not FENCE_RE.match(lines[i].strip()):
                i += 1
            if i < len(lines):
                i += 1
            continue
        seen_paths.add(path)

        # Expect SEARCH marker
        if i >= len(lines) or lines[i].strip() != SEARCH_MARK:
            add_error("Missing <<<<<<< SEARCH marker", line=i + 1, filename=path)
            # Skip to next fence end
            while i < len(lines) and not FENCE_RE.match(lines[i].strip()):
                i += 1
            if i < len(lines):
                i += 1
            continue
        i += 1

        # Collect SEARCH lines until SPLIT_MARK
        search_lines: List[str] = []
        while i < len(lines) and lines[i].strip() != SPLIT_MARK:
            # Stop if fence closed unexpectedly
            if FENCE_RE.match(lines[i].strip()):
                add_error("Missing ======= split marker", line=i + 1, filename=path)
                break
            search_lines.append(lines[i])
            i += 1

        if i >= len(lines) or lines[i].strip() != SPLIT_MARK:
            # Try to fast-forward to fence end
            while i < len(lines) and not FENCE_RE.match(lines[i].strip()):
                i += 1
            if i < len(lines):
                i += 1
            continue
        i += 1  # skip SPLIT_MARK

        # Collect REPLACE lines until REPLACE_MARK
        replace_lines: List[str] = []
        while i < len(lines) and lines[i].strip() != REPLACE_MARK:
            if FENCE_RE.match(lines[i].strip()):
                add_error("Missing >>>>>>> REPLACE marker", line=i + 1, filename=path)
                break
            replace_lines.append(lines[i])
            i += 1

        if i >= len(lines) or lines[i].strip() != REPLACE_MARK:
            while i < len(lines) and not FENCE_RE.match(lines[i].strip()):
                i += 1
            if i < len(lines):
                i += 1
            continue
        i += 1  # skip REPLACE_MARK

        # Expect closing fence
        if i >= len(lines) or not FENCE_RE.match(lines[i].strip()):
            add_error("Missing closing code fence ``", line=i + 1 if i < len(lines) else None, filename=path)
            # Attempt to continue scanning
            while i < len(lines) and not FENCE_RE.match(lines[i].strip()):
                i += 1
            if i < len(lines):
                i += 1
        else:
            i += 1  # consume closing fence

        # Normalize: treat a single empty line as empty content
        if len(search_lines) == 1 and search_lines[0] == "":
            search_lines = []
        if len(replace_lines) == 1 and replace_lines[0] == "":
            replace_lines = []

        # Determine action type (after normalization above)
        if search_lines and not replace_lines:
            action_type = ActionType.DELETE
        elif replace_lines and not search_lines:
            action_type = ActionType.ADD
        elif search_lines and replace_lines:
            action_type = ActionType.UPDATE
        else:
            add_error("Empty patch block (no SEARCH and no REPLACE content)", line=path_line_no, filename=path)
            continue

        patch.actions[path] = PatchAction(
            type=action_type,
            search=search_lines,
            replace=replace_lines,
            start_line=path_line_no,
        )

    return patch, errors


def load_files(paths: List[str], open_fn: Callable[[str], str]) -> Tuple[Dict[str, str], List[PatchError]]:
    files: Dict[str, str] = {}
    errs: List[PatchError] = []
    for path in paths:
        try:
            files[path] = open_fn(path)
        except Exception as e:
            errs.append(
                PatchError(
                    msg=f"Failed to read file: {path}",
                    hint=f"{type(e).__name__}: {e}",
                    filename=path,
                )
            )
    return files, errs


def _join_lines(lines: List[str], *, eol: bool) -> str:
    s = "\n".join(lines)
    return s + ("\n" if eol else "")


def _find_subsequence(hay: List[str], needle: List[str]) -> Optional[int]:
    if not needle:
        return None
    n, m = len(hay), len(needle)
    if m > n:
        return None
    for start in range(0, n - m + 1):
        if hay[start : start + m] == needle:
            return start
    return None


def build_commits(patch: Patch, files: Dict[str, str]) -> Tuple[List[Commit], List[PatchError], Dict[str, FileApplyStatus]]:
    commits: List[Commit] = []
    errors: List[PatchError] = []
    changes: Dict[str, FileChange] = {}
    status_map: Dict[str, FileApplyStatus] = {}

    def add_error(msg: str, *, line: Optional[int] = None, hint: Optional[str] = None, filename: Optional[str] = None):
        errors.append(PatchError(msg=msg, line=line, hint=hint, filename=filename))

    for path, action in patch.actions.items():
        if action.type == ActionType.ADD:
            new_content = _join_lines(action.replace, eol=False)
            changes[path] = FileChange(type=ActionType.ADD, new_content=new_content)
            status_map[path] = FileApplyStatus.Create
            continue

        if action.type == ActionType.DELETE:
            changes[path] = FileChange(type=ActionType.DELETE)
            status_map[path] = FileApplyStatus.Delete
            continue

        # UPDATE
        original = files.get(path)
        if original is None:
            add_error(
                f"No loaded content for file: {path}",
                hint="Ensure the file exists and is readable for update.",
                filename=path,
                line=action.start_line,
            )
            status_map[path] = FileApplyStatus.PartialUpdate
            continue

        had_eol = original.endswith("\n")
        file_lines = original.splitlines()
        start_idx = _find_subsequence(file_lines, action.search)
        if start_idx is None:
            add_error(
                f"Failed to locate exact SEARCH block in {path}",
                hint="SEARCH content must match current file exactly (contiguous block).",
                filename=path,
                line=action.start_line,
            )
            status_map[path] = FileApplyStatus.PartialUpdate
            # Do not add a change if we cannot locate the block
            continue

        end_idx = start_idx + len(action.search)
        new_lines = file_lines[:start_idx] + action.replace + file_lines[end_idx:]
        new_content = _join_lines(new_lines, eol=had_eol)
        changes[path] = FileChange(type=ActionType.UPDATE, old_content=original, new_content=new_content)
        status_map[path] = FileApplyStatus.Update

    if changes:
        commits.append(Commit(changes=changes))

    return commits, errors, status_map


def apply_commits(
    commits: List[Commit],
    write_fn: Callable[[str, str], None],
    delete_fn: Callable[[str], None],
) -> List[PatchError]:
    errors: List[PatchError] = []
    for commit in commits:
        for path, change in commit.changes.items():
            try:
                if change.type == ActionType.ADD or change.type == ActionType.UPDATE:
                    write_fn(path, change.new_content or "")
                elif change.type == ActionType.DELETE:
                    delete_fn(path)
                else:
                    errors.append(
                        PatchError(
                            msg=f"Unknown change type for {path}",
                            hint="Supported types are Add/Update/Delete",
                            filename=path,
                        )
                    )
            except Exception as e:
                errors.append(
                    PatchError(
                        msg=f"Failed to apply change to file: {path}",
                        hint=f"{type(e).__name__}: {e}",
                        filename=path,
                    )
                )
    return errors


def process_patch(
    text: str,
    open_fn: Callable[[str], str],
    write_fn: Callable[[str, str], None],
    delete_fn: Callable[[str], None],
) -> Tuple[Dict[str, FileApplyStatus], List[PatchError]]:
    # 1) Parse
    patch, parse_errors = parse_patch(text)
    if parse_errors:
        return {}, parse_errors

    # 2) Load files needed for UPDATE actions only
    to_read = [p for p, a in patch.actions.items() if a.type == ActionType.UPDATE]
    files, read_errors = load_files(to_read, open_fn)
    if read_errors:
        return {}, read_errors

    # 3) Build commits
    commits, build_errors, status_map = build_commits(patch, files)

    # 4) Apply commits (partial application OK)
    apply_errors = apply_commits(commits, write_fn, delete_fn)

    return status_map, [*build_errors, *apply_errors]
