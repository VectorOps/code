from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple
import re
import os
import difflib

from .models import FileApplyStatus


class DiffError(ValueError):
    """Any problem detected while parsing or applying a patch."""


class ActionType(Enum):
    ADD = auto()
    UPDATE = auto()
    DELETE = auto()


@dataclass
class Chunk:
    anchors: List[str] = field(default_factory=list)
    prefix: List[str] = field(default_factory=list)
    suffix: List[str] = field(default_factory=list)
    deletions: List[str] = field(default_factory=list)
    additions: List[str] = field(default_factory=list)
    start_line: Optional[int] = None


@dataclass
class PatchAction:
    type: ActionType = ActionType.UPDATE
    chunks: List[Chunk] = field(default_factory=list)


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
    move_path: Optional[str] = None


@dataclass
class Commit:
    changes: Dict[str, FileChange] = field(default_factory=dict)


BEGIN_MARKER = "*** Begin Patch"
END_MARKER = "*** End Patch"
FILE_HEADER_RE = re.compile(r"^\*\*\*\s+(Add|Update|Delete)\s+File:\s+(.+)$")
ANCHOR_RE = re.compile(r"^@@(.*)$")
ANCHOR_PREFIX = "@@"
DELETE_PREFIX = "-"
ADD_PREFIX = "+"
CONTEXT_PREFIX = " "

@dataclass
class PartialMatch:
    matched: int = -1
    start: Optional[int] = None
    phase: str = ""
    expected: str = ""
    actual: str = ""
    fail_line: Optional[int] = None
    prefix_matched: int = 0
    deletions_matched: int = 0
    suffix_matched: int = 0

@dataclass
class MatchResult:
    ok: bool
    start: int
    matched: int
    expected: str
    actual: str
    fail_index: int  # 0-based index into file_lines where the mismatch/EOF occurred
    prefix_matched: int
    deletions_matched: int
    suffix_matched: int


def _is_relative_path(p: str) -> bool:
    # Reject absolute POSIX and Windows (drive or UNC) paths
    if not p:
        return False
    if p.startswith("/") or p.startswith("\\"):
        return False
    if re.match(r"^[A-Za-z]:[\\/]", p):
        return False
    # Normalize and ensure it doesn't escape upward with absolute root
    norm = os.path.normpath(p)
    # Keep relative; allow ../ segments, but forbid path becoming absolute after norm
    return not os.path.isabs(norm)


def parse_v4a_patch(text: str) -> Tuple[Patch, List[PatchError]]:
    """
    Best-effort, resilient V4A parser.
    - Ignores any text before the first BEGIN marker and after the matching END marker.
    - Validates:
        * One BEGIN and one END marker (records errors if multiples).
        * Each file appears exactly once (duplicates are errors; later ones ignored).
        * Paths must be relative.
        * Delete sections must not contain change/content lines.
        * Context chunk boundaries must be unambiguous (require @@ between multiple chunks).
        * Context chunks may use any number of prefix/suffix lines (3 is typical).
    Returns a Patch model and a list of PatchError.
    """
    errors: List[PatchError] = []
    lines = text.splitlines()

    begin_idxs = [i for i, l in enumerate(lines) if l.strip() == BEGIN_MARKER]
    end_idxs = [i for i, l in enumerate(lines) if l.strip() == END_MARKER]

    def add_error(
        msg: str,
        *,
        line: Optional[int] = None,
        hint: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        errors.append(PatchError(msg=msg, line=line, hint=hint, filename=filename))

    if not begin_idxs:
        return Patch(), [PatchError("Missing *** Begin Patch", line=None, hint="Ensure patch is wrapped with *** Begin Patch / *** End Patch")]
    if not end_idxs:
        return Patch(), [PatchError("Missing *** End Patch", line=None, hint="Ensure patch is wrapped with *** Begin Patch / *** End Patch")]
    if len(begin_idxs) > 1:
        extra_begins = [str(i + 1) for i in begin_idxs[1:]]
        add_error("Multiple *** Begin Patch markers found; using the first",
                  line=begin_idxs[1] + 1,
                  hint=f"Extra BEGIN markers at lines: {', '.join(extra_begins)}")
    # Choose the first end after the first begin; if none, error
    first_begin = begin_idxs[0]
    ends_after_begin = [i for i in end_idxs if i > first_begin]
    if not ends_after_begin:
        add_error("No *** End Patch after *** Begin Patch",
                  line=first_begin + 1,
                  hint="Add *** End Patch after this line")
        return Patch(), errors
    if len(end_idxs) > 1:
        first_end_after_begin = ends_after_begin[0]
        extras = [i for i in end_idxs if i != first_end_after_begin]
        add_error("Multiple *** End Patch markers found; using the first after begin",
                  line=(extras[0] + 1) if extras else (first_end_after_begin + 1),
                  hint=f"Extra END markers at lines: {', '.join(str(i + 1) for i in extras)}")
    first_end = ends_after_begin[0]

    content = lines[first_begin + 1 : first_end]

    patch = Patch()
    seen_paths: set[str] = set()

    current_path: Optional[str] = None
    current_action: Optional[PatchAction] = None
    skip_current_file: bool = False

    # Chunk assembly state
    pending_anchors: List[str] = []
    current_chunk: Optional[Chunk] = None
    chunk_has_mods: bool = False  # any +/- seen in current_chunk

    def finish_chunk_if_any():
        nonlocal current_chunk, chunk_has_mods, pending_anchors
        if current_chunk is None:
            # Drop stray anchors if never used
            pending_anchors = []
            return
        # Validate chunk has actual modifications
        if not (current_chunk.additions or current_chunk.deletions):
            add_error(
                f"Chunk in {current_path or '<no file>'} has no modifications",
                line=current_chunk.start_line,
                hint="Each chunk must include at least one +/- line",
                filename=current_path,
            )
            # Drop it
        else:
            # Add action must not include prefix/suffix context
            if current_action is not None and current_action.type == ActionType.ADD:
                if current_chunk.prefix or current_chunk.suffix:
                    add_error(
                        f"Add file section for {current_path} must not contain context",
                        line=current_chunk.start_line,
                        hint="Remove context lines for Add sections; only use + lines",
                        filename=current_path,
                    )
            if current_action is not None:
                current_action.chunks.append(current_chunk)
        current_chunk = None
        chunk_has_mods = False
        pending_anchors = []

    def start_chunk_if_needed():
        nonlocal current_chunk, chunk_has_mods, pending_anchors
        if current_chunk is None:
            current_chunk = Chunk(anchors=list(pending_anchors), start_line=current_line_num)
            pending_anchors = []
            chunk_has_mods = False

    for current_line_num, raw_line in enumerate(content, start=first_begin + 2):
        # Detect new file header
        m_header = FILE_HEADER_RE.match(raw_line)
        if m_header is not None:
            # Close previous file and chunk
            finish_chunk_if_any()
            current_chunk = None
            chunk_has_mods = False
            pending_anchors = []

            action_word, path = m_header.group(1), m_header.group(2).strip()
            current_path = path
            current_action = None
            skip_current_file = False

            # Validate path
            if not _is_relative_path(path):
                add_error(
                    f"Path must be relative: {path!r}",
                    line=current_line_num,
                    hint="Use a relative path, not absolute",
                    filename=path,
                )
                skip_current_file = True
                continue
            # Validate uniqueness
            if path in seen_paths:
                add_error(
                    f"Duplicate file entry: {path}",
                    line=current_line_num,
                    hint="Each file may only appear once in a patch",
                    filename=path,
                )
                skip_current_file = True
                continue
            seen_paths.add(path)

            # Initialize action
            if action_word == "Add":
                current_action = PatchAction(type=ActionType.ADD)
            elif action_word == "Update":
                current_action = PatchAction(type=ActionType.UPDATE)
            elif action_word == "Delete":
                current_action = PatchAction(type=ActionType.DELETE)
            else:
                add_error(
                    f"Unknown action '{action_word}' for file {path}",
                    line=current_line_num,
                    hint="Use Add/Update/Delete",
                    filename=path,
                )
                skip_current_file = True
                continue

            # Record action
            patch.actions[path] = current_action
            continue

        # Ignore anything until we have a current file
        if current_path is None or current_action is None or skip_current_file:
            continue

        # Handle Delete action: it must not have content
        if current_action.type == ActionType.DELETE:
            if raw_line.startswith(DELETE_PREFIX) or raw_line.startswith(ADD_PREFIX) or raw_line.startswith(ANCHOR_PREFIX):
                add_error(
                    f"Delete file section for {current_path} must not contain changes",
                    line=current_line_num,
                    hint="Delete sections must not include anchors or +/- lines",
                    filename=current_path,
                )
            # Otherwise ignore lines inside delete section
            continue

        # Non-delete: Update/Add have content blocks
        if raw_line.startswith(ANCHOR_PREFIX):
            # Starting a new chunk or setting anchors for the next chunk
            # If there is an open chunk, finish it before starting a new one
            if current_chunk is not None:
                finish_chunk_if_any()
            anchor_text = raw_line[len(ANCHOR_PREFIX):]
            # Normalize: remove one leading space if present
            if anchor_text.startswith(" "):
                anchor_text = anchor_text[1:]
            pending_anchors.append(anchor_text)
            continue

        # Diff lines
        if raw_line.startswith(DELETE_PREFIX) or raw_line.startswith(ADD_PREFIX):
            # If we are inside a chunk and have already recorded modifications,
            # and we encounter new +/- without an intervening anchor, treat as ambiguous multi-chunk.
            if current_chunk is not None and chunk_has_mods and current_chunk.suffix:
                add_error(
                    f"Ambiguous or overlapping blocks in {current_path}: missing @@ between chunks",
                    line=current_line_num,
                    hint="Separate adjacent change blocks with an @@ anchor",
                    filename=current_path,
                )
                finish_chunk_if_any()

            start_chunk_if_needed()

            if raw_line.startswith(DELETE_PREFIX):
                current_chunk.deletions.append(raw_line[len(DELETE_PREFIX):])
                chunk_has_mods = True
            else:
                current_chunk.additions.append(raw_line[len(ADD_PREFIX):])
                chunk_has_mods = True
            continue

        # Context line or invalid starter
        # Allow blank lines as empty context; otherwise require a single leading space.
        if raw_line.strip() == "":
            # Blank line -> empty context
            if current_chunk is None and pending_anchors:
                start_chunk_if_needed()
            if current_chunk is None:
                start_chunk_if_needed()
            ctx_line = ""
            if not chunk_has_mods:
                current_chunk.prefix.append(ctx_line)
            else:
                current_chunk.suffix.append(ctx_line)
            continue
        if raw_line.startswith(CONTEXT_PREFIX):
            if current_chunk is None and pending_anchors:
                start_chunk_if_needed()
            if current_chunk is None:
                start_chunk_if_needed()
            ctx_line = raw_line[len(CONTEXT_PREFIX):]
            if not chunk_has_mods:
                current_chunk.prefix.append(ctx_line)
            else:
                current_chunk.suffix.append(ctx_line)
            continue
        # Any other non-blank line is invalid inside a file section
        add_error(
            f"Invalid patch line in {current_path}: must start with @@, -, +, or a space",
            line=current_line_num,
            hint="Lines inside file sections must start with '@@', '-', '+', or a single leading space for context. Blank lines are allowed as empty context.",
            filename=current_path,
        )
        # Skip this line and continue parsing
        continue

    # End of content: close out any open chunk
    finish_chunk_if_any()

    return patch, errors


def load_files(paths: List[str], open_fn: Callable[[str], str]) -> Tuple[Dict[str, str], List[PatchError]]:
    """
    Read a set of files, returning a (files, errors) tuple.
    files: mapping path -> content for successfully read files
    errors: list of PatchError for files that failed to read
    """
    files: Dict[str, str] = {}
    errs: List[PatchError] = []
    for path in paths:
        try:
            files[path] = open_fn(path)
        except Exception as e:
            errs.append(
                PatchError(
                    msg=f"Failed to read file: {path}",
                    line=None,
                    hint=f"{type(e).__name__}: {e}",
                    filename=path,
                )
            )
    return files, errs


def build_commits(patch: Patch, files: Dict[str, str]) -> Tuple[List["Commit"], List[PatchError], Dict[str, FileApplyStatus]]:
    """
    Produce a list of Commit objects describing concrete file changes
    from a parsed Patch and a mapping of already loaded file contents.
    Returns (commits, errors).
    """
    commits: List[Commit] = []
    errors: List[PatchError] = []
    changes: Dict[str, FileChange] = {}
    status_map: Dict[str, FileApplyStatus] = {}

    def add_error(
        msg: str,
        *,
        line: Optional[int] = None,
        hint: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        errors.append(PatchError(msg=msg, line=line, hint=hint, filename=filename))

    def join_lines(lines: List[str], *, eol: bool) -> str:
        s = "\n".join(lines)
        return s + ("\n" if eol else "")

    def find_block(
        file_lines: List[str],
        chunk: Chunk,
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Try to find a contiguous block equal to prefix + deletions + suffix.
        Returns (start_index, None) on success.
        Returns (None, hint) on failure with a detailed human-readable hint.
        Preference: if anchors are provided, search candidates closest to anchor lines first.
        """
        # Extract pattern components from the chunk
        prefix, deletions, suffix = chunk.prefix, chunk.deletions, chunk.suffix
        pattern: List[str] = prefix + deletions + suffix

        pre_len, del_len, suf_len = len(prefix), len(deletions), len(suffix)
        pat_len = len(pattern)
        n_lines = len(file_lines)

        # Candidate starts: align on first token of the pattern
        if pat_len == 0:
            return None, "Empty change block (no prefix/deletions/suffix)"
        first_token = prefix[0] if pre_len > 0 else (deletions[0] if del_len > 0 else suffix[0])
        candidate_starts: List[int] = [
            i for i in range(0, max(0, n_lines - pat_len + 1)) if file_lines[i] == first_token
        ]
        if not candidate_starts:
            # No direct anchor match; scan whole feasible range
            candidate_starts = list(range(0, max(0, n_lines - pat_len + 1)))

        # If anchors are provided and non-empty, prioritize candidate starts near lines containing anchor text
        anchor_idxs: List[int] = []
        for a in chunk.anchors:
            if a:
                for i, line in enumerate(file_lines):
                    if a in line:
                        anchor_idxs.append(i)
        if anchor_idxs:
            # Try candidates closest to any anchor line first
            def _dist_to_anchors(i: int) -> int:
                return min(abs(i - ai) for ai in anchor_idxs)
            candidate_starts.sort(key=lambda i: (_dist_to_anchors(i), i))

        best_partial = PartialMatch()

        def try_match_at(start: int) -> MatchResult:
            matched = 0
            p_matched = d_matched = s_matched = 0
            for j in range(pat_len):
                i = start + j
                if i >= n_lines:
                    return MatchResult(
                        ok=False,
                        start=start,
                        matched=matched,
                        expected=pattern[j],
                        actual="<EOF>",
                        fail_index=i,
                        prefix_matched=p_matched,
                        deletions_matched=d_matched,
                        suffix_matched=s_matched,
                    )
                if file_lines[i] != pattern[j]:
                    return MatchResult(
                        ok=False,
                        start=start,
                        matched=matched,
                        expected=pattern[j],
                        actual=file_lines[i],
                        fail_index=i,
                        prefix_matched=p_matched,
                        deletions_matched=d_matched,
                        suffix_matched=s_matched,
                    )
                matched += 1
                if j < pre_len:
                    p_matched += 1
                elif j < pre_len + del_len:
                    d_matched += 1
                else:
                    s_matched += 1
            return MatchResult(
                ok=True,
                start=start,
                matched=matched,
                expected="",
                actual="",
                fail_index=start + pat_len - 1,
                prefix_matched=p_matched,
                deletions_matched=d_matched,
                suffix_matched=s_matched,
            )

        for start in candidate_starts:
            res = try_match_at(start)
            if res.ok:
                return start, None
            if res.matched > best_partial.matched:
                best_partial.matched = res.matched
                best_partial.start = res.start
                # Derive phase from how many were matched in each section
                if res.prefix_matched < pre_len:
                    best_partial.phase = "prefix"
                elif res.deletions_matched < del_len:
                    best_partial.phase = "deletions"
                else:
                    best_partial.phase = "suffix"
                best_partial.expected = res.expected
                best_partial.actual = res.actual
                best_partial.fail_line = res.fail_index + 1  # human lines
                best_partial.prefix_matched = res.prefix_matched
                best_partial.deletions_matched = res.deletions_matched
                best_partial.suffix_matched = res.suffix_matched

        # Build human-readable hint describing where it failed
        bp = best_partial
        hint_lines: List[str] = []
        hint_lines.append(f"Matched {bp.matched}/{pat_len} lines.")
        # Include the exact source lines that matched before the failure to aid debugging
        if bp.start is not None and bp.matched > 0:
            matched_src = file_lines[bp.start : bp.start + bp.matched]
            hint_lines.append("Matched source lines (from file):")
            for s in matched_src:
                hint_lines.append(f"  {s!r}")
        if bp.expected != "" or bp.actual != "":
            hint_lines.append(f"Next expected line: {bp.expected!r}")
            hint_lines.append(f"File has: {bp.actual!r}")

        # Similarity suggestions for the next expected line
        if bp.expected:
            sims: List[Tuple[float, int, str]] = []
            for idx, sline in enumerate(file_lines):
                ratio = difflib.SequenceMatcher(None, bp.expected, sline).ratio()
                sims.append((ratio, idx, sline))
            # Sort by similarity (desc), then by line number (asc)
            sims.sort(key=lambda t: (-t[0], t[1]))
            hint_lines.append("Similar lines to expected next line (by similarity, then line number):")
            for ratio, idx, sline in sims:
                hint_lines.append(f"  L{idx + 1}: sim={ratio:.3f}: {sline!r}")

        hint_lines.append("Tips: ensure exact whitespace and blank lines match; remove escaping; do not trim indentation.")
        return None, "\n".join(hint_lines)

        # End find_block

    # Process actions (compute all matches first per file, then check overlaps, then apply)
    for path, action in patch.actions.items():
        if action.type == ActionType.ADD:
            # Build content from additions across all chunks
            add_lines: List[str] = []
            for ch in action.chunks:
                add_lines.extend(ch.additions)
            new_content = join_lines(add_lines, eol=False)
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
                hint="Load files before building commits or ensure the file exists for update.",
                filename=path,
            )
            continue
        # Preserve trailing newline status
        had_eol = original.endswith("\n")
        lines = original.splitlines()

        # Phase 1: locate all blocks for this file
        located: List[Tuple[int, int, Chunk]] = []
        any_failed = False
        for ch in action.chunks:
            start_idx, hint = find_block(lines, ch)
            if start_idx is None:
                add_error(
                    f"Failed to locate change block in {path}",
                    line=ch.start_line,
                    hint=hint,
                    filename=path,
                )
                any_failed = True
                # Do not break; collect all failures for better reporting
                continue
            pre_len, del_len, suf_len = len(ch.prefix), len(ch.deletions), len(ch.suffix)
            pattern_len = pre_len + del_len + suf_len
            end_idx = start_idx + pattern_len
            located.append((start_idx, end_idx, ch))

        # If some chunks could not be found, still apply the ones we did find.
        # If none located, skip updating this file.
        if not located:
            continue

        # Phase 2: detect overlaps (using original file indices)
        located.sort(key=lambda t: t[0])
        overlaps_found = False
        for (s1, e1, ch1), (s2, e2, ch2) in zip(located, located[1:]):
            if not (e1 <= s2):  # overlap if e1 > s2
                overlaps_found = True
                add_error(
                    f"Overlapping change blocks detected in {path}",
                    line=min((ch1.start_line or 0), (ch2.start_line or 0)) or None,
                    hint=(
                        "Two change blocks overlap in their context/deletion ranges. "
                        f"First block covers [{s1}, {e1}), second covers [{s2}, {e2}). "
                        "Reorder the chunks or regenerate the patch to avoid overlapping contexts."
                    ),
                    filename=path,
                )
        if overlaps_found:
            # Do not apply any changes to this file if overlaps exist
            continue

        # Phase 3: apply non-overlapping blocks to produce new content
        result: List[str] = []
        cursor = 0
        for start_idx, end_idx, ch in located:
            pre_len, del_len, suf_len = len(ch.prefix), len(ch.deletions), len(ch.suffix)
            # Copy up to the block start
            result.extend(lines[cursor:start_idx])
            # Keep prefix as-is from original
            result.extend(lines[start_idx : start_idx + pre_len])
            # Replace deletions with additions
            result.extend(ch.additions)
            # Keep suffix as-is from original
            result.extend(lines[start_idx + pre_len + del_len : start_idx + pre_len + del_len + suf_len])
            cursor = end_idx
        # Append remainder of file
        result.extend(lines[cursor:])
        lines = result

        new_content = join_lines(lines, eol=had_eol)
        changes[path] = FileChange(
            type=ActionType.UPDATE,
            old_content=original,
            new_content=new_content,
        )
        # Mark status based on whether any chunks for this file failed to locate
        status_map[path] = FileApplyStatus.PartialUpdate if any_failed else FileApplyStatus.Update

    if changes:
        commits.append(Commit(changes=changes))

    return commits, errors, status_map


def apply_commits(
    commits: List[Commit],
    write_fn: Callable[[str, str], None],
    delete_fn: Callable[[str], None],
) -> List[PatchError]:
    """
    Apply a list of commits using provided IO functions.
    Always attempts all writes/deletes, collecting errors for any failures.
    """
    errors: List[PatchError] = []

    for commit in commits:
        for path, change in commit.changes.items():
            try:
                if change.type == ActionType.ADD or change.type == ActionType.UPDATE:
                    write_fn(path, change.new_content or "")
                elif change.type == ActionType.DELETE:
                    delete_fn(path)
                else:
                    # Future-proof: unknown types
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
    patch, errors = parse_v4a_patch(text)
    if errors:
        return {}, errors

    # Load files needed for application (Update only). Add/Delete do not require reading here.
    paths = [p for p, a in patch.actions.items() if a.type == ActionType.UPDATE]
    files, read_errors = load_files(paths, open_fn)
    if read_errors:
        return {}, read_errors
    # Build commits (may be partial) and apply; return combined errors with statuses.
    commits, build_errors, status_map = build_commits(patch, files)
    apply_errors = apply_commits(commits, write_fn, delete_fn)
    return status_map, [*build_errors, *apply_errors]


__all__ = [
    "ActionType",
    "Chunk",
    "PatchAction",
    "Patch",
    "PatchError",
    "FileChange",
    "Commit",
    "parse_v4a_patch",
    "process_patch",
    "load_files",
    "build_commits",
    "apply_commits",
    "FileApplyStatus",
]
