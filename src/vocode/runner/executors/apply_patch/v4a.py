"""
A self-contained **pure-Python 3.9+** utility for applying human-readable
“pseudo-diff” patch files to a collection of text files.
"""
import pathlib
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)


# --------------------------------------------------------------------------- #
#  Domain objects
# --------------------------------------------------------------------------- #
class ActionType(str, Enum):
    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"


@dataclass
class FileChange:
    type: ActionType
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    move_path: Optional[str] = None


@dataclass
class Commit:
    changes: Dict[str, FileChange] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Exceptions
# --------------------------------------------------------------------------- #
class DiffError(ValueError):
    """Any problem detected while parsing or applying a patch."""


def _format_error_message(
    message: str,
    *,
    file: Optional[str] = None,
    line_no: Optional[int] = None,
    got_line: Optional[str] = None,
    hints: Optional[List[str]] = None,
) -> str:
    parts: List[str] = [f"Patch error: {message}"]
    if file:
        parts.append(f"File: {file}")
    if line_no is not None:
        parts.append(f"Line: {line_no}")
    if got_line is not None:
        # Show a safe representation of the line for clarity
        got = repr(got_line)
        # Limit extremely long lines to avoid flooding the LLM
        if len(got) > 500:
            got = got[:500] + "... (truncated)"
        parts.append(f"Got line: {got}")
    if hints:
        parts.append("Hints:")
        for h in hints:
            parts.append(f"- {h}")
    return "\n".join(parts)


def _detect_literal_escapes(text: str) -> List[str]:
    """
    Detect common literal escape sequences that indicate the patch
    was pasted as an escaped string (e.g., '\\n' instead of real newlines).
    """
    found: List[str] = []
    tokens = ["\\r\\n", "\\n", "\\r", "\\t", "\\\""]
    for tok in tokens:
        if tok in text:
            found.append(tok)
    # Stronger signal: sentinel with literal \n embedded
    if "*** Begin Patch\\n" in text or "\\n*** End Patch" in text:
        if "\\n" not in found:
            found.append("\\n")
    return found


# --------------------------------------------------------------------------- #
#  Helper dataclasses used while parsing patches
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    orig_index: int = -1
    del_lines: List[str] = field(default_factory=list)
    ins_lines: List[str] = field(default_factory=list)


@dataclass
class PatchAction:
    type: ActionType
    new_file: Optional[str] = None
    chunks: List[Chunk] = field(default_factory=list)
    move_path: Optional[str] = None


@dataclass
class Patch:
    actions: Dict[str, PatchAction] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Patch text parser
# --------------------------------------------------------------------------- #
@dataclass
class Parser:
    current_files: Dict[str, str]
    lines: List[str]
    index: int = 0
    patch: Patch = field(default_factory=Patch)
    fuzz: int = 0
    current_file: Optional[str] = None

    def _make_err(self, message: str, *, hints: Optional[List[str]] = None) -> DiffError:
        got_line: Optional[str] = None
        if 0 <= self.index < len(self.lines):
            got_line = self.lines[self.index]
        return DiffError(
            _format_error_message(
                message,
                file=self.current_file,
                line_no=self.index + 1,
                got_line=got_line,
                hints=hints,
            )
        )

    # ------------- low-level helpers -------------------------------------- #
    def _cur_line(self) -> str:
        if self.index >= len(self.lines):
            raise DiffError(
                _format_error_message(
                    "Unexpected end of input while parsing patch",
                    file=self.current_file,
                    line_no=self.index + 1,
                    hints=[
                        "Ensure the patch contains all required sections and ends with '*** End Patch'.",
                        "Do not truncate the patch; include the full content.",
                    ],
                )
            )
        return self.lines[self.index]

    @staticmethod
    def _norm(line: str) -> str:
        """Strip CR so comparisons work for both LF and CRLF input."""
        return line.rstrip("\r")

    # ------------- scanning convenience ----------------------------------- #
    def is_done(self, prefixes: Optional[Tuple[str, ...]] = None) -> bool:
        if self.index >= len(self.lines):
            return True
        if (
            prefixes
            and len(prefixes) > 0
            and self._norm(self._cur_line()).startswith(prefixes)
        ):
            return True
        return False

    def startswith(self, prefix: Union[str, Tuple[str, ...]]) -> bool:
        return self._norm(self._cur_line()).startswith(prefix)

    def read_str(self, prefix: str) -> str:
        """
        Consume the current line if it starts with *prefix* and return the text
        **after** the prefix.  Raises if prefix is empty.
        """
        if prefix == "":
            raise ValueError("read_str() requires a non-empty prefix")
        if self._norm(self._cur_line()).startswith(prefix):
            text = self._cur_line()[len(prefix) :]
            self.index += 1
            return text
        return ""

    def read_line(self) -> str:
        """Return the current raw line and advance."""
        line = self._cur_line()
        self.index += 1
        return line

    # ------------- public entry point -------------------------------------- #
    def parse(self) -> None:
        while not self.is_done(("*** End Patch",)):
            # ---------- UPDATE ---------- #
            path = self.read_str("*** Update File: ")
            if path:
                self.current_file = path
                if path in self.patch.actions:
                    raise self._make_err(
                        f"Duplicate update for file: {path}",
                        hints=[
                            "Only one '*** Update File:' section per file is allowed.",
                            "Merge multiple hunks into a single update section for the file.",
                        ],
                    )
                move_to = self.read_str("*** Move to: ")
                if path not in self.current_files:
                    raise self._make_err(
                        f"Update File Error - missing file: {path}",
                        hints=[
                            f"If this is a new file, use '*** Add File: {path}' instead of update.",
                            "If the file should exist, verify the path and ensure it is loaded.",
                        ],
                    )
                text = self.current_files[path]
                action = self._parse_update_file(text)
                action.move_path = move_to or None
                self.patch.actions[path] = action
                continue

            # ---------- DELETE ---------- #
            path = self.read_str("*** Delete File: ")
            if path:
                self.current_file = path
                if path in self.patch.actions:
                    raise self._make_err(
                        f"Duplicate delete for file: {path}",
                        hints=[
                            "Only one '*** Delete File:' section per file is allowed.",
                            "Remove duplicate delete sections.",
                        ],
                    )
                if path not in self.current_files:
                    raise self._make_err(
                        f"Delete File Error - missing file: {path}",
                        hints=[
                            "Delete can only target existing files.",
                            "Remove this section or correct the file path.",
                        ],
                    )
                self.patch.actions[path] = PatchAction(type=ActionType.DELETE)
                continue

            # ---------- ADD ---------- #
            path = self.read_str("*** Add File: ")
            if path:
                self.current_file = path
                if path in self.patch.actions:
                    raise self._make_err(
                        f"Duplicate add for file: {path}",
                        hints=[
                            "Only one '*** Add File:' section per file is allowed.",
                            "Consolidate the file content in a single add section.",
                        ],
                    )
                if path in self.current_files:
                    raise self._make_err(
                        f"Add File Error - file already exists: {path}",
                        hints=[
                            "Use '*** Update File:' for modifying existing files.",
                            "If you intended to replace it, delete first or update in-place.",
                        ],
                    )
                self.patch.actions[path] = self._parse_add_file()
                continue

            raise self._make_err(
                f"Unknown line while parsing: {self._cur_line()}",
                hints=[
                    "Expected one of: '*** Update File:', '*** Add File:', '*** Delete File:', or '*** End Patch'.",
                    "If your patch contains literal escape sequences like '\\n', send the patch with real newlines (e.g., in a code block).",
                ],
            )

        if not self.startswith("*** End Patch"):
            raise self._make_err(
                "Missing *** End Patch sentinel",
                hints=[
                    "Ensure the patch ends with a line exactly: *** End Patch",
                    "If the patch was truncated, include the full content.",
                ],
            )
        self.index += 1  # consume sentinel

    # ------------- section parsers ---------------------------------------- #
    def _parse_update_file(self, text: str) -> PatchAction:
        action = PatchAction(type=ActionType.UPDATE)
        lines = text.split("\n")
        index = 0
        while not self.is_done(
            (
                "*** End Patch",
                "*** Update File:",
                "*** Delete File:",
                "*** Add File:",
                "*** End of File",
            )
        ):
            def_str = self.read_str("@@ ")
            section_str = ""
            if not def_str and self._norm(self._cur_line()) == "@@":
                section_str = self.read_line()

            if not (def_str or section_str or index == 0):
                raise self._make_err(
                    f"Invalid line in update section: {self._cur_line()}",
                    hints=[
                        "Each hunk must start with a line beginning '@@ '.",
                        "Within a hunk, lines must start with ' ', '+', or '-'.",
                    ],
                )

            if def_str.strip():
                found = False
                if def_str not in lines[:index]:
                    for i, s in enumerate(lines[index:], index):
                        if s == def_str:
                            index = i + 1
                            found = True
                            break
                if not found and def_str.strip() not in [
                    s.strip() for s in lines[:index]
                ]:
                    for i, s in enumerate(lines[index:], index):
                        if s.strip() == def_str.strip():
                            index = i + 1
                            self.fuzz += 1
                            found = True
                            break

            next_ctx, chunks, end_idx, eof = peek_next_section(
                self.lines, self.index, file_path=self.current_file
            )
            new_index, fuzz = find_context(lines, next_ctx, index, eof)
            if new_index == -1:
                ctx_txt = "\n".join(next_ctx)
                raise self._make_err(
                    f"Invalid {'EOF ' if eof else ''}context at {index}: {ctx_txt}",
                    hints=[
                        "The patch context doesn't match the current file content.",
                        "Regenerate the patch against the latest version of the file.",
                        "Make sure that the code in the patch is not additionally escaped."
                    ],
                )
            self.fuzz += fuzz
            for ch in chunks:
                ch.orig_index += new_index
                action.chunks.append(ch)
            index = new_index + len(next_ctx)
            self.index = end_idx
        return action

    def _parse_add_file(self) -> PatchAction:
        lines: List[str] = []
        while not self.is_done(
            ("*** End Patch", "*** Update File:", "*** Delete File:", "*** Add File:")
        ):
            s = self.read_line()
            if not s.startswith("+"):
                raise self._make_err(
                    f"Invalid Add File line (missing '+'): {s}",
                    hints=[
                        "Each line in an Add section must begin with '+'.",
                        "For a blank line, include a line with just '+'.",
                    ],
                )
            lines.append(s[1:])  # strip leading '+'
        return PatchAction(type=ActionType.ADD, new_file="\n".join(lines))


# --------------------------------------------------------------------------- #
#  Helper functions
# --------------------------------------------------------------------------- #
def find_context_core(
    lines: List[str], context: List[str], start: int
) -> Tuple[int, int]:
    if not context:
        return start, 0

    for i in range(start, len(lines)):
        if lines[i : i + len(context)] == context:
            return i, 0
    for i in range(start, len(lines)):
        if [s.rstrip() for s in lines[i : i + len(context)]] == [
            s.rstrip() for s in context
        ]:
            return i, 1
    for i in range(start, len(lines)):
        if [s.strip() for s in lines[i : i + len(context)]] == [
            s.strip() for s in context
        ]:
            return i, 100
    return -1, 0


def find_context(
    lines: List[str], context: List[str], start: int, eof: bool
) -> Tuple[int, int]:
    if eof:
        new_index, fuzz = find_context_core(lines, context, len(lines) - len(context))
        if new_index != -1:
            return new_index, fuzz
        new_index, fuzz = find_context_core(lines, context, start)
        return new_index, fuzz + 10_000
    return find_context_core(lines, context, start)


def peek_next_section(
    lines: List[str], index: int, file_path: Optional[str] = None
) -> Tuple[List[str], List[Chunk], int, bool]:
    old: List[str] = []
    del_lines: List[str] = []
    ins_lines: List[str] = []
    chunks: List[Chunk] = []
    mode = "keep"
    orig_index = index

    while index < len(lines):
        s = lines[index]
        if s.startswith(
            (
                "@@",
                "*** End Patch",
                "*** Update File:",
                "*** Delete File:",
                "*** Add File:",
                "*** End of File",
            )
        ):
            break
        if s == "***":
            break
        if s.startswith("***"):
            raise DiffError(
                _format_error_message(
                    f"Unexpected section/header marker inside a hunk: {s}",
                    file=file_path,
                    line_no=index + 1,
                    got_line=s,
                    hints=[
                        "Section headers ('*** ...') must start at the top level, not inside a hunk.",
                        "Close the current hunk or finish the section before starting a new one.",
                    ],
                )
            )
        index += 1

        last_mode = mode
        if s == "":
            s = " "
        if s[0] == "+":
            mode = "add"
        elif s[0] == "-":
            mode = "delete"
        elif s[0] == " ":
            mode = "keep"
        else:
            raise DiffError(
                _format_error_message(
                    f"Invalid hunk line prefix: {s}",
                    file=file_path,
                    line_no=index,
                    got_line=s,
                    hints=[
                        "Lines inside a hunk must start with one of: ' ' (context), '+' (insert), '-' (delete).",
                        "Add a hunk header '@@ ' before hunk lines, if missing.",
                    ],
                )
            )
        s = s[1:]

        if mode == "keep" and last_mode != mode:
            if ins_lines or del_lines:
                chunks.append(
                    Chunk(
                        orig_index=len(old) - len(del_lines),
                        del_lines=del_lines,
                        ins_lines=ins_lines,
                    )
                )
            del_lines, ins_lines = [], []

        if mode == "delete":
            del_lines.append(s)
            old.append(s)
        elif mode == "add":
            ins_lines.append(s)
        elif mode == "keep":
            old.append(s)

    if ins_lines or del_lines:
        chunks.append(
            Chunk(
                orig_index=len(old) - len(del_lines),
                del_lines=del_lines,
                ins_lines=ins_lines,
            )
        )

    if index < len(lines) and lines[index] == "*** End of File":
        index += 1
        return old, chunks, index, True

    if index == orig_index:
        raise DiffError(
            _format_error_message(
                "Nothing in this section",
                file=file_path,
                line_no=index + 1,
                hints=[
                    "After a hunk header '@@', include at least one line starting with ' ', '+', or '-'.",
                    "If you intended to indicate end-of-file context, add a '*** End of File' marker.",
                ],
            )
        )
    return old, chunks, index, False


# --------------------------------------------------------------------------- #
#  Patch → Commit and Commit application
# --------------------------------------------------------------------------- #
def _get_updated_file(text: str, action: PatchAction, path: str) -> str:
    if action.type is not ActionType.UPDATE:
        raise DiffError("_get_updated_file called with non-update action")
    orig_lines = text.split("\n")
    dest_lines: List[str] = []
    orig_index = 0

    for chunk in action.chunks:
        if chunk.orig_index > len(orig_lines):
            raise DiffError(
                _format_error_message(
                    f"Chunk start index {chunk.orig_index} exceeds file length",
                    file=path,
                    hints=[
                        "The patch context likely did not match the file.",
                        "Regenerate the patch against the current file contents.",
                    ],
                )
            )
        if orig_index > chunk.orig_index:
            raise DiffError(
                _format_error_message(
                    f"Overlapping chunks at {orig_index} > {chunk.orig_index}",
                    file=path,
                    hints=[
                        "Combine or reorder hunks so they don't overlap.",
                        "Regenerate the patch to ensure chunk ordering is correct.",
                    ],
                )
            )

        dest_lines.extend(orig_lines[orig_index : chunk.orig_index])
        orig_index = chunk.orig_index

        dest_lines.extend(chunk.ins_lines)
        orig_index += len(chunk.del_lines)

    dest_lines.extend(orig_lines[orig_index:])
    return "\n".join(dest_lines)


def patch_to_commit(patch: Patch, orig: Dict[str, str]) -> Commit:
    commit = Commit()
    for path, action in patch.actions.items():
        if action.type is ActionType.DELETE:
            commit.changes[path] = FileChange(
                type=ActionType.DELETE, old_content=orig[path]
            )
        elif action.type is ActionType.ADD:
            if action.new_file is None:
                raise DiffError(
                    _format_error_message(
                        "ADD action without file content",
                        hints=[
                            "Provide the new file content after '*** Add File:' with each line prefixed by '+'.",
                            "Include a '+' line even for blank lines.",
                        ],
                    )
                )
            commit.changes[path] = FileChange(
                type=ActionType.ADD, new_content=action.new_file
            )
        elif action.type is ActionType.UPDATE:
            new_content = _get_updated_file(orig[path], action, path)
            commit.changes[path] = FileChange(
                type=ActionType.UPDATE,
                old_content=orig[path],
                new_content=new_content,
                move_path=action.move_path,
            )
    return commit


# --------------------------------------------------------------------------- #
#  User-facing helpers
# --------------------------------------------------------------------------- #
def text_to_patch(text: str, orig: Dict[str, str]) -> Tuple[Patch, int]:
    lines = text.splitlines()  # preserves blank lines, no strip()
    has_begin = len(lines) >= 1 and Parser._norm(lines[0]).startswith("*** Begin Patch")
    has_end = len(lines) >= 1 and Parser._norm(lines[-1]) == "*** End Patch"
    if not (has_begin and has_end):
        escapes = _detect_literal_escapes(text)
        if escapes:
            raise DiffError(
                _format_error_message(
                    "Invalid patch text - looks like it contains literal escape sequences instead of real newlines.",
                    hints=[
                        f"Detected escape sequences: {', '.join(escapes)}",
                        "Paste the patch with real newlines (e.g., surrounded by a Markdown code block).",
                        "If you have a JSON-escaped string, decode it before applying.",
                    ],
                )
            )
        raise DiffError(
            _format_error_message(
                "Invalid patch text - missing '*** Begin Patch' and/or '*** End Patch' sentinels.",
                hints=[
                    "Ensure the first line starts with '*** Begin Patch' and the last line is exactly '*** End Patch'.",
                ],
            )
        )

    parser = Parser(current_files=orig, lines=lines, index=1)
    parser.parse()
    return parser.patch, parser.fuzz


def identify_files_needed(text: str) -> List[str]:
    lines = text.splitlines()
    return [
        line[len("*** Update File: ") :]
        for line in lines
        if line.startswith("*** Update File: ")
    ] + [
        line[len("*** Delete File: ") :]
        for line in lines
        if line.startswith("*** Delete File: ")
    ]


def identify_files_added(text: str) -> List[str]:
    lines = text.splitlines()
    return [
        line[len("*** Add File: ") :]
        for line in lines
        if line.startswith("*** Add File: ")
    ]


# --------------------------------------------------------------------------- #
#  File-system helpers
# --------------------------------------------------------------------------- #
def load_files(paths: List[str], open_fn: Callable[[str], str]) -> Dict[str, str]:
    return {path: open_fn(path) for path in paths}


def apply_commit(
    commit: Commit,
    write_fn: Callable[[str, str], None],
    remove_fn: Callable[[str], None],
) -> None:
    for path, change in commit.changes.items():
        if change.type is ActionType.DELETE:
            remove_fn(path)
        elif change.type is ActionType.ADD:
            if change.new_content is None:
                raise DiffError(
                    _format_error_message(
                        "ADD change has no content",
                        file=path,
                        hints=[
                            "Provide content lines in the Add section, each starting with '+'.",
                        ],
                    )
                )
            write_fn(path, change.new_content)
        elif change.type is ActionType.UPDATE:
            if change.new_content is None:
                raise DiffError(
                    _format_error_message(
                        "UPDATE change has no new content",
                        file=path,
                        hints=[
                            "Verify hunks produce output; ensure at least one insertion or deletion is present.",
                        ],
                    )
                )
            target = change.move_path or path
            write_fn(target, change.new_content)
            if change.move_path:
                remove_fn(path)


def process_patch(
    text: str,
    open_fn: Callable[[str], str],
    write_fn: Callable[[str, str], None],
    remove_fn: Callable[[str], None],
) -> str:
    if not text.startswith("*** Begin Patch"):
        escapes = _detect_literal_escapes(text)
        if escapes:
            raise DiffError(
                _format_error_message(
                    "Patch text appears to be escaped (contains literal '\\n' etc.).",
                    hints=[
                        f"Detected escape sequences: {', '.join(escapes)}",
                        "Send the patch with real newlines or decode the escaped string first.",
                        "The first line must be '*** Begin Patch'.",
                    ],
                )
            )
        raise DiffError(
            _format_error_message(
                "Patch text must start with *** Begin Patch",
                hints=[
                    "Ensure the header line is exactly: *** Begin Patch",
                ],
            )
        )
    paths = identify_files_needed(text)
    orig = load_files(paths, open_fn)
    patch, _fuzz = text_to_patch(text, orig)
    commit = patch_to_commit(patch, orig)
    apply_commit(commit, write_fn, remove_fn)
    return "Done!"
