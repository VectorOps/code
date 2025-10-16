from typing import AsyncIterator, Optional, Any, List, Dict, Tuple
from pathlib import Path
import hashlib

from pydantic import BaseModel, Field

from vocode.runner.runner import Executor
from vocode.models import Node, ResetPolicy, PreprocessorSpec
from vocode.state import Message
from vocode.runner.models import ReqPacket, ReqFinalMessage, ExecRunInput
from vocode.commands import CommandContext
from vocode.runner.executors.llm.preprocessors.base import register_preprocessor

STRICT_DEFAULT_PROMPT = (
    "The following files were explicitly added by the developer to your context. "
    "Discard and ignore any prior versions of these files you received earlier. "
    "Use only the versions provided below."
)


class FileStateNode(Node):
    type: str = "file_state"
    prompt: str = Field(default=STRICT_DEFAULT_PROMPT)
    reset_policy: ResetPolicy = Field(default=ResetPolicy.keep_state)


class FileStateState(BaseModel):
    hashes: Dict[str, str] = Field(default_factory=dict)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 64), b""):
            h.update(chunk)
    return h.hexdigest()


class FileStateContext(BaseModel):
    files: Dict[str, str] = Field(default_factory=dict)

    def _validate_relpath(self, rel: str, project) -> str:
        p = Path(rel)
        if p.is_absolute():
            raise ValueError(f"Path must be relative to project: {rel}")
        base = project.base_path.resolve()
        full = (project.base_path / p).resolve()
        try:
            _ = full.relative_to(base)
        except Exception:
            raise ValueError(f"Path escapes project root: {rel}")
        if not full.exists():
            raise FileNotFoundError(f"File does not exist: {rel}")
        if not full.is_file():
            raise IsADirectoryError(f"Not a file: {rel}")
        return p.as_posix()

    def add(self, rel_paths: List[str], project) -> tuple[int, int]:
        added = 0
        skipped = 0
        for rel in rel_paths:
            try:
                norm_rel = self._validate_relpath(rel, project)
            except Exception:
                skipped += 1
                continue
            if norm_rel in self.files:
                skipped += 1
                continue
            # Do not compute hash at add time. Only track presence.
            self.files[norm_rel] = ""
            added += 1
        return added, skipped

    def remove(self, rel_paths: List[str]) -> tuple[int, int]:
        removed = 0
        skipped = 0
        for rel in rel_paths:
            if rel in self.files:
                self.files.pop(rel)
                removed += 1
            else:
                skipped += 1
        return removed, skipped

    def list(self) -> List[str]:
        return list(self.files.keys())


def get_file_state_ctx(project) -> FileStateContext:
    key = "file_state_ctx"
    ctx = project.project_state.get(key)
    if not isinstance(ctx, FileStateContext):
        ctx = FileStateContext()
        project.project_state.set(key, ctx)
    return ctx


def _build_file_state_injection(
    project,
    prompt: str,
    prev_hashes: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict[str, str], List[str]]:
    """
    Shared logic to:
      - enumerate tracked files from FileStateContext
      - compute current hashes
      - determine which files to include (first run: all; otherwise: changed only)
      - render injection text
    Returns: (injection_text, current_hashes, include_rels)
    """
    ctx_files = get_file_state_ctx(project).list()
    base = project.base_path

    prev_hashes = dict(prev_hashes or {})

    # Compute current hashes for tracked files that exist
    cur_hashes: Dict[str, str] = {}
    for rel in ctx_files:
        full = base / rel
        if not full.exists() or not full.is_file():
            continue
        try:
            cur_hashes[rel] = _hash_file(full)
        except Exception:
            continue

    # Which files to include
    if prev_hashes:
        include_rels = [
            rel for rel, h in cur_hashes.items() if prev_hashes.get(rel) != h
        ]
    else:
        include_rels = list(cur_hashes.keys())

    parts: List[str] = [prompt]
    for rel in include_rels:
        full = base / rel
        try:
            content = full.read_text(encoding="utf-8", errors="replace")
        except Exception:
            try:
                content = full.read_bytes().decode("utf-8", errors="replace")
            except Exception:
                content = "[Error reading file content]"
        parts.append(f"File: {rel}\n```\n{content}\n```")

    final_text = "\n\n".join(parts)
    return final_text, cur_hashes, include_rels


class FileStateExecutor(Executor):
    type = "file_state"

    def __init__(self, config: Node, project):
        super().__init__(config=config, project=project)
        if not isinstance(config, FileStateNode):
            raise TypeError("FileStateExecutor requires config to be a FileStateNode")
        self._ensure_commands_registered()

    def _ensure_commands_registered(self) -> None:
        # Idempotent: register only if missing. Survives CommandManager.clear().
        existing = {
            name
            for name in ("fadd", "fdel", "flist")
            if self.project.commands.get(name)
        }
        if existing == {"fadd", "fdel", "flist"}:
            return

        async def cmd_fadd(ctx: CommandContext, args: List[str]) -> Optional[str]:
            if not args:
                return "usage: /fadd <relative-path> [more ...]"
            state = get_file_state_ctx(ctx.project)
            to_add: List[str] = []
            added_list: List[str] = []
            skipped_details: List[tuple[str, str]] = []

            # Validate, normalize, and dedupe for clearer reporting
            for raw in args:
                try:
                    norm = state._validate_relpath(raw, ctx.project)
                except Exception as e:
                    skipped_details.append((raw, str(e)))
                    continue
                if norm in state.files or norm in to_add:
                    skipped_details.append((raw, "already added"))
                    continue
                to_add.append(norm)

            if to_add:
                # Perform the actual addition
                state.add(to_add, ctx.project)
                added_list = list(to_add)

            # Human-readable report (no counts; show individual files and errors; always show current files)
            lines: List[str] = []
            if added_list:
                lines.append("Added:")
                lines.extend(f"- {p}" for p in added_list)
            if skipped_details:
                lines.append("Errors:")
                lines.extend(f"- {raw}: {reason}" for raw, reason in skipped_details)
            files_now = state.list()
            lines.append("Files in context:")
            if files_now:
                lines.extend(f"- {p}" for p in files_now)
            else:
                lines.append("(none)")
            return "\n".join(lines)

        async def cmd_fdel(ctx: CommandContext, args: List[str]) -> Optional[str]:
            if not args:
                return "usage: /fdel <relative-path> [more ...]"
            state = get_file_state_ctx(ctx.project)
            removed_list: List[str] = []
            skipped_details: List[tuple[str, str]] = []

            # Normalize relative paths to project (without requiring file existence)
            base = ctx.project.base_path.resolve()
            for raw in args:
                p = Path(raw)
                if p.is_absolute():
                    skipped_details.append(
                        (raw, f"Path must be relative to project: {raw}")
                    )
                    continue
                try:
                    full = (ctx.project.base_path / p).resolve()
                    # Ensure path is within project
                    norm_rel = full.relative_to(base).as_posix()
                except Exception:
                    skipped_details.append((raw, f"Path escapes project root: {raw}"))
                    continue

                # Remove normalized path if present
                if norm_rel in state.files:
                    state.files.pop(norm_rel)
                    removed_list.append(norm_rel)
                else:
                    skipped_details.append((raw, "not in context"))

            lines: List[str] = []
            if removed_list:
                lines.append("Removed:")
                lines.extend(f"- {p}" for p in removed_list)
            if skipped_details:
                lines.append("Errors:")
                lines.extend(f"- {raw}: {reason}" for raw, reason in skipped_details)
            files_now = state.list()
            lines.append("Files in context:")
            if files_now:
                lines.extend(f"- {p}" for p in files_now)
            else:
                lines.append("(none)")
            return "\n".join(lines)

        async def cmd_flist(ctx: CommandContext, args: List[str]) -> Optional[str]:
            files = get_file_state_ctx(ctx.project).list()
            if not files:
                return "(file_state empty)"
            return "\n".join(files)

        self.project.commands.register(
            "fadd",
            "Add file(s) to FileState context",
            cmd_fadd,
            "usage: /fadd <relative-path> [more ...]",
            autocompleter="filelist",
        )
        self.project.commands.register(
            "fdel",
            "Remove file(s) from FileState context",
            cmd_fdel,
            "usage: /fdel <relative-path> [more ...]",
            autocompleter="filelist",
        )
        self.project.commands.register(
            "flist", "List files in FileState context", cmd_flist, "usage: /flist"
        )

    async def run(
        self, inp: ExecRunInput
    ) -> AsyncIterator[tuple[ReqPacket, Optional[Any]]]:
        cfg: FileStateNode = self.config  # type: ignore[assignment]

        # Previous state (hashes) if available
        prev_hashes: Dict[str, str] = {}
        if isinstance(inp.state, FileStateState):
            prev_hashes = dict(inp.state.hashes)
        elif isinstance(inp.state, dict):
            # Back-compat if raw dict was persisted
            prev_hashes = {k: str(v) for k, v in inp.state.items()}

        final_text, cur_hashes, _include_rels = _build_file_state_injection(
            project=self.project,
            prompt=cfg.prompt,
            prev_hashes=prev_hashes,
        )
        final = Message(role="agent", text=final_text)
        state_out = FileStateState(hashes=cur_hashes)
        yield ReqFinalMessage(message=final), state_out


from typing import Any, List, Optional

from ...models import Mode, PreprocessorSpec
from ...state import Message
from .llm.preprocessors.base import (
    register_preprocessor,
)


def _file_state_preprocessor(
    project: Any, spec: PreprocessorSpec, messages: List[Message]
) -> List[Message]:
    opts = spec.options or {}
    prompt = opts.get("prompt", STRICT_DEFAULT_PROMPT)

    # Persist previous hashes in project-scoped state
    prev_key = "file_state_hashes"
    prev_hashes = project.project_state.get(prev_key)
    if not isinstance(prev_hashes, dict):
        prev_hashes = {}

    injection, cur_hashes, include_rels = _build_file_state_injection(
        project=project, prompt=prompt, prev_hashes=prev_hashes
    )

    # Save current snapshot for next invocation
    project.project_state.set(prev_key, cur_hashes)

    # Only inject on first run (no prev_hashes) or if the state has changed.
    if prev_hashes and cur_hashes == prev_hashes:
        return messages


    target_message: Optional[Message] = None
    if not messages:
        role = "system" if spec.mode == Mode.System else "user"
        target_message = Message(text="", role=role)
        messages.append(target_message)
    else:
        if spec.mode == Mode.System:
            target_message = next(
                (msg for msg in messages if msg.role == "system"), None
            )
        elif spec.mode == Mode.User:
            target_message = next(
                (msg for msg in reversed(messages) if msg.role == "user"), None
            )

    if target_message:
        base_text = target_message.text or ""
        if injection in base_text:
            return messages
        if spec.prepend:
            target_message.text = f"{injection}{base_text}"
        else:
            target_message.text = f"{base_text}{injection}"
    return messages


register_preprocessor("file_state", _file_state_preprocessor)
