from typing import AsyncIterator, Optional, Any, List, Dict
from pathlib import Path
import hashlib

from pydantic import BaseModel, Field

from vocode.runner.runner import Executor
from vocode.models import Node, ResetPolicy
from vocode.state import Message
from vocode.runner.models import ReqPacket, ReqFinalMessage, ExecRunInput
from vocode.commands import CommandContext

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
            full = (project.base_path / norm_rel).resolve()
            try:
                file_hash = _hash_file(full)
            except Exception:
                # If hashing fails, still record presence with empty hash
                file_hash = ""
            self.files[norm_rel] = file_hash
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

            # Human-readable report
            lines: List[str] = [
                f"/fadd: added={len(added_list)} skipped={len(skipped_details)}"
            ]
            if added_list:
                lines.append("Added:")
                lines.extend(f"- {p}" for p in added_list)
            if skipped_details:
                lines.append("Skipped:")
                lines.extend(f"- {raw} ({reason})" for raw, reason in skipped_details)
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

            lines: List[str] = [
                f"/fdel: removed={len(removed_list)} skipped={len(skipped_details)}"
            ]
            if removed_list:
                lines.append("Removed:")
                lines.extend(f"- {p}" for p in removed_list)
            if skipped_details:
                lines.append("Skipped:")
                lines.extend(f"- {raw} ({reason})" for raw, reason in skipped_details)
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
        ctx_files = get_file_state_ctx(self.project).list()
        base = self.project.base_path

        # Previous state (hashes) if available
        prev_hashes: Dict[str, str] = {}
        if isinstance(inp.state, FileStateState):
            prev_hashes = dict(inp.state.hashes)
        elif isinstance(inp.state, dict):
            # Back-compat if raw dict was persisted
            prev_hashes = {k: str(v) for k, v in inp.state.items()}

        # Compute current hashes for tracked files that exist
        cur_hashes: Dict[str, str] = {}
        for rel in ctx_files:
            full = base / rel
            if not full.exists() or not full.is_file():
                continue
            try:
                cur_hashes[rel] = _hash_file(full)
            except Exception:
                # Skip files we cannot hash
                continue

        # Which files to include:
        # - First run (no state): include all
        # - Subsequent runs: include only new/changed files
        if prev_hashes:
            include_rels = [
                rel for rel, h in cur_hashes.items() if prev_hashes.get(rel) != h
            ]
        else:
            include_rels = list(cur_hashes.keys())

        parts: List[str] = [cfg.prompt]
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
        final = Message(role="agent", text=final_text)

        # Persist current hashes for next run
        state_out = FileStateState(hashes=cur_hashes)
        yield ReqFinalMessage(message=final), state_out
