from typing import AsyncIterator, Optional, Any, List
from pathlib import Path

from pydantic import BaseModel, Field

from vocode.runner.runner import Executor
from vocode.models import Node
from vocode.state import Message
from vocode.runner.models import ReqPacket, ReqFinalMessage, ExecRunInput
from vocode.commands import CommandContext

STRICT_DEFAULT_PROMPT = (
    "STRICT: The following files were explicitly added by the developer to your context. "
    "Discard and ignore any prior versions of these files you received earlier. "
    "Use only the versions provided below."
)


class FileStateNode(Node):
    type: str = "file_state"
    prompt: str = Field(default=STRICT_DEFAULT_PROMPT)


class FileStateContext(BaseModel):
    files: List[str] = Field(default_factory=list)

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
            self.files.append(norm_rel)
            added += 1
        return added, skipped

    def remove(self, rel_paths: List[str]) -> tuple[int, int]:
        removed = 0
        skipped = 0
        for rel in rel_paths:
            try:
                self.files.remove(rel)
                removed += 1
            except ValueError:
                skipped += 1
        return removed, skipped

    def list(self) -> List[str]:
        return list(self.files)


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
        if getattr(self.project, "_file_state_cmds_registered", False):
            return

        async def cmd_fadd(ctx: CommandContext, args: List[str]) -> Optional[str]:
            if not args:
                return "usage: /fadd <relative-path> [more ...]"
            added, skipped = get_file_state_ctx(ctx.project).add(args, ctx.project)
            return f"/fadd: added={added} skipped={skipped}"

        async def cmd_fdel(ctx: CommandContext, args: List[str]) -> Optional[str]:
            if not args:
                return "usage: /fdel <relative-path> [more ...]"
            removed, skipped = get_file_state_ctx(ctx.project).remove(args)
            return f"/fdel: removed={removed} skipped={skipped}"

        async def cmd_flist(ctx: CommandContext, args: List[str]) -> Optional[str]:
            files = get_file_state_ctx(ctx.project).list()
            if not files:
                return "(file_state empty)"
            return "\n".join(files)

        self.project.commands.register(
            "/fadd",
            "Add file(s) to FileState context",
            cmd_fadd,
            "usage: /fadd <relative-path> [more ...]",
        )
        self.project.commands.register(
            "/fdel",
            "Remove file(s) from FileState context",
            cmd_fdel,
            "usage: /fdel <relative-path> [more ...]",
        )
        self.project.commands.register(
            "/flist", "List files in FileState context", cmd_flist, "usage: /flist"
        )
        setattr(self.project, "_file_state_cmds_registered", True)
 
    async def run(
        self, inp: ExecRunInput
    ) -> AsyncIterator[tuple[ReqPacket, Optional[Any]]]:
        cfg: FileStateNode = self.config  # type: ignore[assignment]
        files = get_file_state_ctx(self.project).list()
        base = self.project.base_path

        parts: List[str] = [cfg.prompt]
        for rel in files:
            full = (base / rel)
            if not full.exists() or not full.is_file():
                # If file disappeared since addition, skip silently
                continue
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
        yield ReqFinalMessage(message=final), None
