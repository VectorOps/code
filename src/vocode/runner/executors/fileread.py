from typing import AsyncIterator, List, Optional, Any
from pathlib import Path

from vocode.runner.runner import Executor
from vocode.models import Node
from vocode.state import Message
from vocode.runner.models import ReqPacket, ReqFinalMessage, ExecRunInput


class FilereadNode(Node):
    type: str = "fileread"
    files: List[str]


class FilereadExecutor(Executor):
    type = "fileread"

    def __init__(self, config: Node, project) -> None:
        super().__init__(config=config, project=project)
        if not isinstance(config, FilereadNode):
            raise TypeError("FilereadExecutor requires config to be a FilereadNode")

    def _validate_relpath(self, rel: str) -> str:
        p = Path(rel)
        if p.is_absolute():
            raise ValueError(f"Path must be relative to project: {rel}")
        base = self.project.base_path.resolve()
        full = (self.project.base_path / p).resolve()
        try:
            _ = full.relative_to(base)
        except Exception:
            raise ValueError(f"Path escapes project root: {rel}")
        if not full.exists():
            raise FileNotFoundError(f"File does not exist: {rel}")
        if not full.is_file():
            raise IsADirectoryError(f"Not a file: {rel}")
        return full.as_posix()

    async def run(self, inp: ExecRunInput) -> AsyncIterator[tuple[ReqPacket, Optional[Any]]]:
        cfg: FilereadNode = self.config  # type: ignore[assignment]

        contents: List[str] = []
        for rel in cfg.files:
            full_posix = self._validate_relpath(rel)
            full_path = Path(full_posix)
            try:
                text = full_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                text = full_path.read_bytes().decode("utf-8", errors="replace")
            contents.append(text)

        final = Message(role="agent", text="".join(contents))
        yield ReqFinalMessage(message=final), None
