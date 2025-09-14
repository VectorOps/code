from __future__ import annotations

import pathlib
from typing import AsyncIterator, List, Optional, Any

from vocode.runner.runner import Executor
from vocode.graph.models import ApplyPatchNode, ResetPolicy
from vocode.state import Message
from vocode.runner.models import (
    ReqPacket,
    ReqMessageRequest,
    ReqFinalMessage,
    ExecRunInput,
    PACKET_MESSAGE,
)

from .v4a import process_patch, DiffError


class ApplyPatchExecutor(Executor):
    # Must match ApplyPatchNode.type
    type = "apply_patch"

    def __init__(self, config, project):
        super().__init__(config=config, project=project)
        if not isinstance(config, ApplyPatchNode):
            raise TypeError("ApplyPatchExecutor requires config to be an ApplyPatchNode")
        if config.reset_policy != ResetPolicy.always_reset:
            raise ValueError("ApplyPatchExecutor supports only ResetPolicy.always_reset")

    def _resolve_safe_path(self, rel: str) -> pathlib.Path:
        if rel.startswith("/") or rel.startswith("~"):
            raise DiffError(f"Absolute paths are not allowed: {rel}")
        base = self.project.base_path  # type: ignore[attr-defined]
        abs_path = (base / rel).resolve()
        base_resolved = base.resolve()
        if str(abs_path).startswith(str(base_resolved)):
            return abs_path
        raise DiffError(f"Path escapes project root: {rel}")

    def _open_file(self, rel: str) -> str:
        path = self._resolve_safe_path(rel)
        try:
            with path.open("rt", encoding="utf-8") as fh:
                return fh.read()
        except FileNotFoundError as e:
            raise DiffError(f"File not found: {rel}") from e

    def _write_file(self, rel: str, content: str) -> None:
        path = self._resolve_safe_path(rel)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wt", encoding="utf-8") as fh:
                fh.write(content)
        except OSError as e:
            raise DiffError(f"Failed to write file: {rel}: {e}") from e

    def _remove_file(self, rel: str) -> None:
        path = self._resolve_safe_path(rel)
        try:
            path.unlink(missing_ok=True)
        except OSError as e:
            raise DiffError(f"Failed to remove file: {rel}: {e}") from e

    def _extract_patch_text(self, messages: List[Message]) -> Optional[str]:
        # Find the last message containing a V4A patch block
        for m in reversed(messages):
            txt = (m.text or "").strip()
            if "*** Begin Patch" in txt:
                start = txt.find("*** Begin Patch")
                # Prefer up to the last End Patch if present
                end_idx = txt.rfind("*** End Patch")
                if end_idx != -1:
                    end = end_idx + len("*** End Patch")
                    return txt[start:end]
                return txt[start:]
        return None

    async def run(self, inp: ExecRunInput) -> AsyncIterator[tuple[ReqPacket, Optional[Any]]]:
        cfg: ApplyPatchNode = self.config  # type: ignore[assignment]

        # Enforce supported patch format
        if (cfg.patch_format or "v4a").lower() != "v4a":
            final = Message(role="agent", text=f"Unsupported patch format: {cfg.patch_format}")
            yield (ReqFinalMessage(message=final, outcome_name="fail"), inp.state)
            return

        # Determine patch text from provided messages or latest response
        patch_text: Optional[str] = self._extract_patch_text(inp.messages)
        if patch_text is None:
            if inp.response is not None and inp.response.kind == PACKET_MESSAGE and inp.response.message is not None:
                # Accept either a proper patch block or raw text
                patch_text = self._extract_patch_text([inp.response.message]) or (inp.response.message.text or "")
            else:
                # Ask user for a message containing a patch
                yield (ReqMessageRequest(), inp.state)
                return


        outcome = "success"
        try:
            _ = process_patch(
                patch_text,
                open_fn=self._open_file,
                write_fn=self._write_file,
                remove_fn=self._remove_file,
            )
            final = Message(role="agent", text="Done!")
        except DiffError as e:
            final = Message(role="agent", text=f"Error applying patch: {e}")
            outcome = "fail"

        yield (ReqFinalMessage(message=final, outcome_name=outcome), inp.state)
        return
