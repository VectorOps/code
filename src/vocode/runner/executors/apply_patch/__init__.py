from __future__ import annotations

import pathlib
from typing import AsyncIterator, List, Optional, Tuple

from vocode.runner.runner import Executor
from vocode.graph.models import ApplyPatchNode
from vocode.state import Message
from vocode.runner.models import (
    ReqPacket,
    ReqMessageRequest,
    ReqFinalMessage,
)

from .v4a import process_patch, DiffError


class ApplyPatchExecutor(Executor):
    # Must match ApplyPatchNode.type
    type = "apply_patch"

    def __init__(self, config, project):
        super().__init__(config=config, project=project)
        if not isinstance(config, ApplyPatchNode):
            raise TypeError("ApplyPatchExecutor requires config to be an ApplyPatchNode")
        # Buffers for dummy FS helpers (dry-run writes/removes)
        self._writes: List[Tuple[str, str]] = []
        self._removes: List[str] = []

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
        # Dummy write: record only, do not touch disk
        self._writes.append((rel, content))

    def _remove_file(self, rel: str) -> None:
        # Dummy remove: record only, do not touch disk
        self._removes.append(rel)

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

    async def run(self, messages: List[Message]) -> AsyncIterator[ReqPacket]:
        cfg: ApplyPatchNode = self.config  # type: ignore[assignment]
        if (cfg.patch_format or "v4a").lower() != "v4a":
            final = Message(role="agent", text=f"Unsupported patch format: {cfg.patch_format}")
            _ = (yield ReqFinalMessage(message=final, outcome_name="fail"))
            return

        patch_text = self._extract_patch_text(messages)
        if not patch_text:
            # Ask for a patch message
            reply = (yield ReqMessageRequest())
            # Runner guarantees a RespMessage here; extract patch from the returned message
            try:
                msg = reply.message  # type: ignore[attr-defined]
            except Exception:
                msg = None
            patch_text = self._extract_patch_text([msg] if msg is not None else [])
            if not patch_text:
                final = Message(role="agent", text="No patch provided.")
                _ = (yield ReqFinalMessage(message=final, outcome_name="fail"))
                return

        outcome = "success"
        try:
            _ = process_patch(
                patch_text,
                open_fn=self._open_file,
                write_fn=self._write_file,
                remove_fn=self._remove_file,
            )
            # Intentionally do not modify the file system; this is a dry-run with dummy helpers.
            final = Message(role="agent", text="Done!")
        except DiffError as e:
            final = Message(role="agent", text=f"Error applying patch: {e}")
            outcome = "fail"

        _ = (yield ReqFinalMessage(message=final, outcome_name=outcome))
        return
