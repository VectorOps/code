from __future__ import annotations

import pathlib
from typing import AsyncIterator, List, Optional, Tuple

from vocode.runner.runner import Executor
from vocode.graph.models import ApplyPatchNode, ResetPolicy
from vocode.state import Message
import asyncio
from vocode.runner.models import (
    ReqPacket,
    ReqMessageRequest,
    ReqFinalMessage,
    RespMessage,
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
        next_patch_text: Optional[str] = None

        while True:
            # Enforce supported patch format
            if (cfg.patch_format or "v4a").lower() != "v4a":
                final = Message(role="agent", text=f"Unsupported patch format: {cfg.patch_format}")
                post = (yield ReqFinalMessage(message=final, outcome_name="fail"))
                if isinstance(post, RespMessage):
                    # Format is fixed; ignore additional input and present the same final again
                    continue
                await asyncio.Event().wait()

            # Determine patch text: either from carried-over post-final message or inputs/history
            if next_patch_text is None:
                patch_text = self._extract_patch_text(messages)
                if not patch_text:
                    reply = (yield ReqMessageRequest())
                    try:
                        msg = reply.message  # type: ignore[attr-defined]
                    except Exception:
                        msg = None
                    patch_text = self._extract_patch_text([msg] if msg is not None else [])
                    if not patch_text:
                        final = Message(role="agent", text="No patch provided.")
                        post = (yield ReqFinalMessage(message=final, outcome_name="fail"))
                        if isinstance(post, RespMessage):
                            # Try to extract patch from the post-final message; if none, use raw text
                            next_patch_text = self._extract_patch_text([post.message]) or (post.message.text or "")
                            continue
                        await asyncio.Event().wait()
            else:
                patch_text = next_patch_text
                next_patch_text = None

            # Reset dummy FS buffers for this attempt
            self._writes.clear()
            self._removes.clear()

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

            post = (yield ReqFinalMessage(message=final, outcome_name=outcome))
            if isinstance(post, RespMessage):
                # Continue in-loop using the new patch content
                next_patch_text = self._extract_patch_text([post.message]) or (post.message.text or "")
                continue

            # Pause until runner closes/resets this executor
            await asyncio.Event().wait()
