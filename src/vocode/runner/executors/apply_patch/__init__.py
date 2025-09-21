from __future__ import annotations

import pathlib
from typing import AsyncIterator, List, Optional, Any, Dict
import asyncio

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
from .models import FileApplyStatus


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
            # Lazy import to avoid potential import cycles
            from vocode.project import FileChangeModel, FileChangeType

            # Track file changes with precedence: DELETED > UPDATED > CREATED
            changes_map: Dict[str, FileChangeType] = {}

            def _record(rel: str, new_type: FileChangeType) -> None:
                prev = changes_map.get(rel)
                if prev is None:
                    changes_map[rel] = new_type
                    return
                # precedence: deleted overrides anything; updated overrides created unless created is present; created does not override
                if new_type == FileChangeType.DELETED:
                    changes_map[rel] = new_type
                elif new_type == FileChangeType.UPDATED:
                    if prev != FileChangeType.CREATED and prev != FileChangeType.DELETED:
                        changes_map[rel] = new_type

            def tracking_write(rel: str, content: str) -> None:
                # classify as CREATED vs UPDATED based on pre-existence
                try:
                    existed = self._resolve_safe_path(rel).exists()
                except DiffError:
                    # If resolution fails, default to assuming it existed so we mark UPDATED; underlying write will raise
                    existed = True
                self._write_file(rel, content)
                _record(rel, FileChangeType.UPDATED if existed else FileChangeType.CREATED)

            def tracking_remove(rel: str) -> None:
                self._remove_file(rel)
                _record(rel, FileChangeType.DELETED)

            statuses, errs = process_patch(
                patch_text,
                open_fn=self._open_file,
                write_fn=tracking_write,
                delete_fn=tracking_remove,
            )
            # Compose result message based on statuses and errors
            created = sorted([f for f, s in statuses.items() if s == FileApplyStatus.Create])
            updated = sorted([f for f, s in statuses.items() if s in (FileApplyStatus.Update, FileApplyStatus.PartialUpdate)])
            deleted = sorted([f for f, s in statuses.items() if s == FileApplyStatus.Delete])

            if errs:
                # Failure: include summary of what did get applied, then list errors
                parts = []
                if created:
                    parts.append(f"{len(created)} added")
                if updated:
                    parts.append(f"{len(updated)} updated")
                if deleted:
                    parts.append(f"{len(deleted)} deleted")
                summary = ""
                if parts:
                    summary = " Applied: " + ", ".join(parts) + "."

                lines = ["Failed to apply patch fully. Please analyze following error messages and provide fixed patch." + summary]
                for e in errs:
                    loc = ""
                    if e.filename and e.line:
                        loc = f"{e.filename}:{e.line}: "
                    elif e.filename:
                        loc = f"{e.filename}: "
                    lines.append(f"- {loc}{e.msg}")
                    if e.hint:
                        lines.append(f"  Hint: {e.hint}")
                final = Message(role="agent", text="\n".join(lines))
                outcome = "fail"
            else:
                # Success summary
                parts = []
                if created:
                    parts.append(f"{len(created)} added: {', '.join(created)}")
                if updated:
                    parts.append(f"{len(updated)} updated: {', '.join(updated)}")
                if deleted:
                    parts.append(f"{len(deleted)} deleted: {', '.join(deleted)}")
                summary = "Applied patch successfully."
                if parts:
                    summary += " Changes: " + "; ".join(parts) + "."
                final = Message(role="agent", text=summary)

            # Build the accumulated list for refresh
            changed_files = [
                FileChangeModel(type=chg_type, relative_filename=rel)
                for rel, chg_type in changes_map.items()
            ]
            # Schedule a non-blocking project refresh in the background (do not await).
            asyncio.create_task(self.project.refresh(files=changed_files))  # type: ignore[attr-defined]
        except Exception as e:
            final = Message(role="agent", text=f"Error applying patch: {e}")
            outcome = "fail"

        yield (ReqFinalMessage(message=final, outcome_name=outcome), inp.state)
        return
