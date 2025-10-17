from __future__ import annotations

import pathlib
from typing import AsyncIterator, List, Optional, Any, Dict, Callable, Tuple
from pydantic import BaseModel, Field
from dataclasses import dataclass
import asyncio

from vocode.runner.runner import Executor
from vocode.models import Node, ResetPolicy
from vocode.state import Message
from vocode.runner.models import (
    ReqPacket,
    ReqMessageRequest,
    ReqFinalMessage,
    ExecRunInput,
    PACKET_MESSAGE,
)

from .v4a import (
    process_patch as v4a_process_patch,
    DiffError,
    DIFF_SYSTEM_INSTRUCTION as V4A_SYSTEM_PROMPT,
)
from .patch import (
    process_patch as sr_process_patch,
    DIFF_PATCH_SYSTEM_INSTRUCTION as PATCH_SYSTEM_PROMPT,
)


# Node
class ApplyPatchNode(Node):
    type: str = "apply_patch"
    format: str = Field(
        default="v4a",
        description="Patch format identifier ('v4a' or 'patch')",
    )


# Internal state
@dataclass(frozen=True)
class PatchFormat:
    name: str
    system_prompt: str
    handler: Callable[
        [str, Callable[[str], str], Callable[[str, str], None], Callable[[str], None]],
        Tuple[Dict[str, "FileApplyStatus"], List[Any]],
    ]


SUPPORTED_PATCH_FORMATS: Dict[str, PatchFormat] = {
    "v4a": PatchFormat(
        name="v4a",
        system_prompt=V4A_SYSTEM_PROMPT,
        handler=v4a_process_patch,
    ),
    "patch": PatchFormat(
        name="patch",
        system_prompt=PATCH_SYSTEM_PROMPT,
        handler=sr_process_patch,
    ),
}
from .models import FileApplyStatus


class ApplyPatchExecutor(Executor):
    # Must match ApplyPatchNode.type
    type = "apply_patch"

    def __init__(self, config, project):
        super().__init__(config=config, project=project)
        if not isinstance(config, ApplyPatchNode):
            raise TypeError(
                "ApplyPatchExecutor requires config to be an ApplyPatchNode"
            )
        if config.reset_policy != ResetPolicy.always_reset:
            raise ValueError(
                "ApplyPatchExecutor supports only ResetPolicy.always_reset"
            )

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

    async def run(
        self, inp: ExecRunInput
    ) -> AsyncIterator[tuple[ReqPacket, Optional[Any]]]:
        cfg: ApplyPatchNode = self.config  # type: ignore[assignment]

        # Enforce supported patch formats
        fmt = (cfg.format or "v4a").lower()

        fmt_entry = SUPPORTED_PATCH_FORMATS.get(fmt)
        if fmt_entry is None:
            supported = ", ".join(sorted(SUPPORTED_PATCH_FORMATS.keys()))
            final = Message(
                role="agent",
                text=f"Unsupported patch format: {cfg.format}. Supported formats: {supported}",
            )
            yield (ReqFinalMessage(message=final, outcome_name="fail"), inp.state)
            return

        # Determine source text: let patchers extract actual patch content
        source_text: Optional[str] = None
        if (
            inp.response is not None
            and inp.response.kind == PACKET_MESSAGE
            and inp.response.message is not None
        ):
            source_text = inp.response.message.text or ""
        elif inp.messages:
            # Use the last message text as the source
            source_text = inp.messages[-1].text or ""
        if not source_text or not source_text.strip():
            final = Message(
                role="agent",
                text="No patch was provided. The patch application has failed.",
            )
            yield (ReqFinalMessage(message=final, outcome_name="fail"), inp.state)
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
                    if (
                        prev != FileChangeType.CREATED
                        and prev != FileChangeType.DELETED
                    ):
                        changes_map[rel] = new_type

            def tracking_write(rel: str, content: str) -> None:
                # classify as CREATED vs UPDATED based on pre-existence
                try:
                    existed = self._resolve_safe_path(rel).exists()
                except DiffError:
                    # If resolution fails, default to assuming it existed so we mark UPDATED; underlying write will raise
                    existed = True
                self._write_file(rel, content)
                _record(
                    rel, FileChangeType.UPDATED if existed else FileChangeType.CREATED
                )

            def tracking_remove(rel: str) -> None:
                self._remove_file(rel)
                _record(rel, FileChangeType.DELETED)

            handler = fmt_entry.handler
            statuses, errs = handler(
                source_text,
                open_fn=self._open_file,
                write_fn=tracking_write,
                delete_fn=tracking_remove,
            )
            # Categorize by status
            created = sorted(
                [f for f, s in statuses.items() if s == FileApplyStatus.Create]
            )
            updated_full = sorted(
                [f for f, s in statuses.items() if s == FileApplyStatus.Update]
            )
            updated_partial = sorted(
                [f for f, s in statuses.items() if s == FileApplyStatus.PartialUpdate]
            )
            deleted = sorted(
                [f for f, s in statuses.items() if s == FileApplyStatus.Delete]
            )

            if errs:
                # Determine what actually got applied (successful IO only)
                applied_files = set(changes_map.keys())
                applied_any = bool(applied_files)

                # Intersect status categories with actually applied changes to remove ambiguity
                applied_created = sorted([f for f in created if f in applied_files])
                applied_updated_full = sorted(
                    [f for f in updated_full if f in applied_files]
                )
                applied_updated_partial = sorted(
                    [f for f in updated_partial if f in applied_files]
                )
                applied_deleted = sorted([f for f in deleted if f in applied_files])

                # Files that had errors and did not result in any applied change
                failed_files = sorted(
                    {e.filename for e in errs if getattr(e, "filename", None)}
                )
                not_applied_failed = sorted(
                    [f for f in failed_files if f not in applied_files]
                )

                # Compose precise summary
                lines: list[str] = []
                if not applied_any:
                    lines.append("Patch application failed. No changes were applied.")
                else:
                    lines.append("Patch application completed with errors. Summary:")
                    if applied_created:
                        lines.append("Added files (fully applied):")
                        for f in applied_created:
                            lines.append(f"* {f}")
                    if applied_updated_full:
                        lines.append("Fully updated files:")
                        for f in applied_updated_full:
                            lines.append(f"* {f}")
                    if applied_updated_partial:
                        lines.append("Partially updated files (some chunks failed):")
                        for f in applied_updated_partial:
                            lines.append(f"* {f}")
                    if applied_deleted:
                        lines.append("Deleted files (fully applied):")
                        for f in applied_deleted:
                            lines.append(f"* {f}")

                # Guidance: target regeneration for failed parts and files with no applied changes
                targets_for_fix = sorted(
                    set(applied_updated_partial) | set(not_applied_failed)
                )
                if targets_for_fix:
                    lines.append(
                        "Please regenerate patch chunks for the failed parts in these files:"
                    )
                    for f in targets_for_fix:
                        lines.append(f"* {f}")
                    lines.append(
                        "If there were other files that were not mentioned in"
                        "this response and not successfully applied, regenerate chunks for them as well."
                    )

                # Append detailed error list with filename/line/hint
                lines.append("Errors:")
                for e in errs:
                    loc = ""
                    if getattr(e, "filename", None) and getattr(e, "line", None):
                        loc = f"{e.filename}:{e.line}: "
                    elif getattr(e, "filename", None):
                        loc = f"{e.filename}: "
                    lines.append(f"* {loc}{e.msg}")
                    if getattr(e, "hint", None):
                        lines.append(f"  Hint: {e.hint}")

                final = Message(role="agent", text="\n".join(lines))
                outcome = "fail"
            else:
                # Success summary
                lines = ["Applied patch successfully."]
                if created:
                    lines.append("Added files:")
                    for f in created:
                        lines.append(f"* {f}")
                if updated_full:
                    lines.append("Fully updated files:")
                    for f in updated_full:
                        lines.append(f"* {f}")
                if updated_partial:
                    # Normally shouldn't occur on success (errs == []), but include for completeness.
                    lines.append("Partially updated files:")
                    for f in updated_partial:
                        lines.append(f"* {f}")
                if deleted:
                    lines.append("Deleted files:")
                    for f in deleted:
                        lines.append(f"* {f}")
                final = Message(role="agent", text="\n".join(lines))

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
