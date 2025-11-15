from __future__ import annotations
from dataclasses import dataclass

import pathlib
from typing import AsyncIterator, List, Optional, Any, Dict
from pydantic import BaseModel, Field
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
from vocode.patch import get_supported_formats, apply_patch


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


# No local registry or IO helpers; delegate to vocode.patch helpers


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

    async def run(
        self, inp: ExecRunInput
    ) -> AsyncIterator[tuple[ReqPacket, Optional[Any]]]:
        cfg: ApplyPatchNode = self.config  # type: ignore[assignment]

        # Enforce supported patch formats
        fmt = (cfg.format or "v4a").lower()

        supported_set = set(get_supported_formats())
        if fmt not in supported_set:
            supported = ", ".join(sorted(supported_set))
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

        try:
            # Lazy import to avoid potential import cycles for refresh types
            from vocode.project import FileChangeModel, FileChangeType

            base_path = self.project.base_path  # type: ignore[attr-defined]
            summary, outcome, changes_map, _statuses, _errs = apply_patch(
                fmt, source_text, base_path
            )
            final = Message(role="agent", text=summary)

            # Build the accumulated list for refresh based on helper's changes_map
            change_type_map: Dict[str, FileChangeType] = {
                "created": FileChangeType.CREATED,
                "updated": FileChangeType.UPDATED,
                "deleted": FileChangeType.DELETED,
            }
            changed_files = [
                FileChangeModel(type=change_type_map[kind], relative_filename=rel)
                for rel, kind in changes_map.items()
                if kind in change_type_map
            ]
            # Schedule a non-blocking project refresh in the background (do not await).
            asyncio.create_task(self.project.refresh(files=changed_files))  # type: ignore[attr-defined]
        except Exception as e:
            final = Message(role="agent", text=f"Error applying patch: {e}")
            outcome = "fail"

        yield (ReqFinalMessage(message=final, outcome_name=outcome), inp.state)
        return
