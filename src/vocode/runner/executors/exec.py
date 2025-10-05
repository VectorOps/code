from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional, Any, List, Set
from pydantic import model_validator

from vocode.runner.runner import Executor
from vocode.models import Node
from vocode.state import Message
from vocode.runner.models import ReqFinalMessage, ExecRunInput
from vocode.proc.manager import ProcessManager


class ExecNode(Node):
    type: str = "exec"

    command: str
    timeout_s: Optional[float] = None
    expected_return_code: Optional[int] = None
    message: Optional[str] = None

    @model_validator(mode="after")
    def _validate_outcomes_vs_expected_code(self) -> "ExecNode":
        exp = self.expected_return_code
        if exp is None:
            # At most one outcome allowed
            if len(self.outcomes) > 1:
                raise ValueError(
                    "ExecNode: when 'expected_return_code' is not provided, at most one outcome is allowed"
                )
        else:
            # Must have exactly two outcomes: success and fail
            names: Set[str] = {o.name for o in self.outcomes}
            if names != {"success", "fail"}:
                raise ValueError(
                    "ExecNode: when 'expected_return_code' is provided, outcomes must be exactly {'success', 'fail'}"
                )
        return self


class ExecExecutor(Executor):
    # Must match ExecNode.type
    type = "exec"

    def __init__(self, config, project):
        super().__init__(config=config, project=project)
        if not isinstance(config, ExecNode):
            raise TypeError("ExecExecutor requires config to be an ExecNode")

    async def run(
        self, inp: ExecRunInput
    ) -> AsyncIterator[tuple[ReqFinalMessage, Optional[Any]]]:
        cfg: ExecNode = self.config  # type: ignore[assignment]
        timeout_s = cfg.timeout_s if cfg.timeout_s is not None else 120.0

        pm: ProcessManager = self.project.processes

        # Spawn a one-off process
        handle = await pm.spawn(
            command=cfg.command, name=f"exec:{cfg.name}", shell=True, use_pty=False
        )

        # Collect stdout and stderr concurrently (append stderr after stdout)
        stdout_parts: List[str] = []
        stderr_parts: List[str] = []

        async def _read_stdout():
            async for chunk in handle.iter_stdout():
                stdout_parts.append(chunk)

        async def _read_stderr():
            async for chunk in handle.iter_stderr():
                stderr_parts.append(chunk)

        readers = [
            asyncio.create_task(_read_stdout()),
            asyncio.create_task(_read_stderr()),
        ]

        timed_out = False
        try:
            rc = await asyncio.wait_for(handle.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            timed_out = True
            # Kill the process, then wait for it to exit
            try:
                await handle.kill()
            finally:
                try:
                    rc = await handle.wait()
                except Exception:
                    rc = None  # not expected to be used further

        # Ensure readers finished
        await asyncio.gather(*readers, return_exceptions=True)

        output = "".join(stdout_parts) + "".join(stderr_parts)

        # Build final message text: message (optional), then command header, then output
        header_parts: List[str] = []
        if cfg.message:
            header_parts.append(cfg.message)
        header_parts.append(f"> {cfg.command}")
        header = "\n".join(header_parts)
        final_text = f"{header}\n{output}"

        # Outcome selection
        outcome_name: Optional[str] = None
        if cfg.expected_return_code is not None:
            if (not timed_out) and (rc == cfg.expected_return_code):
                outcome_name = "success"
            else:
                outcome_name = "fail"
        else:
            # At most one outcome allowed; if present, take that one.
            if len(cfg.outcomes) == 1:
                outcome_name = cfg.outcomes[0].name

        final_msg = Message(role="agent", text=final_text)
        yield ReqFinalMessage(message=final_msg, outcome_name=outcome_name), None