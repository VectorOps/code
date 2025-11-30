from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional, Any, List, Set, Tuple
from pydantic import model_validator

from vocode.runner.runner import Executor
from vocode.models import Node
from vocode.state import Message
from vocode.runner.models import (
    ReqPacket,
    ReqInterimMessage,
    ReqFinalMessage,
    ExecRunInput,
)
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
    ) -> AsyncIterator[tuple[ReqPacket, Optional[Any]]]:
        cfg: ExecNode = self.config  # type: ignore[assignment]
        timeout_s = cfg.timeout_s if cfg.timeout_s is not None else 120.0
        # Debounce window: if the command finishes quickly, do not stream interims
        STREAM_DEBOUNCE_S = 0.25

        pm: ProcessManager = self.project.processes

        # Spawn a one-off process
        handle = await pm.spawn(
            command=cfg.command, name=f"exec:{cfg.name}", shell=True, use_pty=False
        )

        # Prepare streaming buffers and pumps
        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
        queue: asyncio.Queue[Tuple[str, str]] = asyncio.Queue()
        pending_chunks: List[str] = []

        async def _pump_stdout():
            async for chunk in handle.iter_stdout():
                await queue.put(("stdout", chunk))

        async def _pump_stderr():
            async for chunk in handle.iter_stderr():
                await queue.put(("stderr", chunk))

        pump_out = asyncio.create_task(_pump_stdout())
        pump_err = asyncio.create_task(_pump_stderr())
        wait_task = asyncio.create_task(handle.wait())
        # Build header text used for both streaming and final message
        header_parts: List[str] = []
        if cfg.message:
            header_parts.append(cfg.message)
        header_parts.append(f"> {cfg.command}")
        header = "\n".join(header_parts)

        # Stream chunks as they arrive while observing timeout
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        timed_out = False
        rc: Optional[int] = None
        streaming_started = False

        while True:
            # Handle timeout
            if not wait_task.done() and timeout_s is not None:
                if loop.time() - start_time > timeout_s:
                    timed_out = True
                    await handle.kill()

            # Drain available output
            try:
                src, chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
                if src == "stdout":
                    stdout_parts.append(chunk)
                else:
                    stderr_parts.append(chunk)
                # Decide when to start streaming: after debounce window and if process is still running
                if not streaming_started:
                    pending_chunks.append(chunk)
                    if (
                        loop.time() - start_time
                    ) >= STREAM_DEBOUNCE_S and not wait_task.done():
                        streaming_started = True
                        # Emit header first, then any pending chunks accumulated so far
                        _ = yield (
                            ReqInterimMessage(
                                message=Message(
                                    role="agent", text=f"{header}\n", node=cfg.name
                                )
                            ),
                            None,
                        )
                        for p in pending_chunks:
                            _ = yield (
                                ReqInterimMessage(
                                    message=Message(role="agent", text=p, node=cfg.name)
                                ),
                                None,
                            )
                        pending_chunks.clear()
                else:
                    # Stream this chunk immediately
                    _ = yield (
                        ReqInterimMessage(
                            message=Message(role="agent", text=chunk, node=cfg.name)
                        ),
                        None,
                    )
            except asyncio.TimeoutError:
                pass

            # Exit condition: process finished, pumps done, and queue drained
            if (
                wait_task.done()
                and pump_out.done()
                and pump_err.done()
                and queue.empty()
            ):
                break

        # Ensure pumps finished
        await asyncio.gather(pump_out, pump_err, return_exceptions=True)
        # Get return code
        try:
            rc = wait_task.result()
        except Exception:
            # In case wait_task raised due to cancellation/kill sequencing
            try:
                rc = await handle.wait()
            except Exception:
                rc = None

        output = "".join(stdout_parts) + "".join(stderr_parts)
        # If we streamed, the terminal already saw 'header' and all chunks exactly as they arrived.
        # Avoid printing twice by matching the streamed text exactly (no extra newline).
        if streaming_started:
            final_text = f"{header}\n{output}" if output else header
        else:
            # No streaming: include a newline between header and output for readability
            if output:
                final_text = f"{header}\n{output}"
            else:
                final_text = header

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

        final_msg = Message(role="agent", text=final_text, node=cfg.name)
        yield ReqFinalMessage(message=final_msg, outcome_name=outcome_name), None
