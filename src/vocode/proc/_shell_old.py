from __future__ import annotations

import asyncio
import uuid
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict
import contextlib

from vocode.proc.manager import ProcessManager
from vocode.proc.base import ProcessHandle


@dataclass
class PosixShellSpec:
    # Program and arguments used to start the long-lived shell.
    # Defaults to a clean bash session without user profiles.
    program: str = "bash"
    args: list[str] = field(default_factory=lambda: ["--noprofile", "--norc"])

    def start_command(self) -> str:
        parts = [self.program, *self.args]
        return " ".join(shlex.quote(p) for p in parts)

    def wrap_command_with_marker(self, command: str, marker: str) -> str:
        # Execute the user command in a fresh subshell to insulate parsing errors,
        # capture its exit code, redirect stderr->stdout for payload, and always print
        # a single-line marker with the exit code appended.
        # Build inner invocation: <program> <args...> -c '<command>'
        tokens: list[str] = [self.program, *self.args, "-c", command]
        inner = " ".join(shlex.quote(t) for t in tokens)
        # Initialize rc to a fallback, run the inner, save rc, then print marker:rc
        # Emit a single marker line as "<marker>:<rc>"
        return (
            "rc=127; "
            f"{{ {inner}; rc=$?; }} 2>&1; "
            "echo "
            + shlex.quote(marker)
            + ':"$rc"\n'
        )


class ShellTimeoutError(asyncio.TimeoutError):
    pass


class ShellRunner:
    """
    Owns a long-lived POSIX shell process and runs commands sequentially.
    Usage:
        runner = ShellRunner(process_manager)
        await runner.start()
        out = await runner.run("ls -la")
        await runner.aclose()
    """

    def __init__(
        self,
        process_manager: ProcessManager,
        *,
        spec: Optional[PosixShellSpec] = None,
        cwd: Optional[Path] = None,
        env_overlay: Optional[Dict[str, str]] = None,
        default_timeout_s: float = 120.0,
        name: Optional[str] = "shell",
    ) -> None:
        self._pm = process_manager
        self._spec = spec or PosixShellSpec()
        self._cwd = cwd
        self._env_overlay = env_overlay
        self._default_timeout_s = default_timeout_s
        self._name = name
        self._handle: Optional[ProcessHandle] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._handle is not None and self._handle.alive():
            return
        self._handle = await self._pm.spawn(
            command=self._spec.start_command(),
            name=self._name,
            cwd=self._cwd,
            env_overlay=self._env_overlay,
        )

    async def aclose(self) -> None:
        if self._handle is not None and self._handle.alive():
            try:
                # Close stdin first to ensure the write transport is released
                await self._handle.terminate(grace_s=1.0)
            finally:
                if self._handle.alive():
                    await self._handle.kill()
            # Ensure the subprocess and its transports are fully torn down
            with contextlib.suppress(Exception):
                await self._handle.wait()
        self._handle = None

    async def _ensure_started(self) -> None:
        if self._handle is None or not self._handle.alive():
            await self.start()

    async def _restart(self) -> None:
        await self.aclose()
        await self.start()

    async def run(
        self, command: str, *, timeout_s: Optional[float] = None
    ) -> tuple[str, int]:
        """
        Run a command in the owned shell and return (combined stdout+stderr text, exit_code).
        Serialized via an internal lock to ensure one-at-a-time execution.
        """
        async with self._lock:
            await self._ensure_started()
            assert self._handle is not None

            marker = f"VOCODE_MARK_{uuid.uuid4().hex}"
            wrapped = self._spec.wrap_command_with_marker(command, marker)

            # Send command
            await self._handle.write(wrapped)

            # Collect output until marker is seen
            timeout = timeout_s if timeout_s is not None else self._default_timeout_s
            try:
                collect_task = asyncio.create_task(self._collect_until_marker(marker))
                return await asyncio.wait_for(collect_task, timeout=timeout)
            except asyncio.TimeoutError as e:
                # On timeout, terminate and respawn the shell to ensure the command is stopped
                # Make sure the collector task is cancelled and cleaned up
                with contextlib.suppress(asyncio.CancelledError):
                    collect_task.cancel()
                    await collect_task
                await self._restart()
                raise ShellTimeoutError(f"Command timed out after {timeout}s") from e

    async def _collect_until_marker(self, marker: str) -> tuple[str, int]:
        """
        Read stdout line-by-line until the stop marker line is seen.
        Uses only stdout because the wrapper redirects stderr to stdout.
        Returns (exact concatenation of lines prior to the marker, exit_code).
        """
        assert self._handle is not None
        out_chunks: list[str] = []
        exit_code: int = 0

        # Create a local iterator for this read; subsequent calls create new ones continuing from stream position.
        stdout_iter = self._handle.iter_stdout()
        try:
            async for line in stdout_iter:
                # Preserve exact output
                text = line.rstrip("\r\n")
                if text.startswith(marker):
                    # Parse "<marker>:<rc>"
                    suffix = text[len(marker) :]
                    if suffix.startswith(":"):
                        with contextlib.suppress(ValueError):
                            exit_code = int(suffix[1:])
                    break
                out_chunks.append(line)
        finally:
            # Ensure the iterator is closed even on cancellation to release any reader tasks/transports
            try:
                await stdout_iter.aclose()  # async-generator path
            except AttributeError:
                pass
            except Exception:
                pass
        return "".join(out_chunks), exit_code
