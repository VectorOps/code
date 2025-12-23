from __future__ import annotations

import asyncio
import contextlib
import shlex
import uuid
from pathlib import Path
from typing import AsyncIterator, Optional

from vocode import settings as vsettings

from .manager import ProcessManager
from .base import ProcessHandle
from .shell_base import ShellCommandHandle, ShellProcessor


class PersistentShellCommand(ShellCommandHandle):
    """
    Represents a single command executed inside the long-lived shell.

    - iter_stdout streams lines until the marker is seen (marker is not yielded).
    - wait() waits until the marker is seen and returns the parsed exit code.
    """

    def __init__(
        self,
        *,
        processor: "PersistentShellProcessor",
        marker: str,
        name: Optional[str],
    ) -> None:
        self._processor = processor
        self._marker = marker
        self._returncode: Optional[int] = None
        self._done = asyncio.Event()
        self._stdout_consumed = False
        self.id = str(uuid.uuid4())
        self.name = name

    @property
    def pid(self) -> Optional[int]:
        handle = self._processor.handle
        return handle.pid if handle is not None else None

    @property
    def returncode(self) -> Optional[int]:
        return self._returncode

    def alive(self) -> bool:
        handle = self._processor.handle
        return (handle is not None and handle.alive()) and not self._done.is_set()

    async def write(self, data: str | bytes) -> None:
        handle = self._processor.handle
        if handle is None:
            return
        await handle.write(data)

    async def close_stdin(self) -> None:
        handle = self._processor.handle
        if handle is None:
            return
        await handle.close_stdin()

    async def iter_stdout(self) -> AsyncIterator[str]:
        handle = self._processor.handle
        if handle is None:
            return
        if self._stdout_consumed:
            return
        self._stdout_consumed = True

        stdout_iter = handle.iter_stdout()
        try:
            async for line in stdout_iter:
                text = line.rstrip("\r\n")
                if text.startswith(self._marker):
                    suffix = text[len(self._marker) :]
                    if suffix.startswith(":"):
                        with contextlib.suppress(ValueError):
                            self._returncode = int(suffix[1:])
                    if self._returncode is None:
                        self._returncode = 0
                    self._done.set()
                    self._processor.on_command_finished(self)
                    break
                yield line
        finally:
            with contextlib.suppress(Exception):
                await stdout_iter.aclose()
            if not self._done.is_set():
                # Shell exited without emitting the marker; treat as unknown non-zero.
                self._returncode = self._returncode if self._returncode is not None else 1
                self._done.set()
                self._processor.on_command_finished(self)

    async def iter_stderr(self) -> AsyncIterator[str]:
        # stderr for wrapped commands is redirected to stdout (2>&1), but we
        # still forward any shell-level stderr if present.
        handle = self._processor.handle
        if handle is None:
            return
        async for line in handle.iter_stderr():
            yield line

    async def terminate(self, grace_s: float = 5.0) -> None:
        handle = self._processor.handle
        if handle is None:
            return
        await handle.terminate(grace_s=grace_s)
        if not self._done.is_set():
            self._returncode = self._returncode if self._returncode is not None else 1
            self._done.set()
            self._processor.on_command_finished(self)

    async def kill(self) -> None:
        handle = self._processor.handle
        if handle is None:
            return
        await handle.kill()
        if not self._done.is_set():
            self._returncode = self._returncode if self._returncode is not None else 1
            self._done.set()
            self._processor.on_command_finished(self)

    async def wait(self) -> int:
        await self._done.wait()
        assert self._returncode is not None
        return self._returncode


class PersistentShellProcessor(ShellProcessor):
    """
    Shell mode: maintain a long-lived shell process and run commands within it.

    Commands are wrapped with a marker line "<marker>:<rc>" that is consumed
    internally and not yielded to callers.
    """

    def __init__(
        self,
        *,
        process_manager: ProcessManager,
        settings: vsettings.ShellSettings,
        default_cwd: Optional[Path],
        env_overlay: dict[str, str],
    ) -> None:
        self._pm = process_manager
        self._settings = settings
        self._default_cwd = default_cwd
        self._env_overlay = env_overlay
        self._handle: Optional[ProcessHandle] = None
        self._active_cmd: Optional[PersistentShellCommand] = None
        self._lock = asyncio.Lock()

    @property
    def handle(self) -> Optional[ProcessHandle]:
        return self._handle

    async def start(self) -> None:
        async with self._lock:
            await self._ensure_started()

    async def stop(self) -> None:
        async with self._lock:
            handle = self._handle
            self._handle = None
            self._active_cmd = None

        if handle is None:
            return

        try:
            await handle.terminate(grace_s=1.0)
        except Exception:
            with contextlib.suppress(Exception):
                await handle.kill()
        finally:
            with contextlib.suppress(Exception):
                await handle.wait()

    async def run(self, command: str) -> ShellCommandHandle:
        async with self._lock:
            await self._ensure_started()
            if self._active_cmd is not None and self._active_cmd.alive():
                raise RuntimeError("Another shell command is already running")

            assert self._handle is not None
            marker = f"VOCODE_MARK_{uuid.uuid4().hex}"
            wrapped = self._wrap_command_with_marker(command, marker)
            cmd = PersistentShellCommand(
                processor=self,
                marker=marker,
                name="shell",
            )
            self._active_cmd = cmd
            await self._handle.write(wrapped)
            return cmd

    async def _ensure_started(self) -> None:
        if self._handle is not None and self._handle.alive():
            return
        start_cmd = self._start_command()
        self._handle = await self._pm.spawn(
            command=start_cmd,
            name="shell",
            cwd=self._default_cwd,
            env_overlay=self._env_overlay or None,
        )

    def _start_command(self) -> str:
        parts = [self._settings.program, *self._settings.args]
        return " ".join(shlex.quote(p) for p in parts)

    def _wrap_command_with_marker(self, command: str, marker: str) -> str:
        # Execute the user command in a fresh subshell to insulate parsing
        # errors, capture its exit code, redirect stderr->stdout for payload,
        # and always print a single-line marker with the exit code appended.
        # Build inner invocation: <program> <args...> -c '<command>'
        tokens: list[str] = [
            self._settings.program,
            *self._settings.args,
            "-c",
            command,
        ]
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

    def on_command_finished(self, cmd: PersistentShellCommand) -> None:
        if self._active_cmd is cmd:
            self._active_cmd = None