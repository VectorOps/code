from __future__ import annotations

import asyncio
import shlex
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import contextlib

from vocode.proc.manager import ProcessManager
from vocode.proc.base import ProcessHandle


@dataclass
class DockerShellSpec:
    """Specification for starting a long-lived shell inside a Docker container.

    This mirrors PosixShellSpec but wraps the shell invocation in a `docker run`
    command that mounts the project directory and switches to a configured
    working directory inside the container.
    """

    docker_binary: str = "docker"
    docker_args: list[str] = field(default_factory=lambda: ["run", "-i", "--rm"])
    image: str = "bash:latest"
    workdir: str = "/workspace"
    shell_program: str = "bash"
    shell_args: list[str] = field(default_factory=lambda: ["--noprofile", "--norc"])

    def start_command(self, project_dir: Path) -> str:
        """Build the full docker command used to start the interactive shell.

        The project directory is mounted into the container at self.workdir and
        the shell starts in that directory.
        """

        # docker <docker_args...> -v <host>:<container> -w <workdir> <image> <shell_program> <shell_args...>
        parts: list[str] = [self.docker_binary, *self.docker_args]

        # Map project directory into container; always use absolute path
        host_path = str(project_dir.resolve())
        volume_arg = f"{host_path}:{self.workdir}"
        parts.extend(["-v", volume_arg, "-w", self.workdir])

        parts.append(self.image)
        parts.append(self.shell_program)
        parts.extend(self.shell_args)

        return " ".join(shlex.quote(p) for p in parts)

    def wrap_command_with_marker(self, command: str, marker: str) -> str:
        """Wrap a user command to capture combined output and exit code.

        This uses the same strategy as PosixShellSpec.wrap_command_with_marker:
        run the user command in a subshell, capture its exit code, redirect
        stderr to stdout, and then print a single-line marker containing the
        exit code.
        """

        tokens: list[str] = [self.shell_program, *self.shell_args, "-c", command]
        inner = " ".join(shlex.quote(t) for t in tokens)
        return (
            "rc=127; "
            f"{{ {inner}; rc=$?; }} 2>&1; "
            "echo " + shlex.quote(marker) + ':"$rc"\n'
        )


class DockerShellTimeoutError(asyncio.TimeoutError):
    pass


class DockerShellRunner:
    """Owns a long-lived Docker-backed shell process and runs commands.

    The API is intentionally compatible with ShellRunner so callers can
    switch between host shell and Docker-backed execution based on settings.
    """

    def __init__(
        self,
        process_manager: ProcessManager,
        *,
        spec: Optional[DockerShellSpec] = None,
        project_dir: Optional[Path] = None,
        env_overlay: Optional[Dict[str, str]] = None,
        default_timeout_s: float = 120.0,
        name: Optional[str] = "docker-shell",
    ) -> None:
        self._pm = process_manager
        self._spec = spec or DockerShellSpec()
        # Prefer explicit project_dir; otherwise fall back to the process
        # manager's default_cwd, which is the project base path in normal
        # Project wiring.
        if project_dir is not None:
            self._project_dir = project_dir
        else:
            # Access private attribute directly inside the proc package.
            self._project_dir = process_manager._default_cwd  # type: ignore[attr-defined]
        self._env_overlay = env_overlay
        self._default_timeout_s = default_timeout_s
        self._name = name
        self._handle: Optional[ProcessHandle] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._handle is not None and self._handle.alive():
            return
        assert self._project_dir is not None
        self._handle = await self._pm.spawn(
            command=self._spec.start_command(self._project_dir),
            name=self._name,
            cwd=self._project_dir,
            env_overlay=self._env_overlay,
        )

    async def aclose(self) -> None:
        if self._handle is not None and self._handle.alive():
            try:
                await self._handle.terminate(grace_s=1.0)
            finally:
                if self._handle.alive():
                    await self._handle.kill()
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
        async with self._lock:
            await self._ensure_started()
            assert self._handle is not None

            marker = f"VOCODE_DOCKER_MARK_{uuid.uuid4().hex}"
            wrapped = self._spec.wrap_command_with_marker(command, marker)

            await self._handle.write(wrapped)

            effective_timeout = (
                timeout_s if timeout_s is not None else self._default_timeout_s
            )
            try:
                collect_task = asyncio.create_task(self._collect_until_marker(marker))
                return await asyncio.wait_for(collect_task, timeout=effective_timeout)
            except asyncio.TimeoutError as e:  # pragma: no cover - exact timing
                with contextlib.suppress(asyncio.CancelledError):
                    collect_task.cancel()
                    await collect_task
                await self._restart()
                raise DockerShellTimeoutError(
                    f"Command timed out after {effective_timeout}s"
                ) from e

    async def _collect_until_marker(self, marker: str) -> tuple[str, int]:
        assert self._handle is not None
        out_chunks: list[str] = []
        exit_code = 0

        stdout_iter = self._handle.iter_stdout()
        try:
            async for line in stdout_iter:
                text = line.rstrip("\r\n")
                if text.startswith(marker):
                    suffix = text[len(marker) :]
                    if suffix.startswith(":"):
                        with contextlib.suppress(ValueError):
                            exit_code = int(suffix[1:])
                    break
                out_chunks.append(line)
        finally:
            try:
                await stdout_iter.aclose()  # type: ignore[attr-defined]
            except Exception:
                pass
        return "".join(out_chunks), exit_code
