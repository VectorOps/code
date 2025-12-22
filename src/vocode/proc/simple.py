from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from .base import EnvPolicy, ProcessBackend, ProcessHandle, SpawnOptions
from .local import LocalProcessHandle, _build_env


class SimpleSubprocessBackend(ProcessBackend):
    """A minimal backend similar to LocalSubprocessBackend.

    This implementation intentionally mirrors LocalSubprocessBackend but does
    not place subprocesses into their own process group. It is useful for
    environments where process group semantics are not desired.
    """

    def __init__(self, env_policy: Optional[EnvPolicy] = None) -> None:
        self.env_policy: EnvPolicy = env_policy or EnvPolicy()

    async def spawn(self, opts: SpawnOptions) -> ProcessHandle:
        if opts.use_pty:
            raise NotImplementedError("PTY not supported by SimpleSubprocessBackend")
        cwd: Optional[str | Path] = opts.cwd
        env = _build_env(self.env_policy, opts.env_overlay)

        proc = await asyncio.create_subprocess_shell(
            opts.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
        )
        # use_process_group is ignored in this backend; it always uses the
        # default OS behavior for subprocesses.
        return LocalProcessHandle(proc, opts.name, use_process_group=False)
