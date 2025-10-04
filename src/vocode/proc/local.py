from __future__ import annotations
import asyncio
import os
import uuid
from pathlib import Path
from typing import AsyncIterator, Optional, Dict
from .base import ProcessBackend, ProcessHandle, SpawnOptions, EnvPolicy, register_backend

def _build_env(policy: EnvPolicy, overlay: Optional[Dict[str, str]]) -> Dict[str, str]:
    base: Dict[str, str] = {}
    if policy.inherit_parent:
        base = dict(os.environ)
        if policy.allowlist is not None:
            allow = set(policy.allowlist)
            base = {k: v for k, v in base.items() if k in allow}
        if policy.denylist is not None:
            for k in policy.denylist:
                base.pop(k, None)
    base.update(policy.defaults or {})
    if overlay:
        base.update(overlay)
    return base

class LocalProcessHandle(ProcessHandle):
    def __init__(self, proc: asyncio.subprocess.Process, name: Optional[str]) -> None:
        self._proc = proc
        self.id = str(uuid.uuid4())
        self.name = name

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid

    @property
    def returncode(self) -> Optional[int]:
        return self._proc.returncode

    def alive(self) -> bool:
        return self._proc.returncode is None

    async def write(self, data: str | bytes) -> None:
        if self._proc.stdin is None:
            return
        if isinstance(data, str):
            data = data.encode("utf-8")
        self._proc.stdin.write(data)
        await self._proc.stdin.drain()

    async def close_stdin(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
            try:
                await self._proc.stdin.wait_closed()
            except Exception:
                pass

    async def iter_stdout(self) -> AsyncIterator[str]:
        if self._proc.stdout is None:
            return
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                break
            yield line.decode("utf-8", errors="replace")

    async def iter_stderr(self) -> AsyncIterator[str]:
        if self._proc.stderr is None:
            return
        while True:
            line = await self._proc.stderr.readline()
            if not line:
                break
            yield line.decode("utf-8", errors="replace")

    async def terminate(self, grace_s: float = 5.0) -> None:
        if self._proc.returncode is not None:
            return
        try:
            self._proc.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=grace_s)
        except asyncio.TimeoutError:
            await self.kill()

    async def kill(self) -> None:
        if self._proc.returncode is not None:
            return
        try:
            self._proc.kill()
        except ProcessLookupError:
            return
        await self._proc.wait()

    async def wait(self) -> int:
        return await self._proc.wait()

class LocalSubprocessBackend(ProcessBackend):
    def __init__(self, env_policy: Optional[EnvPolicy] = None) -> None:
        # Public property per requirement
        self.env_policy: EnvPolicy = env_policy or EnvPolicy()

    async def spawn(self, opts: SpawnOptions) -> ProcessHandle:
        if opts.use_pty:
            # Reserved for future PTY support
            raise NotImplementedError("PTY not supported by LocalSubprocessBackend yet")
        cwd: Optional[str | Path] = opts.cwd
        env = _build_env(self.env_policy, opts.env_overlay)
        # Shell-string execution
        proc = await asyncio.create_subprocess_shell(
            opts.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
        )
        return LocalProcessHandle(proc, opts.name)

# Register backend under 'local'
register_backend("local", lambda: LocalSubprocessBackend())
