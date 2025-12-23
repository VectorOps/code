from __future__ import annotations

from pathlib import Path
from typing import Optional, Type

from vocode import settings as vsettings

from .manager import ProcessManager
from .shell_base import ShellCommandHandle, ShellProcessor
from .shell_direct import DirectShellProcessor
from .shell_persistent import PersistentShellProcessor


PROCESSOR_FACTORY: dict[vsettings.ShellMode, Type[ShellProcessor]] = {
    vsettings.ShellMode.direct: DirectShellProcessor,
    vsettings.ShellMode.shell: PersistentShellProcessor,
}


class ShellManager:
    """
    High-level manager for running shell commands in either:
      - direct mode: each command is its own subprocess
      - shell mode: commands run via a long-lived shell process with wrapped markers
    """

    def __init__(
        self,
        process_manager: ProcessManager,
        *,
        settings: Optional[vsettings.ShellSettings] = None,
        default_cwd: Optional[Path] = None,
        env_overlay: Optional[dict[str, str]] = None,
    ) -> None:
        self._pm = process_manager
        self._settings = settings or vsettings.ShellSettings()
        self._mode = self._settings.mode
        self._default_cwd = default_cwd
        self._env_overlay: dict[str, str] = dict(env_overlay or {})
        self._processor: Optional[ShellProcessor] = None

    @property
    def mode(self) -> vsettings.ShellMode:
        return self._mode

    async def start(self) -> None:
        if self._processor is not None:
            return

        processor_cls = PROCESSOR_FACTORY.get(self._mode)
        if processor_cls is None:
            raise ValueError(f"Unsupported shell mode: {self._mode}")

        self._processor = processor_cls(
            process_manager=self._pm,
            settings=self._settings,
            default_cwd=self._default_cwd,
            env_overlay=self._env_overlay,
        )
        await self._processor.start()

    async def stop(self) -> None:
        if self._processor is None:
            return
        processor = self._processor
        self._processor = None
        await processor.stop()

    async def run(self, command: str) -> ShellCommandHandle:
        """
        Run a shell command using the configured processor.

        Ensures the underlying processor is started on first use.
        """
        if self._processor is None:
            await self.start()
        assert self._processor is not None
        return await self._processor.run(command)


__all__ = ["ShellManager", "ShellCommandHandle"]