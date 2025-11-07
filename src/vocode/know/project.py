from typing import Any, Callable, Optional
import asyncio

from knowlt import init_project as know_init_project
from knowlt.models import Repo
from knowlt.project import ProjectManager as KnowProjectManager
from knowlt.settings import ProjectSettings as KnowProjectSettings
from ..proto import (
    Packet,
    PacketProjectOpFinish,
    PacketProjectOpProgress,
    PacketProjectOpStart,
)


class KnowProject:
    """Async wrapper around knowlt.ProjectManager."""

    pm: KnowProjectManager

    async def start(self, settings: KnowProjectSettings) -> None:
        """Initialize the ProjectManager (no auto-refresh)."""
        self.pm = await know_init_project(settings, refresh=False)

    async def shutdown(self) -> None:
        """Shut down the project manager."""
        await self.pm.destroy()

    @property
    def data(self):
        """Direct access to the data repository."""
        return self.pm.data

    async def _report_op(
        self,
        op_func: Callable,
        op_name: str,
        progress_sender: Optional[Callable[[Packet], Any]] = None,
        progress_supported: bool = False,
        **kwargs,
    ):
        """Generic wrapper to report operation start, progress, and finish."""
        if progress_sender:
            await progress_sender(PacketProjectOpStart(message=op_name))
            if progress_supported:

                def progress_callback(progress):
                    packet = PacketProjectOpProgress(
                        progress=progress.processed_files, total=progress.total_files
                    )
                    # Fire-and-forget; schedule the async progress send.
                    asyncio.ensure_future(progress_sender(packet))

                kwargs["progress_callback"] = progress_callback

        try:
            await op_func(**kwargs)
        finally:
            if progress_sender:
                await progress_sender(PacketProjectOpFinish())

    async def refresh(
        self,
        repo: Optional[Repo] = None,
        progress_sender: Optional[Callable[[Packet], Any]] = None,
    ) -> None:
        """Asynchronously refresh a repository with optional progress reporting."""
        await self._report_op(
            self.pm.refresh,
            "Refreshing project",
            progress_sender,
            progress_supported=True,
            repo=repo,
        )

    async def refresh_all(
        self, progress_sender: Optional[Callable[[Packet], Any]] = None
    ) -> None:
        """Asynchronously refresh all repositories."""
        await self._report_op(
            self.pm.refresh_all, "Refreshing all projects", progress_sender
        )

    async def maybe_refresh(
        self, progress_sender: Optional[Callable[[Packet], Any]] = None
    ) -> None:
        """Asynchronously refresh if cooldown has passed."""
        await self._report_op(
            self.pm.maybe_refresh, "Checking for project refresh", progress_sender
        )
