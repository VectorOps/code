import asyncio
from typing import Any, Callable, Optional

from know import init_project as know_init_project
from know.models import Repo
from know.project import ProjectManager as KnowProjectManager
from know.settings import ProjectSettings as KnowProjectSettings

from ..lib.threads import RpcThread
from ..proto import (
    Packet,
    PacketProjectOpFinish,
    PacketProjectOpProgress,
    PacketProjectOpStart,
)


know_thread = RpcThread(name="know-project")


class KnowProject:
    """An async wrapper around know.ProjectManager that runs it in a dedicated thread."""

    pm: KnowProjectManager

    def _init(self, settings: KnowProjectSettings) -> None:
        """This runs on the RpcThread and initializes the ProjectManager."""
        # Initialize without triggering an automatic refresh; refresh is handled by Project.start()
        self.pm = know_init_project(settings, refresh=False)

    def start(self, settings: KnowProjectSettings) -> None:
        """
        Initializes the project manager on a dedicated thread.
        This call is synchronous and will block until initialization is complete.
        """
        know_thread.start()
        know_thread.proxy()(self._init)(settings)

    async def shutdown(self) -> None:
        """Shuts down the project manager and its thread."""
        if know_thread._started.is_set() and not know_thread._stopped.is_set():
            # Destroy the manager on the RPC thread
            await know_thread.async_proxy()(self.pm.destroy)()
        # Ensure the background thread is terminated (idempotent if already stopped)
        # TODO: This locks up
        # know_thread.shutdown()

    @property
    def data(self):
        """Direct, thread-safe access to the data repository."""
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
                loop = asyncio.get_running_loop()

                def progress_callback(progress):
                    packet = PacketProjectOpProgress(
                        progress=progress.processed_files, total=progress.total_files
                    )
                    asyncio.run_coroutine_threadsafe(progress_sender(packet), loop)

                kwargs["progress_callback"] = progress_callback

        try:
            await know_thread.async_proxy()(op_func)(**kwargs)
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
