from typing import Optional

from know import init_project as know_init_project
from know.models import Repo
from know.project import ProjectManager as KnowProjectManager
from know.settings import ProjectSettings as KnowProjectSettings

from ..lib.threads import RpcThread


know_thread = RpcThread(name="know-project")


class KnowProject:
    """An async wrapper around know.ProjectManager that runs it in a dedicated thread."""

    pm: KnowProjectManager

    def _init(self, settings: KnowProjectSettings) -> None:
        """This runs on the RpcThread and initializes the ProjectManager."""
        self.pm = know_init_project(settings)

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

    async def refresh(self, repo: Optional[Repo] = None) -> None:
        """Asynchronously refresh a repository."""
        await know_thread.async_proxy()(self.pm.refresh)(repo)

    async def refresh_all(self) -> None:
        """Asynchronously refresh all repositories."""
        await know_thread.async_proxy()(self.pm.refresh_all)()

    async def maybe_refresh(self) -> None:
        """Asynchronously refresh if cooldown has passed."""
        await know_thread.async_proxy()(self.pm.maybe_refresh)()
