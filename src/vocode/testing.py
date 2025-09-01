from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .project import Project
from .settings import Settings, KnowProjectSettings
from .know import KnowProject
from .know.tools import register_know_tools
from .tools import get_all_tools

if TYPE_CHECKING:
    from .tools import BaseTool


class ProjectSandbox:
    """
    A helper for creating Project instances for tests.
    It properly initializes a Project without creating any config files or directories.
    It can be used as an async context manager to handle shutdown.

    Usage:
        async with ProjectSandbox.create(tmp_path) as project:
            # use project instance
            ...
    """

    def __init__(self, project: Project):
        self._project = project

    @classmethod
    def create(cls, base_path: Path) -> "ProjectSandbox":
        """
        Creates a Project instance for testing, avoiding filesystem writes for config.
        """
        settings = Settings()

        # Initialize `know` with an in-memory database.
        know_settings = KnowProjectSettings(
            project_name="test-project",
            repo_name="test-repo",
            repo_path=str(base_path),
            repository_connection=":memory:",
        )
        know_project = KnowProject()
        know_project.start(know_settings)

        # Register and collect all tools.
        register_know_tools()
        all_tools = get_all_tools()

        # Create the project instance.
        project = Project(
            base_path=base_path,
            config_relpath=Path(".vocode/config.yaml"),  # Dummy path
            settings=settings,
            tools=all_tools,
            know=know_project,
        )

        return cls(project)

    async def shutdown(self) -> None:
        """Gracefully shut down project components."""
        await self._project.shutdown()

    async def __aenter__(self) -> Project:
        return self._project

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()
