from __future__ import annotations

from pathlib import Path
from typing import Optional

from vocode.project import Project, init_project


class ProjectSandbox:
    """
    Async context manager that initializes a real Project in a temporary directory,
    starts async subsystems (e.g., MCP), and shuts it down on exit.

    Usage:
      async with ProjectSandbox.create(tmp_path) as project:
          ...
    """

    def __init__(self, base_path: Path) -> None:
        self._base_path = Path(base_path)
        self.project: Optional[Project] = None

    @classmethod
    def create(cls, base_path: Path) -> "ProjectSandbox":
        return cls(base_path)

    async def __aenter__(self) -> Project:
        # Initialize a project in base_path; do not search ancestors or SCM
        proj = init_project(
            self._base_path,
            search_ancestors=False,
            use_scm=False,
        )
        await proj.start()
        self.project = proj
        return proj

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.project is not None:
            await self.project.shutdown()
            self.project = None