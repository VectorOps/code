from pathlib import Path
from typing import Optional, Union, Dict, Any, TYPE_CHECKING, List
from enum import Enum
from pydantic import BaseModel

if TYPE_CHECKING:
    from .tools import BaseTool
    from know.models import Repo

from .know import KnowProject
from .scm.git import GitSCM
from .know.tools import register_know_tools
from .tools import get_all_tools
from .settings import KnowProjectSettings, Settings, load_settings
from .templates import write_default_config
from vocode.runner.models import TokenUsageTotals
from vocode.commands import CommandManager
from .mcp.manager import MCPManager
from .proc.manager import ProcessManager
from .proc.base import EnvPolicy


class ProjectState:
    """
    Ephemeral, process-local project-level state shared across executors.
    Not persisted across runs.
    """

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def clear(self) -> None:
        self._data.clear()


class FileChangeType(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"


class FileChangeModel(BaseModel):
    type: FileChangeType
    # Relative filename within the project root
    relative_filename: str


class Project:
    def __init__(
        self,
        base_path: Path,
        config_relpath: Path,
        settings: Optional[Settings],
        tools: Dict[str, "BaseTool"],
        know: KnowProject,
    ):
        self.base_path: Path = base_path
        self.config_relpath: Path = config_relpath
        self.settings: Optional[Settings] = settings
        self.tools: Dict[str, "BaseTool"] = tools
        self.know: KnowProject = know
        self.mcp_manager: Optional[MCPManager] = None
        # Project-level shared state for executors
        self.project_state: ProjectState = ProjectState()
        # Workflow-scoped custom commands (cleared on workflow changes)
        self.commands: CommandManager = CommandManager()
        # Ephemeral (per-process) LLM usage totals
        self.llm_usage: TokenUsageTotals = TokenUsageTotals()
        # Process manager
        self.processes: Optional[ProcessManager] = None

    @property
    def config_path(self) -> Path:
        # Do not resolve symlinks; return the composed path as-is
        return self.base_path / self.config_relpath

    @classmethod
    def from_base_path(
        cls,
        base_path: Union[str, Path],
        *,
        search_ancestors: bool = True,
        use_scm: bool = True,
    ) -> "Project":
        return init_project(
            base_path,
            search_ancestors=search_ancestors,
            use_scm=use_scm,
        )

    async def shutdown(self) -> None:
        """Gracefully shut down project components."""
        # Stop processes first to release IO and resources
        if self.processes is not None:
            await self.processes.shutdown()
            self.processes = None
        # Stop MCP first to allow cleanup of any tool proxies
        if self.mcp_manager is not None:
            await self.mcp_manager.stop()
            self.mcp_manager = None
        await self.know.shutdown()

    async def refresh(
        self,
        repo: Optional["Repo"] = None,
        files: Optional[List[FileChangeModel]] = None,
    ) -> None:
        """
        Asynchronously refresh the underlying 'know' project (optionally for a specific Repo).
        Note: The 'files' parameter is currently ignored since the 'know' backend
        does not support partial refresh yet.
        """
        await self.know.refresh(repo)

    # ---------------
    # LLM usage totals
    # ---------------
    def add_llm_usage(
        self, prompt_delta: int, completion_delta: int, cost_delta: float
    ) -> None:
        """Increment aggregate LLM usage totals for this project."""
        self.llm_usage.prompt_tokens += int(prompt_delta or 0)
        self.llm_usage.completion_tokens += int(completion_delta or 0)
        self.llm_usage.cost_dollars += float(cost_delta or 0.0)

    def refresh_tools_from_registry(self) -> None:
        """
        Refresh self.tools from the global registry, excluding disabled tools per settings.
        """
        from .tools import get_all_tools

        disabled_tool_names = (
            {
                entry.name
                for entry in (self.settings.tools or [])
                if entry.enabled is False
            }
            if self.settings
            else set()
        )
        all_tools = get_all_tools()
        self.tools = {
            name: tool
            for name, tool in all_tools.items()
            if name not in disabled_tool_names
        }

    async def start(self) -> None:
        """
        Start project subsystems that require async initialization (e.g., MCP).
        """
        if self.settings and self.settings.mcp:
            # Create and start MCP manager; it will register tool proxies and refresh the tool map.
            self.mcp_manager = MCPManager(self.settings.mcp)
            await self.mcp_manager.start(self)
        # Initialize process manager (idempotent)
        if self.processes is None:
            backend_name = (
                self.settings.process.backend
                if (self.settings and self.settings.process)
                else "local"
            )
            env = (
                self.settings.process.env
                if (self.settings and self.settings.process)
                else None
            )
            env_policy = EnvPolicy(
                inherit_parent=(env.inherit_parent if env else True),
                allowlist=(env.allowlist if env else None),
                denylist=(env.denylist if env else None),
                defaults=(env.defaults if env else {}),
            )
            self.processes = ProcessManager(
                backend_name=backend_name,
                default_cwd=self.base_path,
                env_policy=env_policy,
            )


def _find_project_root_with_config(start: Path, rel_config: Path) -> Optional[Path]:
    """
    Walk upwards from 'start' to filesystem root looking for rel_config (e.g., '.vocode/config.yaml').
    Returns the directory that contains rel_config if found; otherwise None.
    Note: Ignore '.vocode' dirs that don't contain the config file.
    """
    current = start
    while True:
        candidate = current / rel_config
        if candidate.is_file():
            return current
        if current.parent == current:
            # Reached filesystem root
            return None
        current = current.parent


def init_project(
    base_path: Union[str, Path],
    config_relpath: Union[str, Path] = ".vocode/config.yaml",
    *,
    search_ancestors: bool = True,
    use_scm: bool = True,
) -> Project:
    """
    Initialize a Project by:
    1) Searching upwards for an existing .vocode/config.yaml (nearest ancestor) if search_ancestors is True.
    2) Otherwise, if use_scm is True, detecting a Git repository root and using it as the base; create .vocode/config.yaml there if missing.
    3) Otherwise, creating .vocode/config.yaml at the provided start directory.
    """
    start_path = Path(base_path)
    # If a file path is provided, start from its parent; otherwise the directory itself.
    start_dir = start_path if start_path.is_dir() else start_path.parent
    start_dir = start_dir.resolve()

    rel = Path(config_relpath)
    base = None
    config_path = None

    # 1) Search upwards for an existing config file (nearest ancestor)
    found_base = (
        _find_project_root_with_config(start_dir, rel) if search_ancestors else None
    )
    if found_base is not None:
        base = found_base
        config_path = base / rel
    else:
        # 2) Try SCM (git) if enabled
        if use_scm:
            repo_root = GitSCM().find_repo(start_dir)
            if repo_root is not None:
                base = Path(repo_root)
                config_path = base / rel
                if not config_path.exists():
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    write_default_config(config_path)

        # 3) Fall back to the start directory
        if base is None:
            base = start_dir
            config_path = base / rel
            config_path.parent.mkdir(parents=True, exist_ok=True)
            if not config_path.exists():
                write_default_config(config_path)

    # Load merged settings (supports include + YAML/JSON5)
    settings = load_settings(str(config_path))

    # Initialize `know` project.
    if settings.know:
        # Create a mutable copy of know settings to populate defaults.
        know_settings = settings.know.model_copy(deep=True)
    else:
        # Create default settings if 'know' section is missing from config.
        # Required fields are given placeholder values that will be immediately
        # cleared to trigger the defaulting logic below.
        know_settings = KnowProjectSettings(project_name="_", repo_name="_")
        know_settings.project_name = ""
        know_settings.repo_name = ""

    # Default project/repo names if not set.
    if not know_settings.project_name:
        know_settings.project_name = "my-project"
    if not know_settings.repo_name:
        know_settings.repo_name = base.name
    if not know_settings.repo_path:
        know_settings.repo_path = str(base)

    # Default database path.
    if not know_settings.repository_connection:
        know_data_path = base / ".vocode/data"
        know_data_path.mkdir(parents=True, exist_ok=True)
        know_settings.repository_connection = str(know_data_path / "know.duckdb")

    know_project = KnowProject()
    know_project.start(know_settings)

    register_know_tools()
    proj = Project(
        base_path=base,
        config_relpath=rel,
        settings=settings,
        tools={},  # will be refreshed from registry below
        know=know_project,
    )
    # Initialize tools snapshot based on current registry (includes 'know' tools).
    proj.refresh_tools_from_registry()
    return proj
