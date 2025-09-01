from pathlib import Path
from typing import Optional, Union, Dict, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from .tools import BaseTool

from .know import KnowProject
from .know.tools import register_know_tools
from .tools import get_all_tools
from .settings import KnowProjectSettings, Settings, load_settings
from .templates import write_default_config


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

    @property
    def config_path(self) -> Path:
        return (self.base_path / self.config_relpath).resolve()

    @classmethod
    def from_base_path(cls, base_path: Union[str, Path]) -> "Project":
        return init_project(base_path)

    async def shutdown(self) -> None:
        """Gracefully shut down project components."""
        await self.know.shutdown()


def init_project(base_path: Union[str, Path], config_relpath: Union[str, Path] = ".vocode/config.yaml") -> Project:
    """
    Initialize a Project: ensure config exists (write default if missing) and load settings.
    """
    base = Path(base_path).resolve()
    rel = Path(config_relpath)
    config_path = (base / rel).resolve()
    # Ensure .vocode directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    # If missing, write template config (and sample files)
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

    # Tools are enabled by default; settings can disable them.
    all_tools = get_all_tools()
    disabled_tool_names = {entry.name for entry in settings.tools or [] if entry.enabled is False}
    tools = {
        name: tool_instance
        for name, tool_instance in all_tools.items()
        if name not in disabled_tool_names
    }

    return Project(base_path=base, config_relpath=rel, settings=settings, tools=tools, know=know_project)
