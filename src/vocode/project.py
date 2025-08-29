from pathlib import Path
from typing import Optional, Union
from pydantic import BaseModel, Field

from .settings import Settings, load_settings
from .templates import write_default_config


class Project(BaseModel):
    base_path: Path
    config_relpath: Path = Field(default=Path(".vocode/config.yaml"))
    settings: Optional[Settings] = None

    @property
    def config_path(self) -> Path:
        return (self.base_path / self.config_relpath).resolve()

    @classmethod
    def from_base_path(cls, base_path: Union[str, Path]) -> "Project":
        return init_project(base_path)


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
    return Project(base_path=base, config_relpath=rel, settings=settings)
