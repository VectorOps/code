from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .project import Project


# Global registry of tool name -> tool instance
_registry: Dict[str, "BaseTool"] = {}


def register_tool(name: str, tool: "BaseTool") -> None:
    """Registers a tool instance."""
    if name in _registry:
        raise ValueError(f"Tool with name '{name}' already registered.")
    _registry[name] = tool


def unregister_tool(name: str) -> bool:
    """Unregister a tool instance by name. Returns True if removed, False if not present."""
    return _registry.pop(name, None) is not None


def get_tool(name: str) -> Optional["BaseTool"]:
    """Gets a tool instance by name."""
    return _registry.get(name)


def get_all_tools() -> Dict[str, "BaseTool"]:
    """Returns a copy of the tool registry."""
    return dict(_registry)


class BaseTool(ABC):
    # Subclasses must set this to a unique string
    name: str
    # Tools accept Any arguments directly

    def __init__(self) -> None:
        pass

    @abstractmethod
    async def run(self, project: "Project", args: Any) -> Optional[str]:
        """
        Execute this tool within the context of the given Project.
        Args:
            project: The active project context.
            args: Parsed arguments structure (e.g., dict or Pydantic model). Not a JSON string.
        """
        pass

    @abstractmethod
    def openapi_spec(self) -> Dict[str, Any]:
        """
        Return this tool's definition in OpenAI 'function' tool format,
        using JSON Schema for parameters.
        """
        pass
