
from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING, Optional, Union, Type, List

from pydantic import BaseModel

if TYPE_CHECKING:
    from .project import Project


# Global registry of tool name -> tool instance
_registry: Dict[str, "BaseTool"] = {}


def register_tool(name: str, tool: "BaseTool") -> None:
    """Registers a tool instance."""
    if name in _registry:
        raise ValueError(f"Tool with name '{name}' already registered.")
    _registry[name] = tool


def get_tool(name: str) -> Optional["BaseTool"]:
    """Gets a tool instance by name."""
    return _registry.get(name)


def get_all_tools() -> Dict[str, "BaseTool"]:
    """Returns a copy of the tool registry."""
    return dict(_registry)


class BaseTool(ABC):
    # Subclasses must set this to a unique string
    name: str
    input_model: Type[BaseModel]

    def __init__(self) -> None:
        pass

    @abstractmethod
    async def run(self, project: "Project", args: BaseModel) -> Optional[str]:
        """
        Execute this tool within the context of the given Project.
        Subclasses must implement this.
        """
        pass

    @abstractmethod
    def openapi_spec(self) -> Dict[str, Any]:
        """
        Return this tool's definition in OpenAI 'function' tool format,
        using JSON Schema for parameters.
        """
        pass
