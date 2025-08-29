from __future__ import annotations

from typing import Any, ClassVar, Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .project import Project


class BaseTool:
    # Subclasses must set this to a unique string
    name: ClassVar[str]

    # Optional metadata used for OpenAI tool definition
    description: ClassVar[str] = ""
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    # Global registry of tool name -> subclass
    _registry: ClassVar[Dict[str, Type["BaseTool"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Do not register the abstract base itself
        if cls is BaseTool:
            return

        nm = getattr(cls, "name", None)
        if not isinstance(nm, str) or not nm:
            raise TypeError(f"{cls.__name__} must define a class attribute 'name' (non-empty str)")

        # Enforce unique names
        existing = BaseTool._registry.get(nm)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Duplicate tool name '{nm}' already registered by {existing.__name__}"
            )

        BaseTool._registry[nm] = cls

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> "BaseTool":
        """
        Factory: return a new tool instance by registered name.
        Extra args/kwargs are forwarded to the tool subclass constructor.
        """
        tool_cls = cls._registry.get(name)
        if tool_cls is None:
            available = ", ".join(sorted(cls._registry.keys()))
            raise KeyError(f"Unknown tool '{name}'. Available: {available}")
        return tool_cls(*args, **kwargs)

    def run(self, project: "Project", *args: Any, **kwargs: Any) -> Any:
        """
        Execute this tool within the context of the given Project.
        Subclasses must implement this.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.run() not implemented")

    def openapi_spec(self) -> Dict[str, Any]:
        """
        Return this tool's definition in OpenAI 'function' tool format,
        using JSON Schema for parameters.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": getattr(self, "description", "") or "",
                "parameters": getattr(self, "parameters", {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }),
            },
        }

    @classmethod
    def get_registered(cls) -> Dict[str, Type["BaseTool"]]:
        """Return a copy of the registry mapping."""
        return dict(cls._registry)
