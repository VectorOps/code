from typing import List, Tuple, Dict, Set, Optional, Type, ClassVar, Any
from pydantic import BaseModel, Field, root_validator, validator, PrivateAttr


class OutputSlot(BaseModel):
    name: str


class Node(BaseModel):
    name: str = Field(..., description="Unique node name")
    type: str = Field(..., description="Node type identifier")
    outputs: List[OutputSlot] = Field(default_factory=list)

    # Registry keyed by the existing "type" field value
    _registry: ClassVar[Dict[str, Type["Node"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Auto-register subclass if it declares a default for the instance "type" field
        try:
            f = getattr(cls, "__fields__", {}).get("type")  # type: ignore[call-arg]
        except Exception:
            f = None
        default_type = getattr(f, "default", None)
        if isinstance(default_type, str) and default_type:
            Node._registry[default_type] = cls  # type: ignore[assignment]

    @classmethod
    def register(cls, type_name: str, node_cls: Type["Node"]) -> None:
        cls._registry[type_name] = node_cls

    @classmethod
    def get_registered(cls, type_name: str) -> Optional[Type["Node"]]:
        return cls._registry.get(type_name)

    @validator("outputs")
    def _unique_output_names(cls, v: List[OutputSlot]) -> List[OutputSlot]:
        names = [s.name for s in v]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate output slot names found in node.outputs")
        return v

    @classmethod
    def __get_validators__(cls):
        # Enables polymorphic parsing when Node is used as a field type
        yield cls._dispatch_and_validate

    @classmethod
    def _dispatch_and_validate(cls, v):
        # Already a Node (or subclass) instance
        if isinstance(v, Node):
            return v
        if not isinstance(v, dict):
            raise TypeError("Node must be parsed from a mapping/dict")
        node_type = v.get("type")
        if isinstance(node_type, str):
            subcls = cls.get_registered(node_type)
            if subcls and subcls is not cls:
                # Instantiate the registered subclass for this type
                return subcls(**v)
        # Fallback to the base Node model
        return cls(**v)

    @validator("type")
    def _validate_type(cls, v: str) -> str:
        # If this class declares a default for the 'type' field, enforce it
        field = getattr(cls, "__fields__", {}).get("type")
        expected = getattr(field, "default", None)
        if expected is not None and v != expected:
            raise ValueError(f"Invalid type '{v}' for {cls.__name__}; expected '{expected}'")
        return v

    @classmethod
    def from_obj(cls, obj: Any) -> "Node":
        # Convenience factory for manual dispatch from raw dict/obj
        return cls._dispatch_and_validate(obj)

    @classmethod
    def parse_obj(cls, obj: Any) -> "Node":  # type: ignore[override]
        # Allow direct parse with dispatch (e.g., Node.parse_obj(data))
        return cls._dispatch_and_validate(obj)


class Edge(BaseModel):
    source_node: str = Field(..., description="Name of the source node")
    source_slot: str = Field(..., description="Name of the output slot on the source node")
    target_node: str = Field(..., description="Name of the target node")
