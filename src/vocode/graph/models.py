from typing import List, Tuple, Dict, Set, Optional, Type, ClassVar, Any
from pydantic import BaseModel, Field, field_validator, model_validator, AliasChoices
from enum import Enum
import re

EDGE_ALT_SYNTAX_RE = re.compile(r"^\s*([A-Za-z0-9_\-]+)\.([A-Za-z0-9_\-]+)\s*->\s*([A-Za-z0-9_\-]+)\s*$")


class OutcomeSlot(BaseModel):
    name: str
    description: Optional[str] = None

class Confirmation(str, Enum):
    prompt = "prompt"
    auto = "auto"

class ResetPolicy(str, Enum):
    always_reset = "always_reset"
    never_reset = "never_reset"

class PreprocessorSpec(BaseModel):
    name: str
    options: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, v: Any) -> Any:
        # Accept either:
        # - "name" (string)
        # - {"name": "name", "options": {...}}
        if isinstance(v, str):
            return {"name": v, "options": {}}
        if isinstance(v, dict):
            name = v.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("Preprocessor spec mapping must include non-empty 'name'")
            options = v.get("options", {})
            if options is None:
                options = {}
            if not isinstance(options, dict):
                raise TypeError("Preprocessor 'options' must be a mapping/dict if provided")
            return {"name": name, "options": options}
        raise TypeError("Preprocessor spec must be a string or a mapping with 'name' and optional 'options'")


class Node(BaseModel):
    name: str = Field(..., description="Unique node name")
    type: str = Field(..., description="Node type identifier")
    outcomes: List[OutcomeSlot] = Field(default_factory=list)
    pass_all_messages: bool = Field(
        default=False,
        description="If True, pass all messages to the next node; if False, pass only the last message.",
    )
    confirmation: Confirmation = Field(
        default=Confirmation.prompt,
        description="How to handle node final confirmation ('prompt' or 'auto')",
    )
    reset_policy: ResetPolicy = Field(
        default=ResetPolicy.always_reset,
        description="Defines how message history is handled. 'always_reset' (default): use only current input messages. 'never_reset': accumulate messages from all previous executions of this node.",
    )

    # Registry keyed by the existing "type" field value
    _registry: ClassVar[Dict[str, Type["Node"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Prefer direct class attribute (works reliably for runtime-defined plugins)
        default_type = getattr(cls, "type", None)
        if isinstance(default_type, str) and default_type:
            Node._registry[default_type] = cls  # type: ignore[assignment]
            return
        # Fallback: inspect Pydantic model_fields default
        try:
            cls.model_rebuild()
            field = cls.model_fields.get("type")
            default_type = getattr(field, "default", None)
            if isinstance(default_type, str) and default_type:
                Node._registry[default_type] = cls  # type: ignore[assignment]
        except Exception:
            pass

    @classmethod
    def register(cls, type_name: str, node_cls: Type["Node"]) -> None:
        cls._registry[type_name] = node_cls

    @classmethod
    def _ensure_registry_populated(cls) -> None:
        # Lazily scan subclasses defined at runtime and register those with a string 'type' default
        for subcls in cls.__subclasses__():
            # Prefer direct attribute for reliability
            dt = getattr(subcls, "type", None)
            if isinstance(dt, str) and dt:
                cls._registry.setdefault(dt, subcls)
                continue
            # Fallback to Pydantic model_fields default
            try:
                subcls.model_rebuild()
                field = subcls.model_fields.get("type")
                dt2 = getattr(field, "default", None)
            except Exception:
                dt2 = None
            if isinstance(dt2, str) and dt2:
                cls._registry.setdefault(dt2, subcls)

    @classmethod
    def get_registered(cls, type_name: str) -> Optional[Type["Node"]]:
        sub = cls._registry.get(type_name)
        if sub is not None:
            return sub
        # Try to discover subclasses declared after import (e.g., in tests)
        cls._ensure_registry_populated()
        return cls._registry.get(type_name)

    @field_validator("outcomes", mode="after")
    def _unique_outcome_names(cls, v: List[OutcomeSlot]) -> List[OutcomeSlot]:
        names = [s.name for s in v]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate outcome slot names found in node.outcomes")
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

    @field_validator("type", mode="after")
    def _validate_type(cls, v: str) -> str:
        field = getattr(cls, "model_fields", {}).get("type")
        expected = getattr(field, "default", None)
        if isinstance(expected, str) and v != expected:
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


class OutcomeStrategy(str, Enum):
    tag = "tag"
    function_call = "function_call"


class Edge(BaseModel):
    source_node: str = Field(..., description="Name of the source node")
    source_outcome: str = Field(..., description="Name of the outcome slot on the source node")
    target_node: str = Field(..., description="Name of the target node")

    @model_validator(mode="before")
    @classmethod
    def _parse_alt_syntax(cls, v: Any) -> Any:
        # Accept "source.node -> target" string form
        if isinstance(v, str):
            m = EDGE_ALT_SYNTAX_RE.match(v)
            if not m:
                raise ValueError("Edge string must be '<source_node>.<source_outcome> -> <target_node>'")
            return {
                "source_node": m.group(1),
                "source_outcome": m.group(2),
                "target_node": m.group(3),
            }
        return v


class LLMNode(Node):
    type: str = "llm"
    model: str
    system: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    outcome_strategy: OutcomeStrategy = Field(default=OutcomeStrategy.tag)
    tools: List[str] = Field(
        default_factory=list, description="List of enabled tool names for this node."
    )
    extra: Dict[str, Any] = Field(default_factory=dict)
    preprocessors: List[PreprocessorSpec] = Field(
        default_factory=list,
        description="Pre-execution preprocessors applied to the LLM system prompt",
    )

class InputNode(Node):
    type: str = "input"
    message: str

class NoopNode(Node):
    type: str = "noop"
    # Auto-skip without prompting for approval
    confirmation: Confirmation = Field(default=Confirmation.auto, description="No-op auto confirmation")
    # Pass all prior messages through to the next node by default
    pass_all_messages: bool = Field(
        default=True,
        description="No-op defaults to passing all messages to the next node",
    )

class ApplyPatchNode(Node):
    type: str = "apply_patch"
    patch_format: str = Field(default="v4a", description="Patch format identifier (currently only 'v4a' is supported)")
