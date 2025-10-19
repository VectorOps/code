from typing import List, Tuple, Dict, Set, Optional, Type, ClassVar, Any
from pydantic import BaseModel, Field, field_validator, model_validator, AliasChoices
from enum import Enum
import re

EDGE_ALT_SYNTAX_RE = re.compile(
    r"^\s*([A-Za-z0-9_\-]+)\.([A-Za-z0-9_\-]+)\s*->\s*([A-Za-z0-9_\-]+)(?::([A-Za-z0-9_\-]+))?\s*$"
)


class OutcomeSlot(BaseModel):
    name: str
    description: Optional[str] = None


class Confirmation(str, Enum):
    prompt = "prompt"
    auto = "auto"
    confirm = "confirm"  # require explicit Y/N approval


class ResetPolicy(str, Enum):
    always_reset = "always_reset"
    keep_results = "keep_results"
    keep_state = "keep_state"
    keep_final = "keep_final"


class MessageMode(str, Enum):
    final_response = "final_response"
    all_messages = "all_messages"
    concatenate_final = "concatenate_final"


class Mode(str, Enum):
    System = "system"
    User = "user"


class PreprocessorSpec(BaseModel):
    name: str
    options: Dict[str, Any] = Field(default_factory=dict)
    mode: Mode = Field(
        default=Mode.System,
        description="Where to apply this preprocessor: system or last user message",
    )
    prepend: bool = Field(
        default=False,
        description="If true, preprocessor output is prepended to the target text; otherwise the preprocessor transforms/appends.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, v: Any) -> Any:
        # Accept either:
        # - "name" (string) -> defaults to Mode.System
        # - {"name": "name", "options": {...}, "mode": "system"|"user"|Mode}
        if isinstance(v, str):
            return {"name": v, "options": {}, "mode": Mode.System, "prepend": False}
        if isinstance(v, dict):
            name = v.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "Preprocessor spec mapping must include non-empty 'name'"
                )
            options = v.get("options", {})
            if options is None:
                options = {}
            if not isinstance(options, dict):
                raise TypeError(
                    "Preprocessor 'options' must be a mapping/dict if provided"
                )
            raw_mode = v.get("mode", Mode.System)
            # Normalize 'mode' accepting either enum or string (case-insensitive)
            if isinstance(raw_mode, str):
                low = raw_mode.strip().lower()
                if low == "system":
                    mode = Mode.System
                elif low == "user":
                    mode = Mode.User
                else:
                    raise ValueError("Preprocessor 'mode' must be 'system' or 'user'")
            elif isinstance(raw_mode, Mode):
                mode = raw_mode
            else:
                # Allow None -> default
                mode = Mode.System
            prepend_flag = bool(v.get("prepend", False))
            return {
                "name": name,
                "options": options,
                "mode": mode,
                "prepend": prepend_flag,
            }
        raise TypeError(
            "Preprocessor spec must be a string or a mapping with 'name', optional 'options', and optional 'mode'"
        )


class Node(BaseModel):
    name: str = Field(..., description="Unique node name")
    type: str = Field(..., description="Node type identifier")
    description: Optional[str] = Field(
        None, description="Node description for UI display"
    )
    outcomes: List[OutcomeSlot] = Field(default_factory=list)
    skip: bool = Field(
        default=False,
        description="If true, runner will skip executing this node and suppress notifications; executor instance may still be created.",
    )
    max_runs: Optional[int] = Field(
        default=None,
        description="Maximum number of times this node may execute within a single runner session. None = unlimited; 0 = equivalent to skip=True.",
    )
    message_mode: MessageMode = Field(
        default=MessageMode.final_response,
        description=(
            "How to pass messages to the next node.\n"
            "- 'final_response' (default): pass only the final executor message.\n"
            "- 'all_messages': pass all messages from this step (initial inputs + interim + final).\n"
            "- 'concatenate_final': concatenate input message(s) with the final output into a single message."
        ),
    )
    confirmation: Confirmation = Field(
        default=Confirmation.prompt,
        description="How to handle node final confirmation ('prompt' or 'auto')",
    )
    hide_final_output: bool = Field(
        default=False,
        description=(
            "If true, the UI layer will suppress display of the node's final output when no input is requested "
            "(i.e., after auto-confirmation or no confirmation). This flag is ignored when input is required "
            "(prompt/confirm) — the final message is always shown in that case."
        ),
    )
    reset_policy: ResetPolicy = Field(
        default=ResetPolicy.always_reset,
        description=(
            "Defines how executor state is handled.\n"
            "- 'always_reset' (default): create a new executor each run; only current inputs are passed.\n"
            "- 'keep_results': create a new executor each run, but include all previous messages for this node from prior runs except interim executor chunks (includes user inputs and final messages).\n"
            "- 'keep_final': create a new executor each run, and include only the immediately previous final accepted response from this node when looping (no user/interim messages from earlier runs).\n"
            "- 'keep_state': reuse the same long-lived executor and resume it by sending new input messages; executor internal state is preserved."
        ),
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
            raise ValueError(
                f"Invalid type '{v}' for {cls.__name__}; expected '{expected}'"
            )
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
    source_outcome: str = Field(
        ..., description="Name of the outcome slot on the source node"
    )
    target_node: str = Field(..., description="Name of the target node")
    reset_policy: Optional[ResetPolicy] = Field(
        default=None,
        description=(
            "Optional reset policy override applied when traversing this edge. "
            "When set to 'keep_final', a re-run of the same node will include only the previous final accepted response "
            "from that node as context (not interim messages or previous user inputs)."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _parse_alt_syntax(cls, v: Any) -> Any:
        # Accept "source.node -> target[:reset_policy]" string form
        if isinstance(v, str):
            m = EDGE_ALT_SYNTAX_RE.match(v)
            if not m:
                raise ValueError(
                    "Edge string must be '<source_node>.<source_outcome> -> <target_node>[:<reset_policy>]'"
                )
            data = {
                "source_node": m.group(1),
                "source_outcome": m.group(2),
                "target_node": m.group(3),
            }
            rp = m.group(4)
            if rp:
                data["reset_policy"] = rp  # Parsed by pydantic into ResetPolicy
            return data
        return v


class Graph(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)

    @property
    def node_by_name(self) -> Dict[str, "Node"]:
        return {n.name: n for n in self.nodes}

    def _children_names(self, node_name: str) -> List[str]:
        return [e.target_node for e in self.edges if e.source_node == node_name]

    @model_validator(mode="after")
    def _validate_graph(self) -> "Graph":
        nodes: List[Node] = self.nodes or []
        edges: List[Edge] = self.edges or []

        node_by_name: Dict[str, Node] = {n.name: n for n in nodes}
        if len(node_by_name) != len(nodes):
            raise ValueError("Duplicate node names detected in graph.nodes")

        declared_outcomes: Set[Tuple[str, str]] = {
            (n.name, slot.name) for n in nodes for slot in n.outcomes
        }

        edges_by_source: Dict[Tuple[str, str], Edge] = {}
        for e in edges:
            if e.source_node not in node_by_name:
                raise ValueError(
                    f"Edge source_node '{e.source_node}' does not exist in graph.nodes"
                )
            if e.target_node not in node_by_name:
                raise ValueError(
                    f"Edge target_node\n'{e.target_node}' does not exist in graph.nodes"
                )

            source_node = node_by_name[e.source_node]
            if e.source_outcome not in {s.name for s in source_node.outcomes}:
                raise ValueError(
                    f"Edge references unknown source_outcome '{e.source_outcome}' on node '{e.source_node}'"
                )

            key = (e.source_node, e.source_outcome)
            if key in edges_by_source:
                raise ValueError(
                    f"Multiple edges found from the same outcome slot: node='{e.source_node}', slot='{e.source_outcome}'"
                )
            edges_by_source[key] = e

        sources_with_edges = set(edges_by_source.keys())
        missing = declared_outcomes - sources_with_edges
        extra = sources_with_edges - declared_outcomes

        if missing or extra:
            msgs = []
            if missing:
                msgs.append(
                    "Missing edges for declared outcome slots: "
                    + ", ".join(
                        [
                            f"{n}:{s}"
                            for (n, s) in sorted(missing, key=lambda x: (x[0], x[1]))
                        ]
                    )
                )
            if extra:
                msgs.append(
                    "Edges originate from undeclared outcome slots: "
                    + ", ".join(
                        [
                            f"{n}:{s}"
                            for (n, s) in sorted(extra, key=lambda x: (x[0], x[1]))
                        ]
                    )
                )
            raise ValueError("; ".join(msgs))

        return self


class Workflow(BaseModel):
    name: str
    graph: Graph

# New node to start a stacked workflow
class StartWorkflowNode(Node):
    # Enforce node type for registry/dispatch
    type: str = Field(default="start_workflow")
    # Name of the child workflow to start
    workflow: str = Field(..., description="Name of the workflow to start in a new runner frame")
    # Optional initial text for the child workflow (sent as a user message)
    initial_text: Optional[str] = Field(
        default=None, description="Optional initial user message text for the child workflow"
    )
