from typing import List, Tuple, Dict, Set, Optional
from pydantic import BaseModel, Field, model_validator, PrivateAttr
from .models import Node, Edge, OutcomeSlot


class RuntimeNode:
    def __init__(self, model: "Node"):
        self._model = model
        self._children: List["RuntimeNode"] = []
        self._child_by_outcome: Dict[str, "RuntimeNode"] = {}

    def get_child_by_outcome(self, outcome_name: str) -> Optional["RuntimeNode"]:
        return self._child_by_outcome.get(outcome_name)

    @property
    def name(self) -> str:
        return self._model.name

    @property
    def model(self) -> "Node":
        return self._model

    @property
    def type(self) -> str:
        return self._model.type

    @property
    def outcomes(self) -> List["OutcomeSlot"]:
        return self._model.outcomes

    @property
    def children(self) -> List["RuntimeNode"]:
        return self._children


class Graph(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    _runtime_nodes: Dict[str, RuntimeNode] = PrivateAttr(default_factory=dict)
    _root: Optional[RuntimeNode] = PrivateAttr(default=None)

    @classmethod
    def build(cls, nodes: List["Node"], edges: List["Edge"]) -> "Graph":
        """
        Construct and validate a Graph from nodes and edges.
        Assumes the first node in `nodes` is the root.
        """
        nodes = [Node.from_obj(n) for n in nodes]
        graph = cls(nodes=nodes, edges=edges)  # triggers Pydantic validation
        if not graph.nodes:
            raise ValueError("Graph.build requires at least one node to identify the root")
        graph._runtime_nodes = {n.name: RuntimeNode(n) for n in graph.nodes}
        # Wire direct children references
        for e in graph.edges:
            parent = graph._runtime_nodes[e.source_node]
            child = graph._runtime_nodes[e.target_node]
            parent._children.append(child)
            parent._child_by_outcome[e.source_outcome] = child
        graph._root = graph._runtime_nodes[graph.nodes[0].name]
        return graph

    @property
    def node_by_name(self) -> Dict[str, "Node"]:
        # Computed mapping for quick lookups
        return {n.name: n for n in self.nodes}

    def _children_names(self, node_name: str) -> List[str]:
        # Computed list of child node names (targets) for a given source node
        return [e.target_node for e in self.edges if e.source_node == node_name]

    @property
    def root(self) -> "RuntimeNode":
        if self._root is None:
            raise ValueError("Graph root is not initialized. Use Graph.build to construct the graph.")
        return self._root

    def get_runtime_node_by_name(self, name: str) -> Optional[RuntimeNode]:
        if self._root is None:
            raise ValueError("Graph runtime nodes not initialized. Use Graph.build to construct the graph.")
        return self._runtime_nodes.get(name)

    @model_validator(mode="after")
    def _validate_graph(self) -> "Graph":
        nodes: List[Node] = self.nodes or []
        edges: List[Edge] = self.edges or []

        # Ensure node names are unique
        node_by_name: Dict[str, Node] = {n.name: n for n in nodes}
        if len(node_by_name) != len(nodes):
            raise ValueError("Duplicate node names detected in graph.nodes")

        # Collect all declared outcome slots across nodes
        declared_outcomes: Set[Tuple[str, str]] = {
            (n.name, slot.name) for n in nodes for slot in n.outcomes
        }

        # Validate edges refer to existing nodes and declared source slots,
        # and that there is at most one edge per (source_node, source_outcome)
        edges_by_source: Dict[Tuple[str, str], Edge] = {}
        for e in edges:
            if e.source_node not in node_by_name:
                raise ValueError(f"Edge source_node '{e.source_node}' does not exist in graph.nodes")
            if e.target_node not in node_by_name:
                raise ValueError(f"Edge target_node '{e.target_node}' does not exist in graph.nodes")

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

        # Enforce: exactly one edge from each declared outcome slot
        sources_with_edges = set(edges_by_source.keys())
        missing = declared_outcomes - sources_with_edges
        extra = sources_with_edges - declared_outcomes

        if missing or extra:
            msgs = []
            if missing:
                msgs.append(
                    "Missing edges for declared outcome slots: "
                    + ", ".join([f"{n}:{s}" for (n, s) in sorted(missing, key=lambda x: (x[0], x[1]))])
                )
            if extra:
                msgs.append(
                    "Edges originate from undeclared outcome slots: "
                    + ", ".join([f"{n}:{s}" for (n, s) in sorted(extra, key=lambda x: (x[0], x[1]))])
                )
            raise ValueError("; ".join(msgs))

        return self

# TODO: Redo.
class Workflow(BaseModel):
    name: str
    graph: Graph
