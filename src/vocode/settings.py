from typing import List
from pydantic import BaseModel, Field
import yaml

from .graph.models import Node, Edge, Graph


class Tool(BaseModel):
    name: str
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)

    def build_graph(self) -> Graph:
        # Convenience: build a validated runtime Graph from the nodes/edges
        return Graph.build(nodes=self.nodes, edges=self.edges)


class Settings(BaseModel):
    tools: List[Tool] = Field(default_factory=list)


def load_settings_from_yaml(path: str) -> Settings:
    """
    Load Settings from a YAML file located at `path`.

    Expected YAML structure:
    tools:
      - name: example
        nodes:
          - name: Root
            type: some_type
            outcomes:
              - name: next
          # ...
        edges:
          - source_node: Root
            source_slot: next
            target_node: Child
          # ...
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Settings.parse_obj(data)
