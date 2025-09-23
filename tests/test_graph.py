import pytest

from vocode.graph import build
from vocode.models import Node


# Register two simple plugin node types via default "type" values
class RootNode(Node):
    type: str = "root"


class ChildNode(Node):
    type: str = "child"


def test_build_simple_graph_ok():
    nodes = [
        {"name": "A", "type": "root", "outcomes": [{"name": "toB"}]},
        {"name": "B", "type": "child", "outcomes": []},
    ]
    edges = [
        {"source_node": "A", "source_outcome": "toB", "target_node": "B"},
    ]
    rg = build(nodes=nodes, edges=edges)

    # Root and wiring
    assert rg.root.model.name == "A"
    assert len(rg.root.children) == 1
    assert rg.root.get_child_by_outcome("toB").model.name == "B"
    assert rg.root.get_child_by_outcome("missing") is None

    # Node lookup helpers
    assert set(rg.graph.node_by_name.keys()) == {"A", "B"}
    assert rg.graph._children_names("A") == ["B"]

    # Plugin dispatch to subclasses
    print(rg.graph.nodes, repr(RootNode), "!", repr(rg.graph.nodes[0]))
    assert isinstance(rg.graph.nodes[0], RootNode)
    assert isinstance(rg.graph.nodes[1], ChildNode)


def test_duplicate_node_names_error():
    nodes = [
        {"name": "A", "type": "root", "outcomes": []},
        {"name": "A", "type": "child", "outcomes": []},
    ]
    with pytest.raises(ValueError, match="Duplicate node names"):
        build(nodes=nodes, edges=[])


def test_edge_to_unknown_node_error():
    nodes = [{"name": "A", "type": "root", "outcomes": []}]
    edges = [{"source_node": "A", "source_outcome": "x", "target_node": "B"}]
    with pytest.raises(ValueError, match="target_node\n'B' does not exist"):
        build(nodes=nodes, edges=edges)


def test_edge_unknown_source_outcome_error():
    nodes = [
        {"name": "A", "type": "root", "outcomes": [{"name": "x"}]},
        {"name": "B", "type": "child", "outcomes": []},
    ]
    edges = [{"source_node": "A", "source_outcome": "y", "target_node": "B"}]
    with pytest.raises(ValueError, match="unknown source_outcome 'y'"):
        build(nodes=nodes, edges=edges)


def test_duplicate_edges_from_same_slot_error():
    nodes = [
        {"name": "A", "type": "root", "outcomes": [{"name": "x"}]},
        {"name": "B1", "type": "child", "outcomes": []},
        {"name": "B2", "type": "child", "outcomes": []},
    ]
    edges = [
        {"source_node": "A", "source_outcome": "x", "target_node": "B1"},
        {"source_node": "A", "source_outcome": "x", "target_node": "B2"},
    ]
    with pytest.raises(ValueError, match="Multiple edges found from the same outcome slot"):
        build(nodes=nodes, edges=edges)


def test_missing_edges_for_declared_outcome_slot_error():
    nodes = [
        {"name": "A", "type": "root", "outcomes": [{"name": "x"}, {"name": "y"}]},
        {"name": "B", "type": "child", "outcomes": []},
    ]
    edges = [{"source_node": "A", "source_outcome": "x", "target_node": "B"}]
    with pytest.raises(ValueError, match=r"Missing edges for declared outcome slots: A:y"):
        build(nodes=nodes, edges=edges)
