import pytest

from vocode.graph import Graph, Node


# Register two simple plugin node types via default "type" values
class RootNode(Node):
    type: str = "root"


class ChildNode(Node):
    type: str = "child"


def test_build_simple_graph_ok():
    nodes = [
        {"name": "A", "type": "root", "outputs": [{"name": "toB"}]},
        {"name": "B", "type": "child", "outputs": []},
    ]
    edges = [
        {"source_node": "A", "source_slot": "toB", "target_node": "B"},
    ]
    g = Graph.build(nodes=nodes, edges=edges)

    # Root and wiring
    assert g.root.model.name == "A"
    assert len(g.root.children) == 1
    assert g.root.get_child_by_output("toB").model.name == "B"
    assert g.root.get_child_by_output("missing") is None

    # Node lookup helpers
    assert set(g.node_by_name.keys()) == {"A", "B"}
    assert g._children_names("A") == ["B"]

    # Plugin dispatch to subclasses
    print(g.nodes, repr(RootNode), "!", repr(g.nodes[0]))
    assert isinstance(g.nodes[0], RootNode)
    assert isinstance(g.nodes[1], ChildNode)


def test_duplicate_node_names_error():
    nodes = [
        {"name": "A", "type": "root", "outputs": []},
        {"name": "A", "type": "child", "outputs": []},
    ]
    with pytest.raises(ValueError, match="Duplicate node names"):
        Graph.build(nodes=nodes, edges=[])


def test_edge_to_unknown_node_error():
    nodes = [{"name": "A", "type": "root", "outputs": []}]
    edges = [{"source_node": "A", "source_slot": "x", "target_node": "B"}]
    with pytest.raises(ValueError, match="target_node 'B' does not exist"):
        Graph.build(nodes=nodes, edges=edges)


def test_edge_unknown_source_slot_error():
    nodes = [
        {"name": "A", "type": "root", "outputs": [{"name": "x"}]},
        {"name": "B", "type": "child", "outputs": []},
    ]
    edges = [{"source_node": "A", "source_slot": "y", "target_node": "B"}]
    with pytest.raises(ValueError, match="unknown source_slot 'y'"):
        Graph.build(nodes=nodes, edges=edges)


def test_duplicate_edges_from_same_slot_error():
    nodes = [
        {"name": "A", "type": "root", "outputs": [{"name": "x"}]},
        {"name": "B1", "type": "child", "outputs": []},
        {"name": "B2", "type": "child", "outputs": []},
    ]
    edges = [
        {"source_node": "A", "source_slot": "x", "target_node": "B1"},
        {"source_node": "A", "source_slot": "x", "target_node": "B2"},
    ]
    with pytest.raises(ValueError, match="Multiple edges found from the same output slot"):
        Graph.build(nodes=nodes, edges=edges)


def test_missing_edges_for_declared_output_slot_error():
    nodes = [
        {"name": "A", "type": "root", "outputs": [{"name": "x"}, {"name": "y"}]},
        {"name": "B", "type": "child", "outputs": []},
    ]
    edges = [{"source_node": "A", "source_slot": "x", "target_node": "B"}]
    with pytest.raises(ValueError, match=r"Missing edges for declared output slots: A:y"):
        Graph.build(nodes=nodes, edges=edges)
