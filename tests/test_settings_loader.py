import pytest
from pathlib import Path

from vocode.settings import load_settings
from vocode.graph.models import LLMNode


def _w(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_yaml_include_and_deep_merge(tmp_path: Path):
    base = _w(
        tmp_path,
        "base.yml",
        """
workflows:
  greet:
    nodes:
      - name: A
        type: a
        outcomes: []
    edges: []
  other:
    nodes: []
    edges: []
""",
    )

    child = _w(
        tmp_path,
        "child.yml",
        """
include: base.yml
workflows:
  greet:
    # Replace nodes list (list replacement expected)
    nodes:
      - name: B
        type: b
        outcomes: []
    edges: []
  newtool:
    nodes: []
    edges: []
""",
    )

    settings = load_settings(str(child))

    # Merged tool keys from base and child
    assert set(settings.workflows.keys()) == {"greet", "other", "newtool"}

    # 'greet' nodes list replaced by child's definition
    greet = settings.workflows["greet"]
    assert len(greet.nodes) == 1
    assert greet.nodes[0].name == "B"
    assert greet.nodes[0].type == "b"

    # Tool.name is synchronized to dict key
    assert greet.name == "greet"

    # 'other' preserved from base
    assert settings.workflows["other"].name == "other"
    assert settings.workflows["other"].nodes == []
    assert settings.workflows["other"].edges == []

    # 'newtool' added by child
    assert settings.workflows["newtool"].name == "newtool"


def test_include_dict_files_list_and_override(tmp_path: Path):
    f1 = _w(
        tmp_path,
        "file1.yml",
        """
workflows:
  t1:
    nodes: []
    edges: []
""",
    )
    f2 = _w(
        tmp_path,
        "file2.yaml",
        """
workflows:
  t2:
    nodes: []
    edges: []
""",
    )
    root = _w(
        tmp_path,
        "root.yml",
        f"""
include:
  files:
    - {f1.name}
    - {f2.name}
workflows:
  # Override/define t2 after includes
  t2:
    nodes:
      - name: C
        type: c
        outcomes: []
    edges: []
""",
    )

    settings = load_settings(str(root))

    assert set(settings.workflows.keys()) == {"t1", "t2"}
    assert settings.workflows["t1"].name == "t1"
    assert settings.workflows["t2"].name == "t2"
    assert len(settings.workflows["t2"].nodes) == 1
    assert settings.workflows["t2"].nodes[0].name == "C"
    assert settings.workflows["t2"].nodes[0].type == "c"


def test_json5_loader(tmp_path: Path):
    pytest.importorskip("json5")

    cfg = _w(
        tmp_path,
        "config.json5",
        """
{
  // JSON5 with comments and trailing commas
  workflows: {
    hello: {
      nodes: [
        {name: "A", type: "a", outcomes: []},
      ],
      edges: [],
    },
  },
}
""",
    )

    settings = load_settings(str(cfg))
    assert "hello" in settings.workflows
    assert settings.workflows["hello"].name == "hello"
    assert len(settings.workflows["hello"].nodes) == 1
    assert settings.workflows["hello"].nodes[0].name == "A"
    assert settings.workflows["hello"].nodes[0].type == "a"


def test_include_cycle_detection(tmp_path: Path):
    a = _w(
        tmp_path,
        "a.yml",
        """
include: b.yml
workflows: {}
""",
    )
    b = _w(
        tmp_path,
        "b.yml",
        """
include: a.yml
workflows: {}
""",
    )

    with pytest.raises(ValueError, match="include cycle"):
        _ = load_settings(str(a))


def test_include_array_of_strings(tmp_path: Path):
    f1 = _w(
        tmp_path,
        "file1.yml",
        """
workflows:
  t1:
    nodes: []
    edges: []
""",
    )
    f2 = _w(
        tmp_path,
        "file2.yml",
        """
workflows:
  t2:
    nodes: []
    edges: []
""",
    )
    root = _w(
        tmp_path,
        "root.yml",
        f"""
include:
  - {f1.name}
  - {f2.name}
""",
    )
    settings = load_settings(str(root))
    assert set(settings.workflows.keys()) == {"t1", "t2"}


def test_include_array_of_objects_local(tmp_path: Path):
    f1 = _w(
        tmp_path,
        "a.yml",
        """
workflows:
  a:
    nodes: []
    edges: []
""",
    )
    f2 = _w(
        tmp_path,
        "b.yml",
        """
workflows:
  b:
    nodes: []
    edges: []
""",
    )
    root = _w(
        tmp_path,
        "root.yml",
        f"""
include:
  - local: {f1.name}
  - local: {f2.name}
""",
    )
    settings = load_settings(str(root))
    assert set(settings.workflows.keys()) == {"a", "b"}


def test_include_glob_patterns(tmp_path: Path):
    inc = tmp_path / "inc"
    inc.mkdir()
    _w(
        inc,
        "one.yml",
        """
workflows:
  one:
    nodes: []
    edges: []
""",
    )
    _w(
        inc,
        "two.yml",
        """
workflows:
  two:
    nodes: []
    edges: []
""",
    )
    root = _w(
        tmp_path,
        "root.yml",
        """
include: inc/*.yml
""",
    )
    settings = load_settings(str(root))
    assert set(settings.workflows.keys()) == {"one", "two"}


def test_variables_mapping_and_interpolation(tmp_path: Path):
    cfg = _w(
        tmp_path,
        "vars.yml",
        """
variables:
  NAME: world
  MODEL: gpt-4o-mini
workflows:
  hello:
    nodes:
      - name: "Hello ${NAME}"
        type: llm
        model: ${MODEL}
        outcomes: []
    edges: []
""",
    )
    settings = load_settings(str(cfg))
    hello = settings.workflows["hello"]
    assert len(hello.nodes) == 1
    n = hello.nodes[0]
    assert isinstance(n, LLMNode)
    assert n.name == "Hello world"
    assert n.type == "llm"
    # LLMNode should parse and carry model after interpolation
    assert getattr(n, "model", None) == "gpt-4o-mini"


def test_variables_list_one_key_mappings(tmp_path: Path):
    cfg = _w(
        tmp_path,
        "vars_list.yml",
        """
variables:
  - FOO: bar
  - BAZ: qux
workflows:
  varlist:
    nodes:
      - name: "${FOO}-${BAZ}"
        type: a
        outcomes: []
    edges: []
""",
    )
    settings = load_settings(str(cfg))
    n = settings.workflows["varlist"].nodes[0]
    assert n.name == "bar-qux"


def test_variables_list_key_value_entries(tmp_path: Path):
    cfg = _w(
        tmp_path,
        "vars_kv.yml",
        """
variables:
  - key: X
    value: 42
workflows:
  kv:
    nodes:
      - name: "The answer is ${X}"
        type: a
        outcomes: []
    edges: []
""",
    )
    settings = load_settings(str(cfg))
    n = settings.workflows["kv"].nodes[0]
    assert n.name == "The answer is 42"


def test_variables_env_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("NAME", "envy")
    cfg = _w(
        tmp_path,
        "vars_env.yml",
        """
variables:
  NAME: default
workflows:
  hello:
    nodes:
      - name: "${NAME}"
        type: a
        outcomes: []
    edges: []
""",
    )
    settings = load_settings(str(cfg))
    n = settings.workflows["hello"].nodes[0]
    # Environment variables should not override file-defined variables
    assert n.name == "default"


def test_variables_unknown_left_unmodified(tmp_path: Path):
    cfg = _w(
        tmp_path,
        "vars_unknown.yml",
        """
workflows:
  t:
    nodes:
      - name: "${NOPE} and ${ALSO_NOPE}"
        type: a
        outcomes: []
    edges: []
""",
    )
    settings = load_settings(str(cfg))
    n = settings.workflows["t"].nodes[0]
    assert n.name == "${NOPE} and ${ALSO_NOPE}"


def test_include_disallow_parent_traversal(tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside.yml"
    outside.write_text(
        """
workflows:
  out:
    nodes: []
    edges: []
""",
        encoding="utf-8",
    )
    root = base / "root.yml"
    root.write_text(
        """
include: ../outside.yml
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"may not contain '\.\.'"):
        _ = load_settings(str(root))


def test_include_template_glob(tmp_path: Path):
    # Use packaged templates under vocode/config_templates
    root = _w(
        tmp_path,
        "root.yml",
        """
include:
  - template: sample/*.yaml
""",
    )
    settings = load_settings(str(root))
    # At least confirm the loader returned a Settings object with a workflows mapping
    assert isinstance(settings.workflows, dict)


def test_edge_alternative_string_syntax(tmp_path: Path):
    cfg = _w(
        tmp_path,
        "edges_str.yml",
        """
workflows:
  t:
    nodes: []
    edges:
      - requirements.done -> validate-requirements
""",
    )
    settings = load_settings(str(cfg))
    edges = settings.workflows["t"].edges
    assert len(edges) == 1
    e = edges[0]
    assert e.source_node == "requirements"
    assert e.source_outcome == "done"
    assert e.target_node == "validate-requirements"


def test_variables_structured_values_array_and_object(tmp_path: Path):
    cfg = _w(
        tmp_path,
        "vars_struct.yml",
        """
variables:
  LIST: [1, 2, 3]
  MAP:
    a: 1
    b: 2
workflows:
  t:
    config:
      arr: ${LIST}
      obj: ${MAP}
      msg: "pre-${LIST}-post"
    nodes: []
    edges: []
""",
    )
    settings = load_settings(str(cfg))
    wf = settings.workflows["t"]
    assert wf.config["arr"] == [1, 2, 3]
    assert wf.config["obj"] == {"a": 1, "b": 2}
    # Interpolation of structured values becomes JSON string
    assert wf.config["msg"] == "pre-[1, 2, 3]-post"


def test_variables_env_override_structured(monkeypatch, tmp_path: Path):
    # Override LIST variable with an env var holding JSON array
    monkeypatch.setenv("LIST", '["x","y"]')
    cfg = _w(
        tmp_path,
        "vars_env_struct.yml",
        """
variables:
  LIST: [1, 2]
workflows:
  t:
    config:
      arr: ${LIST}
    nodes: []
    edges: []
""",
    )
    settings = load_settings(str(cfg))
    wf = settings.workflows["t"]
    # Environment variables should not override file-defined variables
    assert wf.config["arr"] == [1, 2]
