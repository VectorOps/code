import pytest
from pathlib import Path

from vocode.settings import load_settings
from vocode.runner.executors.llm import LLMNode


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
$include: base.yml
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
name: C1
type: c
outcomes: []
""",
    )
    f2 = _w(
        tmp_path,
        "file2.yaml",
        """
name: C2
type: c
outcomes: []
""",
    )
    root = _w(
        tmp_path,
        "root.yml",
        f"""
workflows:
  t2:
    nodes:
      $include:
        files:
          - {f1.name}
          - {f2.name}
    edges: []
""",
    )

    settings = load_settings(str(root))

    assert set(settings.workflows.keys()) == {"t2"}
    t2_nodes = settings.workflows["t2"].nodes
    assert len(t2_nodes) == 2
    assert {n.name for n in t2_nodes} == {"C1", "C2"}


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
$include: b.yml
workflows: {}
""",
    )
    b = _w(
        tmp_path,
        "b.yml",
        """
$include: a.yml
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
name: N1
type: a
outcomes: []
""",
    )
    f2 = _w(
        tmp_path,
        "file2.yml",
        """
name: N2
type: a
outcomes: []
""",
    )
    root = _w(
        tmp_path,
        "root.yml",
        f"""
workflows:
  t:
    nodes:
      $include:
        - {f1.name}
        - {f2.name}
    edges: []
""",
    )
    settings = load_settings(str(root))
    nodes = settings.workflows["t"].nodes
    assert len(nodes) == 2
    assert {n.name for n in nodes} == {"N1", "N2"}


def test_include_array_of_objects_local(tmp_path: Path):
    f1 = _w(
        tmp_path,
        "a.yml",
        """
name: A
type: a
outcomes: []
""",
    )
    f2 = _w(
        tmp_path,
        "b.yml",
        """
name: B
type: b
outcomes: []
""",
    )
    root = _w(
        tmp_path,
        "root.yml",
        f"""
workflows:
  t:
    nodes:
      $include:
        - local: {f1.name}
        - local: {f2.name}
    edges: []
""",
    )
    settings = load_settings(str(root))
    nodes = settings.workflows["t"].nodes
    assert len(nodes) == 2
    assert {n.name for n in nodes} == {"A", "B"}


def test_include_glob_patterns(tmp_path: Path):
    inc = tmp_path / "inc"
    inc.mkdir()
    _w(
        inc,
        "one.yml",
        """
name: One
type: a
outcomes: []
""",
    )
    _w(
        inc,
        "two.yml",
        """
name: Two
type: b
outcomes: []
""",
    )
    root = _w(
        tmp_path,
        "root.yml",
        """
workflows:
  t:
    nodes:
      $include: inc/*.yml
    edges: []
""",
    )
    settings = load_settings(str(root))
    nodes = settings.workflows["t"].nodes
    assert {n.name for n in nodes} == {"One", "Two"}


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
$include: ../outside.yml
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"may not contain '\.\.'"):
        _ = load_settings(str(root))




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


def test_nested_include_under_nodes_list(tmp_path: Path):
    inc = tmp_path / "inc"
    inc.mkdir()
    (inc / "one.yml").write_text(
        """
name: A
type: a
outcomes: []
""",
        encoding="utf-8",
    )
    (inc / "two.yml").write_text(
        """
name: B
type: b
outcomes: []
""",
        encoding="utf-8",
    )
    root = _w(
        tmp_path,
        "root.yml",
        """
workflows:
  t:
    nodes:
      $include: inc/*.yml
    edges: []
""",
    )
    settings = load_settings(str(root))
    nodes = settings.workflows["t"].nodes
    assert [n.name for n in nodes] == ["A", "B"]


def test_include_single_dict_override(tmp_path: Path):
    inc = _w(
        tmp_path,
        "cfg.yml",
        """
foo: bar
alpha:
  beta: 1
  gamma: 2
""",
    )
    root = _w(
        tmp_path,
        "root.yml",
        f"""
workflows:
  t:
    config:
      $include: {inc.name}
      foo: baz
      alpha:
        beta: 10
    nodes: []
    edges: []
""",
    )
    settings = load_settings(str(root))
    cfg = settings.workflows["t"].config
    # Top-level key override
    assert cfg["foo"] == "baz"
    # Deep override and preservation of other keys
    assert cfg["alpha"]["beta"] == 10
    assert cfg["alpha"]["gamma"] == 2


def test_include_imports_variables_and_parent_overrides(tmp_path: Path):
    base = _w(
        tmp_path,
        "base.yml",
        """
variables:
  MODEL: gpt-4o-mini
workflows:
  w:
    nodes:
      - name: "Use ${MODEL}"
        type: llm
        model: ${MODEL}
        outcomes: []
    edges: []
""",
    )
    root = _w(
        tmp_path,
        "root.yml",
        f"""
$include: {base.name}
variables:
  MODEL: gpt-4o-pro
""",
    )
    settings = load_settings(str(root))
    n = settings.workflows["w"].nodes[0]
    assert n.name == "Use gpt-4o-pro"
    assert isinstance(n, LLMNode)
    assert getattr(n, "model", None) == "gpt-4o-pro"


def test_include_inline_vars_override(tmp_path: Path):
    base = _w(
        tmp_path,
        "base.yml",
        """
variables:
  MODEL: gpt-4o-mini
workflows:
  w:
    nodes:
      - name: "Model ${MODEL}"
        type: llm
        model: ${MODEL}
        outcomes: []
    edges: []
""",
    )
    root = _w(
        tmp_path,
        "root.yml",
        f"""
$include:
  local: {base.name}
  vars:
    MODEL: gpt-4o-pro
""",
    )
    settings = load_settings(str(root))
    n = settings.workflows["w"].nodes[0]
    assert n.name == "Model gpt-4o-pro"
    assert isinstance(n, LLMNode)
    assert getattr(n, "model", None) == "gpt-4o-pro"


def test_include_import_vars_false_disables_defaults(tmp_path: Path):
    base = _w(
        tmp_path,
        "base.yml",
        """
variables:
  MODEL: gpt-4o-mini
workflows:
  w:
    nodes:
      - name: "Model ${MODEL}"
        type: llm
        model: ${MODEL}
        outcomes: []
    edges: []
""",
    )
    root = _w(
        tmp_path,
        "root.yml",
        f"""
$include:
  local: {base.name}
  import_vars: false
""",
    )
    settings = load_settings(str(root))
    n = settings.workflows["w"].nodes[0]
    # Defaults from included file are not imported, so placeholders remain
    assert n.name == "Model ${MODEL}"
    assert isinstance(n, LLMNode)
    assert getattr(n, "model", None) == "${MODEL}"


def test_include_var_prefix_and_inline_vars(tmp_path: Path):
    inc_vars = _w(
        tmp_path,
        "vars.yml",
        """
variables:
  MODEL: mini
""",
    )
    root = _w(
        tmp_path,
        "root.yml",
        f"""
$include:
  local: {inc_vars.name}
  var_prefix: base_
  vars:
    MODEL: pro
variables:
  base_MODEL: ultra
workflows:
  t:
    nodes:
      - name: "N ${{base_MODEL}}"
        type: a
        outcomes: []
    edges: []
""",
    )
    settings = load_settings(str(root))
    n = settings.workflows["t"].nodes[0]
    # Precedence: included defaults (base_MODEL=mini) -> inline include vars (base_MODEL=pro) -> root vars (base_MODEL=ultra)
    assert n.name == "N ultra"


def test_include_order_precedence_last_wins(tmp_path: Path):
    a = _w(
        tmp_path,
        "a.yml",
        """
variables:
  FOO: A
""",
    )
    b = _w(
        tmp_path,
        "b.yml",
        """
variables:
  FOO: B
""",
    )
    root = _w(
        tmp_path,
        "root.yml",
        f"""
$include:
  - local: {a.name}
  - local: {b.name}
workflows:
  t:
    nodes:
      - name: "${{FOO}}"
        type: a
        outcomes: []
    edges: []
""",
    )
    settings = load_settings(str(root))
    n = settings.workflows["t"].nodes[0]
    # Later include overrides earlier include
    assert n.name == "B"


def test_nested_includes_export_variables(tmp_path: Path):
    varpack = _w(
        tmp_path,
        "varpack.yml",
        """
variables:
  X: 123
""",
    )
    primary = _w(
        tmp_path,
        "primary.yml",
        f"""
$include: {varpack.name}
""",
    )
    root = _w(
        tmp_path,
        "root.yml",
        f"""
$include: {primary.name}
workflows:
  t:
    config:
      val: ${{X}}
    nodes: []
    edges: []
""",
    )
    settings = load_settings(str(root))
    wf = settings.workflows["t"]
    assert wf.config["val"] == 123
