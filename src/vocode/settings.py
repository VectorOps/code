from typing import List, Dict, Optional, Any, Union, Set, Final, Type, Literal
import re
from pathlib import Path
from os import PathLike
import os
import json
from pydantic import BaseModel, Field, model_validator
import yaml
import json5  # type: ignore
from .models import Node, Edge
from .state import LogLevel

from know.settings import ProjectSettings as KnowProjectSettings


# Base path for packaged template configs, e.g. include: { vocode: "nodes/requirements.yaml" }
VOCODE_TEMPLATE_BASE: Path = (Path(__file__).resolve().parent / "config_templates").resolve()
# Include spec keys for bundled templates. Support GitLab 'template', legacy 'vocode', and 'templates'
TEMPLATE_INCLUDE_KEYS: Final[Set[str]] = {"template", "templates", "vocode"}
# Variable replacement pattern: only support ${ABC}
VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

INCLUDE_KEY: Final[str] = "$include"


class Workflow(BaseModel):
    name: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _dispatch_nodes(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        nodes = data.get("nodes")
        if isinstance(nodes, list):
            # Convert dicts to Node instances using the registry-based dispatcher
            data = dict(data)
            data["nodes"] = [Node.from_obj(n) if isinstance(n, dict) else n for n in nodes]
        return data


class ToolSettings(BaseModel):
    name: str
    enabled: bool = True
    auto_approve: Optional[bool] = None

class UISettings(BaseModel):
    # When true, PromptSession accepts multi-line input (Enter can insert newlines)
    multiline: bool = True
    # Optional editing mode override. None => use prompt_toolkit default (Emacs)
    edit_mode: Optional[Literal["emacs", "vim"]] = None
    # Minimum log level to display in the terminal.
    log_level: LogLevel = LogLevel.info

class Settings(BaseModel):
    workflows: Dict[str, Workflow] = Field(default_factory=dict)
    tools: List[ToolSettings] = Field(default_factory=list)
    know: Optional[KnowProjectSettings] = Field(default=None)
    ui: Optional[UISettings] = Field(default=None)

    @model_validator(mode="after")
    def _sync_workflow_names(self) -> "Settings":
        for key, wf in self.workflows.items():
            wf.name = key
        return self


# Configuration loading
def _deep_merge_dicts(a: Dict[str, Any], b: Dict[str, Any], *, concat_lists: bool = False) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge_dicts(out[k], v, concat_lists=concat_lists)
        elif k in out and isinstance(out[k], list) and isinstance(v, list):
            if concat_lists:
                # Concatenate lists when merging includes so multiple include files can add to arrays
                out[k] = [*out[k], *v]
            else:
                # Default behavior: replacement by the including file
                out[k] = v
        else:
            out[k] = v
    return out


def _collect_variables(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collect variables from the merged config. Supports:
      - mapping: variables: { KEY: default }
      - list of one-key mappings: variables: [ {KEY: default}, ... ]
      - list of entries with explicit keys: variables: [ {key: KEY, value: default}, ... ]
    """
    out: Dict[str, Any] = {}
    vars_spec = doc.get("variables")
    if vars_spec is None:
        return out
    if isinstance(vars_spec, dict):
        for k, v in vars_spec.items():
            if isinstance(k, str):
                out[k] = v  # keep original type (can be list/dict/etc)
    elif isinstance(vars_spec, list):
        for item in vars_spec:
            if isinstance(item, dict):
                if "key" in item and "value" in item and isinstance(item["key"], str):
                    out[item["key"]] = item["value"]
                else:
                    for k, v in item.items():
                        if isinstance(k, str):
                            out[k] = v
    return out

def _apply_var_prefix_to_map(vars_map: Dict[str, Any], prefix: Optional[str]) -> Dict[str, Any]:
    if not prefix:
        return dict(vars_map)
    return {f"{prefix}{k}": v for k, v in vars_map.items()}

def _interpolate_string(s: str, vars_map: Dict[str, Any]) -> str:
    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name not in vars_map:
            return m.group(0)
        val = vars_map[name]
        if val is None:
            return ""
        if isinstance(val, (dict, list)):
            return json.dumps(val, ensure_ascii=False)
        return str(val)
    return VAR_PATTERN.sub(repl, s)

def _apply_variables(obj: Any, vars_map: Dict[str, Any]) -> Any:
    if isinstance(obj, str):
        m = VAR_PATTERN.fullmatch(obj)
        if m:
            name = m.group(1)
            if name in vars_map:
                return vars_map[name]
            return obj
        return _interpolate_string(obj, vars_map)
    if isinstance(obj, dict):
        return {k: _apply_variables(v, vars_map) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_apply_variables(v, vars_map) for v in obj]
    return obj

def _expand_include_patterns(base: Path, pattern: str) -> List[Path]:
    """
    Expand a relative glob pattern under 'base' and return matching files.
    Security:
      - disallow absolute patterns
      - disallow parent traversal ('..')
      - ensure every matched file resolves within the base directory
    """
    if not isinstance(pattern, str):
        raise TypeError("include path must be a string")
    if os.path.isabs(pattern):
        raise ValueError(f"Include pattern must be relative: '{pattern}'")
    # Normalize separators for globbing and validate no parent traversal
    norm = pattern.replace("\\", "/")
    # Reject any explicit parent traversal
    parts = [p for p in norm.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ValueError(f"Include pattern may not contain '..': '{pattern}'")
    # Expand pattern relative to base
    matches: List[Path] = []
    for cand in base.glob(norm):
        if not cand.is_file():
            continue
        try:
            cand.resolve().relative_to(base.resolve())
        except Exception:
            # Skip anything that is not within base (defense-in-depth)
            continue
        matches.append(cand.resolve())
    if not matches:
        raise ValueError(f"Include pattern '{pattern}' under base '{base}' did not match any files")
    return matches


def _collect_include_paths(spec: Any, base_dir: Path) -> List[tuple[Path, Dict[str, Any]]]:
    if spec is None:
        return []

    def parse_opts(item: Dict[str, Any]) -> Dict[str, Any]:
        opts: Dict[str, Any] = {
            "import_vars": item.get("import_vars", True),
            "vars": item.get("vars", {}) or {},
            "var_prefix": item.get("var_prefix"),
        }
        if not isinstance(opts["import_vars"], bool):
            raise TypeError("import_vars must be a boolean")
        if not isinstance(opts["vars"], dict):
            raise TypeError("vars must be a mapping/object")
        if opts["var_prefix"] is not None and not isinstance(opts["var_prefix"], str):
            raise TypeError("var_prefix must be a string")
        return opts

    def make_entries(paths: List[Path], base_opts: Dict[str, Any]) -> List[tuple[Path, Dict[str, Any]]]:
        return [(p, dict(base_opts)) for p in paths]

    def norm_one(item: Any) -> List[tuple[Path, Dict[str, Any]]]:
        if isinstance(item, str):
            return make_entries(_expand_include_patterns(base_dir, item), {"import_vars": True, "vars": {}, "var_prefix": None})
        if isinstance(item, dict):
            opts = parse_opts(item)

            paths: List[Path] = []
            if "local" in item:
                loc = item["local"]
                if isinstance(loc, list):
                    for p in loc:
                        paths.extend(_expand_include_patterns(base_dir, p))
                else:
                    paths.extend(_expand_include_patterns(base_dir, loc))
            elif any(k in item for k in TEMPLATE_INCLUDE_KEYS):
                key = next(k for k in TEMPLATE_INCLUDE_KEYS if k in item)
                loc = item[key]
                if isinstance(loc, list):
                    for p in loc:
                        paths.extend(_expand_include_patterns(VOCODE_TEMPLATE_BASE, p))
                else:
                    paths.extend(_expand_include_patterns(VOCODE_TEMPLATE_BASE, loc))
            elif "file" in item:
                loc = item["file"]
                if isinstance(loc, list):
                    for p in loc:
                        paths.extend(_expand_include_patterns(base_dir, p))
                else:
                    paths.extend(_expand_include_patterns(base_dir, loc))
            elif "files" in item:
                for p in item["files"]:
                    paths.extend(_expand_include_patterns(base_dir, p))
            else:
                raise ValueError(f"Unsupported include dict keys for location: {list(item.keys())}")

            return make_entries(paths, opts)
        if isinstance(item, list):
            acc: List[tuple[Path, Dict[str, Any]]] = []
            for sub in item:
                acc.extend(norm_one(sub))
            return acc
        raise TypeError(f"Unsupported include item type: {type(item).__name__}")

    return norm_one(spec)

def _combine_included_values(values: List[Any]) -> Any:
    if not values:
        return None
    if all(isinstance(v, dict) for v in values):
        acc: Dict[str, Any] = {}
        for v in values:
            acc = _deep_merge_dicts(acc, v, concat_lists=False)
        return acc
    if all(isinstance(v, list) for v in values):
        acc_list: List[Any] = []
        for v in values:
            acc_list.extend(v)
        return acc_list
    raise ValueError("All included files must produce the same type (all dicts or all lists)")

def _preprocess_includes(node: Any, base_dir: Path, seen: Set[Path]) -> tuple[Any, Dict[str, Any]]:
    if isinstance(node, dict):
        # If this dict contains a $include, expand it and merge/replace as appropriate
        if INCLUDE_KEY in node:
            include_spec = node[INCLUDE_KEY]
            # Resolve include entries relative to this file's base_dir (or templates base)
            entries = _collect_include_paths(include_spec, base_dir)
            included_payloads: List[Any] = []
            acc_vars: Dict[str, Any] = {}

            for inc_path, opts in entries:
                inc_path = inc_path.resolve()
                if inc_path in seen:
                    raise ValueError(f"Detected include cycle at {inc_path}")
                seen.add(inc_path)
                try:
                    inc_data = _load_raw_file(inc_path)
                    # Extract top-level variables from the included file
                    inc_top_vars: Dict[str, Any] = {}
                    if isinstance(inc_data, dict):
                        inc_top_vars = _collect_variables(inc_data)
                        inc_data = dict(inc_data)
                        inc_data.pop("variables", None)

                    inc_proc, inc_vars_nested = _preprocess_includes(inc_data, inc_path.parent, seen)

                    # Merge nested include variables then the included file's own top-level variables
                    inc_all_vars = dict(inc_vars_nested)
                    inc_all_vars.update(inc_top_vars)

                    # Apply optional prefix
                    var_prefix = opts.get("var_prefix")
                    if var_prefix:
                        inc_all_vars = _apply_var_prefix_to_map(inc_all_vars, var_prefix)

                    # Import included variables unless disabled
                    if opts.get("import_vars", True):
                        # Include order: later entries override earlier ones
                        acc_vars.update(inc_all_vars)

                    # Inline include-specific overrides applied after imported defaults
                    inline_vars = opts.get("vars") or {}
                    if var_prefix and inline_vars:
                        inline_vars = _apply_var_prefix_to_map(inline_vars, var_prefix)
                    acc_vars.update(inline_vars)

                    included_payloads.append(inc_proc)
                finally:
                    seen.remove(inc_path)

            # New semantics from previous code:
            # - every included file must resolve to a dict (mapping)
            for i_val, (i_path, _) in zip(included_payloads, entries):
                if not isinstance(i_val, dict):
                    raise ValueError(f"Included file must be a mapping/object: {i_path}")

            # Process the rest of this dict (other keys beside $include)
            rest = {k: v for k, v in node.items() if k != INCLUDE_KEY}

            # If multiple includes and additional keys are present, merge included dicts
            if len(included_payloads) == 1:
                combined: Any = included_payloads[0]
            else:
                if rest:
                    merged: Dict[str, Any] = {}
                    for v in included_payloads:
                        merged = _deep_merge_dicts(merged, v, concat_lists=False)
                    combined = merged
                else:
                    combined = included_payloads

            if rest:
                rest_proc, rest_vars = _preprocess_includes(rest, base_dir, seen)
                # Merge variables discovered in the rest of this node (e.g., from nested includes)
                acc_vars.update(rest_vars)

                if isinstance(combined, dict) and isinstance(rest_proc, dict):
                    return _deep_merge_dicts(combined, rest_proc, concat_lists=False), acc_vars
                # If combined were a list, we would have merged above when 'rest' is present
                return rest_proc, acc_vars

            # Only include present -> return combined payload directly
            return combined, acc_vars

        # No include at this level; recurse into values
        out: Dict[str, Any] = {}
        acc_vars: Dict[str, Any] = {}
        for k, v in node.items():
            child_proc, child_vars = _preprocess_includes(v, base_dir, seen)
            out[k] = child_proc
            acc_vars.update(child_vars)
        return out, acc_vars

    if isinstance(node, list):
        out_list: List[Any] = []
        acc_vars: Dict[str, Any] = {}
        for v in node:
            child_proc, child_vars = _preprocess_includes(v, base_dir, seen)
            out_list.append(child_proc)
            acc_vars.update(child_vars)
        return out_list, acc_vars

    return node, {}

def _load_and_preprocess(path: Union[str, Path], seen: Optional[Set[Path]] = None) -> tuple[Any, Dict[str, Any], Dict[str, Any]]:
    p = Path(path).resolve()
    if seen is None:
        seen = set()
    data = _load_raw_file(p)

    root_vars: Dict[str, Any] = {}
    if isinstance(data, dict):
        root_vars = _collect_variables(data)
        data = dict(data)
        data.pop("variables", None)

    processed, included_vars = _preprocess_includes(data, p.parent, seen)
    return processed, included_vars, root_vars


def _load_raw_file(path: Path) -> Any:
    ext = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    data: Any = None
    if ext in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif ext in {".json5", ".jsonc", ".json"}:
        data = json5.loads(text)
    else:
        raise ValueError(f"Unsupported config file extension: {ext}")
    if data is None:
        return {}
    return data


# (removed: _load_with_includes)


def load_settings(path: str) -> Settings:
    data_any, included_vars, root_vars = _load_and_preprocess(path)
    if not isinstance(data_any, dict):
        raise ValueError("Root configuration must be a mapping/object")

    # Build final variable map: included defaults first, then root-level overrides
    vars_map: Dict[str, Any] = {}
    vars_map.update(included_vars)
    vars_map.update(root_vars)

    # Interpolate and build models
    data = _apply_variables(data_any, vars_map)
    return Settings.model_validate(data)

def build_model_from_settings(data: Optional[Dict[str, Any]], model_cls: Type[BaseModel]) -> BaseModel:
    """
    Populate a Pydantic model from a settings dict.
    Raises ValidationError if the configuration is incorrect.
    """
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict for {model_cls.__name__} settings, got {type(data).__name__}")
    return model_cls.model_validate(data)
