from typing import List, Dict, Optional, Any, Union, Set, Final, Type
import re
from pathlib import Path
from os import PathLike
import os
import json
from pydantic import BaseModel, Field, model_validator
import yaml
import json5  # type: ignore
from .graph.models import Node, Edge

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
    config: Dict[str, Any] = Field(default_factory=dict)

class Settings(BaseModel):
    workflows: Dict[str, Workflow] = Field(default_factory=dict)
    tools: List[ToolSettings] = Field(default_factory=list)

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


def _collect_include_paths(spec: Any, base_dir: Path) -> List[Path]:
    if spec is None:
        return []
    def norm_one(item: Any) -> List[Path]:
        if isinstance(item, str):
            return _expand_include_patterns(base_dir, item)
        if isinstance(item, dict):
            paths: List[Path] = []
            if "local" in item:
                loc = item["local"]
                if isinstance(loc, list):
                    for p in loc:
                        paths.extend(_expand_include_patterns(base_dir, p))
                else:
                    paths.extend(_expand_include_patterns(base_dir, loc))
            elif any(k in item for k in TEMPLATE_INCLUDE_KEYS):
                # Resolve relative to the packaged vocode/config_templates directory
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
                raise ValueError(f"Unsupported include dict keys: {list(item.keys())}")
            return paths
        if isinstance(item, list):
            acc: List[Path] = []
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

def _preprocess_includes(node: Any, base_dir: Path, seen: Set[Path]) -> Any:
    if isinstance(node, dict):
        # If this dict contains a $include, expand it and merge/replace as appropriate
        if INCLUDE_KEY in node:
            include_spec = node[INCLUDE_KEY]
            # Resolve include paths relative to this file's base_dir (or templates base)
            paths = _collect_include_paths(include_spec, base_dir)
            included: List[Any] = []
            for inc in paths:
                inc = inc.resolve()
                if inc in seen:
                    raise ValueError(f"Detected include cycle at {inc}")
                seen.add(inc)
                inc_data = _load_raw_file(inc)
                inc_proc = _preprocess_includes(inc_data, inc.parent, seen)
                seen.remove(inc)
                included.append(inc_proc)

            # New semantics:
            # - every included file must resolve to a dict (mapping)
            # - if exactly one path matched, return that dict
            # - if multiple matched, return a list of dicts
            for i_val, i_path in zip(included, paths):
                if not isinstance(i_val, dict):
                    raise ValueError(f"Included file must be a mapping/object: {i_path}")

            combined: Any = included[0] if len(included) == 1 else included

            # Process the rest of this dict (other keys beside $include)
            rest = {k: v for k, v in node.items() if k != INCLUDE_KEY}
            if rest:
                rest_proc = _preprocess_includes(rest, base_dir, seen)
                if isinstance(combined, dict) and isinstance(rest_proc, dict):
                    return _deep_merge_dicts(combined, rest_proc, concat_lists=False)
                if isinstance(combined, list):
                    raise ValueError("$include produced a list but additional keys are present to merge")
                return rest_proc
            # Only include present -> return combined payload directly
            return combined
        # No include at this level; recurse into values
        return {k: _preprocess_includes(v, base_dir, seen) for k, v in node.items()}
    if isinstance(node, list):
        return [_preprocess_includes(v, base_dir, seen) for v in node]
    return node

def _load_and_preprocess(path: Union[str, Path], seen: Optional[Set[Path]] = None) -> Any:
    p = Path(path).resolve()
    if seen is None:
        seen = set()
    data = _load_raw_file(p)
    processed = _preprocess_includes(data, p.parent, seen)
    return processed


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
    data_any = _load_and_preprocess(path)
    if not isinstance(data_any, dict):
        raise ValueError("Root configuration must be a mapping/object")
    data = data_any
    # Collect variables and interpolate
    vars_map = _collect_variables(data)
    data.pop("variables", None)
    data = _apply_variables(data, vars_map)
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
