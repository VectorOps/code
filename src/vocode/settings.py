from typing import List, Dict, Optional, Any, Union, Set, Final
import re
from pathlib import Path
from os import PathLike
import os
from pydantic import BaseModel, Field, model_validator
import yaml
import json5  # type: ignore

from .graph.models import Node, Edge

# Base path for packaged template configs, e.g. include: { vocode: "nodes/requirements.yaml" }
VOCODE_TEMPLATE_BASE: Path = (Path(__file__).resolve().parent / "config_templates").resolve()
# Include spec keys for bundled templates. Support GitLab 'template', legacy 'vocode', and 'templates'
TEMPLATE_INCLUDE_KEYS: Final[Set[str]] = {"template", "templates", "vocode"}
# Variable replacement pattern
VAR_PATTERN = re.compile(r"\$(\w+)|\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class Tool(BaseModel):
    name: Optional[str] = None
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


class Settings(BaseModel):
    tools: Dict[str, Tool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sync_tool_names(self) -> "Settings":
        for key, tool in self.tools.items():
            tool.name = key
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

def _collect_variables(doc: Dict[str, Any]) -> Dict[str, str]:
    """
    Collect variables from the merged config. Supports:
      - mapping: variables: { KEY: default }
      - list of one-key mappings: variables: [ {KEY: default}, ... ]
      - list of entries with explicit keys: variables: [ {key: KEY, value: default}, ... ]
    Environment variables with the same name override the defaults.
    """
    vars_spec = doc.get("variables")
    out: Dict[str, str] = {}
    if vars_spec is None:
        return out
    if isinstance(vars_spec, dict):
        for k, v in vars_spec.items():
            if isinstance(k, str):
                out[k] = "" if v is None else str(v)
    elif isinstance(vars_spec, list):
        for item in vars_spec:
            if isinstance(item, dict):
                if "key" in item and "value" in item and isinstance(item["key"], str):
                    out[item["key"]] = "" if item["value"] is None else str(item["value"])
                else:
                    # Merge one-key mappings
                    for k, v in item.items():
                        if isinstance(k, str):
                            out[k] = "" if v is None else str(v)
    # Overlay with environment if present (only for declared keys)
    for k in list(out.keys()):
        envv = os.getenv(k)
        if envv is not None:
            out[k] = envv
    return out

def _interpolate_string(s: str, vars_map: Dict[str, str]) -> str:
    def repl(m: re.Match) -> str:
        name = m.group(1) or m.group(2)
        return vars_map.get(name, m.group(0))
    return VAR_PATTERN.sub(repl, s)

def _apply_variables(obj: Any, vars_map: Dict[str, str]) -> Any:
    if isinstance(obj, str):
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


def _load_raw_file(path: Path) -> Dict[str, Any]:
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
    if not isinstance(data, dict):
        raise ValueError(f"Top-level config in {path} must be a mapping/object")
    return data


def _load_with_includes(path: Union[str, Path], seen: Optional[Set[Path]] = None) -> Dict[str, Any]:
    p = Path(path).resolve()
    if seen is None:
        seen = set()
    if p in seen:
        raise ValueError(f"Detected include cycle at {p}")
    seen.add(p)
    data = _load_raw_file(p)
    base_dir = p.parent
    include_spec = data.pop("include", None)
    merged: Dict[str, Any] = {}
    # Merge included files first (concatenate lists across multiple include fragments)
    for inc in _collect_include_paths(include_spec, base_dir):
        inc_data = _load_with_includes(inc, seen)
        merged = _deep_merge_dicts(merged, inc_data, concat_lists=True)
    # Then overlay the including file's own content (lists replace by default)
    merged = _deep_merge_dicts(merged, data, concat_lists=False)
    # Collect variables (defaults from config, overridden by environment) and interpolate
    vars_map = _collect_variables(merged)
    merged.pop("variables", None)
    merged = _apply_variables(merged, vars_map)
    seen.remove(p)
    return merged


def load_settings(path: str) -> Settings:
    data = _load_with_includes(path)
    return Settings.model_validate(data)
