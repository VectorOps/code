from typing import List, Dict, Optional, Any, Union, Set, Final, Type, Literal
import re
from pathlib import Path
from os import PathLike
import os
import json
from pydantic import BaseModel, Field
from pydantic import model_validator, field_validator
import yaml
import json5  # type: ignore
from .models import Node, Edge
from .state import LogLevel
from .lib.validators import get_value_by_dotted_key, regex_matches_value


from knowlt.settings import ProjectSettings as KnowProjectSettings


# Base path for packaged template configs, e.g. include: { vocode: "nodes/requirements.yaml" }
VOCODE_TEMPLATE_BASE: Path = (
    Path(__file__).resolve().parent / "config_templates"
).resolve()
# Include spec keys for bundled templates. Support GitLab 'template', legacy 'vocode', and 'templates'
TEMPLATE_INCLUDE_KEYS: Final[Set[str]] = {"template", "templates", "vocode"}
# Variable replacement pattern: only support ${ABC}
VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

INCLUDE_KEY: Final[str] = "$include"

# Default maximum combined stdout/stderr characters returned by the exec tool.
# Individual projects can override this via Settings.exec_tool.max_output_chars.
EXEC_TOOL_MAX_OUTPUT_CHARS_DEFAULT: Final[int] = 10 * 1024


class WorkflowConfig(BaseModel):
    name: Optional[str] = None
    # Human-readable purpose/summary for this workflow; used in tool descriptions.
    description: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    # Optional allowlist of workflows that may be started as agents from this
    # workflow via the run_agent tool. When None, no additional restriction is
    # applied.
    agent_workflows: Optional[List[str]] = None

    @model_validator(mode="before")
    @classmethod
    def _dispatch_nodes(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # Backwards compatibility: support legacy "child_workflows" key by
        # mapping it into the new "agent_workflows" field when the latter is
        # not explicitly provided.
        legacy = data.get("child_workflows")
        if "agent_workflows" not in data and legacy is not None:
            data = dict(data)
            data["agent_workflows"] = legacy

        nodes = data.get("nodes")
        if isinstance(nodes, list):
            # Convert dicts to Node instances using the registry-based dispatcher
            data = dict(data)
            data["nodes"] = [
                Node.from_obj(n) if isinstance(n, dict) else n for n in nodes
            ]
        return data


class ToolAutoApproveRule(BaseModel):
    """Rule for automatically approving a tool call based on its JSON arguments.

    - key: dot-separated path inside the arguments dict (e.g. "resource.action").
    - pattern: regular expression applied to the stringified value at that key.
    """

    key: str
    pattern: str

    @field_validator("pattern")
    @classmethod
    def _validate_pattern(cls, v: str) -> str:
        """Validate that 'pattern' is a syntactically correct regular expression."""
        try:
            re.compile(v)
        except (
            re.error
        ) as exc:  # pragma: no cover - exact message is implementation detail
            raise ValueError(f"Invalid regex pattern {v!r}: {exc}") from exc
        return v


class ToolSpec(BaseModel):
    """
    Tool specification usable both globally (Settings.tools) and per-node (LLMNode.tools).
    - name: required tool name
    - enabled: global enable/disable (ignored for node-local specs)
    - auto_approve: optional auto-approval flag for UI behavior
    - auto_approve_rules: optional list of rules that allow auto-approval when
      any rule matches the tool call arguments
    - config: free-form configuration for tool implementations
    Accepts shorthand string form: "tool_name".
    Extra fields are ignored.

    When updating the model, make sure that build_effective_tool_specs helper function
    is also updated.
    """

    name: str
    enabled: bool = True
    auto_approve: Optional[bool] = None
    auto_approve_rules: List[ToolAutoApproveRule] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, v: Any) -> Any:
        if isinstance(v, str):
            return {"name": v}
        if isinstance(v, dict):
            # Permit extra fields; validator ignores unknowns via Pydantic defaults
            name = v.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("Tool spec must include non-empty 'name'")
            out = {
                "name": name,
                "enabled": v.get("enabled", True),
                "auto_approve": v.get("auto_approve", None),
                "auto_approve_rules": v.get("auto_approve_rules", []) or [],
                "config": v.get("config", {}) or {},
            }
            return out
        return v


class ToolCallFormatter(BaseModel):
    """
    Configures how to display a tool call in the terminal.
    - title: what to display as the function name
    - formatter: registered formatter implementation name (e.g. "generic")
    - show_output: whether to show tool output details by default
    - options: free-form formatter-specific configuration
    """

    title: str
    formatter: str = "generic"
    show_output: bool = False
    options: Dict[str, Any] = Field(default_factory=dict)


class UISettings(BaseModel):
    # When true, PromptSession accepts multi-line input (Enter can insert newlines)
    multiline: bool = True
    # Optional editing mode override. None => use prompt_toolkit default (Emacs)
    edit_mode: Optional[Literal["emacs", "vim"]] = None
    # Show banner
    show_banner: bool = True


class LoggingSettings(BaseModel):
    # Default level for our primary loggers (vocode, knowlt) if not overridden.
    default_level: LogLevel = LogLevel.info
    # Mapping of logger name -> level override (e.g., {"asyncio": "debug"})
    enabled_loggers: Dict[str, LogLevel] = Field(default_factory=dict)


class MCPServerSettings(BaseModel):
    # FastMCP-compatible server config. One of url OR command must be provided.
    url: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)


class MCPSettings(BaseModel):
    """
    FastMCP-compatible MCP configuration.
    - servers: mapping of server-name -> MCPServerSettings
    - tools_whitelist: optional exact-name allowlist for discovered tools
    Backward compatible with legacy fields (url/command/env) by mapping to a default 'mcp' server.
    """

    servers: Dict[str, MCPServerSettings] = Field(default_factory=dict)
    tools_whitelist: Optional[List[str]] = None


class ProcessEnvSettings(BaseModel):
    inherit_parent: bool = True
    allowlist: Optional[List[str]] = None
    denylist: Optional[List[str]] = None
    defaults: Dict[str, str] = Field(default_factory=dict)


class ShellSettings(BaseModel):
    # POSIX-only in v1; reserved for future shells
    type: Literal["bash"] = "bash"
    # Program and args to start the long-lived shell process
    program: str = "bash"
    args: List[str] = Field(default_factory=lambda: ["--noprofile", "--norc"])
    # Default per-command timeout (seconds)
    default_timeout_s: int = 120


class ProcessSettings(BaseModel):
    # Backend key in the process backend registry
    backend: str = "local"
    env: ProcessEnvSettings = Field(default_factory=ProcessEnvSettings)
    shell: ShellSettings = Field(default_factory=ShellSettings)


class ExecToolSettings(BaseModel):
    # Maximum characters of combined stdout/stderr returned by the exec tool.
    # This guards against excessive subprocess output overwhelming callers.
    max_output_chars: int = EXEC_TOOL_MAX_OUTPUT_CHARS_DEFAULT


class ToolRuntimeSettings(BaseModel):
    """Runtime configuration for tool execution.

    - max_concurrent: maximum number of tools to execute concurrently for a
      single PACKET_TOOL_CALL. None or <= 0 means unlimited concurrency.
    """

    max_concurrent: Optional[int] = None


class Settings(BaseModel):
    workflows: Dict[str, WorkflowConfig] = Field(default_factory=dict)
    # Optional name of the workflow to auto-start in interactive UIs
    default_workflow: Optional[str] = Field(default=None)
    tools: List[ToolSpec] = Field(default_factory=list)
    know: Optional[KnowProjectSettings] = Field(default=None)
    ui: Optional[UISettings] = Field(default=None)
    # Optional logging configuration (per-logger overrides).
    logging: Optional[LoggingSettings] = Field(default=None)
    # Optional Model Context Protocol (MCP) configuration
    mcp: Optional[MCPSettings] = Field(default=None)
    # Optional process subsystem settings
    process: Optional[ProcessSettings] = Field(default=None)
    # Optional global exec tool configuration
    exec_tool: Optional[ExecToolSettings] = Field(default=None)
    # Optional runtime behavior for tool execution
    tools_runtime: Optional[ToolRuntimeSettings] = Field(default=None)
    # Mapping of tool name -> formatter configuration
    tool_call_formatters: Dict[str, ToolCallFormatter] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sync_workflow_names(self) -> "Settings":
        for key, wf in self.workflows.items():
            wf.name = key
        return self
