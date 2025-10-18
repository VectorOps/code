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
VOCODE_TEMPLATE_BASE: Path = (
    Path(__file__).resolve().parent / "config_templates"
).resolve()
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
            data["nodes"] = [
                Node.from_obj(n) if isinstance(n, dict) else n for n in nodes
            ]
        return data


class ToolSettings(BaseModel):
    name: str
    enabled: bool = True
    auto_approve: Optional[bool] = None


class ToolCallFormatter(BaseModel):
    """
    Configures how to display a tool call in the terminal.
    - title: what to display as the function name
    - rule: field name in the tool request arguments to extract for display
    """

    title: str
    rule: str

    @model_validator(mode="after")
    def _validate_rule(self) -> "ToolCallFormatter":
        # No-op: allow any string as the field name.
        return self


class UISettings(BaseModel):
    # When true, PromptSession accepts multi-line input (Enter can insert newlines)
    multiline: bool = True
    # Optional editing mode override. None => use prompt_toolkit default (Emacs)
    edit_mode: Optional[Literal["emacs", "vim"]] = None
    # Minimum log level to display in the terminal.
    log_level: LogLevel = LogLevel.info
    # Show banner
    show_banner: bool = True


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

    @model_validator(mode="after")
    def _backcompat_legacy_fields(self) -> "MCPSettings":
        # Support legacy schemas by mapping into a default 'mcp' server
        # Accept legacy keys if present in raw input (model already validated)
        data = self.model_dump(mode="python")
        # Legacy at root: url, command (list), env
        legacy_url = data.get("url")
        legacy_command = data.get("command")
        legacy_env = data.get("env") or {}
        if (legacy_url or legacy_command) and not self.servers:
            server = MCPServerSettings()
            if legacy_url:
                server.url = legacy_url
            if legacy_command:
                # Legacy command was a list; split into command + args
                if isinstance(legacy_command, list) and legacy_command:
                    server.command = str(legacy_command[0])
                    server.args = [str(a) for a in legacy_command[1:]]
            if legacy_env:
                server.env = dict(legacy_env)
            self.servers = {"mcp": server}
        return self


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


class Settings(BaseModel):
    workflows: Dict[str, Workflow] = Field(default_factory=dict)
    tools: List[ToolSettings] = Field(default_factory=list)
    know: Optional[KnowProjectSettings] = Field(default=None)
    ui: Optional[UISettings] = Field(default=None)
    # Optional Model Context Protocol (MCP) configuration
    mcp: Optional[MCPSettings] = Field(default=None)
    # Optional process subsystem settings
    process: Optional[ProcessSettings] = Field(default=None)
    # Mapping of tool name -> formatter configuration
    tool_call_formatters: Dict[str, ToolCallFormatter] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sync_workflow_names(self) -> "Settings":
        for key, wf in self.workflows.items():
            wf.name = key
        return self
