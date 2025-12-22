from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Optional dependency: fastmcp
try:
    from fastmcp import Client as FastMCPClient  # type: ignore
except Exception:
    FastMCPClient = None  # type: ignore[assignment]

from ..settings import MCPSettings, MCPServerSettings, ToolSpec

if TYPE_CHECKING:
    from ..project import Project


class MCPManager:
    """
    Manages the lifecycle of an MCP (Model Context Protocol) client. Responsible for:
    - Starting the FastMCP client via its async context manager
    - Discovering available tools and registering MCPToolProxy instances
    - Forwarding tool calls and cleaning up on shutdown
    """

    def __init__(self, settings: MCPSettings) -> None:
        self._settings = settings
        self._client: Optional[Any] = None  # fastmcp.Client instance (entered)
        self._tool_schemas: Dict[str, Dict[str, Any]] = {}
        self._started: bool = False
        self._project: Optional["Project"] = None

    async def start(self, project: "Project") -> None:
        if self._started:
            return
        self._project = project
        self._client = await self._create_client()
        # Enter the FastMCP client's async context
        await self._client.__aenter__()

        tools = await self._list_tools()
        # Optional whitelist
        whitelist = self._settings.tools_whitelist or []
        # Reset discovered tool schemas on (re)start.
        self._tool_schemas.clear()

        for tool in tools:
            # fastmcp list_tools() entries are expected to expose 'name' and 'inputSchema'.
            name = getattr(tool, "name", None)
            if not name:
                continue
            if whitelist and name not in whitelist:
                continue

            parameters = getattr(tool, "inputSchema", None) or {
                "type": "object",
                "properties": {},
            }

            # If multiple MCP servers expose the same tool name, keep the first one.
            if name in self._tool_schemas:
                continue

            self._tool_schemas[name] = parameters

        # Update project tool enablement snapshot
        project.refresh_tools_from_registry()

        self._started = True

    async def stop(self) -> None:
        # Clear discovered MCP tool schemas so they are no longer exposed via the project.
        self._tool_schemas.clear()

        # Notify project of tool set changes
        if self._project:
            self._project.refresh_tools_from_registry()

        # Exit the FastMCP client context
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            await self._client.close()
            self._client = None

        self._started = False
        self._project = None

    async def call_tool(self, name: str, args: Dict[str, Any], spec: ToolSpec) -> Any:
        if not self._client:
            raise RuntimeError("MCPManager is not started")
        return await self._client.call_tool(name=name, arguments=args)

    def get_tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a copy of the discovered MCP tool schemas keyed by tool name.
        Empty if the MCP client has not been started or if discovery returned no tools.
        """
        return dict(self._tool_schemas)

    async def _create_client(self) -> Any:
        """
        Build a FastMCP Client from MCPSettings. The Client manages connecting/spawning.
        """
        if FastMCPClient is None:
            raise RuntimeError("FastMCP client not available; pip install fastmcp")
        if self._project is None:
            raise RuntimeError("Cannot create MCP client without a project context")

        # Build FastMCP-compatible config
        mcp_servers: Dict[str, Dict[str, Any]] = {}
        for server_name, server in (self._settings.servers or {}).items():
            entry: Dict[str, Any] = {}
            if server.url:
                entry["url"] = server.url
                if server.headers:
                    # Configure explicit transport so HTTP/S remote servers receive headers.
                    entry["transport"] = {
                        "type": "sse",
                        "url": server.url,
                        "headers": dict(server.headers),
                    }
            if server.command:
                entry["command"] = server.command
                if server.args:
                    entry["args"] = list(server.args)
            if server.env:
                entry["env"] = dict(server.env)
            # Ensure servers spawned via command run in the project root
            # Note: harmless if URL-based; fastmcp should ignore cwd when not spawning.
            entry["cwd"] = str(self._project.base_path)
            if not entry:
                continue
            mcp_servers[server_name] = entry

        if not mcp_servers:
            # Nothing configured -> error out
            raise RuntimeError(
                "MCP settings require at least one server (url or command)."
            )

        config: Dict[str, Any] = {"mcpServers": mcp_servers}
        # Ensure servers spawned via command run in the project root via per-server cwd
        return FastMCPClient(config)

    async def _list_tools(self):
        if not self._client:
            raise RuntimeError("MCP client is not started")
        return await self._client.list_tools()


async def _maybe_await(obj):
    if obj is None:
        return None
    if asyncio.iscoroutine(obj):
        return await obj
    if callable(obj):
        val = obj()
        if asyncio.iscoroutine(val):
            return await val
        return val
    return obj
