from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..settings import MCPSettings, MCPServerSettings
from ..tools import register_tool, unregister_tool
from .proxy import MCPToolProxy

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
        self._registered_tools: List[str] = []
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
        for tool in tools:
            name = tool.name
            if not name:
                continue
            if whitelist and name not in whitelist:
                continue
            parameters = tool.inputSchema or {"type": "object", "properties": {}}
            proxy = MCPToolProxy(name=name, parameters_schema=parameters, manager=self)
            try:
                register_tool(name, proxy)
                self._registered_tools.append(name)
            except ValueError:
                # Already registered; skip
                continue

        # Update project tool enablement snapshot
        project.refresh_tools_from_registry()

        self._started = True

    async def stop(self) -> None:
        # Unregister tools first
        for name in list(self._registered_tools):
            unregister_tool(name)
        self._registered_tools.clear()

        # Notify project of tool set changes
        if self._project:
            self._project.refresh_tools_from_registry()

        # Exit the FastMCP client context
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None

        self._started = False
        self._project = None

    async def call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        if not self._client:
            raise RuntimeError("MCPManager is not started")
        return await self._client.call_tool(name=name, arguments=args)

    async def _create_client(self) -> Any:
        """
        Build a FastMCP Client from MCPSettings. The Client manages connecting/spawning.
        """
        try:
            from fastmcp import Client  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "FastMCP client not available; pip install fastmcp"
            ) from e

        # Build FastMCP-compatible config
        mcp_servers: Dict[str, Dict[str, Any]] = {}
        for server_name, server in (self._settings.servers or {}).items():
            entry: Dict[str, Any] = {}
            if server.url:
                entry["url"] = server.url
            if server.command:
                entry["command"] = server.command
                if server.args:
                    entry["args"] = list(server.args)
            if server.env:
                entry["env"] = dict(server.env)
            if not entry:
                continue
            mcp_servers[server_name] = entry

        if not mcp_servers:
            # Nothing configured -> error out
            raise RuntimeError(
                "MCP settings require at least one server (url or command)."
            )

        config: Dict[str, Any] = {"mcpServers": mcp_servers}
        return Client(config)

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
