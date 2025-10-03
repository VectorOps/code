from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import asyncio

from ..settings import MCPSettings
from ..tools import register_tool, unregister_tool
from .proxy import MCPToolProxy

if TYPE_CHECKING:
    from ..project import Project


class MCPManager:
    """
    Manages the lifecycle of an MCP (Model Context Protocol) client. Responsible for:
    - Starting (connecting/spawning) the FastMCP client
    - Discovering available tools and registering MCPToolProxy instances
    - Forwarding tool calls and cleaning up on shutdown
    """

    def __init__(self, settings: MCPSettings) -> None:
        self._settings = settings
        self._client: Optional[Any] = None  # opaque fastmcp client
        self._registered_tools: List[str] = []
        self._started: bool = False
        self._project: Optional["Project"] = None

    async def start(self, project: "Project") -> None:
        if self._started:
            return
        self._project = project
        self._client = await self._create_client()

        tools = await _maybe_await(self._list_tools())
        # Optional whitelist
        whitelist = set(self._settings.tools_whitelist or [])
        for tool in tools:
            name = tool.get("name")
            if not name:
                continue
            if whitelist and name not in whitelist:
                continue
            parameters = tool.get("parameters") or {"type": "object", "properties": {}}
            proxy = MCPToolProxy(name=name, parameters_schema=parameters, manager=self)
            try:
                register_tool(name, proxy)
                self._registered_tools.append(name)
            except ValueError:
                # Already registered; skip
                continue

        # Update project tool enablement snapshot
        if hasattr(project, "refresh_tools_from_registry"):
            project.refresh_tools_from_registry()

        self._started = True

    async def stop(self) -> None:
        # Unregister tools first
        for name in list(self._registered_tools):
            unregister_tool(name)
        self._registered_tools.clear()

        # Notify project of tool set changes
        if self._project and hasattr(self._project, "refresh_tools_from_registry"):
            self._project.refresh_tools_from_registry()

        # Close the client if available
        if self._client is not None:
            # Try a few common methods; tolerate either sync or async
            await _maybe_await(getattr(self._client, "aclose", None))
            await _maybe_await(getattr(self._client, "close", None))
            await _maybe_await(getattr(self._client, "shutdown", None))
            # For spawned processes, try 'terminate'
            await _maybe_await(getattr(self._client, "terminate", None))
            self._client = None

        self._started = False
        self._project = None

    async def call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        if not self._client:
            raise RuntimeError("MCPManager is not started")
        call = getattr(self._client, "call_tool", None)
        if call is None:
            raise RuntimeError("MCP client does not support 'call_tool'")
        return await _maybe_await(call(name=name, arguments=args))

    async def _create_client(self) -> Any:
        """
        Connect or spawn a FastMCP client instance based on settings.
        This method is intentionally overridable in tests.
        """
        try:
            # Lazy import to avoid hard dependency for tests
            import fastmcp  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "FastMCP client not available; install 'fastmcp' or provide a test client"
            ) from e

        if self._settings.url:
            connect = getattr(fastmcp, "connect", None) or getattr(
                fastmcp, "Client", None
            )
            if connect is None:
                raise RuntimeError("FastMCP library missing 'connect' or 'Client'")
            client = connect(self._settings.url)  # type: ignore[call-arg]
            return await _maybe_await(client)

        if self._settings.command:
            spawn = getattr(fastmcp, "spawn", None)
            if spawn is None:
                raise RuntimeError("FastMCP library missing 'spawn'")
            client = spawn(self._settings.command, env=self._settings.env or {})
            return await _maybe_await(client)

        raise RuntimeError("MCP settings require either 'url' or 'command'")

    def _list_tools(self):
        if not self._client:
            raise RuntimeError("MCP client is not started")
        lister = getattr(self._client, "list_tools", None) or getattr(
            self._client, "tools", None
        )
        if callable(lister):
            return lister()
        # If it's an attribute with a list-like
        return lister


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