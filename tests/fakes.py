from __future__ import annotations
import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Callable, Awaitable


class FakeMCPClient:
    """
    A mock fastmcp.Client for testing without a real MCP server.
    """

    def __init__(
        self,
        tools: Optional[List[Dict[str, Any]]] = None,
        call_handler: Optional[Callable[[str, Dict[str, Any]], Awaitable[Any]]] = None,
    ):
        self._tools_data = tools or []
        self._call_handler = call_handler or self._default_call_handler

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def list_tools(self) -> List[SimpleNamespace]:
        return [
            SimpleNamespace(name=t.get("name"), inputSchema=t.get("inputSchema"))
            for t in self._tools_data
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> SimpleNamespace:
        result_data = await self._call_handler(name, arguments)
        return SimpleNamespace(data=result_data)

    async def _default_call_handler(self, name: str, arguments: Dict[str, Any]) -> str:
        # A simple default behavior for echo-like tools
        if "echo" in name:
            return arguments.get("text", "")
        return json.dumps({"error": "unhandled in fake"})


def make_fake_mcp_client_creator(client_instance: FakeMCPClient) -> callable:
    """
    Returns a function suitable for monkeypatching MCPManager._create_client.
    """

    async def fake_create_client(self):
        return client_instance

    return fake_create_client
