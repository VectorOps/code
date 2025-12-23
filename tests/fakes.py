from __future__ import annotations
import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Callable, Awaitable

from vocode.settings import Settings, ExecToolSettings
from vocode.state import LLMUsageStats
from vocode.proc.manager import ProcessManager
from vocode.proc.shell import ShellManager


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

    async def close(self):
        pass

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


class TestProject:
    """
    Lightweight test double for vocode.project.Project.

    Provides:
      - settings: vocode.settings.Settings (with exec_tool defaults if not supplied)
      - tools: mapping of tool name -> tool instance
      - processes: ProcessManager, for tools like ExecTool
      - llm_usage: LLMUsageStats, plus add_llm_usage() similar to real Project
    """

    # Prevent pytest from treating this helper as a test class.
    __test__ = False

    def __init__(
        self,
        settings: Optional[Settings] = None,
        tools: Optional[Dict[str, Any]] = None,
        process_manager: Optional[ProcessManager] = None,
    ) -> None:
        # Ensure exec_tool settings exist by default for ExecTool tests.
        self.settings: Settings = settings or Settings(exec_tool=ExecToolSettings())
        # Tools registry used by Runner._run_tools and similar code paths.
        self.tools: Dict[str, Any] = tools or {}
        # Process and shell managers used by tools such as ExecTool.
        self.processes: Optional[ProcessManager] = process_manager
        self.shells = (
            ShellManager(process_manager=process_manager)
            if process_manager is not None
            else None
        )
        # Aggregate LLM usage for LLMExecutor and related code paths.
        self.llm_usage: LLMUsageStats = LLMUsageStats()

    def add_llm_usage(
        self,
        prompt_delta: int = 0,
        completion_delta: int = 0,
        cost_delta: float = 0.0,
    ) -> None:
        stats = self.llm_usage
        stats.prompt_tokens += int(prompt_delta or 0)
        stats.completion_tokens += int(completion_delta or 0)
        stats.cost_dollars += float(cost_delta or 0.0)
