from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from ..tools import BaseTool

if TYPE_CHECKING:
    from ..project import Project
    from .manager import MCPManager

class MCPToolProxy(BaseTool):
    """
    Proxy BaseTool that forwards calls to an MCP manager/client.
    """

    def __init__(
        self, *, name: str, parameters_schema: Dict[str, Any], manager: "MCPManager"
    ):
        super().__init__()
        self.name = name
        self._parameters_schema = parameters_schema or {"type": "object", "properties": {}}
        self._manager = manager
        # Tools accept Any/dict directly; no dynamic Pydantic model is constructed.

    def openapi_spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": "",
            "parameters": self._parameters_schema or {"type": "object", "properties": {}},
        }

    async def run(self, project: "Project", args: Any) -> Optional[str]:
        """
        Accept parsed arguments (typically a dict) and forward to MCP manager.
        Falls back to empty dict for unsupported types/None.
        """
        payload: Dict[str, Any] = args if isinstance(args, dict) else {}
        return await self._manager.call_tool(self.name, payload)
