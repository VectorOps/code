from __future__ import annotations

import json
from typing import Any, Optional, TYPE_CHECKING, Union
from pydantic import BaseModel

from know.tools.base import BaseTool as KnowBaseTool, ToolRegistry as KnowToolRegistry
from ..tools import BaseTool, register_tool, ToolTextResponse
from ..settings import ToolSpec

from .project import know_thread

if TYPE_CHECKING:
    from vocode.project import Project


class _KnowToolWrapper(BaseTool):
    """A wrapper to make a know tool compatible with the vocode tool system."""

    def __init__(self, know_tool: KnowBaseTool):
        super().__init__()
        self._know_tool = know_tool
        self.name = self._know_tool.tool_name
        # Propagate input/output model types from the Know tool
        self.input_model = self._know_tool.tool_input

    def openapi_spec(self, project: "Project", spec: ToolSpec) -> dict[str, Any]:
        return self._know_tool.get_openai_schema()

    async def run(
        self, project: "Project", spec: ToolSpec, args: Any
    ) -> ToolTextResponse:
        def do_execute():
            return self._know_tool.execute(project.know.pm, args)

        try:
            result = await know_thread.async_proxy()(do_execute)()
            return ToolTextResponse(text=result if result is not None else None)
        except Exception as e:
            # Return a structured error instead of propagating.
            return ToolTextResponse(
                text=json.dumps({"error": f"Know tool '{self.name}' failed: {e}"})
            )


_know_tools_registered = False


def register_know_tools() -> None:
    """
    Registers all 'know' tools with the vocode tool registry.
    This is idempotent.
    """
    global _know_tools_registered
    if _know_tools_registered:
        return

    for tool_instance in KnowToolRegistry._tools.values():
        wrapper = _KnowToolWrapper(tool_instance)
        try:
            register_tool(wrapper.name, wrapper)
        except ValueError:
            # Tool with this name is already registered, skipping.
            pass

    _know_tools_registered = True
