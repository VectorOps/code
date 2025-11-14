from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from vocode.proc.manager import ProcessManager
from vocode.tools.base import BaseTool, ToolTextResponse
from vocode.settings import ToolSpec

if TYPE_CHECKING:
    from vocode.project import Project

# Fixed timeout for all exec tool invocations (seconds)
EXEC_TOOL_TIMEOUT_S: float = 1.0


class ExecTool(BaseTool):
    """
    Execute a command in a subprocess using the project's ProcessManager.
    Collects combined stdout/stderr, enforces a fixed timeout, and returns a JSON string payload.
    """
    name = "exec"

    async def run(self, project: "Project", spec: ToolSpec, args: Any):
        if not isinstance(spec, ToolSpec):
            raise TypeError("ExecTool requires a resolved ToolSpec")
        if project is None:
            raise RuntimeError("ExecTool requires a project")
        if project.processes is None:
            raise RuntimeError("ExecTool requires project.processes (ProcessManager)")

        pm: ProcessManager = project.processes

        # Parse args
        command: Optional[str] = None
        if isinstance(args, str):
            command = args
        elif isinstance(args, dict):
            arg_cmd = args.get("command")
            if isinstance(arg_cmd, str):
                command = arg_cmd
        if not command:
            raise ValueError("ExecTool requires 'command' (string) argument")

        # Spawn a one-off process (shell=True)
        handle = await pm.spawn(command=command, name="tool:exec", shell=True, use_pty=False)

        stdout_parts: List[str] = []
        stderr_parts: List[str] = []

        async def _read_stdout():
            async for chunk in handle.iter_stdout():
                stdout_parts.append(chunk)

        async def _read_stderr():
            async for chunk in handle.iter_stderr():
                stderr_parts.append(chunk)

        readers = [
            asyncio.create_task(_read_stdout()),
            asyncio.create_task(_read_stderr()),
        ]

        timed_out = False
        rc: Optional[int] = None
        try:
            rc = await asyncio.wait_for(handle.wait(), timeout=EXEC_TOOL_TIMEOUT_S)
        except asyncio.TimeoutError:
            timed_out = True
            try:
                await handle.kill()
            finally:
                with contextlib.suppress(Exception):
                    await handle.wait()
                rc = None
        finally:
            await asyncio.gather(*readers, return_exceptions=True)

        output = "".join(stdout_parts) + "".join(stderr_parts)
        payload = {
            "output": output,
            "exit_code": rc,
            "timed_out": timed_out,
        }
        return ToolTextResponse(text=json.dumps(payload))

    async def openapi_spec(self, project: "Project", spec: ToolSpec) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                f"Execute a shell command and return combined stdout/stderr, exit code, and timeout status. "
                f"This tool enforces a fixed timeout of {EXEC_TOOL_TIMEOUT_S} seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to run (executed via system shell).",
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        }


# Register tool on import
try:
    from .base import register_tool
    register_tool(ExecTool.name, ExecTool())
except Exception:
    pass