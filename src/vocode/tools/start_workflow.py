from typing import Any, Dict, Optional
from .base import BaseTool, ToolStartWorkflowResponse
from vocode.settings import ToolSpec


class StartWorkflowTool(BaseTool):
    """
    A tool that requests starting a child workflow.

    - The target workflow name must be provided via args["workflow"].
    - Optional initial user text can be provided in args as "text" (preferred) or "initial_text".
    """

    name = "start_workflow"

    async def run(self, spec: ToolSpec, args: Any):
        if not isinstance(spec, ToolSpec):
            raise TypeError("StartWorkflowTool requires a resolved ToolSpec")
        if not isinstance(args, dict):
            raise TypeError(
                "StartWorkflowTool requires dict args with a 'workflow' key"
            )

        workflow = args.get("workflow")
        if not workflow or not isinstance(workflow, str):
            raise ValueError("StartWorkflowTool requires 'workflow' argument (string)")

        # Enforce WorkflowConfig.child_workflows allowlist when a parent workflow
        # context is available on the Project. This prevents arbitrary nested
        # workflow starts unless explicitly whitelisted.
        prj = self.prj
        parent_name: Optional[str] = getattr(prj, "current_workflow", None)
        settings = getattr(prj, "settings", None)
        if parent_name and settings and settings.workflows:
            parent_cfg = settings.workflows.get(parent_name)
            if parent_cfg and parent_cfg.child_workflows is not None:
                if workflow not in parent_cfg.child_workflows:
                    raise ValueError(
                        f"Workflow '{workflow}' is not allowed as a child of '{parent_name}'"
                    )

        initial_text: Optional[str] = None
        # Prefer "text" to align with existing tests and conventions; accept "initial_text" as a fallback.
        if isinstance(args.get("text"), str):
            initial_text = args.get("text")
        elif isinstance(args.get("initial_text"), str):
            initial_text = args.get("initial_text")

        return ToolStartWorkflowResponse(workflow=workflow, initial_text=initial_text)

    async def openapi_spec(self, spec: ToolSpec) -> Dict[str, Any]:
        # Static OpenAPI schema: workflow name is provided via args, not ToolSpec.config.
        return {
            "name": self.name,
            "description": (
                "Start a child workflow by name. "
                "Provide 'workflow' as the workflow name and optional 'text' as "
                "the user's first message in the child workflow."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow": {
                        "type": "string",
                        "description": "Name of the workflow to start",
                    },
                    "text": {
                        "type": "string",
                        "description": "Initial input for the child workflow",
                    },
                },
                "required": ["workflow"],
                "additionalProperties": False,
            },
        }


try:
    from .base import register_tool

    register_tool(StartWorkflowTool.name, StartWorkflowTool)
except Exception:
    pass
