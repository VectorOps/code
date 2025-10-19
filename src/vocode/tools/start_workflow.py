from typing import Any, Dict, Optional

from .base import BaseTool, ToolStartWorkflowResponse
from vocode.settings import ToolSpec


class StartWorkflowTool(BaseTool):
    """
    A tool that requests starting a child workflow.

    - The target workflow name must be provided via ToolSpec.config["workflow"].
    - Optional initial user text can be provided in args as "text" (preferred) or "initial_text".
    """
    name = "start_workflow"

    async def run(self, project, spec: ToolSpec, args: Any):
        if not isinstance(spec, ToolSpec):
            raise TypeError("StartWorkflowTool requires a resolved ToolSpec")
        workflow = (spec.config or {}).get("workflow")
        if not workflow or not isinstance(workflow, str):
            raise ValueError(
                "StartWorkflowTool requires ToolSpec.config['workflow'] (string)"
            )

        initial_text: Optional[str] = None
        if isinstance(args, dict):
            # Prefer "text" to align with existing tests and conventions; accept "initial_text" as a fallback.
            if isinstance(args.get("text"), str):
                initial_text = args.get("text")
            elif isinstance(args.get("initial_text"), str):
                initial_text = args.get("initial_text")

        return ToolStartWorkflowResponse(workflow=workflow, initial_text=initial_text)

    def openapi_spec(self, project, spec: ToolSpec) -> Dict[str, Any]:
        workflow_name = (spec.config or {}).get("workflow")
        wf_desc: Optional[str] = None
        try:
            if project and getattr(project, "settings", None) and workflow_name:
                wf = project.settings.workflows.get(workflow_name)
                wf_desc = getattr(wf, "description", None) if wf else None
        except Exception:
            wf_desc = None

        if workflow_name:
            if wf_desc:
                desc = (
                    f"Start the '{workflow_name}' workflow. {wf_desc} "
                    "Optionally pass 'text' to send as the user's first message in the child workflow."
                )
            else:
                desc = (
                    f"Start the '{workflow_name}' workflow. "
                    "Optionally pass 'text' to send as the user's first message in the child workflow."
                )
        else:
            desc = (
                "Start a child workflow. "
                "Optionally pass 'text' to send as the user's first message in the child workflow."
            )

        return {
            "name": self.name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Optional initial user text for the child workflow",
                    },
                },
                "additionalProperties": False,
            },
        }