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

    async def run(self, spec: ToolSpec, args: Any):
        if not isinstance(spec, ToolSpec):
            raise TypeError("StartWorkflowTool requires a resolved ToolSpec")

        workflow = (spec.config or {}).get("workflow")
        if not workflow or not isinstance(workflow, str):
            raise ValueError(
                "StartWorkflowTool requires ToolSpec.config['workflow'] (string)"
            )

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
        if isinstance(args, dict):
            # Prefer "text" to align with existing tests and conventions; accept "initial_text" as a fallback.
            if isinstance(args.get("text"), str):
                initial_text = args.get("text")
            elif isinstance(args.get("initial_text"), str):
                initial_text = args.get("initial_text")

        return ToolStartWorkflowResponse(workflow=workflow, initial_text=initial_text)

    async def openapi_spec(self, spec: ToolSpec) -> Dict[str, Any]:
        workflow_name = (spec.config or {}).get("workflow")
        wf_desc: Optional[str] = None
        try:
            if self.prj and workflow_name:
                wf = self.prj.settings.workflows.get(workflow_name)
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
