from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import UIState

# Canonical autocompletion provider names
PROVIDER_WORKFLOW_LIST = "workflow_list"
PROVIDER_FILELIST = "filelist"


async def ac_workflow_list(ui: "UIState", params: Dict[str, Any]) -> List[str]:
    names = ui.list_workflows()
    prefix = str(params.get("prefix", "") or "")
    if prefix:
        return [n for n in names if n.startswith(prefix)]
    return names


async def ac_filelist(ui: "UIState", params: Dict[str, Any]) -> List[str]:
    try:
        kp = ui.project.know
        needle = str(params.get("prefix") or params.get("needle") or "").strip()
        if not needle:
            return []
        limit = int(params.get("limit") or 5)
        # Pass repo_ids from KnowProjectManager
        repo_ids: Optional[List[str]] = kp.pm.repo_ids
        files = await kp.data.file.filename_complete(
            needle=needle, repo_ids=repo_ids, limit=limit
        )
        return [f.path for f in files if f.path]
    except Exception:
        # TODO: How to return an error
        return []
