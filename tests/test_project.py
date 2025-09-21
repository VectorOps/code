import pytest
from pathlib import Path
from pydantic import BaseModel
from typing import Any, Dict, Union, TYPE_CHECKING, Optional

from vocode.project import init_project
from vocode.tools import BaseTool, register_tool

if TYPE_CHECKING:
    from vocode.project import Project


def _write_config(base: Path, content: str) -> Path:
    cfg_dir = base / ".vocode"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "config.yaml"
    cfg_path.write_text(content, encoding="utf-8")
    return cfg_path


def test_project_loads_settings_and_instantiates_enabled_tools(tmp_path, monkeypatch):
    # Isolate tool registry for this test
    monkeypatch.setattr("vocode.tools._registry", {}, raising=False)

    # Define simple tools
    class EchoTool(BaseTool):
        name = "echo"

        async def run(self, project: "Project", args: BaseModel) -> Optional[str]:
            pass

        def openapi_spec(self) -> Dict[str, Any]:
            return {}

    class NeedsTool(BaseTool):
        name = "needs"

        async def run(self, project: "Project", args: BaseModel) -> Optional[str]:
            pass

        def openapi_spec(self) -> Dict[str, Any]:
            return {}

    class DisabledTool(BaseTool):
        name = "disabled"

        async def run(self, project: "Project", args: BaseModel) -> Optional[str]:
            pass

        def openapi_spec(self) -> Dict[str, Any]:
            return {}

    register_tool("echo", EchoTool())
    register_tool("needs", NeedsTool())
    register_tool("disabled", DisabledTool())

    # Write project config
    _write_config(
        tmp_path,
        """
workflows:
  t:
    nodes: []
    edges: []
tools:
  - name: echo
    enabled: true
  - name: disabled
    enabled: false
  - name: needs
    enabled: true
""",
    )

    proj = init_project(tmp_path)

    # Workflows loaded
    assert "t" in proj.settings.workflows
    assert proj.settings.workflows["t"].nodes == []
    assert proj.settings.workflows["t"].edges == []

    # Tools: only enabled ones are instantiated
    assert set(proj.tools.keys()) & {"echo", "needs"}

    echo = proj.tools["echo"]
    assert isinstance(echo, EchoTool)

    needs = proj.tools["needs"]
    assert isinstance(needs, NeedsTool)


