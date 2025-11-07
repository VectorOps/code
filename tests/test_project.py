import pytest
from pathlib import Path
from pydantic import BaseModel
from typing import Any, Dict, Union, TYPE_CHECKING, Optional

from vocode.project import init_project
from vocode.tools import BaseTool, register_tool
from .fakes import FakeMCPClient, make_fake_mcp_client_creator
from vocode import project

if TYPE_CHECKING:
    from vocode.project import Project


def _write_config(base: Path, content: str) -> Path:
    cfg_dir = base / ".vocode"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "config.yaml"
    cfg_path.write_text(content, encoding="utf-8")
    return cfg_path


class DummyKnow:
    async def start(self, *_args, **_kwargs):
        return None

    async def shutdown(self):
        return None

    async def refresh(self, repo=None, **_kwargs):
        return None


@pytest.fixture(autouse=True)
def patch_know(monkeypatch):
    # Prevent KnowProject from doing heavy initialization in tests.
    monkeypatch.setattr(project, "KnowProject", lambda *args, **kwargs: DummyKnow())
    yield


def test_project_loads_settings_and_instantiates_enabled_tools(tmp_path, monkeypatch):
    # Isolate tool registry for this test
    monkeypatch.setattr("vocode.tools._registry", {}, raising=False)

    # Define simple tools
    class EchoTool(BaseTool):
        name = "echo"

        async def run(self, project: "Project", spec, args: BaseModel):
            pass

        async def openapi_spec(self, project: "Project", spec) -> Dict[str, Any]:
            return {}

    class NeedsTool(BaseTool):
        name = "needs"

        async def run(self, project: "Project", spec, args: BaseModel):
            pass

        async def openapi_spec(self, project: "Project", spec) -> Dict[str, Any]:
            return {}

    class DisabledTool(BaseTool):
        name = "disabled"

        async def run(self, project: "Project", spec, args: BaseModel):
            pass

        async def openapi_spec(self, project: "Project", spec) -> Dict[str, Any]:
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


@pytest.mark.asyncio
async def test_project_parses_mcp_settings_and_starts_manager(tmp_path, monkeypatch):
    # Isolate tool registry
    monkeypatch.setattr("vocode.tools._registry", {}, raising=False)

    # Configure the fake client
    fake_client = FakeMCPClient(
        tools=[
            {
                "name": "mcp_echo",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            }
        ]
    )

    monkeypatch.setattr(
        "vocode.mcp.manager.MCPManager._create_client",
        make_fake_mcp_client_creator(fake_client),
        raising=True,
    )

    # Write config enabling MCP (connect mode via URL) and allow all tools
    _write_config(
        tmp_path,
        """
workflows:
  t:
    nodes: []
    edges: []
mcp:
  servers:
    mcp:
      url: "tcp://localhost:9999"
tools:
  - name: mcp_echo
    enabled: true
""",
    )

    from vocode.project import init_project

    proj = init_project(tmp_path)
    assert proj.settings is not None and proj.settings.mcp is not None
    assert "mcp" in proj.settings.mcp.servers
    assert proj.settings.mcp.servers["mcp"].url == "tcp://localhost:9999"

    # Start project to initialize MCP manager and register tools
    await proj.start()
    # Tool should be visible via project.tools (filtered by Settings.tools)
    assert "mcp_echo" in proj.tools

    # Shutdown cleans up
    await proj.shutdown()
    assert "mcp_echo" not in proj.tools
