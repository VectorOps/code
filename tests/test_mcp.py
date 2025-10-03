import pytest
from pathlib import Path
from typing import Any, Dict

from vocode.project import init_project


def _write_cfg(base: Path, content: str) -> Path:
    d = base / ".vocode"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "config.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def test_settings_mcp_parsing(tmp_path: Path):
    _write_cfg(
        tmp_path,
        """
workflows: {}
mcp:
  url: "tcp://127.0.0.1:8000"
  tools_whitelist: ["mcp_echo"]
""",
    )
    proj = init_project(tmp_path)
    assert proj.settings is not None and proj.settings.mcp is not None
    assert proj.settings.mcp.url == "tcp://127.0.0.1:8000"
    assert proj.settings.mcp.tools_whitelist == ["mcp_echo"]


@pytest.mark.asyncio
async def test_mcp_manager_registers_and_cleans_tools(tmp_path: Path, monkeypatch):
    # Fake MCP client that exposes one tool
    class FakeMCPClient:
        def list_tools(self):
            return [
                {
                    "name": "mcp_echo",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                }
            ]

        async def call_tool(self, name: str, arguments: Dict[str, Any]):
            return arguments.get("text", "")

        async def aclose(self):
            pass

    async def fake_create_client(self):
        return FakeMCPClient()

    monkeypatch.setattr("vocode.mcp.manager.MCPManager._create_client", fake_create_client, raising=True)
    # Isolate registry
    monkeypatch.setattr("vocode.tools._registry", {}, raising=False)

    _write_cfg(
        tmp_path,
        """
workflows:
  w:
    nodes: []
    edges: []
mcp:
  url: "tcp://localhost:9999"
tools:
  - name: mcp_echo
    enabled: true
""",
    )

    from vocode.testing import ProjectSandbox
    async with ProjectSandbox.create(tmp_path) as project:
        # Tools registered
        assert "mcp_echo" in project.tools
        tool = project.tools["mcp_echo"]
        # Run the tool with empty args (tools accept Any; dict preferred)
        res = await tool.run(project, {})
        assert res == ""

    # Post-exit, the tool should be unregistered from the global registry
    from vocode.tools import get_all_tools
    assert "mcp_echo" not in get_all_tools()