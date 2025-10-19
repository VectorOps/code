import pytest
from pathlib import Path
from typing import Any, Dict
import sys
import os

from vocode.project import init_project
from .fakes import FakeMCPClient, make_fake_mcp_client_creator


TESTS_DIR = Path(__file__).parent.resolve()


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
  servers:
    mcp:
      url: "tcp://127.0.0.1:8000"
  tools_whitelist: ["mcp_echo"]
""",
    )
    proj = init_project(tmp_path)
    assert proj.settings is not None and proj.settings.mcp is not None
    assert "mcp" in proj.settings.mcp.servers
    assert proj.settings.mcp.servers["mcp"].url == "tcp://127.0.0.1:8000"
    assert proj.settings.mcp.tools_whitelist == ["mcp_echo"]


@pytest.mark.asyncio
async def test_mcp_manager_registers_and_cleans_tools(tmp_path: Path, monkeypatch):
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
  servers:
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
        # Support both legacy str and new ToolTextResponse
        res_text = res.text if hasattr(res, "text") else res
        assert res_text == ""

    # Post-exit, the tool should be unregistered from the global registry
    from vocode.tools import get_all_tools

    assert "mcp_echo" not in get_all_tools()


@pytest.mark.asyncio
async def test_mcp_manager_spawns_real_fastmcp_echo_and_cleans_up(
    tmp_path: Path, monkeypatch
):
    # Skip if FastMCP is not available in the environment
    fastmcp = pytest.importorskip("fastmcp")

    # Isolate registry for this test
    monkeypatch.setattr("vocode.tools._registry", {}, raising=False)

    # 1) Locate the pre-written echo server and define path for its PID file.
    srv_path = TESTS_DIR / "fixtures" / "mcp_echo_server.py"
    pidfile = tmp_path / "echo_server.pid"

    # 2) Write project config to use MCP command mode to spawn the above server.
    cfg_dir = tmp_path / ".vocode"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.yaml").write_text(
        f"""
workflows:
  w:
    nodes: []
    edges: []
mcp:
  servers:
    assistant:
      command: "{sys.executable}"
      args: ["-u", "{str(srv_path)}", "{str(pidfile)}"]
tools:
  - name: mcp_echo
    enabled: true
""",
        encoding="utf-8",
    )

    # 3) Start a sandboxed project; it should spawn the FastMCP server via fastmcp.spawn(...)
    from vocode.testing import ProjectSandbox

    async with ProjectSandbox.create(tmp_path) as project:
        # Tool is registered under its name from the server
        assert "mcp_echo" in project.tools

        # Ensure server started by checking pidfile exists and contains a PID
        assert pidfile.exists(), "Server pidfile not created"
        try:
            pid = int(pidfile.read_text(encoding="utf-8").strip())
        except Exception:
            pytest.fail("Invalid server pidfile contents")

        # Optionally check process appears alive (best-effort, POSIX only)
        if hasattr(os, "kill") and os.name != "nt":
            try:
                os.kill(pid, 0)  # does not signal, checks existence/permission
                alive = True
            except OSError:
                alive = False
            assert alive, "Spawned server process not alive"

        # Invoke the MCP tool through the proxy
        res = await project.tools["mcp_echo"].run(project, {"text": "hello"})
        # Support both legacy str and new ToolTextResponse
        res_text = res.text if hasattr(res, "text") else res
        assert res_text == "hello"

    # 4) After sandbox exit, project.shutdown() should have stopped MCPManager and terminated the server.
    #    The server script removes the pidfile on exit, so its absence implies process shutdown.
    assert (
        not pidfile.exists()
    ), "Server did not clean up pidfile (likely still running)"

    # And the tool should be unregistered from global registry
    from vocode.tools import get_all_tools

    assert "mcp_echo" not in get_all_tools()
