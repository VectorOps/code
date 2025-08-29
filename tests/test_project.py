import pytest
from pathlib import Path
from pydantic import BaseModel, ValidationError

from vocode.project import init_project
from vocode.tools import BaseTool


def _write_config(base: Path, content: str) -> Path:
    cfg_dir = base / ".vocode"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "config.yaml"
    cfg_path.write_text(content, encoding="utf-8")
    return cfg_path


def test_project_loads_settings_and_instantiates_enabled_tools(tmp_path, monkeypatch):
    # Isolate tool registry for this test
    monkeypatch.setattr(BaseTool, "_registry", {}, raising=False)

    # Define a simple tool that records the received config
    class EchoTool(BaseTool):
        name = "echo"
        def __init__(self, config=None):
            super().__init__(config)
            self.seen = dict(self.config)

    # Define a tool that validates its config with a pydantic model
    class NeedsConfig(BaseTool):
        name = "needs"
        class Cfg(BaseModel):
            url: str
            timeout: int = 10
        def __init__(self, config=None):
            super().__init__(config)
            from vocode.settings import build_model_from_settings
            self.cfg = build_model_from_settings(self.config, NeedsConfig.Cfg)

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
    config:
      foo: bar
  - name: disabled
    enabled: false
    config:
      whatever: 1
  - name: needs
    enabled: true
    config:
      url: https://example.com
      timeout: 5
""",
    )

    proj = init_project(tmp_path)

    # Workflows loaded
    assert "t" in proj.settings.workflows
    assert proj.settings.workflows["t"].nodes == []
    assert proj.settings.workflows["t"].edges == []

    # Tools: only enabled ones are instantiated
    assert set(proj.tools.keys()) == {"echo", "needs"}

    echo = proj.tools["echo"]
    assert isinstance(echo, EchoTool)
    assert echo.config == {"foo": "bar"}
    assert echo.seen == {"foo": "bar"}

    needs = proj.tools["needs"]
    assert isinstance(needs, NeedsConfig)
    assert needs.cfg.url == "https://example.com"
    assert needs.cfg.timeout == 5


def test_project_tool_validation_error(tmp_path, monkeypatch):
    # Isolate tool registry
    monkeypatch.setattr(BaseTool, "_registry", {}, raising=False)

    class NeedsConfig(BaseTool):
        name = "needs"
        class Cfg(BaseModel):
            url: str
        def __init__(self, config=None):
            super().__init__(config)
            from vocode.settings import build_model_from_settings
            # Missing 'url' should raise ValidationError
            self.cfg = build_model_from_settings(self.config, NeedsConfig.Cfg)

    _write_config(
        tmp_path,
        """
workflows:
  w:
    nodes: []
    edges: []
tools:
  - name: needs
    enabled: true
    config: {}
""",
    )

    with pytest.raises(ValidationError):
        _ = init_project(tmp_path)
