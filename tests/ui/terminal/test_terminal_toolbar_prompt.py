import types

from prompt_toolkit.formatted_text import to_formatted_text

from vocode.state import RunnerStatus
from vocode.ui.terminal.toolbar import build_toolbar


class _DummyUI:
    def __init__(self) -> None:
        self.status = RunnerStatus.idle
        self.selected_workflow_name = None
        self.current_node_name = None
        self.project_op = types.SimpleNamespace(message=None, progress=None, total=None)

    def is_active(self) -> bool:  # pragma: no cover - not used in this test
        return False


class _DummyStackFrame:
    def __init__(self, workflow: str, node: str | None = None) -> None:
        self.workflow = workflow
        self.node = node
        self.node_description = None


class _DummyApp:
    def __init__(self) -> None:
        self.ui = _DummyUI()
        self.workflow_stack = None
        self.llm_usage_global = None
        self.llm_usage_session = None
        self.llm_usage_node = None
        self.llm_usage_context = None


def _render_toolbar_text(app: _DummyApp) -> str:
    fragments = build_toolbar(app, pending_req=None)
    return "".join(text for _, text in to_formatted_text(fragments))


def test_toolbar_uses_innermost_workflow_from_stack():
    app = _DummyApp()
    # Simulate nested workflows: outer -> inner
    app.workflow_stack = [
        _DummyStackFrame("outer-wf", node="outer-node"),
        _DummyStackFrame("inner-wf", node="inner-node"),
    ]
    app.ui.status = RunnerStatus.running

    text = _render_toolbar_text(app)

    # Full stack should be rendered as "outer-wf@outer-node > inner-wf@inner-node"
    assert "outer-wf@outer-node" in text
    assert "inner-wf@inner-node" in text
    assert "@outer-node > inner-wf@inner-node" in text
