import asyncio
import contextlib
from types import SimpleNamespace
import pytest
from prompt_toolkit.document import Document
from vocode.ui.base import UIState
from vocode.ui.proto import UIPacketCompletionRequest, PACKET_COMPLETION_RESULT
from vocode.ui.rpc import RpcHelper
from vocode.ui.terminal.ac_client import make_canned_provider


class FakeProject:
    def __init__(self, settings=None):
        wf_cfg = SimpleNamespace(nodes=[], edges=[])
        self.settings = settings or SimpleNamespace(
            workflows={"wf1": wf_cfg, "wf2": wf_cfg}
        )
        self.commands = SimpleNamespace(clear=lambda: None)
        self.project_state = SimpleNamespace(clear=lambda: None)
        self.llm_usage = SimpleNamespace(
            prompt_tokens=0, completion_tokens=0, cost_dollars=0.0
        )


@pytest.mark.asyncio
async def test_workflow_list_completion_rpc():
    ui = UIState(FakeProject())
    rpc = RpcHelper(ui.send, "test-ac", id_generator=ui.next_client_msg_id)

    async def _pump_ui_to_rpc():
        while True:
            env = await ui.recv()
            # Resolve pending RPCs if this is a response; ignore other packets.
            rpc.handle_response(env)

    pump_task = asyncio.create_task(_pump_ui_to_rpc())

    # Direct RPC call
    res = await rpc.call(
        UIPacketCompletionRequest(name="workflow_list", params={"prefix": "wf"})
    )
    assert res is not None
    assert res.kind == PACKET_COMPLETION_RESULT
    assert res.ok
    assert set(res.suggestions) >= {"wf1", "wf2"}
    # Client-side provider wrapper
    provider = make_canned_provider(rpc, "workflow_list")
    doc = Document(text="/run ", cursor_position=5)
    suggestions = await provider(ui, doc, [], "")
    assert isinstance(suggestions, list)
    assert set(suggestions) >= {"wf1", "wf2"}

    pump_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await pump_task
