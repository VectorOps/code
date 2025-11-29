import asyncio
from types import SimpleNamespace
import pytest
from vocode.ui.base import UIState
from vocode.ui.proto import UIPacketEnvelope, UIPacketRunEvent, UIPacketStatus, UIPacketRunInput, UIPacketUIReset
from vocode.runner.models import ReqMessageRequest, ReqFinalMessage, RunInput, RespMessage
from vocode.state import RunnerStatus, Message
from vocode.testing.ui import (
    FakeProject,
    FakeRunner,
    recv_skip_node_status,
    respond_message,
    respond_approval,
    mk_interim,
    mk_final,
)


_recv_skip_node_status = recv_skip_node_status


def test_ui_state_basic_flow(monkeypatch):
    async def scenario():
        # Patch Runner used inside UIState to our fake
        from vocode.ui import base as ui_base

        monkeypatch.setattr(ui_base, "Runner", FakeRunner)

        # Prepare a workflow stub with a scripted sequence of events
        script = [
            # Step 1: interim message, no input, status running
            ("node1", mk_interim("hello"), False, RunnerStatus.running),
            # Step 2: input request, status waiting_input
            ("node2", ReqMessageRequest(), True, RunnerStatus.waiting_input),
            # Step 3: final message, status running
            ("node3", mk_final("bye"), False, RunnerStatus.running),
        ]
        wf = SimpleNamespace(name="wf", script=script)
        project = FakeProject()
        ui = UIState(project)

        await ui.start(wf)

        # 0) Initial UI Reset
        reset_env = await _recv_skip_node_status(ui)
        assert isinstance(reset_env.payload, UIPacketUIReset)

        # 1) First status emitted on entering running
        msg1_env = await _recv_skip_node_status(ui)
        assert isinstance(msg1_env.payload, UIPacketStatus)
        assert msg1_env.payload.prev is None
        assert msg1_env.payload.curr == RunnerStatus.running

        # 2) First run event (node1, no input)
        req1_env = await _recv_skip_node_status(ui)
        assert isinstance(req1_env.payload, UIPacketRunEvent)
        assert req1_env.payload.event.node == "node1"
        assert req1_env.payload.event.input_requested is False

        # 3) Status waiting_input before next event
        msg2_env = await _recv_skip_node_status(ui)
        assert isinstance(msg2_env.payload, UIPacketStatus)
        assert msg2_env.payload.prev == RunnerStatus.running
        assert msg2_env.payload.curr == RunnerStatus.waiting_input

        # 4) Second run event (node2, input requested)
        req2_env = await _recv_skip_node_status(ui)
        assert isinstance(req2_env.payload, UIPacketRunEvent)
        assert req2_env.payload.event.node == "node2"
        assert req2_env.payload.event.input_requested is True
        # While waiting for input, the driver is blocked; current_node_name is stable.
        # This avoids the race present for non-input events.
        assert ui.current_node_name == "node2"

        # Send a mismatched response; UIState should ignore it and keep waiting
        await ui.send(
            UIPacketEnvelope(msg_id=999, payload=UIPacketRunInput(input=RunInput()))
        )
        # Now send the proper response using helper
        await respond_message(ui, req2_env.msg_id, Message(role="user", text="ok"))

        # 5) Back to running before final event
        msg3_env = await _recv_skip_node_status(ui)
        assert isinstance(msg3_env.payload, UIPacketStatus)
        assert msg3_env.payload.prev == RunnerStatus.waiting_input
        assert msg3_env.payload.curr == RunnerStatus.running

        # 6) Third run event (node3, final)
        req3_env = await _recv_skip_node_status(ui)
        assert isinstance(req3_env.payload, UIPacketRunEvent)
        assert req3_env.payload.event.node == "node3"
        assert req3_env.payload.event.input_requested is False
        # Ensure message ids are increasing across run events despite interleaved packets (e.g., logs)
        assert req1_env.msg_id < req2_env.msg_id < req3_env.msg_id

        # 7) Final status finished
        msg4_env = await _recv_skip_node_status(ui)
        assert isinstance(msg4_env.payload, UIPacketStatus)
        assert msg4_env.payload.prev == RunnerStatus.running
        assert msg4_env.payload.curr == RunnerStatus.finished
        assert ui.is_active() is False

        # Verify FakeRunner captured the input
        assert isinstance(ui.runner, FakeRunner)
        assert len(ui.runner.received_inputs) == 1
        run_input = ui.runner.received_inputs[0]
        assert isinstance(run_input, RunInput)
        assert run_input.response is not None

    asyncio.run(scenario())


def test_last_final_message_and_handoff(monkeypatch):
    async def scenario():
        from vocode.ui import base as ui_base

        monkeypatch.setattr(ui_base, "Runner", FakeRunner)

        # One interim, then a final
        script = [
            ("node1", mk_interim("hello"), False, RunnerStatus.running),
            ("node2", mk_final("done"), False, RunnerStatus.running),
        ]
        wf = SimpleNamespace(name="wf-final", script=script)
        project = FakeProject()
        ui = UIState(project)

        await ui.start(wf)

        # Drain reset and first status, interim event and second status
        # Drain reset and first status
        await _recv_skip_node_status(ui)  # UIPacketUIReset
        await _recv_skip_node_status(ui)  # status running

        # Consume events until runner finishes; last_final_message should be
        # updated when the final_message event is processed.
        seen_final = False
        while True:
            env = await _recv_skip_node_status(ui)
            if isinstance(env.payload, UIPacketStatus) and env.payload.curr == RunnerStatus.finished:
                break
            if isinstance(env.payload, UIPacketRunEvent):
                ev = env.payload.event.event
                if isinstance(ev, ReqFinalMessage):
                    assert ev.message is not None
                    assert ev.message.text == "done"
                    seen_final = True

        assert seen_final
        assert ui.last_final_message is not None
        assert ui.last_final_message.text == "done"

    asyncio.run(scenario())


def test_ui_state_stop_while_waiting(monkeypatch):
    async def scenario():
        from vocode.ui import base as ui_base

        monkeypatch.setattr(ui_base, "Runner", FakeRunner)

        script = [
            # Single step that requests input; status directly waiting_input
            ("node1", ReqMessageRequest(), True, RunnerStatus.waiting_input),
        ]
        wf = SimpleNamespace(name="wf-stop", script=script)
        project = FakeProject()
        ui = UIState(project)
        await ui.start(wf)

        # Consume UI reset packet
        reset_env = await _recv_skip_node_status(ui)
        assert isinstance(reset_env.payload, UIPacketUIReset)

        # First status is waiting_input (no prior running step)
        s1_env = await _recv_skip_node_status(ui)
        assert isinstance(s1_env.payload, UIPacketStatus)
        assert s1_env.payload.prev is None
        assert s1_env.payload.curr == RunnerStatus.waiting_input

        ev1_env = await _recv_skip_node_status(ui)
        assert isinstance(ev1_env.payload, UIPacketRunEvent)
        assert ev1_env.payload.event.input_requested is True

        # Issue stop; driver should emit stopped and exit
        await ui.stop()
        s2_env = await _recv_skip_node_status(ui)
        assert isinstance(s2_env.payload, UIPacketStatus)
        assert s2_env.payload.prev == RunnerStatus.waiting_input
        assert s2_env.payload.curr == RunnerStatus.stopped
        assert ui.is_active() is False

    asyncio.run(scenario())


def test_ui_state_replace_input_guard(monkeypatch):
    async def scenario():
        from vocode.ui import base as ui_base

        monkeypatch.setattr(ui_base, "Runner", FakeRunner)

        script = [
            ("node1", ReqMessageRequest(), True, RunnerStatus.waiting_input),
        ]
        wf = SimpleNamespace(name="wf-guard", script=script)
        project = FakeProject()
        ui = UIState(project)
        await ui.start(wf)

        # Drain reset, status and the input request event
        await _recv_skip_node_status(ui)  # UIPacketUIReset
        await _recv_skip_node_status(ui)  # UIPacketStatus waiting_input
        req_env = await _recv_skip_node_status(ui)  # UIPacketRunEvent (input requested)
        assert isinstance(req_env.payload, UIPacketRunEvent)
        assert req_env.payload.event.input_requested

        # Attempt replacing last user input while waiting for input should fail
        with pytest.raises(RuntimeError):
            await ui.replace_user_input(
                RespMessage(message=Message(role="user", text="new"))
            )

        # Cleanup: cancel the driver to avoid leaks
        await ui.cancel()
        # Receive final canceled status
        s_env = await _recv_skip_node_status(ui)
        assert isinstance(s_env.payload, UIPacketStatus)
        assert s_env.payload.curr == RunnerStatus.canceled

    asyncio.run(scenario())


def test_project_state_reset_clears(monkeypatch):
    async def scenario():
        from vocode.ui import base as ui_base

        monkeypatch.setattr(ui_base, "Runner", FakeRunner)

        # Use a workflow stub with a single final; reset will rebuild from settings via start_by_name
        wf_name = "wf-project-state"
        script = [("node1", mk_final("done"), False, RunnerStatus.running)]
        wf = SimpleNamespace(name=wf_name, script=script)
        project = FakeProject()
        ui = UIState(project)

        await ui.start(wf)

        # Drain first status to ensure driver started
        reset_env = await _recv_skip_node_status(ui)
        assert isinstance(reset_env.payload, UIPacketUIReset)
        s_env = await _recv_skip_node_status(ui)
        assert isinstance(s_env.payload, UIPacketStatus)

        # Basic set/get/delete
        ps = ui.project.project_state
        ps.set("k1", {"v": 1})
        assert ps.get("k1") == {"v": 1}
        ps.delete("k1")
        assert ps.get("k1") is None
        # Set again to test clearing on reset
        ps.set("k2", "persist-me")
        assert ps.get("k2") == "persist-me"

        # Reset should clear project-level state
        await ui.reset()

        assert ps.get("k2") is None
        # After reset, UIState sends a UIReset (and may emit status for the restarted run). Drain them.
        s_end_reset_env = await _recv_skip_node_status(ui)
        assert isinstance(s_end_reset_env.payload, UIPacketUIReset)
        s_post_reset_env = await _recv_skip_node_status(ui)
        assert isinstance(s_post_reset_env.payload, UIPacketStatus)

        # Cleanup: cancel the restarted driver to avoid leaks.
        # Note: cancel() does not guarantee any additional UI messages.
        await ui.cancel()

    asyncio.run(scenario())
