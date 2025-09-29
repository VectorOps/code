import asyncio
from types import SimpleNamespace
from typing import List, Optional, Tuple

import pytest

from vocode.ui.base import UIState
from vocode.ui.proto import (
    UIPacketEnvelope,
    UIPacketRunEvent,
    UIPacketStatus,
    UIPacketRunInput,
)
from vocode.runner.models import (
    ReqMessageRequest,
    ReqInterimMessage,
    ReqFinalMessage,
    RespMessage,
    RespApproval,
    RespPacket,
    RunInput,
    RunEvent,
    TokenUsageTotals,
)
from vocode.state import RunnerStatus, Message, Activity, ActivityType
from vocode.commands import CommandManager
from vocode.project import ProjectState


async def respond_packet(
    ui: UIState, source_msg_id: int, packet: Optional[RespPacket]
) -> None:
    inp = RunInput(response=packet) if packet is not None else RunInput(response=None)
    await ui.send(
        UIPacketEnvelope(
            msg_id=ui.next_client_msg_id(),
            source_msg_id=source_msg_id,
            payload=UIPacketRunInput(input=inp),
        )
    )


async def respond_message(ui: UIState, source_msg_id: int, message: Message) -> None:
    await respond_packet(ui, source_msg_id, RespMessage(message=message))


async def respond_approval(ui: UIState, source_msg_id: int, approved: bool) -> None:
    await respond_packet(ui, source_msg_id, RespApproval(approved=approved))


class FakeProject:
    def __init__(self, settings=None):
        wf_name = "wf-project-state"
        wf_cfg = SimpleNamespace(nodes=[], edges=[])
        self.settings = settings or SimpleNamespace(workflows={wf_name: wf_cfg})
        self.commands = CommandManager()
        self.project_state = ProjectState()
        self.llm_usage = TokenUsageTotals()


class FakeRunner:
    """
    Minimal fake runner to drive UIState driver loop.
    Script is a list of tuples:
      (node_name, req_packet, input_requested: bool, status_before: RunnerStatus)
    """

    def __init__(self, workflow, project, initial_message=None):
        self.status: RunnerStatus = RunnerStatus.idle
        self.script: List[Tuple[str, object, bool, RunnerStatus]] = getattr(
            workflow, "script", []
        )
        self.received_inputs: List[Optional[RunInput]] = []
        self.rewound: Optional[int] = None
        self.replaced_input = None
        # Simple runtime_graph for UIState lookup. workflow may provide node_hide_map mapping node->bool.
        node_hide_map = getattr(workflow, "node_hide_map", {})
        self.runtime_graph = SimpleNamespace(
            get_runtime_node_by_name=lambda name: SimpleNamespace(
                model=SimpleNamespace(hide_final_output=bool(node_hide_map.get(name, False)))
            )
        )

    def cancel(self) -> None:
        self.status = RunnerStatus.canceled

    def stop(self) -> None:
        self.status = RunnerStatus.stopped

    async def rewind(self, task, n: int = 1) -> None:
        self.rewound = n

    def replace_user_input(self, task, response, step_index=None) -> None:
        self.replaced_input = response

    async def run(self, assignment):
        sent = None
        try:
            for node_name, req_packet, input_requested, status_before in self.script:
                # Set status for this step before yielding the event, so UIState emits it first.
                self.status = status_before
                ev = RunEvent(
                    node=node_name,
                    execution=Activity(type=ActivityType.executor),
                    event=req_packet,
                    input_requested=input_requested,
                )
                sent = yield ev
                if input_requested:
                    # UIState sends RunInput (or None) back
                    self.received_inputs.append(sent)
            # Finish cleanly
            self.status = RunnerStatus.finished
        finally:
            return


def _mk_interim(text: str) -> ReqInterimMessage:
    return ReqInterimMessage(message=Message(role="agent", text=text))


def _mk_final(text: str) -> ReqFinalMessage:
    return ReqFinalMessage(message=Message(role="agent", text=text), outcome_name=None)


def test_ui_state_basic_flow(monkeypatch):
    async def scenario():
        # Patch Runner used inside UIState to our fake
        from vocode.ui import base as ui_base

        monkeypatch.setattr(ui_base, "Runner", FakeRunner)

        # Prepare a workflow stub with a scripted sequence of events
        script = [
            # Step 1: interim message, no input, status running
            ("node1", _mk_interim("hello"), False, RunnerStatus.running),
            # Step 2: input request, status waiting_input
            ("node2", ReqMessageRequest(), True, RunnerStatus.waiting_input),
            # Step 3: final message, status running
            ("node3", _mk_final("bye"), False, RunnerStatus.running),
        ]
        wf = SimpleNamespace(name="wf", script=script)
        project = FakeProject()
        ui = UIState(project)

        await ui.start(wf)

        # 1) First status emitted on entering running
        msg1_env = await ui.recv()
        assert isinstance(msg1_env.payload, UIPacketStatus)
        assert msg1_env.payload.prev is None
        assert msg1_env.payload.curr == RunnerStatus.running

        # 2) First run event (node1, no input)
        req1_env = await ui.recv()
        assert isinstance(req1_env.payload, UIPacketRunEvent)
        assert req1_env.payload.event.node == "node1"
        assert req1_env.msg_id == 2
        assert req1_env.payload.event.input_requested is False

        # 3) Status waiting_input before next event
        msg2_env = await ui.recv()
        assert isinstance(msg2_env.payload, UIPacketStatus)
        assert msg2_env.payload.prev == RunnerStatus.running
        assert msg2_env.payload.curr == RunnerStatus.waiting_input

        # 4) Second run event (node2, input requested)
        req2_env = await ui.recv()
        assert isinstance(req2_env.payload, UIPacketRunEvent)
        assert req2_env.payload.event.node == "node2"
        assert req2_env.msg_id == 4
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
        msg3_env = await ui.recv()
        assert isinstance(msg3_env.payload, UIPacketStatus)
        assert msg3_env.payload.prev == RunnerStatus.waiting_input
        assert msg3_env.payload.curr == RunnerStatus.running

        # 6) Third run event (node3, final)
        req3_env = await ui.recv()
        assert isinstance(req3_env.payload, UIPacketRunEvent)
        assert req3_env.payload.event.node == "node3"
        assert req3_env.msg_id == 6
        assert req3_env.payload.event.input_requested is False

        # 7) Final status finished
        msg4_env = await ui.recv()
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

        # First status is waiting_input (no prior running step)
        s1_env = await ui.recv()
        assert isinstance(s1_env.payload, UIPacketStatus)
        assert s1_env.payload.prev is None
        assert s1_env.payload.curr == RunnerStatus.waiting_input

        ev1_env = await ui.recv()
        assert isinstance(ev1_env.payload, UIPacketRunEvent)
        assert ev1_env.payload.event.input_requested is True

        # Issue stop; driver should emit stopped and exit
        await ui.stop()
        s2_env = await ui.recv()
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

        # Drain status and the input request event
        await ui.recv()  # UIPacketStatus waiting_input
        req_env = await ui.recv()  # UIPacketRunEvent (input requested)
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
        s_env = await ui.recv()
        assert isinstance(s_env.payload, UIPacketStatus)
        assert s_env.payload.curr == RunnerStatus.canceled

    asyncio.run(scenario())


def test_project_state_reset_clears(monkeypatch):
    async def scenario():
        from vocode.ui import base as ui_base

        monkeypatch.setattr(ui_base, "Runner", FakeRunner)

        # Use a workflow stub with a single final; reset will rebuild from settings via start_by_name
        wf_name = "wf-project-state"
        script = [
            ("node1", _mk_final("done"), False, RunnerStatus.running),
        ]
        wf = SimpleNamespace(name=wf_name, script=script)
        project = FakeProject()
        ui = UIState(project)

        await ui.start(wf)

        # Drain first status to ensure driver started
        s_env = await ui.recv()
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

        # Cleanup: cancel the restarted driver to avoid leaks
        await ui.cancel()
        s_end_env = await ui.recv()
        assert isinstance(s_end_env.payload, UIPacketStatus)

    asyncio.run(scenario())
