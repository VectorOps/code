import asyncio
from types import SimpleNamespace
from typing import List, Optional, Tuple

import pytest

from vocode.ui.base import UIState
from vocode.ui.proto import UIReqRunEvent, UIReqStatus, UIRespRunInput
from vocode.runner.models import (
    ReqMessageRequest,
    ReqInterimMessage,
    ReqFinalMessage,
    RespMessage,
    RunInput,
    RunEvent,
)
from vocode.state import RunnerStatus, Message, Activity, ActivityType


class FakeRunner:
    """
    Minimal fake runner to drive UIState driver loop.
    Script is a list of tuples:
      (node_name, req_packet, input_requested: bool, status_before: RunnerStatus)
    """

    def __init__(self, workflow, project, initial_message=None):
        self.status: RunnerStatus = RunnerStatus.idle
        self.script: List[
            Tuple[str, object, bool, RunnerStatus]
        ] = getattr(workflow, "script", [])
        self.received_inputs: List[Optional[RunInput]] = []
        self.rewound: Optional[int] = None
        self.replaced_input = None

    def cancel(self) -> None:
        self.status = RunnerStatus.canceled

    def stop(self) -> None:
        self.status = RunnerStatus.stopped

    async def rewind(self, task, n: int = 1) -> None:
        self.rewound = n

    def replace_last_user_input(self, task, response) -> None:
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
        project = SimpleNamespace(settings=None)
        ui = UIState(project)

        await ui.start(wf)

        # 1) First status emitted on entering running
        msg1 = await ui.recv()
        assert isinstance(msg1, UIReqStatus)
        assert msg1.prev is None
        assert msg1.curr == RunnerStatus.running

        # 2) First run event (node1, no input)
        req1 = await ui.recv()
        assert isinstance(req1, UIReqRunEvent)
        assert req1.event.node == "node1"
        assert req1.req_id == 1
        assert req1.event.input_requested is False

        # 3) Status waiting_input before next event
        msg2 = await ui.recv()
        assert isinstance(msg2, UIReqStatus)
        assert msg2.prev == RunnerStatus.running
        assert msg2.curr == RunnerStatus.waiting_input

        # 4) Second run event (node2, input requested)
        req2 = await ui.recv()
        assert isinstance(req2, UIReqRunEvent)
        assert req2.event.node == "node2"
        assert req2.req_id == 2
        assert req2.event.input_requested is True
        # While waiting for input, the driver is blocked; current_node_name is stable.
        # This avoids the race present for non-input events.
        assert ui.current_node_name == "node2"

        # Send a mismatched response; UIState should ignore it and keep waiting
        await ui.send(UIRespRunInput(req_id=999, input=RunInput()))
        # Now send the proper response using helper
        await ui.respond_message(req2.req_id, Message(role="user", text="ok"))

        # 5) Back to running before final event
        msg3 = await ui.recv()
        assert isinstance(msg3, UIReqStatus)
        assert msg3.prev == RunnerStatus.waiting_input
        assert msg3.curr == RunnerStatus.running

        # 6) Third run event (node3, final)
        req3 = await ui.recv()
        assert isinstance(req3, UIReqRunEvent)
        assert req3.event.node == "node3"
        assert req3.req_id == 3
        assert req3.event.input_requested is False

        # 7) Final status finished
        msg4 = await ui.recv()
        assert isinstance(msg4, UIReqStatus)
        assert msg4.prev == RunnerStatus.running
        assert msg4.curr == RunnerStatus.finished
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
        project = SimpleNamespace(settings=None)
        ui = UIState(project)
        await ui.start(wf)

        # First status is waiting_input (no prior running step)
        s1 = await ui.recv()
        assert isinstance(s1, UIReqStatus)
        assert s1.prev is None
        assert s1.curr == RunnerStatus.waiting_input

        ev1 = await ui.recv()
        assert isinstance(ev1, UIReqRunEvent)
        assert ev1.event.input_requested is True

        # Issue stop; driver should emit stopped and exit
        await ui.stop()
        s2 = await ui.recv()
        assert isinstance(s2, UIReqStatus)
        assert s2.prev == RunnerStatus.waiting_input
        assert s2.curr == RunnerStatus.stopped
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
        project = SimpleNamespace(settings=None)
        ui = UIState(project)
        await ui.start(wf)

        # Drain status and the input request event
        await ui.recv()  # UIReqStatus waiting_input
        req = await ui.recv()  # UIReqRunEvent (input requested)
        assert isinstance(req, UIReqRunEvent)
        assert req.event.input_requested

        # Attempt replacing last user input while waiting for input should fail
        with pytest.raises(RuntimeError):
            await ui.replace_last_user_input(RespMessage(message=Message(role="user", text="new")))

        # Cleanup: cancel the driver to avoid leaks
        await ui.cancel()
        # Receive final canceled status
        s = await ui.recv()
        assert isinstance(s, UIReqStatus)
        assert s.curr == RunnerStatus.canceled

    asyncio.run(scenario())
