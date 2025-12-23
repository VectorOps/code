import asyncio
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vocode.ui.base import UIState  # for type hints in helpers
from vocode.ui.proto import (
    UIPacketEnvelope,
    UIPacketRunInput,
    UIPacketStatus,
    UIPacketLog,
)
from vocode.runner.models import (
    ReqInterimMessage,
    ReqFinalMessage,
    ReqMessageRequest,
    RespMessage,
    RespApproval,
    RespPacket,
    RunEvent,
    RunInput,
    TokenUsageTotals,
)
from vocode.settings import Settings, WorkflowConfig, LoggingSettings
from vocode.state import Message, RunnerStatus, Activity, ActivityType
from vocode.project import ProjectState


# -----------------------------
# High-level UI interaction helpers
# -----------------------------
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


async def recv_skip_node_status(ui: UIState) -> UIPacketEnvelope:
    """
    Receive next envelope, skipping:
    - UIPacketLog (emitted by logging interceptor), and
    - UIPacketStatus that include node transition fields.
    """
    while True:
        env = await ui.recv()
        if isinstance(env.payload, UIPacketLog):
            continue
        if isinstance(env.payload, UIPacketStatus):
            if getattr(env.payload, "prev_node", None) or getattr(
                env.payload, "curr_node", None
            ):
                continue
        return env


# -----------------------------
# Packet builders for scripted FakeRunner
# -----------------------------
def mk_interim(text: str) -> ReqInterimMessage:
    return ReqInterimMessage(message=Message(role="agent", text=text))


def mk_final(text: str) -> ReqFinalMessage:
    return ReqFinalMessage(message=Message(role="agent", text=text), outcome_name=None)


# -----------------------------
# Minimal test Project
# -----------------------------
class _TestCommands:
    def __init__(self) -> None:
        self._q: asyncio.Queue = asyncio.Queue()

    def subscribe(self) -> asyncio.Queue:
        return self._q

    async def execute(self, name, ctx, args):
        return ""

    def clear(self):
        pass


class _TestLLMUsage:
    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_dollars = 0.0


class FakeProject:
    """
    Minimal project suitable for UIState tests:
    - in-memory message queue (message_generator/send_message),
    - basic commands manager stub,
    - llm_usage totals,
    - project_state.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.base_path = Path(".")
        # Provide a default minimal workflow to support UIState.reset/start_by_name in tests.
        if settings is None:
            default_wf_name = "wf-project-state"
            settings = Settings(
                workflows={
                    default_wf_name: WorkflowConfig(nodes=[], edges=[], config={})
                },
                logging=LoggingSettings(),
            )
        self.settings = settings
        self.commands = _TestCommands()
        self.llm_usage = _TestLLMUsage()
        self.project_state = ProjectState()
        self._q: asyncio.Queue = asyncio.Queue()
        self.shells = None

    async def message_generator(self):
        while True:
            yield await self._q.get()

    async def send_message(self, pack):
        await self._q.put(pack)

    async def shutdown(self):
        pass

    @classmethod
    def with_workflows(cls, workflows: Dict[str, Any]) -> "FakeProject":
        """
        Build a FakeProject from mapping name -> object with .nodes and .edges.
        """
        wf_map = {
            name: WorkflowConfig(nodes=wf.nodes, edges=wf.edges)
            for name, wf in workflows.items()
        }
        settings = Settings(workflows=wf_map, logging=LoggingSettings())
        return cls(settings=settings)


# -----------------------------
# FakeRunner for deterministic UIState tests
# -----------------------------
class FakeRunner:
    """
    Minimal fake runner to drive UIState driver loop.
    Script is a list of tuples:
      (node_name, req_packet, input_requested: bool, status_before: RunnerStatus)

    - Captures inputs sent by UIState in .received_inputs
    - Records rewind/replace_user_input calls for assertions
    """

    def __init__(self, workflow, project, initial_message=None):
        self.status: RunnerStatus = RunnerStatus.idle
        self.script: List[Tuple[str, object, bool, RunnerStatus]] = getattr(
            workflow, "script", []
        )
        self.received_inputs: List[Optional[RunInput]] = []
        self.rewound: Optional[int] = None
        self.replaced_input: Optional[RespPacket] = None
        node_hide_map = getattr(workflow, "node_hide_map", {})

        def _get_runtime_node_by_name(name: str):
            """Minimal runtime node stub matching UIState expectations.

            UIState may access both `hide_final_output` and `description` on
            `runner.runtime_graph.get_runtime_node_by_name(name).model`.
            """

            return SimpleNamespace(
                model=SimpleNamespace(
                    hide_final_output=bool(node_hide_map.get(name, False)),
                    description=None,
                )
            )

        self.runtime_graph = SimpleNamespace(
            get_runtime_node_by_name=_get_runtime_node_by_name
        )

    def cancel(self) -> None:
        self.status = RunnerStatus.canceled

    def stop(self) -> None:
        self.status = RunnerStatus.stopped

    async def rewind(self, task, n: int = 1) -> None:
        self.rewound = n

    def replace_user_input(
        self, task, response, step_index=None, n: Optional[int] = None
    ) -> None:
        # Accept both legacy and named 'n' arg styles
        self.replaced_input = response

    async def run(self, assignment):
        sent = None
        try:
            for node_name, req_packet, input_requested, status_before in self.script:
                self.status = status_before
                ev = RunEvent(
                    node=node_name,
                    execution=Activity(type=ActivityType.executor),
                    event=req_packet,
                    input_requested=input_requested,
                )
                sent = yield ev
                if input_requested:
                    self.received_inputs.append(sent)
            self.status = RunnerStatus.finished
        finally:
            return


# Also expose commonly referenced request types for convenience in tests
ReqMessageRequestT = ReqMessageRequest
