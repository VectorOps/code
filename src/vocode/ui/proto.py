from typing import Annotated, Optional, Union, Literal
from pydantic import BaseModel, Field

from vocode.runner.models import RunEvent, RunInput
from vocode.state import RunnerStatus

PACKET_RUN_EVENT = "run_event"
PACKET_STATUS = "status"
PACKET_CUSTOM_COMMANDS = "custom_commands"
PACKET_RUN_COMMAND = "run_command"
PACKET_COMMAND_RESULT = "command_result"
PACKET_RUN_INPUT = "run_input"


class UIPacketRunEvent(BaseModel):
    kind: Literal["run_event"] = PACKET_RUN_EVENT
    event: RunEvent


class UIPacketStatus(BaseModel):
    kind: Literal["status"] = PACKET_STATUS
    prev: Optional[RunnerStatus] = None
    curr: RunnerStatus


class UICommand(BaseModel):
    name: str
    help: str
    usage: Optional[str] = None


class UIPacketCustomCommands(BaseModel):
    kind: Literal["custom_commands"] = PACKET_CUSTOM_COMMANDS
    added: list[UICommand] = []
    removed: list[str] = []


class UIPacketRunInput(BaseModel):
    kind: Literal["run_input"] = PACKET_RUN_INPUT
    input: Optional[RunInput] = None


class UIPacketRunCommand(BaseModel):
    kind: Literal["run_command"] = PACKET_RUN_COMMAND
    name: str
    input: list[str] = ""


class UIPacketCommandResult(BaseModel):
    kind: Literal["command_result"] = PACKET_COMMAND_RESULT
    name: str
    ok: bool
    output: Optional[str] = None
    error: Optional[str] = None


UIPacket = Annotated[
    Union[
        UIPacketRunEvent,
        UIPacketStatus,
        UIPacketCustomCommands,
        UIPacketRunInput,
        UIPacketRunCommand,
        UIPacketCommandResult,
    ],
    Field(discriminator="kind"),
]


class UIPacketEnvelope(BaseModel):
    msg_id: int
    payload: UIPacket
