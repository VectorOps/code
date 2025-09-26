from typing import Annotated, Optional, Union, Literal
from pydantic import BaseModel, Field

from vocode.runner.models import RunEvent, RunInput
from vocode.state import RunnerStatus

# UIState -> UI client events (discriminated by 'kind')

UI_PACKET_RUN_EVENT = "run_event"
UI_PACKET_STATUS = "status"
UI_PACKET_CUSTOM_COMMANDS = "custom_commands"
UI_PACKET_RUN_COMMAND = "run_command"
UI_PACKET_COMMAND_RESULT = "command_result"

class UIReqRunEvent(BaseModel):
    """
    Forwarded RunEvent from the runner to the UI client.
    The UI should correlate responses using req_id.
    """
    kind: Literal["run_event"] = UI_PACKET_RUN_EVENT
    req_id: int
    event: RunEvent

class UIReqStatus(BaseModel):
    """
    Notify UI client about runner status changes.
    """
    kind: Literal["status"] = UI_PACKET_STATUS
    prev: Optional[RunnerStatus] = None
    curr: RunnerStatus

class UICommand(BaseModel):
    name: str
    help: str
    usage: Optional[str] = None

class UIReqCustomCommands(BaseModel):
    kind: Literal["custom_commands"] = UI_PACKET_CUSTOM_COMMANDS
    added: list[UICommand] = []
    removed: list[str] = []

UIRequest = Annotated[
    Union[
        UIReqRunEvent,
        UIReqStatus,
        UIReqCustomCommands,
        "UIReqCommandResult",
    ],
    Field(discriminator="kind"),
]

# UI client -> UIState responses (discriminated by 'kind')

UI_PACKET_RUN_INPUT = "run_input"

class UIRespRunInput(BaseModel):
    """
    Response to a specific run_event (by req_id). Wraps a RunInput destined for the runner.
    """
    kind: Literal["run_input"] = UI_PACKET_RUN_INPUT
    req_id: int
    input: Optional[RunInput] = None

class UIRespRunCommand(BaseModel):
    kind: Literal["run_command"] = UI_PACKET_RUN_COMMAND
    name: str
    input: list[str] = ""

class UIReqCommandResult(BaseModel):
    kind: Literal["command_result"] = UI_PACKET_COMMAND_RESULT
    name: str
    ok: bool
    output: Optional[str] = None
    error: Optional[str] = None

UIResponse = Annotated[
    Union[
        UIRespRunInput,
        UIRespRunCommand,
    ],
    Field(discriminator="kind"),
]
