from typing import Annotated, Optional, Union, Literal, Any, Dict, List
from pydantic import BaseModel, Field

from vocode.runner.models import RunEvent, RunInput
from vocode.state import RunnerStatus, LogLevel, Message

PACKET_RUN_EVENT = "run_event"
PACKET_STATUS = "status"
PACKET_UI_RESET = "ui_reset"
PACKET_CUSTOM_COMMANDS = "custom_commands"
PACKET_RUN_COMMAND = "run_command"
PACKET_COMMAND_RESULT = "command_result"
PACKET_RUN_INPUT = "run_input"
PACKET_ACK = "ack"
PACKET_COMPLETION_REQUEST = "completion_request"
PACKET_COMPLETION_RESULT = "completion_result"
PACKET_UI_RELOAD = "ui_reload"
PACKET_PROJECT_OP_START = "project_op_start"
PACKET_PROJECT_OP_PROGRESS = "project_op_progress"
PACKET_PROJECT_OP_FINISH = "project_op_finish"
PACKET_LOG = "log"
PACKET_UI_STOP_ACTION = "ui_stop_action"
PACKET_UI_CANCEL_ACTION = "ui_cancel_action"
PACKET_UI_USE_ACTION = "ui_use_action"
PACKET_UI_RESET_RUN_ACTION = "ui_reset_run_action"
PACKET_UI_RESTART_ACTION = "ui_restart_action"
PACKET_REPLACE_USER_INPUT_ACTION = "replace_user_input_action"


class UIPacketRunEvent(BaseModel):
    kind: Literal["run_event"] = PACKET_RUN_EVENT
    event: RunEvent


class UIPacketUIReset(BaseModel):
    kind: Literal["ui_reset"] = PACKET_UI_RESET
    # No payload for now; acts as a directive.


class UIPacketStatus(BaseModel):
    kind: Literal["status"] = PACKET_STATUS
    prev: Optional[RunnerStatus] = None
    curr: RunnerStatus
    prev_node: Optional[str] = None
    curr_node: Optional[str] = None
    curr_node_description: Optional[str] = None


class UICommand(BaseModel):
    name: str
    help: str
    usage: Optional[str] = None
    autocompleter: Optional[str] = None


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


class UIPacketCompletionRequest(BaseModel):
    kind: Literal["completion_request"] = PACKET_COMPLETION_REQUEST
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class UIPacketCompletionResult(BaseModel):
    kind: Literal["completion_result"] = PACKET_COMPLETION_RESULT
    ok: bool = True
    suggestions: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class UIPacketAck(BaseModel):
    kind: Literal["ack"] = PACKET_ACK


class UIPacketUIReload(BaseModel):
    kind: Literal["ui_reload"] = PACKET_UI_RELOAD

class UIPacketUIStopAction(BaseModel):
    kind: Literal["ui_stop_action"] = PACKET_UI_STOP_ACTION

class UIPacketUICancelAction(BaseModel):
    kind: Literal["ui_cancel_action"] = PACKET_UI_CANCEL_ACTION

class UIPacketUIUseAction(BaseModel):
    kind: Literal["ui_use_action"] = PACKET_UI_USE_ACTION
    name: str

class UIPacketUIResetRunAction(BaseModel):
    kind: Literal["ui_reset_run_action"] = PACKET_UI_RESET_RUN_ACTION

class UIPacketUIRestartAction(BaseModel):
    kind: Literal["ui_restart_action"] = PACKET_UI_RESTART_ACTION

class UIPacketReplaceUserInputAction(BaseModel):
    kind: Literal["replace_user_input_action"] = PACKET_REPLACE_USER_INPUT_ACTION
    # Exactly one of message/approved should be provided.
    message: Optional[Message] = None
    approved: Optional[bool] = None
    n: Optional[int] = 1

class UIPacketProjectOpStart(BaseModel):
    kind: Literal["project_op_start"] = PACKET_PROJECT_OP_START
    message: str

class UIPacketProjectOpProgress(BaseModel):
    kind: Literal["project_op_progress"] = PACKET_PROJECT_OP_PROGRESS
    progress: int
    total: int

class UIPacketProjectOpFinish(BaseModel):
    kind: Literal["project_op_finish"] = PACKET_PROJECT_OP_FINISH

class UIPacketLog(BaseModel):
    kind: Literal["log"] = PACKET_LOG
    level: LogLevel
    message: str
    logger: Optional[str] = None
    pathname: Optional[str] = None
    lineno: Optional[int] = None
    exc_text: Optional[str] = None


UIPacket = Annotated[
    Union[
        UIPacketRunEvent,
        UIPacketUIReset,
        UIPacketStatus,
        UIPacketCustomCommands,
        UIPacketRunInput,
        UIPacketRunCommand,
        UIPacketCompletionRequest,
        UIPacketCompletionResult,
        UIPacketCommandResult,
        UIPacketAck,
        UIPacketUIReload,
        UIPacketUIStopAction,
        UIPacketUICancelAction,
        UIPacketUIUseAction,
        UIPacketUIResetRunAction,
        UIPacketUIRestartAction,
        UIPacketReplaceUserInputAction,
        UIPacketProjectOpStart,
        UIPacketProjectOpProgress,
        UIPacketProjectOpFinish,
        UIPacketLog,
    ],
    Field(discriminator="kind"),
]


class UIPacketEnvelope(BaseModel):
    msg_id: int
    payload: UIPacket
    source_msg_id: Optional[int] = None
