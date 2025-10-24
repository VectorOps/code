from pydantic import BaseModel, Field
from typing import Literal, Type, Union, Literal, Annotated

PACKET_PROJECT_OP_START = "project_op_start"
PACKET_PROJECT_OP_PROGRESS = "project_op_progress"
PACKET_PROJECT_OP_FINISH = "project_op_finish"


class PacketProjectOpStart(BaseModel):
    kind: Literal["project_op_start"] = PACKET_PROJECT_OP_START
    message: str

class PacketProjectOpProgress(BaseModel):
    kind: Literal["project_op_progress"] = PACKET_PROJECT_OP_PROGRESS
    progress: int
    total: int

class PacketProjectOpFinish(BaseModel):
    kind: Literal["project_op_finish"] = PACKET_PROJECT_OP_FINISH


Packet = Annotated[
    Union[
        PacketProjectOpStart,
        PacketProjectOpProgress,
        PacketProjectOpFinish,
    ],
    Field(discriminator="kind"),
]
