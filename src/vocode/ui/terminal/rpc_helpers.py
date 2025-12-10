from __future__ import annotations

from typing import Optional
from vocode.ui.rpc import RpcHelper
from vocode.ui import proto as ui_proto
from vocode.state import Message

# All helpers ACK-only; timeout=None means wait indefinitely where appropriate.


async def rpc_stop(rpc: RpcHelper) -> None:
    await rpc.call(ui_proto.UIPacketUIStopAction(), timeout=None)


async def rpc_cancel(rpc: RpcHelper) -> None:
    await rpc.call(ui_proto.UIPacketUICancelAction(), timeout=None)


async def rpc_use(rpc: RpcHelper, name: str) -> None:
    await rpc.call(ui_proto.UIPacketUIUseAction(name=name), timeout=None)


async def rpc_use_with_input(
    rpc: RpcHelper, name: str, message: Optional[Message]
) -> None:
    await rpc.call(
        ui_proto.UIPacketUIUseWithInputAction(name=name, message=message),
        timeout=None,
    )


async def rpc_reset(rpc: RpcHelper) -> None:
    await rpc.call(ui_proto.UIPacketUIResetRunAction(), timeout=None)


async def rpc_restart(rpc: RpcHelper) -> None:
    await rpc.call(ui_proto.UIPacketUIRestartAction(), timeout=None)


async def rpc_replace_user_input(
    rpc: RpcHelper,
    *,
    message: Optional[Message] = None,
    approved: Optional[bool] = None,
    n: Optional[int] = 1,
) -> None:
    await rpc.call(
        ui_proto.UIPacketReplaceUserInputAction(
            message=message,
            approved=approved,
            n=n,
        ),
        timeout=None,
    )


async def _rpc_repos_common(rpc: RpcHelper, packet: ui_proto.UIPacket):
    res = await rpc.call(packet, timeout=None)
    if res is None:
        raise RuntimeError("No response received for repos action")
    if res.kind != ui_proto.PACKET_COMMAND_RESULT:
        raise RuntimeError(f"Unexpected response kind '{res.kind}' for repos action")
    return res  # UIPacketCommandResult


async def rpc_repos_list(rpc: RpcHelper):
    return await _rpc_repos_common(
        rpc,
        ui_proto.UIPacketUIReposAction(action="list"),
    )


async def rpc_repos_add(rpc: RpcHelper, name: str, path: str):
    return await _rpc_repos_common(
        rpc,
        ui_proto.UIPacketUIReposAction(action="add", name=name, path=path),
    )


async def rpc_repos_remove(rpc: RpcHelper, name: str):
    return await _rpc_repos_common(
        rpc,
        ui_proto.UIPacketUIReposAction(action="remove", name=name),
    )
