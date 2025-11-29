from __future__ import annotations

from typing import Optional
from vocode.ui.rpc import RpcHelper
from vocode.ui.proto import (
    UIPacketUIStopAction,
    UIPacketUICancelAction,
    UIPacketUIUseAction,
    UIPacketUIResetRunAction,
    UIPacketUIRestartAction,
    UIPacketReplaceUserInputAction,
    UIPacketUIUseWithInputAction,
)
from vocode.state import Message

# All helpers ACK-only; timeout=None means wait indefinitely where appropriate.


async def rpc_stop(rpc: RpcHelper) -> None:
    await rpc.call(UIPacketUIStopAction(), timeout=None)


async def rpc_cancel(rpc: RpcHelper) -> None:
    await rpc.call(UIPacketUICancelAction(), timeout=None)


async def rpc_use(rpc: RpcHelper, name: str) -> None:
    await rpc.call(UIPacketUIUseAction(name=name), timeout=None)


async def rpc_use_with_input(
    rpc: RpcHelper, name: str, message: Optional[Message]
) -> None:
    await rpc.call(
        UIPacketUIUseWithInputAction(name=name, message=message), timeout=None
    )


async def rpc_reset(rpc: RpcHelper) -> None:
    await rpc.call(UIPacketUIResetRunAction(), timeout=None)


async def rpc_restart(rpc: RpcHelper) -> None:
    await rpc.call(UIPacketUIRestartAction(), timeout=None)


async def rpc_replace_user_input(
    rpc: RpcHelper,
    *,
    message: Optional[Message] = None,
    approved: Optional[bool] = None,
    n: Optional[int] = 1,
) -> None:
    await rpc.call(
        UIPacketReplaceUserInputAction(message=message, approved=approved, n=n),
        timeout=None,
    )
