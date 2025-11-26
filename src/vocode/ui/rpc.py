import asyncio
from typing import Callable, Awaitable, Optional, Dict

from vocode.logger import logger
from .proto import UIPacket, UIPacketEnvelope, UIPacketAck, PACKET_ACK


class RpcHelper:
    """
    Helper class to manage RPC-style request/response over the UI protocol.
    """

    def __init__(
        self,
        send_callback: Callable[[UIPacketEnvelope], Awaitable[None]],
        name: str,
        id_generator: Optional[Callable[[], int]] = None,
    ):
        self._send_callback = send_callback
        self._name = name
        self._id_generator = id_generator
        self._pending_requests: dict[int, "asyncio.Future[UIPacketEnvelope]"] = {}
        self._msg_id_counter = 0

    def _next_msg_id(self) -> int:
        if self._id_generator:
            return self._id_generator()
        self._msg_id_counter += 1
        return self._msg_id_counter

    async def call(
        self, payload: UIPacket, timeout: Optional[float] = 300.0
    ) -> Optional[UIPacket]:
        """Sends a request payload and waits for a response payload. timeout=None disables timeout."""
        msg_id = self._next_msg_id()
        fut: "asyncio.Future[UIPacketEnvelope]" = (
            asyncio.get_running_loop().create_future()
        )
        self._pending_requests[msg_id] = fut

        envelope = UIPacketEnvelope(msg_id=msg_id, payload=payload)
        await self._send_callback(envelope)

        try:
            response_envelope = await asyncio.wait_for(fut, timeout=timeout)
            if response_envelope.payload.kind == PACKET_ACK:
                return None
            return response_envelope.payload
        except asyncio.TimeoutError:
            logger.error("%s: request %d timed out", self._name, msg_id)
            raise
        finally:
            self._pending_requests.pop(msg_id, None)

    async def reply(self, payload: UIPacket, source_msg_id: int):
        """Sends a response payload; does not wait."""
        # msg_id for a reply doesn't need to be tracked for a response, but should be unique.
        msg_id = self._next_msg_id()
        envelope = UIPacketEnvelope(
            msg_id=msg_id, payload=payload, source_msg_id=source_msg_id
        )
        await self._send_callback(envelope)

    def handle_response(self, envelope: UIPacketEnvelope) -> bool:
        """
        To be called with every incoming envelope that has a source_msg_id.
        Returns True if the response was matched to a pending request.
        """
        if envelope.source_msg_id is None:
            return False  # Should not be called with requests.

        fut = self._pending_requests.get(envelope.source_msg_id)
        if fut and not fut.done():
            fut.set_result(envelope)
            return True

        # No pending waiter; let router decide on logging policy (debug-level).
        return False

    def cancel_all(self):
        for fut in self._pending_requests.values():
            if not fut.done():
                fut.cancel(f"{self._name} RPC client is shutting down")
        self._pending_requests.clear()


class IncomingPacketRouter:
    def __init__(self, rpc: RpcHelper, name: str):
        self._rpc = rpc
        self._name = name
        self._handlers: Dict[
            str, Callable[[UIPacketEnvelope], Awaitable[Optional[UIPacket]]]
        ] = {}

    def register(
        self,
        kind: str,
        handler: Callable[[UIPacketEnvelope], Awaitable[Optional[UIPacket]]],
    ) -> None:
        self._handlers[kind] = handler

    async def handle(self, envelope: UIPacketEnvelope) -> bool:
        # Reply handling
        if envelope.source_msg_id is not None:
            matched = self._rpc.handle_response(envelope)
            if not matched:
                logger.debug(
                    "%s: unmatched response source_msg_id=%s kind=%s",
                    self._name,
                    envelope.source_msg_id,
                    getattr(envelope.payload, "kind", None),
                )
            return True

        # Request dispatch
        kind = envelope.payload.kind
        handler = self._handlers.get(kind)
        if not handler:
            logger.error("%s: no handler for request kind=%s", self._name, kind)
            return False

        resp = await handler(envelope)
        if resp is not None:
            await self._rpc.reply(resp, source_msg_id=envelope.msg_id)
        return True
