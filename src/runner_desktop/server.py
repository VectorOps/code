"""Expose the runner control protocol as JSON WebSocket messages."""

import asyncio
import struct
from contextlib import asynccontextmanager
from typing import AsyncIterator, Union

from websockets.asyncio.client import ClientConnection
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.typing import Origin

from vectorops.runner.control import protocol, server
from vectorops.runner.control.service import RunnerService


class WebSocketStream:
    """Adapt message framing without duplicating runner dispatch or authentication."""

    def __init__(self, socket: Union[ServerConnection, ClientConnection]) -> None:
        self.socket = socket
        self.buffer = bytearray()

    async def readexactly(self, size: int) -> bytes:
        while len(self.buffer) < size:
            try:
                message = await self.socket.recv()
            except ConnectionClosed as exc:
                raise ConnectionError("WebSocket disconnected") from exc
            if not isinstance(message, str):
                raise ValueError("expected a JSON text message")
            raw = message.encode("utf-8")
            if not 0 < len(raw) <= protocol.MAX_FRAME:
                raise ValueError("invalid message size")
            self.buffer.extend(struct.pack("!I", len(raw)) + raw)
        result = bytes(self.buffer[:size])
        del self.buffer[:size]
        return result

    async def write(self, data: bytes) -> None:
        if len(data) < 4 or struct.unpack("!I", data[:4])[0] != len(data) - 4:
            raise ValueError("invalid control frame")
        try:
            await self.socket.send(data[4:].decode("utf-8"))
        except ConnectionClosed as exc:
            raise ConnectionError("WebSocket disconnected") from exc

    async def aclose(self) -> None:
        await self.socket.close()


@asynccontextmanager
async def websocket_server(
    service: RunnerService, token: str, origins: list[str], port: int = 0
) -> AsyncIterator[Server]:
    """Bind only loopback; remote access uses a TLS proxy or SSH tunnel."""
    if not origins or "null" in origins or "*" in origins:
        raise ValueError("explicit HTTP frontend origins are required")
    control = server.RunnerServer(service, token)

    async def handle(socket: ServerConnection) -> None:
        await control.attach(WebSocketStream(socket))

    async with serve(
        handle,
        "127.0.0.1",
        port,
        origins=[Origin(origin) for origin in origins],
        max_size=protocol.MAX_FRAME,
        max_queue=16,
        compression=None,
        ping_interval=20,
        ping_timeout=20,
        close_timeout=5,
    ) as listener:
        try:
            yield listener
        finally:
            listener.close()
            await control.aclose()


async def wait_for_shutdown(service: RunnerService, stop: asyncio.Event) -> None:
    tasks = [
        asyncio.create_task(stop.wait()),
        asyncio.create_task(service.shutdown_requested.wait()),
    ]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
