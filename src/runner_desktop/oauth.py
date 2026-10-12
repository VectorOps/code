"""Native host capability for browser OAuth over the existing control channel."""

import asyncio

from websockets.asyncio.client import connect
from websockets.typing import Origin
from vectorops.runner.control import oauth_host
from vectorops.runner.control.client import RunnerClient

from runner_desktop.server import WebSocketStream


async def authenticate(endpoint: str, token: str, origin: str, identifier: str) -> None:
    """Keep the callback listener local and provider tokens inside the runner."""
    async with connect(
        endpoint, origin=Origin(origin), max_size=8 * 1024 * 1024
    ) as socket:
        client = await RunnerClient.connect(WebSocketStream(socket), token)
        task = asyncio.create_task(oauth_host.authenticate(client, identifier))
        try:
            while not task.done():
                done, _ = await asyncio.wait({task}, timeout=2)
                if done:
                    break
                result = await client.pending_authentication()
                if not any(item.id == identifier for item in result):
                    task.cancel()
                    break
            await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await client.aclose()


class OAuthBridge:
    def __init__(self, origin: str) -> None:
        self.origin = origin

    def authenticate(self, endpoint: str, token: str, identifier: str) -> bool:
        from runner_desktop.desktop import validate_endpoint

        try:
            asyncio.run(
                authenticate(
                    validate_endpoint(endpoint), token, self.origin, identifier
                )
            )
            return True
        except (Exception, asyncio.CancelledError):
            return False
