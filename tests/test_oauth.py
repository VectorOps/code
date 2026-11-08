"""OAuth host delegates browser and callback handling to the runner helper."""

from unittest.mock import AsyncMock, patch
from pathlib import Path
import secrets

import pytest
from vectorops.runner.control.service import RunnerService

from vectorops_code.oauth import OAuthBridge, authenticate
from vectorops_code.server import websocket_server


async def test_oauth_host_uses_authenticated_websocket(tmp_path: Path) -> None:
    token = secrets.token_urlsafe(32)
    origin = "http://localhost:8766"
    async with RunnerService(tmp_path) as service:
        async with websocket_server(service, token, [origin]) as listener:
            port = next(iter(listener.sockets)).getsockname()[1]
            with patch(
                "vectorops_code.oauth.oauth_host.authenticate", new_callable=AsyncMock
            ) as login:
                await authenticate(f"ws://127.0.0.1:{port}", token, origin, "login-1")
                assert login.await_args is not None
                assert login.await_args.args[1] == "login-1"


async def test_host_authenticates_and_closes_client() -> None:
    client = AsyncMock()
    with (
        patch("vectorops_code.oauth.connect") as connect,
        patch("vectorops_code.oauth.RunnerClient.connect", return_value=client),
        patch(
            "vectorops_code.oauth.oauth_host.authenticate", new_callable=AsyncMock
        ) as login,
    ):
        await authenticate(
            "ws://localhost:1234", "token", "http://localhost:8766", "login-1"
        )
        login.assert_awaited_once_with(client, "login-1")
        client.aclose.assert_awaited_once()
        assert connect.call_args.kwargs["origin"] == "http://localhost:8766"


async def test_host_closes_client_after_failure() -> None:
    client = AsyncMock()
    with (
        patch("vectorops_code.oauth.connect"),
        patch("vectorops_code.oauth.RunnerClient.connect", return_value=client),
        patch(
            "vectorops_code.oauth.oauth_host.authenticate",
            side_effect=RuntimeError("failure"),
        ),
    ):
        with pytest.raises(RuntimeError):
            await authenticate(
                "ws://localhost:1234", "token", "http://localhost:8766", "login-1"
            )
        client.aclose.assert_awaited_once()


def test_bridge_rejects_insecure_remote_endpoint() -> None:
    with patch("vectorops_code.oauth.authenticate", new_callable=AsyncMock) as login:
        assert not OAuthBridge("http://localhost:8766").authenticate(
            "ws://remote.example", "token", "login-1"
        )
        login.assert_not_called()
