"""The local browser host delegates OAuth without returning provider tokens."""

import asyncio
from unittest.mock import patch

from aiohttp import ClientSession

from runner_desktop.templates import TemplateServer


async def test_oauth_host_requires_local_authorization() -> None:
    host = TemplateServer("ws://127.0.0.1:8765", "host-secret")
    host.start()
    try:
        async with ClientSession() as session:
            payload = {
                "endpoint": "wss://runner.example",
                "token": "runner-secret",
                "id": "login-1",
            }
            with patch(
                "runner_desktop.templates.OAuthBridge.authenticate", return_value=True
            ) as login:
                async with session.post(host.url + "/oauth", json=payload) as response:
                    assert response.status == 403
                login.assert_not_called()
                headers = {"Authorization": "Bearer host-secret"}
                async with session.post(
                    host.url + "/oauth", json=payload, headers=headers
                ) as response:
                    assert await response.json() == {"success": True}
                login.assert_called_once_with(
                    "wss://runner.example", "runner-secret", "login-1"
                )
                async with session.post(
                    host.url + "/oauth", json={"id": "login-1"}, headers=headers
                ) as response:
                    assert response.status == 400
    finally:
        await asyncio.to_thread(host.close)
