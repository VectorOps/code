import asyncio
import json
from pathlib import Path
import secrets

import pytest
from websockets.asyncio.client import connect, ClientConnection
from websockets.exceptions import ConnectionClosed, InvalidStatus

from vectorops.runner.control.service import RunnerService

from runner_desktop.server import websocket_server
from runner_desktop.worker import Worker

ORIGIN = "http://127.0.0.1:5173"


@pytest.mark.asyncio
async def test_owned_worker_cancels_pending_workflow(tmp_path: Path) -> None:
    (tmp_path / "workflows").mkdir()
    (tmp_path / "workflows" / "wait.py").write_text(
        "from vectorops.runner.api.mailbox import request_input\n"
        "async def workflow(ctx):\n"
        '    return await request_input("user.message", "Hello")\n'
    )
    token = secrets.token_urlsafe(32)
    worker = Worker(tmp_path, token, ORIGIN)
    endpoint = await asyncio.to_thread(worker.start)
    try:
        async with connect(endpoint, origin=ORIGIN) as socket:
            await authenticate(socket, token)
            accepted = await request(
                socket,
                "executions.start",
                {
                    "reference": "wait:workflow",
                    "idempotency_key": "owned-start",
                },
            )
            selected = {"execution_id": accepted["execution_id"]}
            async with asyncio.timeout(5):
                while not (await request(socket, "executions.summary", selected))[
                    "pending_inputs"
                ]:
                    await asyncio.sleep(0.02)
    finally:
        await asyncio.to_thread(worker.close)
    async with RunnerService(tmp_path) as service:
        summary = await service.dispatch("executions.summary", selected)
        assert isinstance(summary, dict)
        assert summary["status"] == "cancelled"
        assert summary["active"] is False


async def authenticate(socket: ClientConnection, token: str) -> None:
    await socket.send(
        json.dumps({"kind": "authenticate", "version": 2, "token": token})
    )
    packet = json.loads(await socket.recv())
    assert packet["kind"] == "authenticated"


async def request(socket: ClientConnection, method: str, params: dict = {}) -> dict:
    await socket.send(
        json.dumps(
            {"kind": "request", "id": "test", "method": method, "params": params}
        )
    )
    async with asyncio.timeout(5):
        while True:
            packet = json.loads(await socket.recv())
            if packet["kind"] != "event":
                assert packet["kind"] == "response", packet
                return packet["result"]


@pytest.mark.asyncio
async def test_auth_origins_and_protocol(tmp_path: Path) -> None:
    token = secrets.token_urlsafe(32)
    async with RunnerService(tmp_path) as service:
        async with websocket_server(service, token, [ORIGIN]) as listener:
            url = f"ws://127.0.0.1:{next(iter(listener.sockets)).getsockname()[1]}"
            for origin in [None, "http://evil.example", "null"]:
                with pytest.raises(InvalidStatus):
                    async with connect(url, origin=origin):
                        pass
            async with connect(url, origin=ORIGIN) as socket:
                await socket.send(
                    json.dumps(
                        {
                            "kind": "authenticate",
                            "version": 2,
                            "token": secrets.token_urlsafe(32),
                        }
                    )
                )
                with pytest.raises(ConnectionClosed):
                    await socket.recv()
            async with connect(url, origin=ORIGIN) as socket:
                await authenticate(socket, token)
                info = await request(socket, "runner.info")
                assert info["event_replay"] is False
                await socket.send(
                    json.dumps(
                        {"kind": "request", "id": "invalid", "method": "unknown"}
                    )
                )
                assert json.loads(await socket.recv())["code"] == "unsupported_method"
            async with connect(url, origin=ORIGIN) as socket:
                await socket.send(b"binary")
                with pytest.raises(ConnectionClosed):
                    await socket.recv()


@pytest.mark.asyncio
async def test_disconnect_preserves_workflow_and_idempotency(tmp_path: Path) -> None:
    (tmp_path / "workflows").mkdir()
    (tmp_path / "workflows" / "wait.py").write_text(
        "from vectorops.runner.api.mailbox import request_input\n"
        "async def workflow(ctx):\n"
        '    return await request_input("user.message", "Hello")\n'
    )
    token = secrets.token_urlsafe(32)
    async with RunnerService(tmp_path) as service:
        async with websocket_server(service, token, [ORIGIN]) as listener:
            url = f"ws://127.0.0.1:{next(iter(listener.sockets)).getsockname()[1]}"
            params = {
                "reference": "wait:workflow",
                "idempotency_key": "start-test",
            }
            async with connect(url, origin=ORIGIN) as socket:
                await authenticate(socket, token)
                await request(socket, "events.subscribe")
                accepted = await request(socket, "executions.start", params)
            async with connect(url, origin=ORIGIN) as socket:
                await authenticate(socket, token)
                retry = await request(socket, "executions.start", params)
                assert retry == accepted
                selected = {"execution_id": accepted["execution_id"]}
                async with asyncio.timeout(5):
                    while (await request(socket, "executions.summary", selected))[
                        "status"
                    ] == "starting":
                        await asyncio.sleep(0.02)
                async with asyncio.timeout(5):
                    while True:
                        pending = await request(socket, "mailbox.list", selected)
                        if pending["requests"]:
                            break
                        await asyncio.sleep(0.02)
                await request(
                    socket,
                    "mailbox.respond",
                    {
                        **selected,
                        "request_id": pending["requests"][0]["request_id"],
                        "response": "Done",
                        "idempotency_key": "reply-test",
                    },
                )
                async with asyncio.timeout(5):
                    while (await request(socket, "executions.summary", selected))[
                        "active"
                    ]:
                        await asyncio.sleep(0.02)
                assert (await request(socket, "executions.summary", selected))[
                    "status"
                ] == "completed"
