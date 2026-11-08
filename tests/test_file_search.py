import asyncio
import json
from pathlib import Path
import secrets

import pydantic
import pytest
from vectorops.runner import file_search
from vectorops.runner.control import protocol
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed
from websockets.typing import Origin

from vectorops_code.file_search import FileSearch
from vectorops_code.server import websocket_server
from vectorops_code.service import DesktopService


async def completed(search: FileSearch, query: str = "") -> file_search.SearchUpdate:
    async with asyncio.timeout(5):
        while True:
            update = file_search.SearchUpdate.model_validate(
                await search.dispatch("file_search.search", {"query": query})
            )
            if not update.refreshing:
                return update
            await asyncio.sleep(0.01)


async def test_search_delegates_to_workspace_index_and_refreshes(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    (tmp_path / "source.py").write_text("source")
    (tmp_path / "ignored.txt").write_text("ignored")
    (tmp_path / "linked.py").symlink_to(tmp_path / "source.py")
    search = FileSearch(tmp_path)
    try:
        result = await completed(search, "sPy")
        assert [item.path for item in result.results] == ["source.py"]
        assert result.results[0].highlights
        assert not result.incomplete
        result = await completed(search)
        assert "ignored.txt" not in [item.path for item in result.results]
        assert "linked.py" not in [item.path for item in result.results]
        (tmp_path / "source.py").unlink()
        (tmp_path / "new.py").write_text("new")
        await search.dispatch("file_search.search", {"refresh": True})
        result = await completed(search, ".py")
        assert [item.path for item in result.results] == ["new.py"]
    finally:
        await search.close()
    await search.close()
    with pytest.raises(RuntimeError, match="closed"):
        await search.dispatch("file_search.search", {})


@pytest.mark.parametrize(
    "params",
    [
        {"query": "x" * 257},
        {"query": 1},
        {"limit": 0},
        {"limit": 21},
        {"limit": True},
        {"refresh": "true"},
        {"root": "/arbitrary"},
    ],
)
async def test_invalid_requests_are_rejected(
    tmp_path: Path, params: dict[str, pydantic.JsonValue]
) -> None:
    search = FileSearch(tmp_path)
    try:
        with pytest.raises(protocol.RemoteError) as error:
            await search.dispatch("file_search.search", params)
        assert error.value.code == "invalid_argument"
        assert not search.started
    finally:
        await search.close()


async def test_results_are_bounded_and_unknown_methods_rejected(tmp_path: Path) -> None:
    for index in range(25):
        (tmp_path / f"file-{index}.txt").touch()
    search = FileSearch(tmp_path)
    try:
        result = await completed(search, "file")
        assert len(result.results) == 20
        assert result.truncated
        with pytest.raises(protocol.RemoteError) as error:
            await search.dispatch("file_search.unknown", {})
        assert error.value.code == "unsupported_method"
    finally:
        await search.close()


async def test_search_uses_authenticated_transport_and_closes_with_desktop(
    tmp_path: Path,
) -> None:
    (tmp_path / "file.py").touch()
    token = secrets.token_urlsafe(32)
    origin = Origin("http://127.0.0.1:5173")
    async with DesktopService(tmp_path) as service:
        async with websocket_server(service, token, [origin]) as listener:
            port = next(iter(listener.sockets)).getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}", origin=origin) as socket:
                await socket.send(
                    json.dumps(
                        {
                            "kind": "request",
                            "id": "no-auth",
                            "method": "file_search.search",
                            "params": {},
                        }
                    )
                )
                with pytest.raises(ConnectionClosed):
                    await socket.recv()
                assert service.file_search is None
            async with connect(f"ws://127.0.0.1:{port}", origin=origin) as socket:
                await socket.send(
                    json.dumps({"kind": "authenticate", "version": 2, "token": token})
                )
                assert isinstance(
                    protocol.PACKET.validate_json(await socket.recv()),
                    protocol.Authenticated,
                )
                async with asyncio.timeout(5):
                    while True:
                        await socket.send(
                            json.dumps(
                                {
                                    "kind": "request",
                                    "id": "search",
                                    "method": "file_search.search",
                                    "params": {"query": "file.py"},
                                }
                            )
                        )
                        packet = protocol.PACKET.validate_json(await socket.recv())
                        assert isinstance(packet, protocol.Response)
                        snapshot = file_search.SearchUpdate.model_validate(
                            packet.result
                        )
                        if not snapshot.refreshing:
                            break
                        await asyncio.sleep(0.01)
                assert [item.path for item in snapshot.results] == ["file.py"]
    assert service.file_search is not None
    with pytest.raises(RuntimeError, match="closed"):
        await service.file_search.dispatch("file_search.search", {})
