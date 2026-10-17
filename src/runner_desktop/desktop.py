"""Main-thread native shell; all runner operations use WebSocket."""

from pathlib import Path
import secrets
from typing import Optional
from urllib.parse import urlencode, urlsplit

from vectorops.runner.control.auth import validate_token

from runner_desktop.assets import AssetServer
from runner_desktop.oauth import OAuthBridge
from runner_desktop.worker import Worker


def validate_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in ("ws", "wss") or not parsed.hostname:
        raise ValueError("endpoint must be a ws:// or wss:// URL")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("endpoint cannot contain credentials or a fragment")
    if parsed.scheme == "ws" and parsed.hostname not in (
        "localhost",
        "127.0.0.1",
        "::1",
    ):
        raise ValueError("remote endpoints require WSS or a loopback SSH tunnel")
    return endpoint


def launch(
    workspace: Path,
    endpoint: Optional[str] = None,
    token: Optional[str] = None,
    asset_port: int = 8766,
) -> None:
    import webview

    if endpoint is not None:
        endpoint = validate_endpoint(endpoint)
        token = validate_token(token or "")
    else:
        token = secrets.token_urlsafe(32)
    with AssetServer(port=asset_port) as assets:
        worker = None if endpoint else Worker(workspace, token, assets.origin)
        try:
            if worker is not None:
                endpoint = worker.start()
            fragment = urlencode({"endpoint": endpoint, "token": token})
            webview.create_window(
                f"VectorOps - {workspace.resolve()}",
                f"{assets.origin}/#{fragment}",
                width=1200,
                height=820,
                js_api=OAuthBridge(assets.origin),
            )
            webview.start()
        finally:
            if worker is not None:
                worker.close()
