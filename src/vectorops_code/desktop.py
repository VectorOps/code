"""Main-thread native shell; all runner operations use WebSocket."""

from pathlib import Path
import secrets
import sys
from threading import Event
from typing import Optional
from urllib.parse import urlencode, urlsplit

from vectorops.runner.control.auth import validate_token

from vectorops_code.assets import AssetServer
from vectorops_code.oauth import OAuthBridge
from vectorops_code.worker import Worker


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
    asset_port: int = 0,
) -> None:
    try:
        import webview
    except ImportError:
        raise RuntimeError(
            "Desktop support is not installed; install vectorops-code[desktop]"
        ) from None

    if endpoint is not None:
        endpoint = validate_endpoint(endpoint)
        token = validate_token(token or "")
        if asset_port == 0:
            raise ValueError(
                "Attaching requires --asset-port matching an origin allowed by the worker"
            )
    else:
        workspace = workspace.expanduser().resolve()
        if not workspace.is_dir():
            raise ValueError(f"Project directory does not exist: {workspace}")
        token = secrets.token_urlsafe(32)
    with AssetServer(port=asset_port) as assets:
        worker = None if endpoint else Worker(workspace, token, assets.origin)
        closed = Event()
        try:
            if worker is not None:
                endpoint = worker.start()
            fragment = urlencode(
                {
                    "endpoint": endpoint,
                    "token": token,
                    "managed": "1" if worker else "0",
                }
            )
            window = webview.create_window(
                f"VectorOps Code - {workspace.resolve()}",
                f"{assets.origin}/#{fragment}",
                width=1200,
                height=820,
                js_api=OAuthBridge(assets.origin),
            )
            if worker is None:
                webview.start()
            else:
                window.events.closed += closed.set

                def monitor() -> None:
                    while not closed.wait(0.1):
                        if worker is not None and worker.finished.is_set():
                            if worker.failure is not None:
                                print(
                                    f"Runner worker failed: {worker.failure}",
                                    file=sys.stderr,
                                )
                            if closed.is_set():
                                return
                            window.load_html(
                                "<h1>Local runner stopped</h1>"
                                "<p>Close this window and reopen the project to reconnect. "
                                "Check the terminal for error details.</p>"
                            )
                            return

                webview.start(monitor)
        finally:
            closed.set()
            if worker is not None:
                worker.close(raise_failure=sys.exc_info()[0] is None)
