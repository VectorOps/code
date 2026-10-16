"""Launch a desktop shell or an independently owned headless worker."""

import argparse
import asyncio
import os
from pathlib import Path
import signal
from typing import Optional

from vectorops.runner.control.auth import validate_token

from runner_desktop.server import wait_for_shutdown, websocket_server
from runner_desktop.service import DesktopService


async def serve(workspace: Path, token: str, origins: list[str], port: int) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: loop.call_soon_threadsafe(stop.set))
    try:
        async with DesktopService(workspace) as service:
            async with websocket_server(service, token, origins, port) as listener:
                actual_port = next(iter(listener.sockets)).getsockname()[1]
                print(
                    f"Runner WebSocket listening at ws://127.0.0.1:{actual_port}",
                    flush=True,
                )
                await wait_for_shutdown(service, stop)
                await service.aclose()
    finally:
        signal.signal(signal.SIGTERM, previous)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("serve", help="Run an independent headless worker")
    worker.add_argument("--workspace", type=Path, default=Path.cwd())
    worker.add_argument("--port", type=int, default=8765)
    worker.add_argument("--origin", action="append", required=True)
    desktop = commands.add_parser("desktop", help="Open the native desktop shell")
    desktop.add_argument("--workspace", type=Path, default=Path.cwd())
    desktop.add_argument(
        "--endpoint", help="Attach to an independent worker; never stop it on exit"
    )
    desktop.add_argument("--asset-port", type=int, default=8766)
    args = parser.parse_args(argv)
    token = os.environ.get("RUNNER_DESKTOP_TOKEN")
    try:
        if args.command == "serve":
            asyncio.run(
                serve(
                    args.workspace, validate_token(token or ""), args.origin, args.port
                )
            )
        else:
            from runner_desktop.desktop import launch

            launch(args.workspace, args.endpoint, token, args.asset_port)
    except KeyboardInterrupt:
        pass
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
