"""Launch a desktop shell or an independently owned headless worker."""

import argparse
import asyncio
import os
from pathlib import Path
import signal
import sys
from typing import Optional

from vectorops.runner.control.auth import validate_token

from vectorops_code.server import wait_for_shutdown, websocket_server
from vectorops_code.service import DesktopService


async def serve(
    workspace: Path,
    token: str,
    origins: list[str],
    port: int,
    *,
    development: bool = False,
) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: loop.call_soon_threadsafe(stop.set))
    try:
        async with DesktopService(workspace, development=development) as service:
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        usage="%(prog)s [PROJECT] | {desktop,serve} ...",
        epilog="Start locally with vectorops-code [PROJECT]. No token or ports required.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("serve", help="Run an independent headless worker")
    worker.add_argument("--workspace", type=Path, default=Path.cwd())
    worker.add_argument("--port", type=int, default=0)
    worker.add_argument("--origin", action="append", required=True)
    worker.add_argument(
        "--development",
        action="store_true",
        help="Enable client diagnostics by default",
    )
    desktop = commands.add_parser("desktop", help="Open the native desktop shell")
    desktop.add_argument("project", type=Path, nargs="?")
    desktop.add_argument("--workspace", type=Path)
    desktop.add_argument(
        "--endpoint", help="Attach to an independent worker; never stop it on exit"
    )
    desktop.add_argument("--asset-port", type=int, default=0)
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in ("desktop", "serve", "-h", "--help"):
        arguments.insert(0, "desktop")
    args = parser.parse_args(arguments)
    token = os.environ.get(
        "VECTOROPS_CODE_TOKEN", os.environ.get("RUNNER_DESKTOP_TOKEN")
    )
    try:
        if args.command == "serve":
            asyncio.run(
                serve(
                    args.workspace,
                    validate_token(token or ""),
                    args.origin,
                    args.port,
                    development=args.development,
                )
            )
        else:
            from vectorops_code.desktop import launch

            if args.project is not None and args.workspace is not None:
                parser.error("specify PROJECT or --workspace, not both")
            launch(
                args.project or args.workspace or Path.cwd(),
                args.endpoint,
                token,
                args.asset_port,
            )
    except KeyboardInterrupt:
        pass
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"{exc}\n")
