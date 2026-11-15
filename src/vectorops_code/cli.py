"""Launch a desktop shell or an independently owned headless worker."""

import argparse
import asyncio
from getpass import getpass
import os
from pathlib import Path
import signal
import sys
from typing import Optional
from urllib.parse import urlsplit

from vectorops.runner.control.auth import validate_token

from vectorops_code import credentials
from vectorops_code.server import wait_for_shutdown, websocket_server
from vectorops_code.service import DesktopService


async def serve(
    workspace: Path,
    token: str,
    origins: Optional[list[str]],
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


def connection_endpoint(address: str, port: Optional[int]) -> str:
    from vectorops_code.desktop import validate_endpoint

    address = address.strip()
    if not address:
        raise ValueError("A remote address is required")
    if "://" in address:
        if port is not None:
            raise ValueError("Specify the port in the endpoint URL, not --port")
        return validate_endpoint(address)
    parsed = urlsplit(f"//{address}")
    if (
        parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Use a hostname, host:port, or a complete WebSocket URL")
    if parsed.port is not None and port is not None:
        raise ValueError("Specify the port in the address or --port, not both")
    port = parsed.port if parsed.port is not None else port
    if port is None:
        if not sys.stdin.isatty():
            raise ValueError(
                "Supply --port or an endpoint URL when connecting non-interactively"
            )
        port = int(input("Remote WebSocket port: "))
    if not 1 <= port <= 65535:
        raise ValueError("Remote port must be between 1 and 65535")
    host = parsed.hostname
    if host is None:
        raise ValueError("A remote hostname is required")
    scheme = "ws" if host in ("127.0.0.1", "localhost", "::1") else "wss"
    if ":" in host:
        host = f"[{host}]"
    return validate_endpoint(f"{scheme}://{host}:{port}")


def connection_token(endpoint: str, token: Optional[str], path: Optional[Path]) -> str:
    if path is not None:
        return credentials.read_token(path.expanduser())
    if token is not None:
        return validate_token(token)
    path = credentials.token_path("connections", endpoint)
    try:
        return credentials.read_token(path)
    except FileNotFoundError:
        if not sys.stdin.isatty():
            raise ValueError("Supply VECTOROPS_CODE_TOKEN or --token-file to connect")
        token = validate_token(getpass(f"Worker token (saved in {path}): "))
        credentials.save_token(path, token)
        return token


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        usage="%(prog)s [PROJECT] | {connect,desktop,serve} ...",
        epilog="Start locally with vectorops-code [PROJECT], or attach with vectorops-code connect.",
    )
    commands = parser.add_subparsers(dest="command", required=True, prog=parser.prog)
    worker = commands.add_parser("serve", help="Run an independent headless worker")
    worker.add_argument("project", type=Path, nargs="?")
    worker.add_argument("--workspace", type=Path)
    worker.add_argument("--port", type=int, default=0)
    worker.add_argument(
        "--origin",
        action="append",
        help="Restrict to exact frontend origins instead of loopback HTTP on any port",
    )
    worker.add_argument(
        "--token-file", type=Path, help="Read or generate a private worker token file"
    )
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
    desktop.add_argument("--token-file", type=Path)
    remote = commands.add_parser(
        "connect", help="Attach to a worker without a local project"
    )
    remote.add_argument(
        "address", nargs="?", help="WebSocket URL or host[:port]; prompts when omitted"
    )
    remote.add_argument(
        "--port", type=int, help="Remote WebSocket port when using a hostname"
    )
    remote.add_argument(
        "--token-file", type=Path, help="Read an existing private token file"
    )
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in (
        "desktop",
        "serve",
        "connect",
        "-h",
        "--help",
    ):
        arguments.insert(0, "desktop")
    args = parser.parse_args(arguments)
    token = os.environ.get(
        "VECTOROPS_CODE_TOKEN", os.environ.get("RUNNER_DESKTOP_TOKEN")
    )
    try:
        if args.command == "connect":
            from vectorops_code.desktop import launch

            if args.address is None and not sys.stdin.isatty():
                parser.error(
                    "supply a remote address when connecting non-interactively"
                )
            endpoint = connection_endpoint(
                args.address or input("Remote address: "), args.port
            )
            launch(None, endpoint, connection_token(endpoint, token, args.token_file))
            return
        if args.project is not None and args.workspace is not None:
            parser.error("specify PROJECT or --workspace, not both")
        workspace = args.project or args.workspace or Path.cwd()
        if args.command == "serve":
            workspace = workspace.expanduser().resolve()
            if not workspace.is_dir():
                raise ValueError(f"Project directory does not exist: {workspace}")
            if not 0 <= args.port <= 65535:
                raise ValueError("Worker port must be between 0 and 65535")
            if args.token_file is not None or token is None:
                path = args.token_file or credentials.token_path(
                    "workers", str(workspace)
                )
                token = credentials.worker_token(path.expanduser())
                print(f"Worker token file: {path}", flush=True)
            asyncio.run(
                serve(
                    workspace,
                    validate_token(token or ""),
                    args.origin,
                    args.port,
                    development=args.development,
                )
            )
        else:
            from vectorops_code.desktop import launch

            if args.endpoint is not None:
                from vectorops_code.desktop import validate_endpoint

                args.endpoint = validate_endpoint(args.endpoint)
                token = connection_token(args.endpoint, token, args.token_file)
            launch(
                workspace,
                args.endpoint,
                token,
                args.asset_port,
            )
    except KeyboardInterrupt:
        pass
    except EOFError:
        parser.exit(
            1, "Connection input ended; supply an address, port, and token file.\n"
        )
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"{exc}\n")
