"""Run Vite and restart the headless worker when Python sources change."""

import argparse
import asyncio
import os
from pathlib import Path
import secrets
import shutil
import sys
from typing import Optional

from vectorops_code.templates import TemplateServer


def watch_paths(root: Path, extra: list[Path]) -> list[Path]:
    candidates = [
        root / "src",
        root.parent / "runner/src",
        root.parent / "connect/src",
        *extra,
    ]
    return list(dict.fromkeys(path.resolve() for path in candidates if path.is_dir()))


async def stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=15)
    except TimeoutError:
        process.kill()
        await process.wait()


async def run(
    root: Path, workspace: Path, port: int, frontend_port: int, extra: list[Path]
) -> None:
    """Own both development processes and serialize worker shutdown and restart."""
    from watchfiles import PythonFilter, awatch

    environment = dict(os.environ)
    environment["VECTOROPS_CODE_TOKEN"] = secrets.token_urlsafe(32)
    environment["VECTOROPS_CODE_LOCAL_ENDPOINT"] = f"ws://127.0.0.1:{port}"
    environment["VECTOROPS_CODE_LOCAL_ORIGIN"] = f"http://127.0.0.1:{frontend_port}"
    node = shutil.which("node")
    vite = root / "frontend/node_modules/vite/bin/vite.js"
    if node is None or not vite.is_file():
        raise ValueError("Install Node.js and run npm ci in frontend/ first")
    paths = watch_paths(root, extra)
    origin = f"http://127.0.0.1:{frontend_port}"
    command = [
        sys.executable,
        "-c",
        "from vectorops_code.cli import main; main()",
        "serve",
        "--development",
        "--workspace",
        str(workspace.resolve()),
        "--port",
        str(port),
        "--origin",
        origin,
    ]
    templates = TemplateServer(
        environment["VECTOROPS_CODE_LOCAL_ENDPOINT"],
        environment["VECTOROPS_CODE_TOKEN"],
        origin,
    )
    environment["VECTOROPS_CODE_TEMPLATE_URL"] = templates.url
    frontend = await asyncio.create_subprocess_exec(
        node,
        str(vite),
        "--host",
        "127.0.0.1",
        "--port",
        str(frontend_port),
        "--strictPort",
        cwd=root / "frontend",
        env=environment,
    )
    worker: Optional[asyncio.subprocess.Process] = None
    templates.start()

    async def reload_worker() -> None:
        nonlocal worker
        async for changes in awatch(*paths, watch_filter=PythonFilter(), debounce=500):
            print(
                f"Python changed ({len(changes)} files); restarting worker", flush=True
            )
            if worker is not None:
                await stop_process(worker)
            worker = await asyncio.create_subprocess_exec(*command, env=environment)

    tasks: list[asyncio.Task[object]] = []
    try:
        worker = await asyncio.create_subprocess_exec(*command, env=environment)
        print(f"Development UI: {origin}; WebSocket: ws://127.0.0.1:{port}", flush=True)
        print("Watching: " + ", ".join(map(str, paths)), flush=True)
        print(
            "Local authentication is automatic. Python reload interrupts workflows.",
            flush=True,
        )
        tasks = [
            asyncio.create_task(reload_worker()),
            asyncio.create_task(frontend.wait()),
        ]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        raise RuntimeError("Development frontend exited")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if worker is not None:
            await stop_process(worker)
        await stop_process(frontend)
        await asyncio.to_thread(templates.close)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--frontend-port", type=int, default=5173)
    parser.add_argument("--watch", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    try:
        asyncio.run(
            run(root, args.workspace, args.port, args.frontend_port, args.watch)
        )
    except KeyboardInterrupt:
        pass
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
