"""An explicitly owned asyncio worker for in-process desktop execution."""

import asyncio
from concurrent.futures import Future
from pathlib import Path
from threading import Thread
from typing import Optional

from vectorops.runner.control import protocol
from vectorops.runner.control.service import RunnerService

from runner_desktop.server import wait_for_shutdown, websocket_server


class Worker:
    def __init__(self, workspace: Path, token: str, origin: str) -> None:
        self.workspace, self.token, self.origin = workspace, token, origin
        self.ready: Future[str] = Future()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.stop: Optional[asyncio.Event] = None
        self.thread = Thread(target=self._run, name="runner-worker")
        self.failure: Optional[BaseException] = None

    async def _serve(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.stop = asyncio.Event()
        async with RunnerService(self.workspace) as service:
            async with websocket_server(service, self.token, [self.origin]) as listener:
                port = next(iter(listener.sockets)).getsockname()[1]
                self.ready.set_result(f"ws://127.0.0.1:{port}")
                await wait_for_shutdown(service, self.stop)
                service.shutdown_mode = "cancel"
                service.shutdown_requested.set()
                after = None
                while True:
                    page = protocol.ActivePage.model_validate(
                        await service.dispatch("executions.active", {"after": after})
                    )
                    for execution in page.executions:
                        try:
                            await service.dispatch(
                                "executions.stop",
                                {"execution_id": execution.execution_id},
                            )
                        except protocol.RemoteError as exc:
                            if exc.code != "execution_conflict":
                                raise
                    after = page.next_after
                    if after is None:
                        break
                await service.aclose()

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:
            self.failure = exc
            if not self.ready.done():
                self.ready.set_exception(exc)

    def start(self) -> str:
        self.thread.start()
        return self.ready.result()

    def close(self) -> None:
        if self.loop is not None and self.stop is not None and self.thread.is_alive():
            try:
                self.loop.call_soon_threadsafe(self.stop.set)
            except RuntimeError:
                pass
        self.thread.join()
        if self.failure is not None:
            raise RuntimeError("Runner worker failed") from self.failure
