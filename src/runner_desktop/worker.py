"""An explicitly owned asyncio worker for in-process desktop execution."""

import asyncio
from concurrent.futures import Future
from pathlib import Path
from threading import Event, Thread
from typing import Final, Optional

from vectorops.runner.control import protocol

from runner_desktop.server import wait_for_shutdown, websocket_server
from runner_desktop.service import DesktopService

START_TIMEOUT: Final = 30.0
STOP_TIMEOUT: Final = 15.0


class Worker:
    def __init__(self, workspace: Path, token: str, origin: str) -> None:
        self.workspace, self.token, self.origin = workspace, token, origin
        self.ready: Future[str] = Future()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.stop: Optional[asyncio.Event] = None
        self.thread = Thread(target=self._run, name="runner-worker", daemon=True)
        self.failure: Optional[BaseException] = None
        self.finished = Event()
        self.stopping = Event()
        self.task: Optional[asyncio.Task[None]] = None

    async def _serve(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.task = asyncio.current_task()
        self.stop = asyncio.Event()
        if self.stopping.is_set():
            return
        async with DesktopService(self.workspace) as service:
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
            if (
                not isinstance(exc, asyncio.CancelledError)
                or not self.stopping.is_set()
            ):
                self.failure = exc
            if not self.ready.done():
                self.ready.set_exception(exc)
        finally:
            self.finished.set()

    def start(self) -> str:
        """Bound readiness waiting; cancellation and joining belong to close()."""
        if self.stopping.is_set():
            raise RuntimeError("Runner worker has already been stopped")
        self.thread.start()
        try:
            return self.ready.result(timeout=START_TIMEOUT)
        except TimeoutError:
            if self.ready.done():
                return self.ready.result()
            self._request_stop(cancel=True)
            raise RuntimeError(
                "Runner startup timed out; reopen the project to retry"
            ) from None

    def _request_stop(self, *, cancel: bool) -> None:
        self.stopping.set()

        def stop() -> None:
            if self.stop is not None:
                self.stop.set()
            if cancel and self.task is not None:
                self.task.cancel()

        if self.loop is not None and self.thread.is_alive():
            try:
                self.loop.call_soon_threadsafe(stop)
            except RuntimeError:
                pass

    def close(self, *, raise_failure: bool = True) -> None:
        """Allow graceful cleanup, then cancel; a stuck daemon cannot block process exit."""
        self._request_stop(cancel=not self.ready.done())
        if self.thread.ident is None:
            return
        self.thread.join(timeout=STOP_TIMEOUT)
        if self.thread.is_alive():
            self._request_stop(cancel=True)
            self.thread.join(timeout=STOP_TIMEOUT)
        if self.thread.is_alive():
            if raise_failure:
                raise RuntimeError(
                    "Runner shutdown timed out; exit the application before reopening this project"
                )
        elif self.failure is not None and raise_failure:
            raise RuntimeError(
                f"Runner worker failed: {self.failure}"
            ) from self.failure
