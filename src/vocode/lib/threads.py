import threading
import queue
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar, Generic, cast, TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio

T = TypeVar("T")


class Result:
    """Holds either a value or an exception."""
    __slots__ = ("value", "exc")

    def __init__(self, value: Any = None, exc: BaseException | None = None):
        self.value = value
        self.exc = exc


@dataclass
class _Call:
    call_id: str
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    done: threading.Event
    result_slot: list[Result]  # single-element list as a mutable cell
    caller_ident: int


# -------- Async-generator plumbing --------

class _Yield:
    __slots__ = ("value",)
    def __init__(self, value: Any):
        self.value = value

class _GenEnd:
    __slots__ = ()

class _GenError:
    __slots__ = ("exc",)
    def __init__(self, exc: BaseException):
        self.exc = exc

@dataclass
class _GenSession:
    """A unidirectional streaming channel from worker -> caller."""
    id: str
    out_q: "queue.Queue[Any]"          # carries _Yield / _GenEnd / _GenError
    cancel: threading.Event            # set when consumer closes
    done: threading.Event              # set by worker when finished

@dataclass
class _GenCall:
    call_id: str
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    session: _GenSession
    caller_ident: int


class RpcThread:
    """
    Dedicated thread that executes proxied calls.

    - proxy():      decorator for normal functions (sync call, returns result)
    - async_proxy():decorator for normal functions returning awaitable (async callers)
    - async_gen_proxy(): decorator for SYNC GENERATOR functions -> becomes ASYNC GENERATOR to caller
    """

    def __init__(self, name: str = "rpc-thread", max_queue: int = 0):
        self._name = name
        self._q: "queue.Queue[_Call | _GenCall | None]" = queue.Queue(max_queue)
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._started = threading.Event()
        self._stopped = threading.Event()
        self._ident_lock = threading.Lock()
        self._thread_ident: Optional[int] = None

    # ---- lifecycle ---------------------------------------------------------

    def start(self) -> "RpcThread":
        if self._started.is_set():
            return self
        self._thread.start()
        self._started.wait()
        return self

    def shutdown(self, *, wait: bool = True) -> None:
        if self._stopped.is_set():
            return
        self._q.put(None)  # sentinel
        if wait:
            self._thread.join()
        self._stopped.set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
        return False

    # ---- decorators: functions --------------------------------------------

    def proxy(self, *, timeout: float | None = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        Decorator: proxy function execution to the RPC thread.
        Blocks caller and re-raises exceptions. If invoked from worker, runs inline.
        """
        def decorate(fn: Callable[..., T]) -> Callable[..., T]:
            def wrapper(*args: Any, **kwargs: Any) -> T:
                if self.on_worker_thread():
                    return fn(*args, **kwargs)

                self._ensure_started()

                done = threading.Event()
                res_slot: list[Result] = [Result()]
                call = _Call(
                    call_id=str(uuid.uuid4()),
                    fn=fn,
                    args=args,
                    kwargs=kwargs,
                    done=done,
                    result_slot=res_slot,
                    caller_ident=threading.get_ident(),
                )
                self._q.put(call)
                ok = done.wait(timeout)
                if not ok:
                    raise TimeoutError(f"RPC call to {fn.__name__} timed out after {timeout} seconds")

                r = res_slot[0]
                if r.exc is not None:
                    raise r.exc
                return cast(T, r.value)
            wrapper.__name__ = getattr(fn, "__name__", "rpc_proxy")
            wrapper.__doc__ = getattr(fn, "__doc__", None)
            wrapper.__wrapped__ = fn
            return wrapper
        return decorate

    def async_proxy(self, *, timeout: float | None = None) -> Callable[[Callable[..., T]], Callable[..., "asyncio.Future[T]"]]:
        """Decorator for asyncio callers: submit to worker and await the result."""
        import asyncio
        def decorate(fn: Callable[..., T]) -> Callable[..., "asyncio.Future[T]"]:
            proxy_sync = self.proxy(timeout=timeout)(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> T:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: proxy_sync(*args, **kwargs))
            wrapper.__name__ = getattr(fn, "__name__", "rpc_async_proxy")
            wrapper.__doc__ = getattr(fn, "__doc__", None)
            wrapper.__wrapped__ = fn
            return wrapper
        return decorate

    # ---- decorators: async generator over a sync generator -----------------

    def async_gen_proxy(
        self,
        *,
        queue_maxsize: int = 1,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator for SYNC GENERATOR functions: exposes them as ASYNC GENERATORS.

        Example:
            @rpc.async_gen_proxy()
            def count(n):
                for i in range(n):
                    yield i

            async for x in count(5):
                ...
        """
        import asyncio

        def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
            def call(*args: Any, **kwargs: Any):
                # If called from worker, just return the original (sync) generator directly.
                if self.on_worker_thread():
                    return fn(*args, **kwargs)

                self._ensure_started()

                session = _GenSession(
                    id=str(uuid.uuid4()),
                    out_q=queue.Queue(maxsize=queue_maxsize),
                    cancel=threading.Event(),
                    done=threading.Event(),
                )
                gcall = _GenCall(
                    call_id=session.id,
                    fn=fn,
                    args=args,
                    kwargs=kwargs,
                    session=session,
                    caller_ident=threading.get_ident(),
                )
                self._q.put(gcall)

                async def agen():
                    loop = asyncio.get_running_loop()
                    try:
                        while True:
                            item = await loop.run_in_executor(None, session.out_q.get)
                            if isinstance(item, _Yield):
                                yield item.value
                            elif isinstance(item, _GenEnd):
                                break
                            elif isinstance(item, _GenError):
                                raise item.exc
                            else:
                                # Defensive: unexpected payload
                                raise RuntimeError("Invalid generator message")
                    finally:
                        # Signal the worker to stop iterating ASAP (if still running)
                        session.cancel.set()
                        # Best-effort: wait briefly for worker to finish flush
                        # (no await here; thread-side event, so just return)
                        # If you want to ensure completion, you can block in executor:
                        await loop.run_in_executor(None, session.done.wait)

                return agen()
            call.__name__ = getattr(fn, "__name__", "rpc_async_gen_proxy")
            call.__doc__ = getattr(fn, "__doc__", None)
            call.__wrapped__ = fn
            return call

        return decorate

    # ---- internals ---------------------------------------------------------

    def _ensure_started(self):
        if not self._started.is_set():
            self.start()

    def on_worker_thread(self) -> bool:
        with self._ident_lock:
            return self._thread_ident is not None and threading.get_ident() == self._thread_ident

    def _run(self) -> None:
        with self._ident_lock:
            self._thread_ident = threading.get_ident()
        self._started.set()

        try:
            while True:
                item = self._q.get()
                if item is None:
                    break

                # Normal function call
                if isinstance(item, _Call):
                    try:
                        value = item.fn(*item.args, **item.kwargs)
                        item.result_slot[0] = Result(value=value)
                    except BaseException as e:
                        item.result_slot[0] = Result(exc=e)
                    finally:
                        item.done.set()
                    continue

                # Generator call
                if isinstance(item, _GenCall):
                    session = item.session
                    try:
                        gen = item.fn(*item.args, **item.kwargs)
                        # Validate it's an iterator/generator
                        it = iter(gen)
                        while True:
                            if session.cancel.is_set():
                                # Cooperative stop
                                try:
                                    if hasattr(it, "close"):
                                        it.close()  # type: ignore[attr-defined]
                                except BaseException as e:
                                    # Send termination error if close fails
                                    session.out_q.put(_GenError(e))
                                break
                            try:
                                val = next(it)
                            except StopIteration:
                                session.out_q.put(_GenEnd())
                                break
                            except BaseException as e:
                                session.out_q.put(_GenError(e))
                                break
                            # Back-pressure: blocks if consumer is slow
                            session.out_q.put(_Yield(val))
                    finally:
                        session.done.set()
                    continue

                # Defensive
                raise RuntimeError("Unknown queue item type")
        finally:
            with self._ident_lock:
                self._thread_ident = None
            self._stopped.set()
