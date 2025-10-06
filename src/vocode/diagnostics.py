from __future__ import annotations

import sys
import threading
import traceback
import asyncio
import gc
import time
from typing import IO, Optional, Iterable, Any


def _write_line(fp: IO[str], line: str = "") -> None:
    fp.write(line + "\n")


def _section(fp: IO[str], title: str) -> None:
    _write_line(fp, "=" * 80)
    _write_line(fp, title)
    _write_line(fp, "=" * 80)


def dump_threads(fp: IO[str]) -> None:
    """
    Dump all threads and their stack traces.
    """
    frames = sys._current_frames()
    threads = threading.enumerate()
    _section(fp, f"Threads (count={len(threads)})")
    for t in sorted(threads, key=lambda th: (th.name or "", th.ident or 0)):
        _write_line(fp, f"- Thread name={t.name!r} ident={t.ident} "
                        f"daemon={t.daemon} alive={t.is_alive()}")
        frame = frames.get(t.ident)
        if frame is None:
            _write_line(fp, "  (no frame)")
            continue
        stack_lines = traceback.format_stack(frame)
        for ln in stack_lines:
            for l in ln.rstrip("\n").splitlines():
                _write_line(fp, "    " + l)
        _write_line(fp)


def _format_task_header(task: "asyncio.Task[Any]") -> str:
    name = None
    try:
        name = task.get_name()  # Python 3.8+
    except Exception:
        name = None
    coro_repr = None
    try:
        coro_repr = repr(task.get_coro())
    except Exception:
        coro_repr = None
    state = (
        "cancelled" if task.cancelled()
        else ("done" if task.done() else "pending")
    )
    return f"Task id={id(task)} name={name!r} state={state} coro={coro_repr}"


def _format_exception(ex: BaseException) -> Iterable[str]:
    for ln in traceback.format_exception(type(ex), ex, ex.__traceback__):
        for l in ln.rstrip("\n").splitlines():
            yield l


def dump_async_tasks(loop: Optional[asyncio.AbstractEventLoop], fp: IO[str]) -> None:
    """
    Dump all asyncio tasks and their stack traces for a given loop.
    If loop is None, attempts to get the running loop; otherwise prints a notice.
    """
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _section(fp, "Async Tasks (no running loop detected)")
            _write_line(fp, "No running event loop; cannot enumerate asyncio tasks.")
            _write_line(fp)
            return

    tasks = list(asyncio.all_tasks(loop))
    _section(fp, f"Async Tasks (count={len(tasks)})")
    for task in sorted(tasks, key=lambda t: id(t)):
        _write_line(fp, "- " + _format_task_header(task))
        try:
            if task.done():
                if task.cancelled():
                    _write_line(fp, "    cancelled")
                else:
                    ex = task.exception()
                    if ex is not None:
                        _write_line(fp, "    exception:")
                        for l in _format_exception(ex):
                            _write_line(fp, "      " + l)
                    else:
                        # Avoid calling result() to not risk heavy object reprs or side effects
                        _write_line(fp, "    finished successfully (no stack available)")
            else:
                stack = task.get_stack(limit=None)
                if not stack:
                    _write_line(fp, "    (no stack frames)")
                for fr in stack:
                    for ln in traceback.format_stack(fr):
                        for l in ln.rstrip("\n").splitlines():
                            _write_line(fp, "      " + l)
        except Exception as e:
            _write_line(fp, f"    <error while dumping task: {e!r}>")
        _write_line(fp)


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _is_asyncio_primitive(obj: Any) -> bool:
    try:
        mod = type(obj).__module__ or ""
        cls = type(obj).__name__
        if not mod.startswith("asyncio"):
            return False
        return cls in {
            "Lock",
            "Event",
            "Condition",
            "Semaphore",
            "BoundedSemaphore",
            "Queue",
        }
    except Exception:
        return False


def _describe_waiters(
    waiters: Any, wait_map: "dict[int, list[asyncio.Task[Any]]]", fp: IO[str]
) -> None:
    try:
        if waiters is None:
            _write_line(fp, "      None")
            return

        try:
            waiter_list = list(waiters)
            ln = len(waiter_list)
        except Exception:
            waiter_list = []
            ln = None

        _write_line(
            fp,
            f"      type={type(waiters).__name__}"
            + (f" len={ln}" if ln is not None else ""),
        )

        if not waiter_list:
            return

        for i, w in enumerate(waiter_list):
            tasks = wait_map.get(id(w), [])
            state = ""
            if isinstance(w, asyncio.Future):
                state = "done" if w.done() else "pending"
            _write_line(
                fp,
                f"      [{i}] {type(w).__name__} at {hex(id(w))}"
                + (f" state={state}" if state else ""),
            )

            if not tasks:
                _write_line(fp, "        (no waiting task found)")
                continue

            for task in tasks:
                _write_line(fp, f"        waiting task: {_format_task_header(task)}")
                try:
                    stack = task.get_stack(limit=None)
                    if not stack:
                        _write_line(fp, "          (no stack frames)")
                    else:
                        for fr in stack:
                            for ln in traceback.format_stack(fr):
                                for l in ln.rstrip("\n").splitlines():
                                    _write_line(fp, "            " + l)
                except Exception as e:
                    _write_line(fp, f"          <error getting stack: {e!r}>")
    except Exception as e:
        _write_line(fp, f"      <error describing waiters: {e!r}>")


def dump_async_locks(loop: Optional[asyncio.AbstractEventLoop], fp: IO[str]) -> None:
    """
    Scan GC for asyncio synchronization primitives and dump basic state.
    """
    wait_map: dict[int, list[asyncio.Task[Any]]] = {}
    if loop:
        for task in asyncio.all_tasks(loop):
            coro = task.get_coro()
            # cr_await exists on coroutine objects, check for it.
            if hasattr(coro, "cr_await"):
                awaited_obj = coro.cr_await
                if awaited_obj is not None:
                    # Using id() for the key as awaitables may not be hashable
                    wait_map.setdefault(id(awaited_obj), []).append(task)

    objs = gc.get_objects()
    primitives = [o for o in objs if _is_asyncio_primitive(o)]
    _section(fp, f"Async Locks/Primitives (count={len(primitives)})")
    for o in primitives:
        cls = type(o).__name__
        _write_line(fp, f"- {cls} at {hex(id(o))}")
        try:
            if cls == "Lock":
                locked = _safe_getattr(o, "locked", None)
                locked_state = (
                    bool(locked())
                    if callable(locked)
                    else bool(_safe_getattr(o, "_locked", False))
                )
                _write_line(fp, f"    locked={locked_state}")
                waiters = _safe_getattr(o, "_waiters", None)
                _write_line(fp, "    waiters:")
                _describe_waiters(waiters, wait_map, fp)
            elif cls == "Event":
                is_set = False
                is_set_fn = _safe_getattr(o, "is_set", None)
                if callable(is_set_fn):
                    try:
                        is_set = bool(is_set_fn())
                    except Exception:
                        pass
                _write_line(fp, f"    is_set={is_set}")
                waiters = _safe_getattr(o, "_waiters", None)
                _write_line(fp, "    waiters:")
                _describe_waiters(waiters, wait_map, fp)
            elif cls in ("Semaphore", "BoundedSemaphore"):
                value = _safe_getattr(o, "_value", None)
                _write_line(fp, f"    value={value}")
                waiters = _safe_getattr(o, "_waiters", None)
                _write_line(fp, "    waiters:")
                _describe_waiters(waiters, wait_map, fp)
            elif cls == "Condition":
                lock = _safe_getattr(o, "_lock", None)
                _write_line(
                    fp,
                    f"    lock={type(lock).__name__ if lock else None} id={hex(id(lock)) if lock else None}",
                )
                waiters = _safe_getattr(o, "_waiters", None)
                _write_line(fp, "    waiters:")
                _describe_waiters(waiters, wait_map, fp)
            else:
                # Other asyncio primitives like Queue
                waiters_put = _safe_getattr(o, "_putters", None)
                waiters_get = _safe_getattr(o, "_getters", None)
                size = _safe_getattr(o, "qsize", None)
                if callable(size):
                    try:
                        size = size()
                    except Exception:
                        pass
                _write_line(fp, f"    size={size}")
                _write_line(fp, "    putters:")
                _describe_waiters(waiters_put, wait_map, fp)
                _write_line(fp, "    getters:")
                _describe_waiters(waiters_get, wait_map, fp)
        except Exception as e:
            _write_line(fp, f"    <error while dumping primitive: {e!r}>")
        _write_line(fp)


def dump_all(loop: Optional[asyncio.AbstractEventLoop] = None, fp: IO[str] = sys.stderr) -> None:
    """
    Orchestrate a complete process dump (threads, async tasks, async locks).
    Safe to call from a prompt_toolkit keybinding via run_in_terminal.
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    _section(fp, f"Process Diagnostics at {ts}")
    dump_threads(fp)
    dump_async_tasks(loop, fp)
    dump_async_locks(loop, fp)
    _write_line(fp, "End of diagnostics.")
    try:
        fp.flush()
    except Exception:
        pass
