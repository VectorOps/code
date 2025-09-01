import pytest
import threading
import time
from vocode.lib.threads import RpcThread


def test_start_returns_self_and_is_idempotent():
    """Tests that start() returns self and can be called multiple times safely."""
    rpc = RpcThread(name="test-idempotent")
    try:
        assert not rpc._started.is_set()

        # First call should start and return self
        r1 = rpc.start()
        assert r1 is rpc
        assert rpc._started.is_set()

        # Second call should be a no-op and return self
        r2 = rpc.start()
        assert r2 is rpc
    finally:
        rpc.shutdown()


def test_start_chaining():
    """Tests that RpcThread can be instantiated and started in a single expression."""
    rpc = RpcThread(name="test-chaining").start()
    try:
        assert isinstance(rpc, RpcThread)
        assert rpc._started.is_set()
        # A simple check to ensure the thread is running
        assert rpc._thread.is_alive()
    finally:
        rpc.shutdown()


def test_automatic_start_on_first_call():
    """Tests that the thread starts automatically when a proxied function is called."""
    rpc = RpcThread(name="test-auto-start")
    try:
        assert not rpc._started.is_set()

        @rpc.proxy()
        def get_ident():
            return threading.get_ident()

        # First call will trigger start()
        get_ident()

        assert rpc._started.is_set()
    finally:
        rpc.shutdown()


def test_proxy_execution_on_worker_thread():
    """Tests that a proxied function executes on the worker thread."""
    with RpcThread(name="test-proxy-exec") as rpc:
        @rpc.proxy()
        def get_ident():
            return threading.get_ident()

        caller_ident = threading.get_ident()
        worker_ident = get_ident()

        assert caller_ident != worker_ident
        assert worker_ident == rpc._thread_ident


def test_proxy_inline_execution_on_worker():
    """Tests that calling a proxied function from the worker thread runs it inline."""
    with RpcThread(name="test-inline-exec") as rpc:
        # This inner function will be called from the outer one, which is already on the worker thread
        @rpc.proxy()
        def get_ident_inner():
            # This should run inline, not be re-queued.
            return threading.get_ident()

        # This outer function gets queued and runs on the worker thread.
        @rpc.proxy()
        def get_ident_outer_and_inner():
            # We are on the worker thread now.
            outer_ident = threading.get_ident()
            # This call should be inline.
            inner_ident = get_ident_inner()
            assert outer_ident == inner_ident
            return inner_ident

        worker_ident = get_ident_outer_and_inner()
        assert worker_ident == rpc._thread_ident


def test_proxy_exception_propagation():
    """Tests that exceptions from the proxied function are re-raised in the caller."""
    with RpcThread(name="test-exceptions") as rpc:
        @rpc.proxy()
        def raise_error():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            raise_error()


def test_context_manager():
    """Tests the __enter__ and __exit__ methods for setup and teardown."""
    with RpcThread(name="test-ctx-mgr") as rpc:
        assert rpc._started.is_set()
        assert rpc._thread.is_alive()
        assert not rpc._stopped.is_set()

    assert rpc._stopped.is_set()
    # After shutdown, a short wait might be needed for the thread to fully terminate
    rpc._thread.join(timeout=1)
    assert not rpc._thread.is_alive()


@pytest.mark.asyncio
async def test_async_proxy():
    """Tests the async_proxy decorator for awaiting sync functions."""
    with RpcThread(name="test-async-proxy") as rpc:
        @rpc.async_proxy()
        def add(a, b):
            time.sleep(0.01)  # Simulate work
            return a + b

        result = await add(5, 10)
        assert result == 15


@pytest.mark.asyncio
async def test_async_gen_proxy():
    """Tests that a sync generator can be consumed as an async generator."""
    with RpcThread(name="test-async-gen") as rpc:
        @rpc.async_gen_proxy()
        def count_up(n):
            for i in range(n):
                yield i

        results = [item async for item in count_up(5)]
        assert results == list(range(5))
