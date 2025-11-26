import asyncio
import io

import pytest

from vocode import diagnostics


async def _sleeping_task(evt: asyncio.Event) -> None:
    await evt.wait()


@pytest.mark.asyncio
async def test_dump_async_tasks_includes_stack_without_errors() -> None:
    loop = asyncio.get_running_loop()
    evt = asyncio.Event()

    task = loop.create_task(_sleeping_task(evt), name="diagnostics-test-task")

    # Let the task start and reach its await point
    await asyncio.sleep(0)

    buf = io.StringIO()
    diagnostics.dump_async_tasks(loop, buf)
    output = buf.getvalue()

    # Clean up the task
    evt.set()
    await task

    # Previously, this would contain a TypeError from traceback.format_list
    assert "<error while dumping task:" not in output

    # Ensure our task and its function name appear in the dump
    assert "diagnostics-test-task" in output
    assert "_sleeping_task" in output