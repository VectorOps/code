import pytest

from pathlib import Path

from vocode.testing import ProjectSandbox
from vocode.runner.executors.result import ResultNode, ResultExecutor
from vocode.runner.models import ExecRunInput, PACKET_FINAL_MESSAGE
from vocode.state import Message


@pytest.mark.asyncio
async def test_result_executor_concatenates_input_messages(tmp_path: Path):
    async with ProjectSandbox.create(tmp_path) as project:
        node = ResultNode(name="res")
        executor = ResultExecutor(config=node, project=project)

        inputs = [
            Message(role="user", text="first"),
            Message(role="system", text="second"),
            Message(role="agent", text="third"),
        ]

        agen = executor.run(ExecRunInput(messages=inputs))
        packet, state = await agen.__anext__()

        assert state is None
        assert packet.kind == PACKET_FINAL_MESSAGE
        assert packet.message is not None
        assert packet.message.text == "first\nsecond\nthird"

        with pytest.raises(StopAsyncIteration):
            await agen.__anext__()