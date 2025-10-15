import pytest
from pathlib import Path

from vocode.testing import ProjectSandbox
from vocode.runner.runner import Runner
from vocode.state import Assignment
from vocode.runner.models import RunInput, PACKET_FINAL_MESSAGE
from vocode.models import Graph, Workflow
from vocode.runner.executors.fileread import FileReadNode


@pytest.mark.asyncio
async def test_fileread_executor_concatenates_file_contents(tmp_path: Path):
    async with ProjectSandbox.create(tmp_path) as project:
        # Create files within the project
        (project.base_path / "src").mkdir(parents=True, exist_ok=True)
        f1 = project.base_path / "src" / "one.txt"
        f2 = project.base_path / "two.txt"
        c1 = "Hello"
        c2 = " World!\n"
        f1.write_text(c1, encoding="utf-8")
        f2.write_text(c2, encoding="utf-8")

        # Build workflow with a single FileRead node; set confirmation=auto
        node = FileReadNode(name="fr", files=["src/one.txt", "two.txt"], confirmation="auto")
        graph = Graph(nodes=[node], edges=[])
        wf = Workflow(name="wf", graph=graph)
        runner = Runner(wf, project)

        assign = Assignment()
        agen = runner.run(assign)
        event = await agen.__anext__()
        assert event.event.kind == PACKET_FINAL_MESSAGE
        text = event.event.message.text

        # Expect exact concatenation in given order
        assert text == c1 + c2

        with pytest.raises(StopAsyncIteration):
            await agen.asend(RunInput())


@pytest.mark.asyncio
async def test_fileread_executor_rejects_invalid_paths(tmp_path: Path):
    async with ProjectSandbox.create(tmp_path) as project:
        node = FileReadNode(name="fr", files=["../escape.txt"], confirmation="auto")
        graph = Graph(nodes=[node], edges=[])
        wf = Workflow(name="wf", graph=graph)
        runner = Runner(wf, project)

        assign = Assignment()
        agen = runner.run(assign)
        with pytest.raises(Exception):
            await agen.__anext__()