import pytest
from pathlib import Path

from vocode.testing import ProjectSandbox
from vocode.runner.runner import Runner
from vocode.state import Assignment
from vocode.runner.models import RunInput, PACKET_FINAL_MESSAGE
from vocode.models import Graph, Workflow
from vocode.runner.executors.file_state import FileStateNode
from vocode.commands import CommandContext


@pytest.mark.asyncio
async def test_file_state_executor_outputs_files_with_fences(tmp_path: Path):
    async with ProjectSandbox.create(tmp_path) as project:
        # Create files within the project
        (project.base_path / "src").mkdir(parents=True, exist_ok=True)
        f1 = project.base_path / "src" / "a.py"
        f1.write_text("print('hi')\n", encoding="utf-8")
        f2 = project.base_path / "README.md"
        f2.write_text("# Title\n\nSome text.\n", encoding="utf-8")

        # Build workflow with a single FileState node; set confirmation=auto
        graph = Graph(nodes=[FileStateNode(name="fs", confirmation="auto")], edges=[])
        wf = Workflow(name="wf", graph=graph)
        runner = Runner(wf, project)

        # Commands are registered by executor initialization in Runner
        ctx = CommandContext(project=project, ui=None)  # ui not used
        msg1 = await project.commands.execute("fadd", ctx, ["src/a.py", "README.md"])
        assert msg1 is not None and "added=2" in msg1

        # Sanity: list shows added files
        lst = await project.commands.execute("flist", ctx, [])
        assert "src/a.py" in (lst or "")
        assert "README.md" in (lst or "")

        # Run the workflow; first event should be final message
        assign = Assignment()
        agen = runner.run(assign)
        event = await agen.__anext__()
        assert event.event.kind == PACKET_FINAL_MESSAGE
        text = event.event.message.text

        # Prompt present
        assert "The following" in text

        # File headers and language fences + content included
        assert "File: src/a.py" in text
        assert "```" in text
        assert "print('hi')" in text

        assert "File: README.md" in text
        assert "```" in text
        assert "# Title" in text

        # Finish the generator
        with pytest.raises(StopAsyncIteration):
            await agen.asend(RunInput())

        # Modify one file and run again; only changed files should be included
        f1.write_text("print('hello again')\n", encoding="utf-8")

        runner2 = Runner(wf, project)
        agen2 = runner2.run(assign)
        event2 = await agen2.__anext__()
        assert event2.event.kind == PACKET_FINAL_MESSAGE
        text2 = event2.event.message.text

        # Prompt present
        assert "The following" in text2

        # Only the changed file should be included
        assert "File: src/a.py" in text2
        assert "print('hello again')" in text2
        assert "File: README.md" not in text2

        with pytest.raises(StopAsyncIteration):
            await agen2.asend(RunInput())

        # Remove a file and verify removal works
        msg2 = await project.commands.execute("fdel", ctx, ["README.md"])
        assert msg2 is not None and "removed=1" in msg2
