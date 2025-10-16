import pytest
from pathlib import Path

from vocode.testing import ProjectSandbox
from vocode.runner.executors.llm.preprocessors.base import get_preprocessor
from vocode.runner.executors.llm.preprocessors import fileread as fileread_mod  # ensure registration on import
from vocode.models import Mode, PreprocessorSpec
from vocode.state import Message


@pytest.mark.asyncio
async def test_fileread_preprocessor_concatenates_and_skips_invalid(tmp_path: Path):
    async with ProjectSandbox.create(tmp_path) as project:
        # Create files and a directory
        (project.base_path / "src").mkdir(parents=True, exist_ok=True)
        (project.base_path / "dir").mkdir(parents=True, exist_ok=True)
        f1 = project.base_path / "src" / "a.txt"
        f2 = project.base_path / "b.txt"
        f1.write_text("A", encoding="utf-8")
        f2.write_text("B", encoding="utf-8")

        pp = get_preprocessor("file_read")
        assert pp is not None

        # Include valid file, missing file, another valid file, and a directory
        spec = PreprocessorSpec(
            name="file_read",
            options={"paths": ["src/a.txt", "missing.txt", "b.txt", "dir"]},
            mode=Mode.User,
        )
        messages = [Message(text="BASE", role="user")]
        out_messages = pp.func(project, spec, messages)
        # Concatenate valid files only, appended to the base text (no separators)
        assert out_messages[0].text == "BASEAB"


@pytest.mark.asyncio
async def test_fileread_preprocessor_prepend(tmp_path: Path):
    async with ProjectSandbox.create(tmp_path) as project:
        f1 = project.base_path / "a.txt"
        f2 = project.base_path / "b.txt"
        f1.write_text("A", encoding="utf-8")
        f2.write_text("B", encoding="utf-8")

        pp = get_preprocessor("file_read")
        assert pp is not None

        spec = PreprocessorSpec(
            name="file_read",
            options={"paths": ["a.txt", "b.txt"]},
            prepend=True,
            mode=Mode.User,
        )
        messages = [Message(text="X", role="user")]
        out_messages = pp.func(project, spec, messages)
        # Prepend concatenated content before the input text
        assert out_messages[0].text == "ABX"


@pytest.mark.asyncio
async def test_fileread_preprocessor_does_not_reinject(tmp_path: Path):
    async with ProjectSandbox.create(tmp_path) as project:
        f1 = project.base_path / "a.txt"
        f1.write_text("-X-", encoding="utf-8")

        pp = get_preprocessor("file_read")
        assert pp is not None

        spec = PreprocessorSpec(
            name="file_read",
            options={"paths": ["a.txt"]},
            mode=Mode.User,
        )
        messages = [Message(text="BASE", role="user")]
        out_messages = pp.func(project, spec, messages)
        assert out_messages[0].text == "BASE-X-"

        # Calling it again should not change the message
        final_messages = pp.func(project, spec, out_messages)
        assert final_messages[0].text == "BASE-X-"
