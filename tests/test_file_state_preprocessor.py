import pytest
from pathlib import Path
from vocode.models import Mode

from vocode.testing import ProjectSandbox
from vocode.runner.executors.llm.preprocessors.base import get_preprocessor
from vocode.runner.executors import file_state as file_state_mod
from vocode.models import PreprocessorSpec
from vocode.state import Message


@pytest.mark.asyncio
async def test_file_state_preprocessor_injects_tracked_files_on_first_run(
    tmp_path: Path,
):
    async with ProjectSandbox.create(tmp_path) as project:
        # Create files
        (project.base_path / "src").mkdir(parents=True, exist_ok=True)
        f1 = project.base_path / "src" / "a.py"
        f1.write_text("print('hi')\n", encoding="utf-8")
        f2 = project.base_path / "README.md"
        f2.write_text("# Title\n\nSome text.\n", encoding="utf-8")

        # Track files via context
        ctx = file_state_mod.get_file_state_ctx(project)
        added, skipped = ctx.add(["src/a.py", "README.md"], project)
        assert added == 2
        assert skipped == 0

        # Ensure preprocessor is registered and callable
        pp = get_preprocessor("file_state")
        assert pp is not None

        # First invocation: both files should be injected
        spec = PreprocessorSpec(name="file_state", options={"prompt": "PROMPT"}, mode=Mode.System)
        messages = [Message(text="", role="system")]
        out_messages = pp.func(project, spec, messages)
        text = out_messages[0].text
        assert "PROMPT" in text
        assert "File: src/a.py" in text
        assert "print('hi')" in text
        assert "File: README.md" in text
        assert "# Title" in text
        assert "```" in text  # fenced


@pytest.mark.asyncio
async def test_file_state_preprocessor_skips_when_unchanged_and_includes_when_changed(
    tmp_path: Path,
):
    async with ProjectSandbox.create(tmp_path) as project:
        (project.base_path / "src").mkdir(parents=True, exist_ok=True)
        f1 = project.base_path / "src" / "a.py"
        f1.write_text("print('hi')\n", encoding="utf-8")
        f2 = project.base_path / "README.md"
        f2.write_text("# Title\n\nSome text.\n", encoding="utf-8")

        ctx = file_state_mod.get_file_state_ctx(project)
        ctx.add(["src/a.py", "README.md"], project)

        pp = get_preprocessor("file_state")
        assert pp is not None

        # First call: inject both
        spec = PreprocessorSpec(name="file_state", options={"prompt": "PROMPT"}, mode=Mode.System)
        messages = [Message(text="", role="system")]
        _ = pp.func(project, spec, messages)

        # Second call with no changes: should return input unchanged (no injection)
        base = "BASE"
        messages_no_change = [Message(text=base, role="user")]
        spec2 = PreprocessorSpec(name="file_state", options={"prompt": "PROMPT"}, mode=Mode.User)
        out_messages_no_change = pp.func(project, spec2, messages_no_change)
        assert out_messages_no_change[0].text == base

        # Modify one file; only that file should be injected on next call
        f1.write_text("print('changed')\n", encoding="utf-8")
        messages_changed = [Message(text="", role="system")]
        spec3 = PreprocessorSpec(name="file_state", options={"prompt": "PROMPT"}, mode=Mode.System)
        out_messages_changed = pp.func(project, spec3, messages_changed)
        out_changed = out_messages_changed[0].text
        assert "PROMPT" in out_changed
        assert "File: src/a.py" in out_changed
        assert "print('changed')" in out_changed
        # Unchanged file should not be included
        assert "File: README.md" not in out_changed

        # Removing a tracked file: it should no longer be considered for injection
        removed, skipped = ctx.remove(["src/a.py"])
        assert removed == 1
        messages_again = [Message(text="", role="system")]
        spec4 = PreprocessorSpec(name="file_state", options={"prompt": "PROMPT"}, mode=Mode.System)
        out_messages_again = pp.func(project, spec4, messages_again)
        again = out_messages_again[0].text
        # No tracked files changed or present now; only the prompt should be there
        assert again == "\n\nPROMPT"


@pytest.mark.asyncio
async def test_file_state_preprocessor_does_not_reinject(tmp_path: Path):
    async with ProjectSandbox.create(tmp_path) as project:
        f1 = project.base_path / "a.py"
        f1.write_text("pass", encoding="utf-8")

        ctx = file_state_mod.get_file_state_ctx(project)
        ctx.add(["a.py"], project)

        pp = get_preprocessor("file_state")
        assert pp is not None

        spec = PreprocessorSpec(name="file_state", mode=Mode.System)

        # Manually inject the content first
        injection, _, _ = file_state_mod._build_file_state_injection(project, "The following files were explicitly added by the developer to your context. Discard and ignore any prior versions of these files you received earlier. Use only the versions provided below.", {})
        messages = [Message(text=injection, role="system")]

        # Running the preprocessor should not change the message
        out_messages = pp.func(project, spec, messages)
        assert out_messages[0].text == injection
