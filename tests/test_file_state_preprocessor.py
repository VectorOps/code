import pytest
from pathlib import Path

from vocode.testing import ProjectSandbox
from vocode.runner.preprocessors.base import get_preprocessor
from vocode.runner.executors import file_state as file_state_mod


@pytest.mark.asyncio
async def test_file_state_preprocessor_injects_tracked_files_on_first_run(tmp_path: Path):
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
        text = pp.func(project, "", {"prompt": "PROMPT"})
        assert "PROMPT" in text
        assert "File: src/a.py" in text
        assert "print('hi')" in text
        assert "File: README.md" in text
        assert "# Title" in text
        assert "```" in text  # fenced


@pytest.mark.asyncio
async def test_file_state_preprocessor_skips_when_unchanged_and_includes_when_changed(tmp_path: Path):
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
        _ = pp.func(project, "", {"prompt": "PROMPT"})

        # Second call with no changes: should return input unchanged (no injection)
        base = "BASE"
        out_no_change = pp.func(project, base, {"prompt": "PROMPT"})
        assert out_no_change == base

        # Modify one file; only that file should be injected on next call
        f1.write_text("print('changed')\n", encoding="utf-8")
        out_changed = pp.func(project, "", {"prompt": "PROMPT"})
        assert "PROMPT" in out_changed
        assert "File: src/a.py" in out_changed
        assert "print('changed')" in out_changed
        # Unchanged file should not be included
        assert "File: README.md" not in out_changed

        # Removing a tracked file: it should no longer be considered for injection
        removed, skipped = ctx.remove(["src/a.py"])
        assert removed == 1
        again = pp.func(project, "", {"prompt": "PROMPT"})
        # No tracked files changed or present now; nothing injected
        assert again == ""
