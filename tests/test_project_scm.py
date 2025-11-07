import os
from pathlib import Path
import tempfile

import pytest

from vocode import project
from vocode.project import Project


class DummyKnow:
    async def start(self, *_args, **_kwargs):
        return None

    async def shutdown(self):
        return None

    async def refresh(self, repo=None):
        return None


@pytest.fixture(autouse=True)
def patch_know(monkeypatch):
    # Prevent KnowProject from doing heavy initialization in tests.
    monkeypatch.setattr(project, "KnowProject", lambda *args, **kwargs: DummyKnow())
    yield

def _write_min_config(base: Path) -> Path:
    cfg = base / ".vocode" / "config.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("tools: []\n", encoding="utf-8")
    return cfg


def test_find_ancestor_config(tmp_path):
    root = tmp_path / "root"
    ancestor = root / "ancestor"
    nested = ancestor / "nested" / "deep"
    nested.mkdir(parents=True)
    cfg_dir = ancestor / ".vocode"
    cfg_dir.mkdir(parents=True)
    cfg_file = cfg_dir / "config.yaml"
    cfg_file.write_text("tools: []\n")

    proj = Project.from_base_path(nested, search_ancestors=True, use_scm=False)
    assert proj.base_path == ancestor
    assert proj.config_path == ancestor / ".vocode" / "config.yaml"


def test_use_git_repo_as_base(tmp_path):
    # initialize a git repo and ensure repo root is chosen when no ancestor config exists
    import git

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    repo = git.Repo.init(str(repo_root))
    # ensure minimal config at the repo root so Settings.tools is a list
    _write_min_config(repo_root)
    nested = repo_root / "a" / "b"
    nested.mkdir(parents=True)

    proj = Project.from_base_path(nested, search_ancestors=True, use_scm=True)
    assert proj.base_path == repo_root
    assert (repo_root / ".vocode" / "config.yaml").exists()


def test_fallback_to_start_dir(tmp_path):
    start = tmp_path / "start" / "here"
    start.mkdir(parents=True)
    # write minimal config at the start dir so validation succeeds
    _write_min_config(start)

    proj = Project.from_base_path(start, search_ancestors=True, use_scm=True)
    assert proj.base_path == start
    assert (start / ".vocode" / "config.yaml").exists()


def test_disable_search_flags(tmp_path):
    root = tmp_path / "root"
    ancestor = root / "ancestor"
    nested = ancestor / "nested"
    nested.mkdir(parents=True)
    cfg_dir = ancestor / ".vocode"
    cfg_dir.mkdir(parents=True)
    cfg_file = cfg_dir / "config.yaml"
    cfg_file.write_text("tools: []\n")
    # since search and SCM are disabled, init should use the start dir; ensure it has a minimal config
    _write_min_config(nested)

    # disable ancestor search and SCM, so base should be the start dir
    proj = Project.from_base_path(nested, search_ancestors=False, use_scm=False)
    assert proj.base_path == nested
    assert (nested / ".vocode" / "config.yaml").exists()