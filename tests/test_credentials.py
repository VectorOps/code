import os
from pathlib import Path
import stat

import pytest

from vectorops_code import credentials


def test_generated_worker_tokens_persist_and_are_private(tmp_path: Path) -> None:
    path = tmp_path / "workers" / "token"
    first = credentials.worker_token(path)
    assert len(first) >= 43
    assert credentials.worker_token(path) == first
    assert credentials.worker_token(tmp_path / "other") != first
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_token_scopes_are_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VECTOROPS_CODE_CONFIG_DIR", str(tmp_path))
    worker = credentials.token_path("workers", "identity")
    connection = credentials.token_path("connections", "identity")
    assert worker != connection
    assert worker.parent == tmp_path / "workers"
    assert connection != credentials.token_path("connections", "other")


def test_invalid_token_does_not_leak_or_get_replaced(tmp_path: Path) -> None:
    path = tmp_path / "token"
    path.touch(mode=0o600)
    path.write_text("secret-invalid-token")
    with pytest.raises(ValueError, match="Invalid connection token") as error:
        credentials.worker_token(path)
    assert "secret-invalid-token" not in str(error.value)
    assert path.read_text() == "secret-invalid-token"


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_shared_token_files_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "token"
    credentials.save_token(path, "x" * 43)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="mode 600"):
        credentials.read_token(path)


def test_save_never_overwrites_existing_tokens(tmp_path: Path) -> None:
    path = tmp_path / "token"
    credentials.save_token(path, "x" * 43)
    with pytest.raises(FileExistsError):
        credentials.save_token(path, "y" * 43)
    assert credentials.read_token(path) == "x" * 43
