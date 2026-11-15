"""Store launcher connection tokens independently of Runner and project data."""

import hashlib
import os
from pathlib import Path
import secrets
import stat
from typing import Literal

from vectorops.runner.control.auth import validate_token


def token_path(kind: Literal["workers", "connections"], identity: str) -> Path:
    root = Path(
        os.environ.get(
            "VECTOROPS_CODE_CONFIG_DIR",
            str(Path.home() / ".config" / "vectorops-code"),
        )
    ).expanduser()
    name = hashlib.sha256(identity.encode()).hexdigest()
    return root / kind / f"{name}.token"


def read_token(path: Path) -> str:
    with path.open() as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Token file must be a regular file: {path}")
        if os.name == "posix" and (
            info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise ValueError(f"Token file must be owned by you with mode 600: {path}")
        try:
            return validate_token(stream.read(4098).strip())
        except ValueError:
            raise ValueError(f"Invalid connection token in {path}") from None


def save_token(path: Path, token: str) -> None:
    token = validate_token(token)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(token + "\n")


def worker_token(path: Path) -> str:
    try:
        return read_token(path)
    except FileNotFoundError:
        token = secrets.token_urlsafe(32)
        try:
            save_token(path, token)
        except FileExistsError:
            return read_token(path)
        return token
