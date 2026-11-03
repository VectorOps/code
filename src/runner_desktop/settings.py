"""Host-owned desktop preferences shared across projects and frontend origins."""

import asyncio
from pathlib import Path
from typing import Literal, Optional

import pydantic
from vectorops.runner import global_config
from vectorops.runner.control import protocol


class HistoryLimits(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", strict=True)

    initial: int = pydantic.Field(default=1000, ge=1, le=9007199254740991)
    maximum: int = pydantic.Field(default=2000, ge=1, le=9007199254740991)
    target: int = pydantic.Field(default=1000, ge=1, le=9007199254740991)
    older: int = pydantic.Field(default=200, ge=1, le=500)

    @pydantic.model_validator(mode="after")
    def retention(self) -> "HistoryLimits":
        if max(self.initial, self.target) > self.maximum:
            raise ValueError("retention ceiling must cover initial and target counts")
        return self


class Settings(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", strict=True)

    appearance: Literal["system", "light", "dark"] = "system"
    default_workflow: str = pydantic.Field(
        default="agent", min_length=1, max_length=4096
    )
    history_limits: HistoryLimits = pydantic.Field(default_factory=HistoryLimits)
    debug_mode: Literal["auto", "on", "off"] = "auto"


class Snapshot(Settings):
    development: bool = False
    debug_enabled: bool = False


class Store:
    def __init__(self, path: Path, *, development: bool = False) -> None:
        self.path = path
        self.development = development

    def _read(self) -> Settings:
        if not self.path.exists():
            return Settings()
        try:
            return Settings.model_validate_json(self.path.read_bytes())
        except pydantic.ValidationError as exc:
            raise OSError("invalid desktop settings file") from exc

    def _access(self, patch: Optional[dict[str, pydantic.JsonValue]]) -> Settings:
        global_config.private_directory(self.path.parent)
        with global_config.file_lock(self.path.with_suffix(".lock")):
            value = self._read()
            if patch is not None:
                value = Settings.model_validate({**value.model_dump(), **patch})
                global_config.atomic_write(self.path, value.model_dump_json().encode())
            return value

    async def dispatch(
        self, method: str, params: dict[str, pydantic.JsonValue]
    ) -> pydantic.JsonValue:
        """Merge field-level updates under the host lock without blocking the loop."""
        if method not in ("desktop.settings.get", "desktop.settings.update"):
            raise protocol.RemoteError(
                "unsupported_method", "unknown desktop settings method"
            )
        if method == "desktop.settings.get" and params:
            raise protocol.RemoteError(
                "invalid_argument", "settings.get takes no parameters"
            )
        try:
            value = await asyncio.to_thread(
                self._access, params if method == "desktop.settings.update" else None
            )
            return Snapshot(
                **value.model_dump(),
                development=self.development,
                debug_enabled=value.debug_mode == "on"
                or (value.debug_mode == "auto" and self.development),
            ).model_dump(mode="json")
        except pydantic.ValidationError:
            raise protocol.RemoteError(
                "invalid_argument", "invalid desktop settings"
            ) from None
        except OSError:
            raise protocol.RemoteError(
                "settings_storage_error", "cannot access desktop settings on host"
            ) from None
