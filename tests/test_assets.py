import os
from pathlib import Path

import pytest

from vectorops_code.assets import AssetServer


def test_missing_assets_explain_how_to_build(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="npm run build"):
        AssetServer(tmp_path)


@pytest.mark.parametrize(
    "changed",
    [
        "src/screens/RunnerScreen.tsx",
        "src/components/ProjectLlmSetup.tsx",
        "src/style.css",
        "index.html",
        "package.json",
        "package-lock.json",
        "tsconfig.json",
        "vite.config.ts",
    ],
)
def test_source_checkout_rejects_stale_assets(tmp_path: Path, changed: str) -> None:
    root = tmp_path / "src" / "vectorops_code" / "web"
    root.mkdir(parents=True)
    index = root / "index.html"
    index.write_text("old interface")
    (tmp_path / "frontend" / "src").mkdir(parents=True)
    source = tmp_path / "frontend" / changed
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("updated source")
    os.utime(source, ns=(index.stat().st_mtime_ns + 1,) * 2)

    with pytest.raises(RuntimeError, match="out of date; run npm run build"):
        AssetServer(root)


@pytest.mark.parametrize("checkout", [False, True])
def test_current_checkout_and_packaged_assets_can_launch(
    tmp_path: Path, checkout: bool
) -> None:
    root = tmp_path / "src" / "vectorops_code" / "web"
    root.mkdir(parents=True)
    if checkout:
        source = tmp_path / "frontend" / "src" / "main.tsx"
        source.parent.mkdir(parents=True)
        source.write_text("source")
    index = root / "index.html"
    index.write_text("current interface")
    if checkout:
        os.utime(index, ns=(source.stat().st_mtime_ns + 1,) * 2)

    with AssetServer(root) as assets:
        assert assets.thread.is_alive()
