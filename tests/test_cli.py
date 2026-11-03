from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from runner_desktop import cli, dev


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["project"],
        ["desktop", "project"],
        ["desktop", "--workspace", "project"],
        ["--workspace", "project"],
    ],
)
def test_local_start_needs_no_token_or_ports(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RUNNER_DESKTOP_TOKEN", raising=False)
    with patch("runner_desktop.desktop.launch") as launch:
        cli.main(arguments)
    launch.assert_called_once_with(
        Path("project") if arguments else Path.cwd(), None, None, 0
    )


def test_conflicting_project_options_are_rejected() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["desktop", "one", "--workspace", "two"])
    assert error.value.code == 2


def test_headless_requires_origin_and_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNNER_DESKTOP_TOKEN", raising=False)
    with pytest.raises(SystemExit) as error:
        cli.main(["serve"])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        cli.main(["serve", "--origin", "http://127.0.0.1:8766"])
    assert error.value.code == 1


def test_headless_port_is_automatic_unless_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNNER_DESKTOP_TOKEN", "x" * 43)
    with patch("runner_desktop.cli.serve", new_callable=AsyncMock) as serve:
        cli.main(["serve", "--origin", "http://127.0.0.1:8766"])
    assert serve.call_args.args[-1] == 0
    assert serve.call_args.kwargs == {"development": False}


def test_headless_development_flag_reaches_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNNER_DESKTOP_TOKEN", "x" * 43)
    with patch("runner_desktop.cli.serve", new_callable=AsyncMock) as serve:
        cli.main(["serve", "--origin", "http://127.0.0.1:5173", "--development"])
    assert serve.call_args.kwargs == {"development": True}


def test_dev_ports_remain_preconfigured() -> None:
    with patch("runner_desktop.dev.run", new_callable=AsyncMock) as run:
        dev.main([])
    assert run.call_args.args[2:4] == (8765, 5173)


def test_runtime_errors_are_actionable_without_tracebacks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch(
        "runner_desktop.desktop.launch",
        side_effect=RuntimeError("another runner owns this project"),
    ):
        with pytest.raises(SystemExit) as error:
            cli.main(["project"])
    assert error.value.code == 1
    assert capsys.readouterr().err == "another runner owns this project\n"
