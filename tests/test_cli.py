from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest

from vectorops_code import cli, credentials, dev


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
    monkeypatch.delenv("VECTOROPS_CODE_TOKEN", raising=False)
    monkeypatch.delenv("RUNNER_DESKTOP_TOKEN", raising=False)
    with patch("vectorops_code.desktop.launch") as launch:
        cli.main(arguments)
    launch.assert_called_once_with(
        Path("project") if arguments else Path.cwd(), None, None, 0
    )


def test_conflicting_project_options_are_rejected() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["desktop", "one", "--workspace", "two"])
    assert error.value.code == 2


def test_headless_generates_persistent_token_and_defaults_origins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VECTOROPS_CODE_TOKEN", raising=False)
    monkeypatch.delenv("RUNNER_DESKTOP_TOKEN", raising=False)
    monkeypatch.setenv("VECTOROPS_CODE_CONFIG_DIR", str(tmp_path / "config"))
    with patch("vectorops_code.cli.serve", new_callable=AsyncMock) as serve:
        cli.main(["serve", str(tmp_path)])
        cli.main(["serve", "--workspace", str(tmp_path)])
    first, second = serve.call_args_list
    assert first == second
    assert first.args == (
        tmp_path.resolve(),
        credentials.read_token(
            credentials.token_path("workers", str(tmp_path.resolve()))
        ),
        None,
        0,
    )


def test_headless_port_is_automatic_unless_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VECTOROPS_CODE_TOKEN", "x" * 43)
    with patch("vectorops_code.cli.serve", new_callable=AsyncMock) as serve:
        cli.main(["serve", "--origin", "http://127.0.0.1:8766"])
    assert serve.call_args.args[-1] == 0
    assert serve.call_args.kwargs == {"development": False}


def test_headless_development_flag_reaches_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VECTOROPS_CODE_TOKEN", "x" * 43)
    with patch("vectorops_code.cli.serve", new_callable=AsyncMock) as serve:
        cli.main(["serve", "--origin", "http://127.0.0.1:5173", "--development"])
    assert serve.call_args.kwargs == {"development": True}


def test_dev_ports_remain_preconfigured() -> None:
    with patch("vectorops_code.dev.run", new_callable=AsyncMock) as run:
        dev.main([])
    assert run.call_args.args[2:4] == (8765, 5173)


def test_runtime_errors_are_actionable_without_tracebacks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch(
        "vectorops_code.desktop.launch",
        side_effect=RuntimeError("another runner owns this project"),
    ):
        with pytest.raises(SystemExit) as error:
            cli.main(["project"])
    assert error.value.code == 1
    assert capsys.readouterr().err == "another runner owns this project\n"


@pytest.mark.parametrize("preferred", [None, "y" * 43])
def test_token_environment_compatibility(
    preferred: Optional[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNNER_DESKTOP_TOKEN", "x" * 43)
    monkeypatch.delenv("VECTOROPS_CODE_TOKEN", raising=False)
    if preferred is not None:
        monkeypatch.setenv("VECTOROPS_CODE_TOKEN", preferred)
    with patch("vectorops_code.cli.serve", new_callable=AsyncMock) as serve:
        cli.main(["serve", "--origin", "http://127.0.0.1:8766"])
    assert serve.call_args.args[1] == (preferred or "x" * 43)


@pytest.mark.parametrize(
    ("address", "port", "expected"),
    [
        ("host.example", 443, "wss://host.example:443"),
        ("host.example:1234", None, "wss://host.example:1234"),
        ("localhost", 1234, "ws://localhost:1234"),
        ("[::1]:1234", None, "ws://[::1]:1234"),
        ("wss://host.example/runner", None, "wss://host.example/runner"),
    ],
)
def test_connection_address(address: str, port: Optional[int], expected: str) -> None:
    assert cli.connection_endpoint(address, port) == expected


@pytest.mark.parametrize(
    ("address", "port"),
    [
        ("ws://host.example:1234", None),
        ("host:1234", 4321),
        ("host", 0),
        ("host", 65536),
        ("user@host", 443),
        ("host/path", 443),
        ("", 443),
        ("wss://host", 443),
    ],
)
def test_invalid_connection_address(address: str, port: Optional[int]) -> None:
    with pytest.raises(ValueError):
        cli.connection_endpoint(address, port)


def test_connect_prompts_and_remembers_token_without_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VECTOROPS_CODE_TOKEN", raising=False)
    monkeypatch.delenv("RUNNER_DESKTOP_TOKEN", raising=False)
    monkeypatch.setenv("VECTOROPS_CODE_CONFIG_DIR", str(tmp_path / "config"))
    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("builtins.input", side_effect=["127.0.0.1", "1234"]),
        patch("vectorops_code.cli.getpass", return_value="x" * 43) as prompt,
        patch("vectorops_code.desktop.launch") as launch,
    ):
        cli.main(["connect"])
        cli.main(["connect", "127.0.0.1:1234"])
    assert launch.call_count == 2
    launch.assert_called_with(None, "ws://127.0.0.1:1234", "x" * 43)
    prompt.assert_called_once()
    assert not (tmp_path / ".voc").exists()


def test_connect_noninteractive_requires_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("VECTOROPS_CODE_TOKEN", raising=False)
    monkeypatch.delenv("RUNNER_DESKTOP_TOKEN", raising=False)
    monkeypatch.setenv("VECTOROPS_CODE_CONFIG_DIR", str(tmp_path))
    with (
        patch("sys.stdin.isatty", return_value=False),
        patch("vectorops_code.cli.getpass") as prompt,
    ):
        with pytest.raises(SystemExit) as error:
            cli.main(["connect", "localhost:1234"])
    assert error.value.code == 1
    prompt.assert_not_called()


def test_explicit_token_file_overrides_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "token"
    credentials.save_token(path, "f" * 43)
    monkeypatch.setenv("VECTOROPS_CODE_TOKEN", "e" * 43)
    with patch("vectorops_code.desktop.launch") as launch:
        cli.main(["connect", "localhost:1234", "--token-file", str(path)])
    launch.assert_called_once_with(None, "ws://localhost:1234", "f" * 43)


def test_legacy_attach_uses_automatic_asset_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VECTOROPS_CODE_TOKEN", "x" * 43)
    with patch("vectorops_code.desktop.launch") as launch:
        cli.main(["desktop", "--endpoint", "ws://localhost:1234"])
    launch.assert_called_once_with(Path.cwd(), "ws://localhost:1234", "x" * 43, 0)


def test_connect_noninteractive_requires_port() -> None:
    with (
        patch("sys.stdin.isatty", return_value=False),
        patch("builtins.input") as prompt,
    ):
        with pytest.raises(ValueError, match="Supply --port"):
            cli.connection_endpoint("localhost", None)
    prompt.assert_not_called()
