import ctypes
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vectorops_code import branding
from vectorops_code.branding import APP_ID, APP_NAME, configure_application


@pytest.mark.parametrize("localized", [False, True])
def test_macos_application_name(localized: bool) -> None:
    foundation = MagicMock()
    info: dict[str, str] = {}
    localized_info: dict[str, str] = {}
    bundle = foundation.NSBundle.mainBundle.return_value
    bundle.infoDictionary.return_value = info
    bundle.localizedInfoDictionary.return_value = localized_info if localized else None
    with (
        patch("vectorops_code.branding.sys.platform", "darwin"),
        patch.dict("sys.modules", {"Foundation": foundation}),
    ):
        icon = Path(configure_application())

    assert info == {"CFBundleName": APP_NAME, "CFBundleDisplayName": APP_NAME}
    if localized:
        assert localized_info == info
    foundation.NSProcessInfo.processInfo.return_value.setProcessName_.assert_called_once_with(
        APP_NAME
    )
    foundation.NSNotificationCenter.defaultCenter.return_value.addObserverForName_object_queue_usingBlock_.assert_called_once_with(
        "NSApplicationDidFinishLaunchingNotification",
        None,
        None,
        branding._set_macos_menu_name,
    )
    assert icon.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_macos_menu_heading() -> None:
    appkit = MagicMock()
    menu = appkit.NSApplication.sharedApplication.return_value.mainMenu.return_value
    menu.numberOfItems.return_value = 3
    with patch.dict("sys.modules", {"AppKit": appkit}):
        branding._set_macos_menu_name(object())

    menu.itemAtIndex_.assert_called_once_with(0)
    menu.itemAtIndex_.return_value.setTitle_.assert_called_once_with(APP_NAME)
    menu.itemAtIndex_.return_value.submenu.return_value.setTitle_.assert_called_once_with(
        APP_NAME
    )


def test_windows_taskbar_identity() -> None:
    with (
        patch("vectorops_code.branding.sys.platform", "win32"),
        patch("vectorops_code.branding.ctypes.OleDLL", create=True) as load_dll,
    ):
        icon = Path(configure_application())

    load_dll.assert_called_once_with("shell32")
    set_app_id = load_dll.return_value.SetCurrentProcessExplicitAppUserModelID
    assert set_app_id.argtypes == [ctypes.c_wchar_p]
    assert set_app_id.restype == ctypes.c_long
    set_app_id.assert_called_once_with(APP_ID)
    assert icon.read_bytes().startswith(b"\x00\x00\x01\x00")


def test_linux_application_name() -> None:
    repository = MagicMock()
    with (
        patch("vectorops_code.branding.sys.platform", "linux"),
        patch.dict("sys.modules", {"gi": MagicMock(), "gi.repository": repository}),
    ):
        icon = Path(configure_application())

    repository.GLib.set_application_name.assert_called_once_with(APP_NAME)
    repository.GLib.set_prgname.assert_called_once_with(APP_ID)
    assert icon.suffix == ".png"
    assert icon.is_file()


def test_linux_without_gtk() -> None:
    with (
        patch("vectorops_code.branding.sys.platform", "linux"),
        patch.dict("sys.modules", {"gi": None, "gi.repository": None}),
    ):
        icon = Path(configure_application())

    assert icon.suffix == ".png"
    assert icon.is_file()
