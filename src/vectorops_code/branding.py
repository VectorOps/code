"""Native desktop application identity, configured before GUI initialization."""

import ctypes
from pathlib import Path
import sys
from typing import Final

APP_NAME: Final = "VectorOps Code"
APP_ID: Final = "com.vectorops.code"


def _set_macos_menu_name(notification: object) -> None:
    import AppKit

    menu = AppKit.NSApplication.sharedApplication().mainMenu()
    if menu is not None and menu.numberOfItems() > 0:
        item = menu.itemAtIndex_(0)
        item.setTitle_(APP_NAME)
        item.submenu().setTitle_(APP_NAME)


def configure_application() -> str:
    if sys.platform == "darwin":
        import Foundation

        bundle = Foundation.NSBundle.mainBundle()
        info = bundle.infoDictionary()
        info["CFBundleName"] = APP_NAME
        info["CFBundleDisplayName"] = APP_NAME
        localized = bundle.localizedInfoDictionary()
        if localized is not None:
            localized["CFBundleName"] = APP_NAME
            localized["CFBundleDisplayName"] = APP_NAME
        Foundation.NSProcessInfo.processInfo().setProcessName_(APP_NAME)
        Foundation.NSNotificationCenter.defaultCenter().addObserverForName_object_queue_usingBlock_(
            "NSApplicationDidFinishLaunchingNotification",
            None,
            None,
            _set_macos_menu_name,
        )
    elif sys.platform == "win32":
        shell32 = ctypes.OleDLL("shell32")
        set_app_id = shell32.SetCurrentProcessExplicitAppUserModelID
        set_app_id.argtypes = [ctypes.c_wchar_p]
        set_app_id.restype = ctypes.c_long
        set_app_id(APP_ID)
    elif sys.platform.startswith("linux"):
        try:
            from gi.repository import GLib
        except ImportError:
            pass
        else:
            GLib.set_application_name(APP_NAME)
            GLib.set_prgname(APP_ID)

    extension = "ico" if sys.platform == "win32" else "png"
    return str(Path(__file__).with_name("icons") / f"app.{extension}")
