"""Serve only packaged frontend assets on a loopback origin."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from os import PathLike
from pathlib import Path
from threading import Thread
from typing import Optional, Union


class AssetHandler(SimpleHTTPRequestHandler):
    def list_directory(self, path: Union[str, PathLike[str]]) -> None:
        self.send_error(404)
        return None

    def log_message(self, format: str, *args: object) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()


class AssetServer:
    def __init__(self, path: Optional[Path] = None, port: int = 0) -> None:
        root = path or Path(__file__).parent / "web"
        if not (root / "index.html").is_file():
            raise FileNotFoundError(
                "Frontend assets missing; run npm run build in frontend/"
            )
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", port), partial(AssetHandler, directory=str(root.resolve()))
        )
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.origin = f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self) -> "AssetServer":
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
