"""Render local frontend bootstrap settings without changing built assets."""

import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from jinja2 import Environment


def render_index(html: str, endpoint: str, token: str) -> str:
    head, separator, body = html.partition("</head>")
    if not separator:
        raise ValueError("Frontend HTML has no head element")
    template = Environment(autoescape=True).from_string(
        '{{ head | safe }}<script id="runner-bootstrap" type="application/json">'
        "{{ bootstrap | tojson }}</script></head>{{ body | safe }}"
    )
    return template.render(
        head=head, body=body, bootstrap={"endpoint": endpoint, "token": token}
    )


class TemplateServer:
    """Provide authenticated HTML rendering to the local Vite process."""

    def __init__(self, endpoint: str, token: str) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if not hmac.compare_digest(
                    self.headers.get("Authorization", ""), f"Bearer {token}"
                ):
                    self.send_error(403)
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 1024 * 1024:
                        raise ValueError("Invalid HTML size")
                    html = self.rfile.read(size).decode("utf-8")
                    rendered = render_index(html, endpoint, token).encode("utf-8")
                except (ValueError, UnicodeError):
                    self.send_error(400)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(rendered)))
                self.end_headers()
                self.wfile.write(rendered)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
