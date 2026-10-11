import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

from runner_desktop.templates import render_index, TemplateServer


def test_bootstrap_escaping() -> None:
    token = '</script><script>alert("test")</script>'
    html = render_index(
        "<html><head></head><body>App</body></html>", "ws://127.0.0.1:8765", token
    )
    value = html.split('type="application/json">')[1].split("</script>")[0]
    assert "<script>" not in value
    assert json.loads(value)["token"] == token
    assert "<body>App</body>" in html


def test_template_server_authentication() -> None:
    server = TemplateServer("ws://127.0.0.1:8765", "secret")
    server.start()
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(Request(server.url, data=b"<head></head>"), timeout=5)
        assert error.value.code == 403
        request = Request(
            server.url,
            data=b"<head></head>",
            headers={"Authorization": "Bearer secret"},
        )
        with urlopen(request, timeout=5) as response:
            assert response.headers["Cache-Control"] == "no-store"
            assert b"runner-bootstrap" in response.read()
    finally:
        server.close()
