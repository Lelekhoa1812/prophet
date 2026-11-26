"""render: a JavaScript page built on load is drawn and read; every browser request passes the address check; a request
to a private or loopback address from inside the page is refused. Skipped when Chromium is not installed.

netguard only allows ports 80 and 443 and refuses 127.0.0.1, so the local test server cannot be reached the normal way.
Tests inject an address check that maps only the test host name to the local server. The URLs the browser and the page
use carry no port (the default 80), and the injected check rewrites only the connection target to the server's real
port. Every other host, including loopback and private literals, goes through netguard's own check. The production
default is never changed.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, urlunsplit

import httpx
import pytest

from prophet_agent.research import fetch as fetch_mod
from prophet_agent.research import render as render_mod
from prophet_agent.research.fetch import fetch_page
from prophet_agent.research.netguard import HostThrottle, resolve_checked
from prophet_agent.research.render import chromium_status, render_page

TEST_HOST = "render-test.invalid"
PUBLIC = "93.184.216.34"
SERVER = {"port": None, "hits": {"page": 0, "pic": 0}}

PAGE = """<!doctype html><html><head><title>Lag table</title></head><body>
<h1>Premier League table</h1><div id="root"></div>
<img src="http://render-test.invalid/pic.png" alt="">
<script>
  const rows = [["Arsenal", 70], ["Chelsea", 60], ["Everton", 40]];
  const table = document.createElement("table");
  for (const [club, points] of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = "<td>" + club + "</td><td>" + points + "</td>";
    table.appendChild(tr);
  }
  document.getElementById("root").appendChild(table);
  const probe = (url) => fetch(url).then(() => "reached").catch(() => "blocked");
  Promise.all([probe("http://127.0.0.1/secret"), probe("http://10.20.30.40/secret")]).then((results) => {
    document.body.dataset.probe = results.join(",");
  });
</script></body></html>"""

STATIC_SHELL = ('<!doctype html><html><head><title>Lag table</title></head><body><div id="root"></div>'
                "<script>a()</script><script>b()</script><script>c()</script></body></html>")

needs_chromium = pytest.mark.skipif(not chromium_status()[0], reason="Chromium is not installed (playwright install)")


def _test_address_check(url, deadline):
    """For the test host only: the connection goes to the local server's real port on 127.0.0.1. Anything else is
    checked by netguard exactly as in production."""
    parts = urlsplit(url)
    if parts.hostname == TEST_HOST:
        real = urlunsplit((parts.scheme, f"127.0.0.1:{SERVER['port']}", parts.path or "/", parts.query, ""))
        return urlsplit(real), ["127.0.0.1"]
    return resolve_checked(url, deadline=deadline)


@pytest.fixture(scope="module", autouse=True)
def local_server():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.startswith("/pic.png"):
                SERVER["hits"]["pic"] += 1
                body, ctype = b"\x89PNG", "image/png"
            else:
                SERVER["hits"]["page"] += 1
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    SERVER["port"] = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


def _reset_hits():
    SERVER["hits"].update(page=0, pic=0)


@needs_chromium
def test_javascript_table_is_drawn_and_private_probes_are_refused():
    _reset_hits()
    result = render_page(f"http://{TEST_HOST}/league", address_check=_test_address_check, total_seconds=30)
    assert result.ok is True, result.note
    assert "<table" in result.html and "Arsenal" in result.html and "Everton" in result.html
    assert result.title == "Lag table"
    assert result.seconds > 0 and result.bytes > 0
    # the page's own probes to a loopback and a private address were refused by the check
    assert 'data-probe="blocked,blocked"' in result.html
    refused = {url: reason for url, reason in result.blocked}
    assert any("127.0.0.1" in url and "loopback" in reason for url, reason in refused.items())
    assert any("10.20.30.40" in url and "private" in reason for url, reason in refused.items())


@needs_chromium
def test_images_are_not_requested():
    _reset_hits()
    render_page(f"http://{TEST_HOST}/league", address_check=_test_address_check, total_seconds=30)
    assert SERVER["hits"]["pic"] == 0


@needs_chromium
def test_production_check_refuses_a_loopback_page():
    result = render_page("http://127.0.0.1/league", total_seconds=20)
    assert result.ok is False
    assert any("loopback" in reason for _, reason in result.blocked)


def test_non_http_scheme_is_refused_before_a_browser_starts():
    result = render_page("file:///etc/hosts")
    assert result.ok is False and "refused" in result.note


def test_missing_chromium_is_a_skipped_render_not_an_error(monkeypatch):
    monkeypatch.setattr(render_mod, "chromium_status", lambda: (False, "Chromium is not installed (test)"))
    result = render_page("https://example.com/")
    assert result.ok is False and "render skipped" in result.note


def _static_client():
    def static_only(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, content=STATIC_SHELL.encode())
    return httpx.Client(transport=httpx.MockTransport(static_only), follow_redirects=False)


@needs_chromium
def test_fetch_page_renders_a_javascript_shell_and_extracts_its_table(tmp_path):
    page = fetch_page(f"http://{TEST_HOST}/league", client=_static_client(), resolver=lambda host, port: [PUBLIC],
                      throttle=HostThrottle(0.0), cache_dir=tmp_path / "cache",
                      renderer=lambda u: render_page(u, address_check=_test_address_check, total_seconds=30))
    assert page.ok is True
    assert "rendered" in page.note
    assert "Arsenal" in page.text
    assert "Arsenal" in json.dumps(page.tables)


def test_fetch_page_without_chromium_names_the_skipped_render(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_mod, "chromium_status", lambda: (False, "Chromium is not installed (test)"))
    page = fetch_page("https://example.com/league", client=_static_client(), resolver=lambda host, port: [PUBLIC],
                      throttle=HostThrottle(0.0), cache_dir=tmp_path / "cache", render=True)
    assert page.ok is False
    assert "render skipped" in page.note and "rendered" not in page.note


def test_fetch_page_without_chromium_keeps_a_page_that_has_text(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_mod, "chromium_status", lambda: (False, "Chromium is not installed (test)"))
    article = ("<html><body><article><p>" + "Brent crude is forecast to average 78.5 dollars per barrel. " * 12
               + "</p></article><script>a()</script></body></html>")

    def serve(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, content=article.encode())

    client = httpx.Client(transport=httpx.MockTransport(serve), follow_redirects=False)
    page = fetch_page("https://example.com/article", client=client, resolver=lambda host, port: [PUBLIC],
                      throttle=HostThrottle(0.0), cache_dir=tmp_path / "cache", render=True)
    assert page.ok is True
    assert "render skipped" in page.note
