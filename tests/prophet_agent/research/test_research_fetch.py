"""fetch: extraction, fallback, redirects re-checked, content types, caps and caching, with httpx.MockTransport."""
import hashlib

import httpx
import pytest

from prophet_agent.research import fetch as fetch_mod
from prophet_agent.research.fetch import MAX_REDIRECTS, fetch_page
from prophet_agent.research.netguard import HostThrottle

PUBLIC = "93.184.216.34"
ARTICLE = """<!doctype html><html><head><title>Crude outlook 2026</title>
<meta property="og:title" content="Crude outlook 2026"></head>
<body><nav>Menu links here</nav><article><h1>Crude outlook 2026</h1>
<p>Brent crude is forecast to average 78.5 dollars per barrel in 2027, the agency said in its monthly report.</p>
<p>Inventories rose by 3.2 million barrels in the latest week, weighing on prices across the region.</p>
</article><footer>Copyright footer</footer></body></html>"""


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))
    return tmp_path / "cache"


def resolver(host, port):
    return [PUBLIC]


def client_for(routes, counter=None):
    """routes maps a path to a response factory or a (status, headers, body) tuple. Unknown paths give 404."""
    def handler(request):
        if counter is not None:
            counter.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        entry = routes.get(request.url.path)
        if entry is None:
            return httpx.Response(404, text="missing")
        if callable(entry):
            return entry(request)
        status, headers, body = entry
        return httpx.Response(status, headers=headers, content=body)
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def fetch(url, client, **kwargs):
    return fetch_page(url, client=client, resolver=resolver, throttle=HostThrottle(0.0),
                      cache_dir=kwargs.pop("cache_dir", None), **kwargs)


HTML = {"/article": (200, {"content-type": "text/html; charset=utf-8"}, ARTICLE.encode())}


def test_extracts_main_text_title_and_fingerprint():
    page = fetch("https://example.com/article", client_for(HTML), use_cache=False)
    assert page.ok is True and page.status == 200
    assert "Brent crude is forecast to average 78.5" in page.text
    assert "Menu links here" not in page.text
    assert page.title.startswith("Crude outlook")
    assert page.sha256 == hashlib.sha256(page.text.encode("utf-8")).hexdigest()


def test_falls_back_to_beautifulsoup_when_trafilatura_returns_nothing(monkeypatch):
    monkeypatch.setattr(fetch_mod.trafilatura, "extract", lambda *a, **k: None)
    page = fetch("https://example.com/article", client_for(HTML), use_cache=False)
    assert page.ok is True
    assert "Inventories rose by 3.2 million barrels" in page.text
    assert "fallback extraction" in page.note
    assert "Copyright footer" not in page.text


def test_pdf_is_refused_with_a_clear_note():
    routes = {"/report.pdf": (200, {"content-type": "application/pdf"}, b"%PDF-1.4 binary")}
    page = fetch("https://example.com/report.pdf", client_for(routes), use_cache=False)
    assert page.ok is False and "PDF" in page.note


def test_other_binary_types_are_refused():
    routes = {"/img": (200, {"content-type": "image/png"}, b"\x89PNG")}
    page = fetch("https://example.com/img", client_for(routes), use_cache=False)
    assert page.ok is False and "not text" in page.note


def test_plain_text_and_json_are_read_as_they_are():
    routes = {"/plain": (200, {"content-type": "text/plain"}, b"Rate: 4.25 per cent"),
              "/data": (200, {"content-type": "application/json"}, b'{"price":82.5}')}
    plain = fetch("https://example.com/plain", client_for(routes), use_cache=False)
    data = fetch("https://example.com/data", client_for(routes), use_cache=False)
    assert plain.text == "Rate: 4.25 per cent"
    assert '"price": 82.5' in data.text


def test_redirect_to_a_public_page_is_followed():
    routes = {"/old": (301, {"location": "/article"}, b""), **HTML}
    page = fetch("https://example.com/old", client_for(routes), use_cache=False)
    assert page.ok is True and page.final_url == "https://example.com/article"


def test_redirect_to_a_private_address_is_refused_on_the_hop():
    routes = {"/go": (302, {"location": "http://127.0.0.1/admin"}, b"")}
    page = fetch("https://example.com/go", client_for(routes), use_cache=False)
    assert page.ok is False and "refused" in page.note and "loopback" in page.note


def test_redirect_loop_gives_up():
    routes = {"/loop": (302, {"location": "/loop"}, b"")}
    page = fetch("https://example.com/loop", client_for(routes), use_cache=False)
    assert page.ok is False and f"more than {MAX_REDIRECTS} redirects" in page.note


def test_http_error_is_reported_not_raised():
    page = fetch("https://example.com/missing", client_for({}), use_cache=False)
    assert page.ok is False and page.status == 404 and "HTTP 404" in page.note


def test_robots_disallow_stops_the_fetch():
    routes = {"/private/x": (200, {"content-type": "text/html"}, b"<p>secret</p>")}

    def handler_client():
        def handler(request):
            if request.url.path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
            return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<p>secret</p>")
        return httpx.Client(transport=httpx.MockTransport(handler))

    page = fetch("https://example.com/private/x", handler_client(), use_cache=False)
    assert page.ok is False and "robots.txt" in page.note


def test_network_failure_never_raises():
    def boom(request):
        raise httpx.ConnectError("unreachable")
    client = httpx.Client(transport=httpx.MockTransport(lambda r: boom(r) if r.url.path != "/robots.txt"
                                                        else httpx.Response(404)))
    page = fetch("https://example.com/x", client, use_cache=False)
    assert page.ok is False and "ConnectError" in page.note


def test_file_urls_are_never_opened():
    page = fetch_page("file:///etc/hosts", use_cache=False)
    assert page.ok is False and "scheme" in page.note


def test_oversized_body_is_cut_at_the_byte_cap():
    big = b"<html><body><article><p>" + b"word " * 5000 + b"</p></article></body></html>"
    routes = {"/big": (200, {"content-type": "text/html"}, big)}
    page = fetch("https://example.com/big", client_for(routes), use_cache=False, max_bytes=1000)
    assert "response cut at 1000 bytes" in page.note


def test_cache_hit_skips_the_network(isolated_cache):
    calls = []
    first = fetch("https://example.com/article", client_for(HTML, counter=calls))
    count = len([c for c in calls if "robots" not in c])
    second = fetch("https://example.com/article", client_for(HTML, counter=calls))
    assert first.cached is False and second.cached is True
    assert len([c for c in calls if "robots" not in c]) == count
    assert second.sha256 == first.sha256


def test_empty_url_and_no_text_page_are_not_ok():
    assert fetch_page("   ", use_cache=False).ok is False
    empty = {"/empty": (200, {"content-type": "text/html"}, b"<html><body></body></html>")}
    page = fetch("https://example.com/empty", client_for(empty), use_cache=False)
    assert page.ok is False and "no readable text" in page.note
