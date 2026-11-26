"""Download guards: address, redirect hops, content type, size and status. The network seam (fetch.open_client) is replaced
with an httpx.MockTransport, and names resolve through a fixed resolver, so no test touches the network."""
import re
from pathlib import Path

import httpx
import pytest

from prophet_agent.opendata import fetch

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CSV_BYTES = (FIXTURES / "open_meteo_daily.csv").read_bytes()
JSON_BYTES = (FIXTURES / "open_meteo_daily.json").read_bytes()
PUBLIC = "93.184.216.34"          # a globally routable address, used wherever a name must resolve
API = "https://api.open-meteo.com/v1/forecast?latitude=52.0&longitude=13.0"


def public_resolver(host, port):
    return [PUBLIC]


@pytest.fixture
def serve(monkeypatch):
    """Route every hop through ``handler``. The handler receives the pinned request: the checked address as the URL host
    and the original name in the Host header."""
    def install(handler):
        def open_client(timeout_s):
            return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
        monkeypatch.setattr(fetch, "open_client", open_client)
    return install


def ok(body=CSV_BYTES, content_type="text/csv", **headers):
    return httpx.Response(200, headers={"content-type": content_type, **headers}, content=body)


def test_csv_body_is_returned_with_type_final_url_and_time(serve):
    seen = []

    def handler(request):
        seen.append((request.url.host, request.headers["host"]))
        return ok(content_type="text/csv; charset=utf-8")

    serve(handler)
    got = fetch.download(API, resolver=public_resolver)
    assert got["bytes"] == CSV_BYTES
    assert got["content_type"] == "text/csv"
    assert got["final_url"] == API
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", got["retrieved_at"])
    assert seen == [(PUBLIC, "api.open-meteo.com")]   # the connection went to the checked address, name kept in Host


def test_json_body_is_accepted(serve):
    serve(lambda request: ok(JSON_BYTES, content_type="application/json"))
    assert fetch.download(API, resolver=public_resolver)["content_type"] == "application/json"


def test_application_csv_is_accepted(serve):
    # FinRes-20261124: FRED's graph CSV answered application/csv and the download refused it.
    serve(lambda request: ok(CSV_BYTES, content_type="application/csv"))
    assert fetch.download(API, resolver=public_resolver)["bytes"] == CSV_BYTES


def test_plain_text_is_accepted(serve):
    serve(lambda request: ok(b"a,b\n1,2\n", content_type="text/plain"))
    assert fetch.download(API, resolver=public_resolver)["bytes"] == b"a,b\n1,2\n"


def test_octet_stream_is_accepted_when_the_body_parses_as_csv(serve):
    serve(lambda request: ok(CSV_BYTES, content_type="application/octet-stream"))
    got = fetch.download(API, resolver=public_resolver)
    assert got["content_type"] == "application/octet-stream" and got["bytes"] == CSV_BYTES


def test_an_sdmx_csv_type_is_accepted(serve):
    # OECD and ABS answer application/vnd.sdmx.data+csv. The body is still a CSV table.
    serve(lambda request: ok(CSV_BYTES, content_type="application/vnd.sdmx.data+csv;version=1.0.0"))
    got = fetch.download(API, resolver=public_resolver)
    assert got["content_type"] == "application/vnd.sdmx.data+csv" and got["bytes"] == CSV_BYTES


def test_octet_stream_is_accepted_when_the_body_parses_as_json(serve):
    serve(lambda request: ok(JSON_BYTES, content_type="application/octet-stream"))
    assert fetch.download(API, resolver=public_resolver)["bytes"] == JSON_BYTES


def test_a_zip_content_type_is_accepted_when_the_body_is_a_zip(serve):
    serve(lambda request: ok(b"PK\x03\x04rest-of-zip", content_type="application/zip"))
    got = fetch.download(API, resolver=public_resolver)
    assert got["content_type"] == "application/zip" and got["bytes"].startswith(b"PK")


def test_a_zip_content_type_with_a_text_body_is_refused(serve):
    serve(lambda request: ok(b"a,b\n1,2\n", content_type="application/zip"))
    with pytest.raises(fetch.DownloadError, match="only when the body is a ZIP"):
        fetch.download(API, resolver=public_resolver)


def test_octet_stream_with_a_binary_body_is_refused(serve):
    serve(lambda request: ok(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", content_type="application/octet-stream"))
    with pytest.raises(fetch.DownloadError, match="accepted only when the body parses as CSV or JSON"):
        fetch.download(API, resolver=public_resolver)


@pytest.mark.parametrize("content_type", ["text/html", "application/xhtml+xml; charset=utf-8"])
def test_an_html_page_is_accepted_and_its_body_returned(serve, content_type):
    # The page's tables are landed by html_landing; the download itself no longer refuses HTML.
    page = b"<!doctype html><html><body><table><tr><td>a</td></tr></table></body></html>"
    serve(lambda request: ok(page, content_type=content_type))
    got = fetch.download(API, resolver=public_resolver)
    assert got["bytes"] == page and got["content_type"] in fetch.HTML_TYPES


def test_an_html_page_that_is_not_utf8_is_refused(serve):
    serve(lambda request: ok(b"<html>\xff\xfe\x00</html>", content_type="text/html"))
    with pytest.raises(fetch.DownloadError, match="not UTF-8"):
        fetch.download(API, resolver=public_resolver)


@pytest.mark.parametrize("content_type", ["image/png", "application/xml"])
def test_other_content_types_are_refused(serve, content_type):
    serve(lambda request: ok(b"<html>x</html>", content_type=content_type))
    with pytest.raises(fetch.DownloadError, match="is not allowed"):
        fetch.download(API, resolver=public_resolver)


def test_a_pdf_is_accepted_as_bytes_when_it_starts_with_the_pdf_header(serve):
    # A PDF's tables are landed by pdf_landing; the download returns the bytes unchanged.
    serve(lambda request: ok(b"%PDF-1.4\n%\xe2\xe3\n", content_type="application/pdf"))
    out = fetch.download(API, resolver=public_resolver)
    assert out["content_type"] == "application/pdf" and out["bytes"].startswith(b"%PDF-")


def test_a_pdf_content_type_with_a_non_pdf_body_is_refused(serve):
    serve(lambda request: ok(b"<html>x</html>", content_type="application/pdf"))
    with pytest.raises(fetch.DownloadError, match="starts with %PDF-"):
        fetch.download(API, resolver=public_resolver)


def test_a_missing_content_type_is_refused(serve):
    serve(lambda request: httpx.Response(200, content=b"a,b\n1,2\n"))
    with pytest.raises(fetch.DownloadError, match="is not allowed"):
        fetch.download(API, resolver=public_resolver)


def test_text_types_must_be_utf8_text(serve):
    serve(lambda request: ok(b"\xff\xfe\x00\x01 not text", content_type="text/csv"))
    with pytest.raises(fetch.DownloadError, match="not UTF-8 text"):
        fetch.download(API, resolver=public_resolver)


def test_an_empty_body_is_refused(serve):
    serve(lambda request: ok(b"", content_type="text/csv"))
    with pytest.raises(fetch.DownloadError, match="empty"):
        fetch.download(API, resolver=public_resolver)


@pytest.mark.parametrize("status", [400, 404, 500])
def test_non_200_responses_are_refused(serve, status):
    serve(lambda request: httpx.Response(status, headers={"content-type": "application/json"},
                                         content=b'{"error": true, "reason": "bad"}'))
    with pytest.raises(fetch.DownloadError, match=f"HTTP {status}"):
        fetch.download(API, resolver=public_resolver)


def test_a_redirect_without_a_location_is_refused(serve):
    serve(lambda request: httpx.Response(301, content=b""))
    with pytest.raises(fetch.DownloadError, match="HTTP 301"):
        fetch.download(API, resolver=public_resolver)


def test_a_declared_length_over_the_cap_is_refused_before_reading(serve):
    serve(lambda request: ok(b"x" * 5000, **{"content-length": "5000"}))
    with pytest.raises(fetch.DownloadError, match="the limit is 1000"):
        fetch.download(API, max_bytes=1000, resolver=public_resolver)


def test_a_streamed_body_is_cut_at_the_cap_and_the_stream_is_not_drained(serve):
    pulled = []

    def chunks():
        while True:                  # an endless body: the cap must stop the read
            pulled.append(1)
            yield b"x" * 100

    serve(lambda request: httpx.Response(200, headers={"content-type": "text/csv"}, content=chunks()))
    with pytest.raises(fetch.DownloadError, match="larger than 1000 bytes"):
        fetch.download(API, max_bytes=1000, resolver=public_resolver)
    assert len(pulled) <= 11


def test_a_body_exactly_at_the_cap_is_accepted(serve):
    serve(lambda request: ok(b"a,b\n" + b"1,2\n" * 3))          # 16 bytes
    assert len(fetch.download(API, max_bytes=16, resolver=public_resolver)["bytes"]) == 16


@pytest.mark.parametrize("resolved, reason", [
    ("10.0.0.5", "private"), ("127.0.0.1", "loopback"), ("169.254.169.254", "metadata"),
    ("192.168.1.9", "private"), ("100.64.0.1", "non-public"),
])
def test_a_name_that_resolves_to_a_non_public_address_is_refused_without_a_request(serve, resolved, reason):
    calls = []
    serve(lambda request: calls.append(request) or ok())
    with pytest.raises(fetch.DownloadError, match=reason):
        fetch.download(API, resolver=lambda host, port: [resolved])
    assert calls == []


@pytest.mark.parametrize("url, reason", [
    ("https://127.0.0.1/x", "loopback"),
    ("https://10.1.2.3/x", "private"),
    ("https://169.254.169.254/latest/meta-data", "metadata"),
    ("http://localhost/x", "internal name"),
    ("http://printer.local/x", "internal name"),
    ("https://user:pass@api.open-meteo.com/v1/forecast", "credentials"),
    ("file:///etc/passwd", "scheme"),
    ("ftp://api.open-meteo.com/x", "scheme"),
    ("https://api.open-meteo.com:8443/x", "port"),
    ("", "empty"),
])
def test_unsafe_urls_are_refused_before_any_request(serve, url, reason):
    calls = []
    serve(lambda request: calls.append(request) or ok())
    with pytest.raises(fetch.DownloadError, match=reason):
        fetch.download(url, resolver=public_resolver)
    assert calls == []


def test_four_redirects_are_followed_and_the_final_url_is_the_last_named_hop(serve):
    def handler(request):
        hop = int(request.url.path.rsplit("/", 1)[-1]) if "/hop/" in request.url.path else 0
        if hop < 4:
            return httpx.Response(302, headers={"location": f"https://api.open-meteo.com/hop/{hop + 1}"})
        return ok()

    serve(handler)
    got = fetch.download("https://api.open-meteo.com/hop/0", resolver=public_resolver)
    assert got["final_url"] == "https://api.open-meteo.com/hop/4"
    assert got["bytes"] == CSV_BYTES


def test_a_fifth_redirect_is_refused(serve):
    calls = []

    def handler(request):
        calls.append(request)
        hop = int(request.url.path.rsplit("/", 1)[-1]) if "/hop/" in request.url.path else 0
        return httpx.Response(302, headers={"location": f"https://api.open-meteo.com/hop/{hop + 1}"})

    serve(handler)
    with pytest.raises(fetch.DownloadError, match="more than 4 redirects"):
        fetch.download("https://api.open-meteo.com/hop/0", resolver=public_resolver)
    assert len(calls) == 5


def test_a_redirect_to_a_private_address_is_refused_and_not_followed(serve):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://10.1.1.1/admin"})

    serve(handler)
    with pytest.raises(fetch.DownloadError, match="private"):
        fetch.download(API, resolver=public_resolver)
    assert len(calls) == 1


def test_a_redirect_to_a_name_that_resolves_privately_is_refused(serve):
    names = {"api.open-meteo.com": PUBLIC, "intranet.example": "192.168.0.9"}
    serve(lambda request: httpx.Response(302, headers={"location": "https://intranet.example/data.csv"}))
    with pytest.raises(fetch.DownloadError, match="private"):
        fetch.download(API, resolver=lambda host, port: [names[host]])


def test_a_redirect_to_an_internal_name_is_refused(serve):
    serve(lambda request: httpx.Response(302, headers={"location": "http://metadata.google.internal/x"}))
    with pytest.raises(fetch.DownloadError, match="internal name"):
        fetch.download(API, resolver=public_resolver)


def test_a_relative_redirect_is_resolved_against_the_current_hop(serve):
    def handler(request):
        if request.url.path == "/v1/forecast":
            return httpx.Response(301, headers={"location": "/v1/other"})
        return ok()

    serve(handler)
    assert fetch.download(API, resolver=public_resolver)["final_url"] == "https://api.open-meteo.com/v1/other"


def test_a_transport_failure_becomes_a_download_error(serve):
    def handler(request):
        raise httpx.ConnectError("refused")

    serve(handler)
    with pytest.raises(fetch.DownloadError, match="transport error: ConnectError"):
        fetch.download(API, resolver=public_resolver)


def test_size_and_time_limits_must_be_positive():
    with pytest.raises(ValueError, match="max_bytes"):
        fetch.download(API, max_bytes=0)
    with pytest.raises(ValueError, match="timeout_s"):
        fetch.download(API, timeout_s=0)
