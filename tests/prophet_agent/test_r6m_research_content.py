"""r6m content: a body is read as text only when it is text.

Defects reproduced here:
- a body labelled text/html, or with no Content-Type at all, was decoded and extracted whatever its bytes were, so a
  binary file (NUL bytes, a PDF signature) could become a "page" full of junk;
- a 3xx response with no Location header was read as a page, as if it were the content.
UTF-16 pages are legitimate and must still be read. No network: httpx.MockTransport throughout.
"""
import httpx

from prophet_agent.research.fetch import fetch_page
from prophet_agent.research.netguard import HostThrottle

PUBLIC = "93.184.216.34"
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"


class AllowAll:
    def allowed(self, url):
        return True


def public_resolver(host, port):
    return [PUBLIC]


def fetch_one(headers, body, status=200):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(status, headers=headers, content=body)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    return fetch_page("https://news.example.com/doc", client=client, resolver=public_resolver, robots=AllowAll(),
                      throttle=HostThrottle(0.0), use_cache=False)


def test_binary_body_labelled_as_html_is_refused():
    page = fetch_one({"content-type": "text/html"}, PNG)
    assert page.ok is False and "binary" in page.note


def test_binary_body_with_no_content_type_is_refused():
    page = fetch_one({}, PNG)
    assert page.ok is False and "binary" in page.note


def test_pdf_bytes_labelled_as_html_are_refused_as_a_pdf():
    page = fetch_one({"content-type": "text/html"}, b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj << >> endobj")
    assert page.ok is False and "PDF" in page.note


def test_redirect_status_without_a_location_is_not_read_as_a_page():
    page = fetch_one({"content-type": "text/html"}, b"<article><p>Choose a regional edition of the report.</p>"
                                                    b"</article>", status=300)
    assert page.ok is False and "HTTP 300" in page.note


def test_utf16_html_is_still_read():
    html = "<html><head><title>Oil</title></head><body><article><p>Brent rose on Monday.</p></article></body>" \
           "</html>"
    page = fetch_one({"content-type": "text/html; charset=utf-16"}, html.encode("utf-16"))
    assert page.ok is True and "Brent rose on Monday." in page.text
