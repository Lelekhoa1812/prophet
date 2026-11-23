"""r6m redirects: every redirect hop is checked, whatever the caller's client is configured to do.

Defects reproduced here:
- a caller's httpx.Client with follow_redirects=True let httpx follow a redirect to a private address without the
  per-hop address check ever running;
- robots.txt was consulted only for the first URL, so a redirect could carry the fetch into an origin that disallows it.
No network: every request goes through httpx.MockTransport.
"""
import httpx

from prophet_agent.research.fetch import fetch_page
from prophet_agent.research.netguard import HostThrottle

PUBLIC = "93.184.216.34"
ARTICLE = b"<!doctype html><html><head><title>Oil</title></head><body><article><p>Brent rose on Monday.</p>" \
          b"</article></body></html>"


def public_resolver(host, port):
    return [PUBLIC]


def test_redirect_to_another_origin_is_checked_against_its_robots_txt():
    def handler(request):
        path, host = request.url.path, request.headers["host"]
        if path == "/robots.txt":
            if host == "other.example.org":
                return httpx.Response(200, text="User-agent: *\nDisallow: /\n")
            return httpx.Response(404)
        if path == "/old":
            return httpx.Response(302, headers={"location": "https://other.example.org/article"})
        return httpx.Response(200, headers={"content-type": "text/html"}, content=ARTICLE)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    page = fetch_page("https://example.com/old", client=client, resolver=public_resolver,
                      throttle=HostThrottle(0.0), use_cache=False)
    assert page.ok is False and "robots.txt" in page.note
