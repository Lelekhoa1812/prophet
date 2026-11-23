"""A DNS lookup that outlasts the request deadline is abandoned at the deadline (GOAL.md Round 6m, defect a).

Both the page lookup and the robots.txt lookup go through the resolver, so both must be bounded. The slow resolver
blocks on an event that the test releases at the end, so no thread is left sleeping after the run."""
import socket
import threading
import time

import httpx

from prophet_agent.research import netguard
from prophet_agent.research.fetch import fetch_page
from prophet_agent.research.netguard import HostThrottle

PUBLIC = "93.184.216.34"
ARTICLE = "<html><head><title>Slow</title></head><body><article><p>" + "Brent crude rose again. " * 20 + "</p></article></body></html>"


def page_client():
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, headers={"content-type": "text/html"}, text=ARTICLE)
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def test_a_slow_resolver_is_abandoned_at_the_deadline(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))
    release = threading.Event()

    def slow_resolver(host, port):
        release.wait(5)   # the lookup outlasts the deadline by a wide margin
        return [PUBLIC]

    try:
        started = time.monotonic()
        page = fetch_page("https://slow-dns.example.com/article", client=page_client(), resolver=slow_resolver,
                          throttle=HostThrottle(0.0), use_cache=False, total_seconds=0.5)
        elapsed = time.monotonic() - started
    finally:
        release.set()
    assert page.ok is False
    assert elapsed < 3.0, f"the fetch took {elapsed:.1f} seconds against a 0.5 second deadline"


def test_the_default_system_lookup_is_bounded_too(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))
    release = threading.Event()

    def slow_getaddrinfo(*args, **kwargs):
        release.wait(5)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC, 443))]

    monkeypatch.setattr(netguard.socket, "getaddrinfo", slow_getaddrinfo)
    try:
        started = time.monotonic()
        page = fetch_page("https://slow-dns.example.com/article", client=page_client(), throttle=HostThrottle(0.0),
                          use_cache=False, total_seconds=0.5)
        elapsed = time.monotonic() - started
    finally:
        release.set()
    assert page.ok is False
    assert elapsed < 3.0, f"the fetch took {elapsed:.1f} seconds against a 0.5 second deadline"
