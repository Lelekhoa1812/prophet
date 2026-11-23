"""The per-host throttle never sleeps past the request deadline (GOAL.md Round 6m, defect e). A wait that would end after
the deadline reserves no slot and returns None, and the fetch fails at the deadline."""
import time

import httpx

from prophet_agent.research.fetch import fetch_page
from prophet_agent.research.netguard import HostThrottle, Robots

PUBLIC = "93.184.216.34"
ARTICLE = "<html><head><title>Late</title></head><body><article><p>" + "Gold eased on Tuesday. " * 20 + "</p></article></body></html>"


def public_resolver(host, port):
    return [PUBLIC]


def test_a_wait_that_ends_after_the_deadline_reserves_nothing():
    sleeps = []
    throttle = HostThrottle(10.0, sleep=sleeps.append)
    assert throttle.wait("news.example.com") == 0
    deadline = time.monotonic() + 1.0
    assert throttle.wait("news.example.com", deadline=deadline) is None
    assert sleeps == []


def test_a_throttled_fetch_fails_at_the_deadline_instead_of_sleeping_past_it(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))
    sleeps = []
    throttle = HostThrottle(10.0, sleep=sleeps.append)
    throttle.wait("news.example.com")   # an earlier request has just used the host's slot

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, headers={"content-type": "text/html"}, text=ARTICLE)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    page = fetch_page("https://news.example.com/a", client=client, resolver=public_resolver,
                      robots=Robots(client=client, resolver=public_resolver), throttle=throttle, use_cache=False,
                      total_seconds=1.0)
    assert page.ok is False
    assert "seconds" in page.note
    assert all(delay <= 1.0 for delay in sleeps), f"slept {sleeps} seconds against a 1 second deadline"
