"""A robots.txt answer with a 5xx status fails closed for that host for the run (GOAL.md Round 6m, defect c). The
earlier behaviour treated it as allowed, so pages were fetched from a host whose rules could not be read."""
import httpx

from prophet_agent.research.fetch import fetch_page
from prophet_agent.research.netguard import HostThrottle, Robots

PUBLIC = "93.184.216.34"
ARTICLE = "<html><head><title>Rules</title></head><body><article><p>" + "Inventories fell last week. " * 20 + "</p></article></body></html>"


def public_resolver(host, port):
    return [PUBLIC]


def client_with_robots(status, calls):
    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(status, text="busy")
        return httpx.Response(200, headers={"content-type": "text/html"}, text=ARTICLE)
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def test_robots_server_error_refuses_the_page(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))
    calls = []
    client = client_with_robots(503, calls)
    page = fetch_page("https://news.example.com/a", client=client, resolver=public_resolver,
                      robots=Robots(client=client, resolver=public_resolver), throttle=HostThrottle(0.0),
                      use_cache=False)
    assert page.ok is False
    assert "robots.txt" in page.note and "503" in page.note
    assert "/a" not in calls


def test_robots_5xx_stays_closed_for_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))
    calls = []
    client = client_with_robots(500, calls)
    robots = Robots(client=client, resolver=public_resolver)
    throttle = HostThrottle(0.0)
    first = fetch_page("https://news.example.com/a", client=client, resolver=public_resolver, robots=robots,
                       throttle=throttle, use_cache=False)
    second = fetch_page("https://news.example.com/b", client=client, resolver=public_resolver, robots=robots,
                        throttle=throttle, use_cache=False)
    assert first.ok is False and second.ok is False
    assert calls.count("/robots.txt") == 1, "robots.txt was asked again within the run"
    assert robots.allowed("https://news.example.com/anything") is False


def test_robots_404_still_allows_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))
    calls = []
    client = client_with_robots(404, calls)
    page = fetch_page("https://news.example.com/a", client=client, resolver=public_resolver,
                      robots=Robots(client=client, resolver=public_resolver), throttle=HostThrottle(0.0),
                      use_cache=False)
    assert page.ok is True
