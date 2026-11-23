"""robots.txt is fetched through the per-host throttle, and a Crawl-delay sets the spacing for that host (GOAL.md Round
6m, defect d). Before the fix, robots.txt was requested without pacing and the Crawl-delay was ignored."""
import httpx

from prophet_agent.research.fetch import fetch_page
from prophet_agent.research.netguard import HostThrottle, Robots

PUBLIC = "93.184.216.34"
ARTICLE = "<html><head><title>Paced</title></head><body><article><p>" + "Copper prices held steady. " * 20 + "</p></article></body></html>"


def public_resolver(host, port):
    return [PUBLIC]


class SpyThrottle(HostThrottle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.waits = []

    def wait(self, host, *args, **kwargs):
        self.waits.append(host)
        return super().wait(host, *args, **kwargs)


def client_with_crawl_delay(robots_calls):
    def handler(request):
        if request.url.path == "/robots.txt":
            robots_calls.append(str(request.url))
            return httpx.Response(200, text="User-agent: *\nCrawl-delay: 3\n")
        return httpx.Response(200, headers={"content-type": "text/html"}, text=ARTICLE)
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def test_crawl_delay_spaces_later_requests_to_the_same_host(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))
    sleeps = []
    throttle = SpyThrottle(0.0, sleep=sleeps.append)
    robots_calls = []
    client = client_with_crawl_delay(robots_calls)
    robots = Robots(client=client, resolver=public_resolver)
    first = fetch_page("https://news.example.com/a", client=client, resolver=public_resolver, robots=robots,
                       throttle=throttle, use_cache=False)
    second = fetch_page("https://news.example.com/b", client=client, resolver=public_resolver, robots=robots,
                        throttle=throttle, use_cache=False)
    assert first.ok and second.ok
    assert len(robots_calls) == 1
    assert sleeps, "the Crawl-delay did not space the second request"
    assert 2.0 < sleeps[-1] <= 3.0


def test_robots_request_goes_through_the_host_throttle(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))
    throttle = SpyThrottle(0.0, sleep=lambda seconds: None)
    client = client_with_crawl_delay([])
    fetch_page("https://news.example.com/a", client=client, resolver=public_resolver,
               robots=Robots(client=client, resolver=public_resolver), throttle=throttle, use_cache=False)
    # one slot for the robots.txt request and one for the page
    assert throttle.waits == ["news.example.com", "news.example.com"]
