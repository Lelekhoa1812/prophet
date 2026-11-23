"""r6m limits: one fetch has an overall time budget, and robots.txt is read under a size cap.

Defects reproduced here:
- httpx timeouts apply to each socket operation, so a server that trickles one byte at a time, or answers each redirect
  slowly, could keep a fetch running for as long as the server liked. The fetch now has one deadline for its redirects
  and body; a body that is still arriving at the deadline is cut, and a fetch with no answer by then fails.
- robots.txt was read whole into memory before the size cap was applied.
The clock is a fake so the tests are deterministic. No network: httpx.MockTransport throughout.
"""
import httpx

from prophet_agent.research import fetch as fetch_mod
from prophet_agent.research import netguard
from prophet_agent.research.fetch import fetch_page
from prophet_agent.research.netguard import ROBOTS_MAX_BYTES, HostThrottle, Robots

PUBLIC = "93.184.216.34"


class Clock:
    """A fake monotonic clock. Handlers and body generators advance it to model time passing."""

    def __init__(self, start=1000.0):
        self.now = start

    def monotonic(self):
        return self.now


class AllowAll:
    def allowed(self, url):
        return True


def public_resolver(host, port):
    return [PUBLIC]


def install_clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(fetch_mod, "time", clock)
    monkeypatch.setattr(netguard, "time", clock)
    return clock


def trickle(clock, pieces, seconds_each):
    """A body that arrives one piece at a time, each taking ``seconds_each`` seconds."""
    for _ in range(pieces):
        clock.now += seconds_each
        yield b"<article><p>Brent crude held steady through the session on Monday.</p></article>"


def test_a_body_still_arriving_at_the_deadline_is_cut_and_said_so(monkeypatch):
    clock = install_clock(monkeypatch)

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html"}, content=trickle(clock, 1000, 10.0))

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    page = fetch_page("https://news.example.com/slow", client=client, resolver=public_resolver, robots=AllowAll(),
                      throttle=HostThrottle(0.0), use_cache=False, total_seconds=30)
    assert page.ok is True
    assert "stopped after 30 seconds" in page.note
    assert clock.now < 1000.0 + 30 + 10 + 1      # the body stopped at the deadline, not after 1000 pieces


def test_redirects_share_one_time_budget(monkeypatch):
    clock = install_clock(monkeypatch)

    def handler(request):
        clock.now += 20.0                         # each hop takes 20 seconds to answer
        if request.url.path == "/hop/4":
            return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<p>x</p>")
        nxt = int(request.url.path.rsplit("/", 1)[1]) + 1
        return httpx.Response(302, headers={"location": f"/hop/{nxt}"})

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    page = fetch_page("https://news.example.com/hop/0", client=client, resolver=public_resolver, robots=AllowAll(),
                      throttle=HostThrottle(0.0), use_cache=False, total_seconds=30)
    assert page.ok is False and "within 30 seconds" in page.note


def test_read_capped_reports_why_it_stopped():
    class Body:
        def __init__(self, pieces):
            self.pieces = pieces

        def iter_bytes(self):
            yield from self.pieces

    assert netguard.read_capped(Body([b"abc", b"def"]), 10) == (b"abcdef", "")
    assert netguard.read_capped(Body([b"abc", b"defgh"]), 6) == (b"abcdef", "size")
