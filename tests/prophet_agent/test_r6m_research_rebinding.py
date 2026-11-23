"""r6m DNS rebinding: the address that is checked is the address that is connected to.

The defect: fetch resolved the host in check_url, resolved it again in _get_checked, and httpx resolved it a third time
when it connected. A host that answers with a public address for the checks and a private one for the connect (TTL 0
rebinding) reached the private address. The fix resolves once per hop and connects to the checked literal address,
keeping the original Host header and TLS server name.

No network: the resolver is injected, httpx uses MockTransport, and the socket layer is replaced by a recorder.
"""
import socket

import httpx
import pytest

from prophet_agent.research import netguard
from prophet_agent.research.fetch import fetch_page
from prophet_agent.research.netguard import HostThrottle

PUBLIC = "93.184.216.34"
OTHER_PUBLIC = "151.101.1.69"
PRIVATE_ANSWERS = ["127.0.0.1", "169.254.169.254", "10.0.0.5"]
ARTICLE = b"<!doctype html><html><head><title>Oil</title></head><body><article><p>Brent rose on Monday.</p>" \
          b"</article></body></html>"


class Answers:
    """A resolver whose answers come in order: the first call gets the first address, and the last answer repeats."""

    def __init__(self, *addresses):
        self.addresses = list(addresses)
        self.calls = []

    def __call__(self, host, port):
        self.calls.append(host)
        return [self.addresses[min(len(self.calls), len(self.addresses)) - 1]]


class AllowAll:
    def allowed(self, url):
        return True


@pytest.mark.parametrize("private", PRIVATE_ANSWERS)
def test_a_later_lookup_cannot_move_the_request_to_a_private_address(private):
    seen = []

    def handler(request):
        seen.append((request.url.host, request.headers.get("host"), request.extensions.get("sni_hostname")))
        return httpx.Response(200, headers={"content-type": "text/html"}, content=ARTICLE)

    resolver = Answers(PUBLIC, private)
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    page = fetch_page("https://news.example.com/oil", client=client, resolver=resolver,
                      robots=AllowAll(), throttle=HostThrottle(0.0), use_cache=False)
    assert page.ok is True
    assert resolver.calls == ["news.example.com"]                 # one lookup per hop
    assert seen == [(PUBLIC, "news.example.com", "news.example.com")]


@pytest.fixture
def socket_recorder(monkeypatch):
    """Replace the socket layer. The system resolver answers with a private address (as a rebinding server would for
    httpx), and every connection attempt is recorded and refused, so nothing is ever contacted."""
    attempts = []
    system_answer = {"address": "127.0.0.1"}

    def fake_getaddrinfo(host, port, *args, **kwargs):
        address = host if host.replace(".", "").isdigit() else system_answer["address"]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    def fake_create_connection(address, *args, **kwargs):
        host, port = address
        target = socket.getaddrinfo(host, port)[0][4][0]
        attempts.append(target)
        raise OSError("test: connections are recorded, not made")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(socket, "create_connection", fake_create_connection)
    return attempts, system_answer


@pytest.mark.parametrize("private", PRIVATE_ANSWERS)
def test_the_connection_is_made_to_the_checked_address_only(socket_recorder, private):
    attempts, system_answer = socket_recorder
    system_answer["address"] = private            # what httpx's own lookup would return
    client = httpx.Client(trust_env=False, follow_redirects=False)
    fetch_page("https://news.example.com/oil", client=client, resolver=Answers(PUBLIC),
               robots=AllowAll(), throttle=HostThrottle(0.0), use_cache=False)
    client.close()
    assert attempts == [PUBLIC]


def test_refused_address_falls_back_to_the_next_checked_address(socket_recorder):
    attempts, _ = socket_recorder
    client = httpx.Client(trust_env=False, follow_redirects=False)
    fetch_page("https://news.example.com/oil", client=client, resolver=lambda host, port: [PUBLIC, OTHER_PUBLIC],
               robots=AllowAll(), throttle=HostThrottle(0.0), use_cache=False)
    client.close()
    assert attempts == [PUBLIC, OTHER_PUBLIC]


def test_a_private_address_in_the_answer_still_refuses_the_host(socket_recorder):
    attempts, _ = socket_recorder
    page = fetch_page("https://news.example.com/oil", resolver=lambda host, port: [PUBLIC, "10.0.0.5"],
                      robots=AllowAll(), throttle=HostThrottle(0.0), use_cache=False)
    assert page.ok is False and "private" in page.note
    assert attempts == []


def test_netguard_pins_the_request_to_an_address_literal():
    parts, addresses = netguard.resolve_checked("https://news.example.com:443/a?b=1", resolver=Answers(PUBLIC))
    target = netguard.pinned(parts, addresses[0])
    assert target.url == f"https://{PUBLIC}:443/a?b=1"
    assert target.host == "news.example.com:443"
    assert target.sni == "news.example.com"


def test_pinned_requests_never_share_a_pooled_connection():
    """Two names on one checked address must not share a TLS session verified for only one of them."""
    seen = []

    def handler(request):
        seen.append((request.headers.get("host"), request.headers.get("connection")))
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://other.example.com/article"})
        return httpx.Response(200, headers={"content-type": "text/html"}, content=ARTICLE)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    page = fetch_page("https://news.example.com/start", client=client, resolver=lambda host, port: [PUBLIC],
                      robots=AllowAll(), throttle=HostThrottle(0.0), use_cache=False)
    assert page.ok is True
    assert seen == [("news.example.com", "close"), ("other.example.com", "close")]
