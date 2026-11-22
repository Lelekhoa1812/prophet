"""netguard: address, scheme and credential checks, host pacing and robots.txt. No network."""
import httpx
import pytest

from prophet_agent.research.netguard import HostThrottle, Robots, UnsafeURL, check_url

PUBLIC = "93.184.216.34"


def public_resolver(host, port):
    return [PUBLIC]


def resolver_to(*addresses):
    return lambda host, port: list(addresses)


@pytest.mark.parametrize("url", [
    "http://localhost/admin",
    "http://127.0.0.1/",
    "http://[::1]/",
    "http://printer.internal/",
    "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/status",
    "http://192.168.1.1/",
    "http://172.16.4.4/",
])
def test_blocks_internal_addresses_and_names(url):
    with pytest.raises(UnsafeURL):
        check_url(url, resolver=public_resolver)


def test_blocks_hostname_that_resolves_to_metadata_address():
    with pytest.raises(UnsafeURL, match="metadata"):
        check_url("http://metadata.example.com/", resolver=resolver_to("169.254.169.254"))


def test_blocks_hostname_that_resolves_to_loopback():
    with pytest.raises(UnsafeURL, match="loopback"):
        check_url("https://looks-public.example.com/", resolver=resolver_to("127.0.0.1"))


def test_one_private_address_among_public_ones_rejects_the_host():
    with pytest.raises(UnsafeURL, match="private"):
        check_url("https://mixed.example.com/", resolver=resolver_to(PUBLIC, "10.1.2.3"))


def test_ipv4_mapped_loopback_is_blocked():
    with pytest.raises(UnsafeURL):
        check_url("https://mapped.example.com/", resolver=resolver_to("::ffff:127.0.0.1"))


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "javascript:alert(1)",
                                 "data:text/html,hi"])
def test_blocks_non_http_schemes(url):
    with pytest.raises(UnsafeURL, match="scheme"):
        check_url(url, resolver=public_resolver)


def test_blocks_credentials_in_url():
    with pytest.raises(UnsafeURL, match="credentials"):
        check_url("https://user:secret@example.com/", resolver=public_resolver)


def test_blocks_non_default_ports():
    with pytest.raises(UnsafeURL, match="port"):
        check_url("https://example.com:8443/", resolver=public_resolver)


def test_allows_public_ip_literal_without_resolver():
    parts = check_url(f"https://{PUBLIC}/page")
    assert parts.hostname == PUBLIC


def test_allows_hostname_resolving_only_to_public_addresses():
    assert check_url("https://news.example.com/a", resolver=resolver_to(PUBLIC, "151.101.1.1")).scheme == "https"


def test_host_without_addresses_is_refused():
    with pytest.raises(UnsafeURL, match="no addresses"):
        check_url("https://empty.example.com/", resolver=resolver_to())


def test_empty_and_overlong_urls_are_refused():
    with pytest.raises(UnsafeURL):
        check_url("   ", resolver=public_resolver)
    with pytest.raises(UnsafeURL, match="longer"):
        check_url("https://example.com/" + "a" * 3000, resolver=public_resolver)


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_throttle_spaces_requests_to_the_same_host():
    clock = FakeClock()
    throttle = HostThrottle(1.0, clock=clock, sleep=clock.sleep)
    assert throttle.wait("example.com") == 0
    throttle.wait("example.com")
    assert clock.sleeps == [pytest.approx(1.0)]


def test_throttle_does_not_delay_a_different_host():
    clock = FakeClock()
    throttle = HostThrottle(1.0, clock=clock, sleep=clock.sleep)
    throttle.wait("a.example.com")
    throttle.wait("b.example.com")
    assert clock.sleeps == []


def robots_client(body=None, status=200, error=None, counter=None):
    def handler(request):
        if counter is not None:
            counter.append(request.url.path)
        if error:
            raise error
        return httpx.Response(status, text=body or "")
    return httpx.Client(transport=httpx.MockTransport(handler))


ROBOTS = "User-agent: *\nDisallow: /private/\n"


def test_robots_disallow_is_honoured():
    robots = Robots(client=robots_client(ROBOTS), resolver=public_resolver)
    assert robots.allowed("https://example.com/private/report") is False
    assert robots.allowed("https://example.com/public/report") is True


def test_robots_network_error_fails_open():
    robots = Robots(client=robots_client(error=httpx.ConnectError("down")), resolver=public_resolver)
    assert robots.allowed("https://example.com/private/x") is True


def test_robots_missing_file_allows_everything():
    robots = Robots(client=robots_client(status=404), resolver=public_resolver)
    assert robots.allowed("https://example.com/anything") is True


def test_robots_is_fetched_once_per_origin():
    calls = []
    robots = Robots(client=robots_client(ROBOTS, counter=calls), resolver=public_resolver)
    robots.allowed("https://example.com/a")
    robots.allowed("https://example.com/b/c")
    robots.allowed("https://other.example.org/a")
    assert calls.count("/robots.txt") == 2
