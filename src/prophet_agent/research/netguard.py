"""Address, scheme and redirect checks for every outbound research request, plus a polite per-host pacing and robots.txt.

Motivation vs Logic
Motivation: a research tool that follows whatever link a search result or page contains can be pointed at the company's
own network (a router admin page, a cloud metadata service that hands out credentials) or at a file on disk. Search
results and redirects are data from the internet, so they must never decide where the agent connects.
Logic: only http and https on the default ports are accepted, with no credentials in the URL. The host is resolved once
per request, and every address it resolves to must be globally routable and not a metadata address; a single bad address
rejects the host. The connection is then made to one of the checked addresses itself, written into the URL as a literal,
with the original name kept in the Host header and the TLS server name. So a DNS answer that changes after the check
(DNS rebinding) cannot move the connection. Redirects are followed by hand so each hop is checked again (the caller does
that loop). Robots.txt is fetched once per origin and honoured; a network failure there fails open, as the brief asks.
A per-host lock spaces requests at least one second apart.
"""
from __future__ import annotations

import ipaddress
import socket
import threading
import time
import urllib.robotparser
import zlib
from collections.abc import Callable
from typing import NamedTuple
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpx

USER_AGENT = "ProphetResearch/1.0"
ALLOWED_SCHEMES = ("http", "https")
ALLOWED_PORTS = (None, 80, 443)
MAX_URL_CHARS = 2048
METADATA_ADDRESSES = {"169.254.169.254", "fd00:ec2::254", "100.100.100.200"}
BLOCKED_HOSTNAMES = {"localhost", "metadata", "metadata.google.internal", "metadata.azure.internal"}
BLOCKED_SUFFIXES = (".localhost", ".local", ".internal", ".home.arpa", ".lan")
ROBOTS_MAX_BYTES = 500_000
NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")        # RFC 6052 well-known prefix: IPv4 in the last 32 bits
NAT64_LOCAL_USE = ipaddress.ip_network("64:ff9b:1::/48")   # RFC 8215 local-use NAT64: not a fixed mapping, never fetched

Resolver = Callable[[str, int], list[str]]


class UnsafeURL(ValueError):
    """Raised when a URL or the address it resolves to is not allowed. The message is safe to show the planner."""


class Pinned(NamedTuple):
    """One connection target. ``url`` has the checked address literal in place of the name; ``host`` is the Host header
    and ``sni`` the TLS server name (None when the URL names an address, or the scheme is not https)."""
    url: str
    host: str
    sni: str | None


def _default_resolver(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError) as exc:
        raise UnsafeURL(f"host {host!r} did not resolve ({type(exc).__name__})") from None
    return [info[4][0] for info in infos]


LOOKUP_SECONDS = 10.0   # the longest a name lookup may take when the caller gives no deadline


class LookupTimeout(httpx.ConnectTimeout):
    """The name lookup did not finish before the deadline. It is an httpx transport error, so callers that already catch
    transport errors (the open-data fetcher, for one) report it without a new branch. The lookup cannot be cancelled, so
    its thread is left to end on its own and its answer is dropped."""


def _bounded_lookup(resolver: Resolver, host: str, port: int, seconds: float) -> list[str]:
    """Run ``resolver`` in a daemon thread and wait at most ``seconds`` for it. The system resolver blocks with no
    timeout of its own, so the wait, not the lookup, is what the deadline bounds."""
    if seconds <= 0:
        raise LookupTimeout(f"host {host!r} did not resolve before the deadline")
    outcome: dict = {}

    def run() -> None:
        try:
            outcome["addresses"] = resolver(host, port)
        except BaseException as exc:   # handed back to the caller, which re-raises it
            outcome["error"] = exc

    worker = threading.Thread(target=run, name="dns-lookup", daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise LookupTimeout(f"host {host!r} did not resolve before the deadline")
    if "error" in outcome:
        raise outcome["error"]
    return list(outcome.get("addresses", []))


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The IPv4 address an IPv6 address carries, for the mapped, NAT64 and IPv4-compatible forms. A NAT64 gateway or a
    dual-stack host connects to that IPv4 address, so it is the address that must be checked."""
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    value = int(ip)
    if ip in NAT64_PREFIX or 1 < value < 2 ** 32:   # ::a.b.c.d is IPv4-compatible; ::0 and ::1 are not
        return ipaddress.IPv4Address(value & 0xFFFFFFFF)
    return None


def blocked_reason(address: str) -> str | None:
    """Why an address may not be contacted, or None when it is a public, globally routable address."""
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return "malformed"
    if isinstance(ip, ipaddress.IPv6Address):
        if ip in NAT64_LOCAL_USE:
            return "non-public"
        embedded = _embedded_ipv4(ip)
        if embedded is not None:
            ip = embedded
    if str(ip) in METADATA_ADDRESSES:
        return "metadata"
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local"
    if ip.is_multicast:
        return "multicast"
    if ip.is_private:
        return "private"
    if ip.is_reserved or ip.is_unspecified:
        return "reserved"
    if getattr(ip, "is_site_local", False):
        return "private"
    if not ip.is_global:
        return "non-public"
    return None


def _is_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _host_and_port(parts: SplitResult) -> tuple[str, int | None, bool]:
    host = (parts.hostname or "").rstrip(".")
    try:
        port = parts.port
    except ValueError:
        raise UnsafeURL("the URL has an invalid port") from None
    return host, port, _is_literal(host)


def check_syntax(url: str) -> SplitResult:
    """The syntax rules (length, scheme, credentials, host, port, internal names) without any DNS lookup. Returns the
    parsed URL; raises UnsafeURL otherwise."""
    if not isinstance(url, str) or not url.strip():
        raise UnsafeURL("the URL is empty")
    url = url.strip()
    if len(url) > MAX_URL_CHARS:
        raise UnsafeURL(f"the URL is longer than {MAX_URL_CHARS} characters")
    parts = urlsplit(url)
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeURL(f"scheme {parts.scheme or '(none)'!r} is not allowed; use http or https")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise UnsafeURL("credentials in the URL are not allowed")
    host, port, literal = _host_and_port(parts)
    if not host:
        raise UnsafeURL("the URL has no host")
    if port not in ALLOWED_PORTS:
        raise UnsafeURL(f"port {port} is not allowed; use the default ports 80 or 443")
    if host in BLOCKED_HOSTNAMES or host.endswith(BLOCKED_SUFFIXES):
        raise UnsafeURL(f"host {host!r} is an internal name")
    if not literal:
        try:
            host.encode("idna")
        except UnicodeError:
            raise UnsafeURL(f"host {host!r} is not a valid name") from None
    return parts


def resolve_checked(url: str, *, resolver: Resolver | None = None,
                    deadline: float | None = None) -> tuple[SplitResult, list[str]]:
    """Check the URL and resolve its host once. Returns the parsed URL and the checked addresses, in the order given;
    raises UnsafeURL when any address is not public, and LookupTimeout when the lookup does not finish by ``deadline``
    (a time.monotonic value; without one the lookup is bounded by LOOKUP_SECONDS). ``resolver(host, port) -> [ip, ...]``
    is injectable so tests never touch the network."""
    parts = check_syntax(url)
    host, port, literal = _host_and_port(parts)
    if literal:
        addresses = [host]
    else:
        seconds = LOOKUP_SECONDS if deadline is None else deadline - time.monotonic()
        addresses = _bounded_lookup(resolver or _default_resolver, host,
                                    port or (443 if parts.scheme.lower() == "https" else 80), seconds)
    if not addresses:
        raise UnsafeURL(f"host {host!r} has no addresses")
    checked: list[str] = []
    for address in addresses:
        reason = blocked_reason(address)
        if reason:
            raise UnsafeURL(f"host {host!r} resolves to a {reason} address, which is not fetched")
        ip = str(ipaddress.ip_address(address.split("%", 1)[0]))
        if ip not in checked:
            checked.append(ip)
    return parts, checked


def check_url(url: str, *, resolver: Resolver | None = None) -> SplitResult:
    """Return the parsed URL if it may be fetched; raise UnsafeURL otherwise."""
    return resolve_checked(url, resolver=resolver)[0]


def pinned(parts: SplitResult, address: str) -> Pinned:
    """The connection target that reaches ``address`` for the URL ``parts``: the address literal in the URL, the
    original name in the Host header, and the name as the TLS server name."""
    host, port, literal = _host_and_port(parts)
    scheme = parts.scheme.lower()
    ip = str(ipaddress.ip_address(address.split("%", 1)[0]))
    netloc = f"[{ip}]" if ":" in ip else ip
    if port is not None:
        netloc += f":{port}"
    url = urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))
    if literal:
        return Pinned(url, parts.netloc, None)
    name = host.encode("idna").decode("ascii")
    return Pinned(url, name if port is None else f"{name}:{port}", name if scheme == "https" else None)


def open_pinned(client: httpx.Client, url: str, *, resolver: Resolver | None, headers: dict[str, str] | None,
                timeout: float, deadline: float | None = None) -> httpx.Response:
    """Send one GET whose connection goes to a checked address. The name is resolved once, every address is checked,
    and each checked address is tried in order until one connects, while the deadline (a time.monotonic value) has not
    passed. Redirects are never followed here. The caller must close the returned response. Raises UnsafeURL for a
    refused address, or the last connect error."""
    parts, addresses = resolve_checked(url, resolver=resolver, deadline=deadline)
    failure: Exception | None = None
    for address in addresses:
        if failure is not None and deadline is not None and time.monotonic() >= deadline:
            break
        target = pinned(parts, address)
        # The pool keys connections by the address in the URL, so two names on one address would share a TLS session
        # verified for the first name. Closing after each response keeps every session matched to its own name.
        request = client.build_request("GET", target.url,
                                       headers={**(headers or {}), "Host": target.host, "Connection": "close"},
                                       timeout=timeout,
                                       extensions={"sni_hostname": target.sni} if target.sni else None)
        try:
            return client.send(request, stream=True, follow_redirects=False)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            failure = exc
    assert failure is not None   # resolve_checked never returns an empty list
    raise failure


READ_CODINGS = ("gzip", "x-gzip", "deflate")


class _Inflater:
    """Decodes a gzip or deflate body piece by piece. ``feed`` returns at most ``room`` decoded bytes, so the decoded size
    is capped at the point of decoding and a decompression bomb is never expanded whole. A deflate body without the zlib
    header (some servers send raw deflate) is read as raw deflate."""

    def __init__(self, coding: str):
        self._gzip = coding in ("gzip", "x-gzip")
        self._raw_deflate_ok = not self._gzip
        self._z = zlib.decompressobj(16 + zlib.MAX_WBITS if self._gzip else zlib.MAX_WBITS)
        self._fed = False

    def feed(self, data: bytes, room: int) -> bytes:
        try:
            out = self._z.decompress(data, room)
        except zlib.error:
            if self._fed or not self._raw_deflate_ok:
                raise
            self._raw_deflate_ok = False
            self._z = zlib.decompressobj(-zlib.MAX_WBITS)
            out = self._z.decompress(data, room)
        self._fed = True
        return out


def _inflater_for(content_encoding: str) -> _Inflater | None:
    """None for an identity body. Raises ValueError for a coding this reader does not decode (br, zstd, stacked codings)."""
    codings = [c.strip().lower() for c in content_encoding.split(",") if c.strip() and c.strip() != "identity"]
    if not codings:
        return None
    if len(codings) == 1 and codings[0] in READ_CODINGS:
        return _Inflater(codings[0])
    raise ValueError(content_encoding)


def read_capped(response, limit: int, deadline: float | None = None) -> tuple[bytes, str]:
    """Read at most ``limit`` decoded bytes of a streamed body, stopping early once the deadline (a time.monotonic value)
    has passed. Returns (body, stop): stop is "" when the whole body was read, "size" when the byte cap cut it, "time"
    when the deadline did, and "encoding" when the body uses a Content-Encoding other than gzip or deflate, or a compressed
    body is damaged (the body is then empty). A gzip or deflate body is decoded while it is read, and each step produces
    no more than the room left under the cap."""
    if isinstance(response, httpx.Response) and not response.is_stream_consumed:
        try:
            inflater = _inflater_for(response.headers.get("content-encoding", ""))
        except ValueError:
            return b"", "encoding"
        pieces = response.iter_raw()
    else:   # a body httpx has already read (a mocked response, say) or a test double: httpx decodes it as before
        inflater = None
        pieces = response.iter_bytes()
    body = bytearray()
    for piece in pieces:
        if deadline is not None and time.monotonic() >= deadline:
            return bytes(body), "time"
        if inflater is not None:
            try:
                piece = inflater.feed(piece, limit - len(body) + 1)   # one more than the room: tells "exactly full" from "cut"
            except zlib.error:
                return b"", "encoding"
        room = limit - len(body)
        if len(piece) > room:
            body.extend(piece[:max(room, 0)])
            return bytes(body), "size"
        body.extend(piece)
    return bytes(body), ""


class HostThrottle:
    """Spaces requests to the same host at least ``min_interval`` seconds apart. Safe across threads: each caller
    reserves its own slot under the lock and then sleeps outside it."""

    def __init__(self, min_interval: float = 1.0, *, clock: Callable[[], float] | None = None,
                 sleep: Callable[[float], None] | None = None):
        # The defaults look up time at call time, so the deadlines a fetch passes in are on the same clock as the slots.
        self.min_interval = float(min_interval)
        self._clock = clock or (lambda: time.monotonic())
        self._sleep = sleep or (lambda seconds: time.sleep(seconds))
        self._next: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str, min_interval: float | None = None, deadline: float | None = None) -> float | None:
        """Reserve the host's next slot and sleep until it comes. ``min_interval`` (robots.txt Crawl-delay) lengthens the
        gap after this request when it is longer than the throttle's own interval. When the slot would come at or after
        ``deadline`` (a time.monotonic value on this throttle's clock), nothing is reserved or slept and None is returned,
        so a throttled request never outlasts its deadline."""
        gap = max(self.min_interval, float(min_interval or 0.0))
        with self._lock:
            now = self._clock()
            slot = max(now, self._next.get(host, now))
            if deadline is not None and slot >= deadline:
                return None
            self._next[host] = slot + gap
        delay = slot - now
        if delay > 0:
            self._sleep(delay)
        return max(delay, 0.0)


DEFAULT_THROTTLE = HostThrottle(1.0)


def origin_of(parts: SplitResult) -> str:
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}"


class _Entry(NamedTuple):
    """What robots.txt said for one origin. ``closed`` is the HTTP status of a 5xx answer: every URL on the origin is
    then refused."""
    parser: urllib.robotparser.RobotFileParser | None
    closed: int = 0


class Robots:
    """robots.txt per origin, fetched once and cached for the life of the object (the process, for DEFAULT_ROBOTS).

    Disallow is honoured. Three cases are decided here, and the choice is deliberate:
    - a network failure (connection refused, reset, timeout) is treated as allowed, as the brief asks;
    - any other status except 200 (404, 401, 403, 3xx) is treated as allowed, as the file does not exist for that host;
    - a 5xx status fails CLOSED: the origin is refused for the rest of the run, because a server error is not evidence
      that the site has no rules. The cache holds that answer like any other, so it lasts as long as the object does.
    A lookup or request cut short by the caller's deadline is not cached, so the next request asks again."""

    def __init__(self, *, client: httpx.Client | None = None, resolver: Resolver | None = None,
                 user_agent: str = USER_AGENT, timeout: float = 10.0):
        self._client = client
        self._resolver = resolver
        self._agent = user_agent
        self._timeout = timeout
        self._cache: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    def check(self, url: str, *, deadline: float | None = None, throttle: HostThrottle | None = None) -> str | None:
        """The reason robots.txt refuses ``url``, or None when it allows it. The robots.txt request takes a slot on
        ``throttle`` like any other request to that host. The lookup and request are bounded by ``deadline`` (a
        time.monotonic value); a lookup cut short by it is not cached, so it is asked again."""
        entry = self._entry(origin_of(urlsplit(url)), deadline, throttle)
        if entry is None:
            return None
        if entry.closed:
            return f"robots.txt answered HTTP {entry.closed}, so the host is not fetched in this run"
        if entry.parser is None:
            return None
        return None if entry.parser.can_fetch(self._agent, url) else "disallowed by robots.txt"

    def allowed(self, url: str, *, deadline: float | None = None, throttle: HostThrottle | None = None) -> bool:
        return self.check(url, deadline=deadline, throttle=throttle) is None

    def crawl_delay(self, url: str) -> float | None:
        """The Crawl-delay robots.txt gives this agent for the origin of ``url``, once that origin's answer is cached."""
        with self._lock:
            entry = self._cache.get(origin_of(urlsplit(url)))
        if entry is None or entry.parser is None:
            return None
        delay = entry.parser.crawl_delay(self._agent)
        return float(delay) if delay is not None else None

    def _entry(self, origin: str, deadline: float | None, throttle: HostThrottle | None) -> _Entry | None:
        """The cached answer for ``origin``; None when the answer is not known yet (the deadline cut it short)."""
        with self._lock:
            if origin in self._cache:
                return self._cache[origin]
        cacheable, entry = self._load(origin, deadline, throttle)
        if not cacheable:
            return None
        with self._lock:
            return self._cache.setdefault(origin, entry)

    def _load(self, origin: str, deadline: float | None, throttle: HostThrottle | None) -> tuple[bool, _Entry]:
        """(cacheable, entry). See the class docstring for each status."""
        robots_url = f"{origin}/robots.txt"
        owned = self._client is None
        client = self._client or httpx.Client(timeout=self._timeout, follow_redirects=False)
        if throttle is not None and throttle.wait(urlsplit(origin).netloc.lower(), deadline=deadline) is None:
            return False, _Entry(None)   # no slot before the deadline: the answer is not known, so it is not cached
        try:
            response = open_pinned(client, robots_url, resolver=self._resolver,
                                   headers={"User-Agent": self._agent, "Accept-Encoding": "gzip, deflate"},
                                   timeout=self._timeout, deadline=deadline)
            try:
                if response.status_code >= 500:
                    return True, _Entry(None, closed=response.status_code)
                if response.status_code != 200:
                    return True, _Entry(None)
                body, _ = read_capped(response, ROBOTS_MAX_BYTES, time.monotonic() + self._timeout)
            finally:
                response.close()
        except LookupTimeout:
            return False, _Entry(None)
        except Exception:
            if deadline is not None and time.monotonic() >= deadline:
                return False, _Entry(None)
            return True, _Entry(None)
        finally:
            if owned:
                client.close()
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(body.decode("utf-8", errors="replace").splitlines())
        return True, _Entry(parser)


DEFAULT_ROBOTS = Robots()
