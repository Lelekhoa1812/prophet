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


def resolve_checked(url: str, *, resolver: Resolver | None = None) -> tuple[SplitResult, list[str]]:
    """Check the URL and resolve its host once. Returns the parsed URL and the checked addresses, in the order given;
    raises UnsafeURL when any address is not public. ``resolver(host, port) -> [ip, ...]`` is injectable so tests never
    touch the network."""
    parts = check_syntax(url)
    host, port, literal = _host_and_port(parts)
    if literal:
        addresses = [host]
    else:
        addresses = (resolver or _default_resolver)(host, port or (443 if parts.scheme.lower() == "https" else 80))
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
                timeout: float) -> httpx.Response:
    """Send one GET whose connection goes to a checked address. The name is resolved once, every address is checked,
    and each checked address is tried in order until one connects. Redirects are never followed here. The caller must
    close the returned response. Raises UnsafeURL for a refused address, or the last connect error."""
    parts, addresses = resolve_checked(url, resolver=resolver)
    failure: Exception | None = None
    for address in addresses:
        target = pinned(parts, address)
        request = client.build_request("GET", target.url, headers={**(headers or {}), "Host": target.host},
                                       timeout=timeout,
                                       extensions={"sni_hostname": target.sni} if target.sni else None)
        try:
            return client.send(request, stream=True, follow_redirects=False)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            failure = exc
    assert failure is not None   # resolve_checked never returns an empty list
    raise failure


class HostThrottle:
    """Spaces requests to the same host at least ``min_interval`` seconds apart. Safe across threads: each caller
    reserves its own slot under the lock and then sleeps outside it."""

    def __init__(self, min_interval: float = 1.0, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.min_interval = float(min_interval)
        self._clock = clock
        self._sleep = sleep
        self._next: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> float:
        with self._lock:
            now = self._clock()
            slot = max(now, self._next.get(host, now))
            self._next[host] = slot + self.min_interval
        delay = slot - now
        if delay > 0:
            self._sleep(delay)
        return max(delay, 0.0)


DEFAULT_THROTTLE = HostThrottle(1.0)


def origin_of(parts: SplitResult) -> str:
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}"


class Robots:
    """robots.txt per origin, fetched once and cached for the process. Disallow is honoured; if robots.txt cannot be
    fetched (network error, 5xx, redirect) the origin is treated as allowed."""

    def __init__(self, *, client: httpx.Client | None = None, resolver: Resolver | None = None,
                 user_agent: str = USER_AGENT, timeout: float = 10.0):
        self._client = client
        self._resolver = resolver
        self._agent = user_agent
        self._timeout = timeout
        self._cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._lock = threading.Lock()

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = origin_of(parts)
        with self._lock:
            known = origin in self._cache
        if not known:
            parser = self._load(origin)
            with self._lock:
                self._cache.setdefault(origin, parser)
        parser = self._cache[origin]
        if parser is None:
            return True
        return parser.can_fetch(self._agent, url)

    def _load(self, origin: str) -> urllib.robotparser.RobotFileParser | None:
        robots_url = f"{origin}/robots.txt"
        owned = self._client is None
        client = self._client or httpx.Client(timeout=self._timeout, follow_redirects=False)
        try:
            response = open_pinned(client, robots_url, resolver=self._resolver, headers={"User-Agent": self._agent},
                                   timeout=self._timeout)
            try:
                if response.status_code != 200:
                    return None
                response.read()
                text = response.text[:ROBOTS_MAX_BYTES]
            finally:
                response.close()
        except Exception:
            return None
        finally:
            if owned:
                client.close()
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(text.splitlines())
        return parser


DEFAULT_ROBOTS = Robots()
