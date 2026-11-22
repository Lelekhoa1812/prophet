"""Address, scheme and redirect checks for every outbound research request, plus a polite per-host pacing and robots.txt.

Motivation vs Logic
Motivation: a research tool that follows whatever link a search result or page contains can be pointed at the company's
own network (a router admin page, a cloud metadata service that hands out credentials) or at a file on disk. Search
results and redirects are data from the internet, so they must never decide where the agent connects.
Logic: only http and https on the default ports are accepted, with no credentials in the URL. The host must resolve,
and every address it resolves to must be globally routable and not a metadata address; a single bad address rejects the
host. Redirects are followed by hand so each hop is checked again (the caller does that loop). Robots.txt is fetched once
per origin and honoured; a network failure there fails open, as the brief asks. A per-host lock spaces requests at least
one second apart. The address check happens at request time and does not pin the connection to the address, so a DNS
rebinding attack that changes the answer between the check and the connect is not fully closed.
"""
from __future__ import annotations

import ipaddress
import socket
import threading
import time
import urllib.robotparser
from collections.abc import Callable
from urllib.parse import SplitResult, urlsplit

import httpx

USER_AGENT = "ProphetResearch/1.0"
ALLOWED_SCHEMES = ("http", "https")
ALLOWED_PORTS = (None, 80, 443)
MAX_URL_CHARS = 2048
METADATA_ADDRESSES = {"169.254.169.254", "fd00:ec2::254", "100.100.100.200"}
BLOCKED_HOSTNAMES = {"localhost", "metadata", "metadata.google.internal", "metadata.azure.internal"}
BLOCKED_SUFFIXES = (".localhost", ".local", ".internal", ".home.arpa", ".lan")
ROBOTS_MAX_BYTES = 500_000

Resolver = Callable[[str, int], list[str]]


class UnsafeURL(ValueError):
    """Raised when a URL or the address it resolves to is not allowed. The message is safe to show the planner."""


def _default_resolver(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError) as exc:
        raise UnsafeURL(f"host {host!r} did not resolve ({type(exc).__name__})") from None
    return [info[4][0] for info in infos]


def blocked_reason(address: str) -> str | None:
    """Why an address may not be contacted, or None when it is a public, globally routable address."""
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return "malformed"
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
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


def check_url(url: str, *, resolver: Resolver | None = None) -> SplitResult:
    """Return the parsed URL if it may be fetched; raise UnsafeURL otherwise. ``resolver(host, port) -> [ip, ...]`` is
    injectable so tests never touch the network."""
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
    host = (parts.hostname or "").rstrip(".")
    if not host:
        raise UnsafeURL("the URL has no host")
    try:
        port = parts.port
    except ValueError:
        raise UnsafeURL("the URL has an invalid port") from None
    if port not in ALLOWED_PORTS:
        raise UnsafeURL(f"port {port} is not allowed; use the default ports 80 or 443")
    if host in BLOCKED_HOSTNAMES or host.endswith(BLOCKED_SUFFIXES):
        raise UnsafeURL(f"host {host!r} is an internal name")
    try:
        ipaddress.ip_address(host)
        literal = True
    except ValueError:
        literal = False
    if literal:
        addresses = [host]
    else:
        addresses = (resolver or _default_resolver)(host, port or (443 if parts.scheme == "https" else 80))
    if not addresses:
        raise UnsafeURL(f"host {host!r} has no addresses")
    for address in addresses:
        reason = blocked_reason(address)
        if reason:
            raise UnsafeURL(f"host {host!r} resolves to a {reason} address, which is not fetched")
    return parts


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
        try:
            check_url(robots_url, resolver=self._resolver)
            if self._client is not None:
                response = self._client.get(robots_url, timeout=self._timeout)
            else:
                with httpx.Client(timeout=self._timeout, follow_redirects=False,
                                  headers={"User-Agent": self._agent}) as client:
                    response = client.get(robots_url)
        except Exception:
            return None
        if response.status_code != 200:
            return None
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(response.text[:ROBOTS_MAX_BYTES].splitlines())
        return parser


DEFAULT_ROBOTS = Robots()
