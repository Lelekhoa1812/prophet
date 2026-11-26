"""Render one page in headless Chromium, so that a page whose text is built by JavaScript can be read.

Motivation vs Logic
Motivation: many data pages (league tables, price widgets, dashboards) ship an empty shell and fill it with a script. A
plain GET reads the shell and finds no table. A headless browser sees the page the way a reader does, so the same
extraction can then run on the finished HTML.
Logic: a browser is the most capable client the agent has, so every request it makes is checked before it leaves. The
page's document, scripts, XHR and frames all pass through one route handler. Each request is checked with the netguard
functions (scheme, credentials, ports, internal names, and every resolved address must be public), and the connection
is made by the handler itself to a checked address, so the browser never resolves or connects by itself, and a DNS
answer that changes later cannot move it. Images, fonts and media are not loaded. Total time and total bytes are capped.
WebSockets are closed. Service workers are off. A fresh context is used for each page, so nothing is kept between
fetches. A normal desktop User-Agent is sent, and no fingerprint, stealth or evasion option is set. A page that shows a
bot check or CAPTCHA is returned as it is, and fetch_page refuses it as an interstitial; it is never worked around.
The address check is a parameter so that tests can allow a local test server. The production default is netguard's own
check, and tests never change that default.
"""
from __future__ import annotations

import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import SplitResult, urlsplit

import httpx

from prophet_agent.research.netguard import (LookupTimeout, UnsafeURL, check_syntax, pinned, read_capped,
                                             resolve_checked)

RENDER_SECONDS = 25.0          # the whole render: launch, every request and the settle wait
RENDER_MAX_BYTES = 8_000_000   # decoded bytes across every response the page receives
SETTLE_SECONDS = 6.0           # the most time to wait for the network to go quiet after the page loads
BLOCKED_TYPES = ("image", "font", "media")     # not needed to read a table; never requested
DROP_REQUEST_HEADERS = {"host", "content-length", "connection", "accept-encoding", "keep-alive", "proxy-connection",
                        "transfer-encoding", "upgrade"}
DROP_RESPONSE_HEADERS = {"content-encoding", "content-length", "transfer-encoding", "connection", "keep-alive"}

AddressCheck = Callable[[str, float | None], tuple[SplitResult, list[str]]]


@dataclass
class Rendered:
    """ok is True when a page was drawn and its HTML is in ``html``. ``blocked`` lists (url, reason) for every request
    the render refused. ``note`` says why a render did not happen."""
    ok: bool
    final_url: str = ""
    html: str = ""
    title: str = ""
    status: int | None = None
    seconds: float = 0.0
    bytes: int = 0
    blocked: list = field(default_factory=list)
    note: str = ""


@lru_cache(maxsize=1)
def chromium_status() -> tuple[bool, str]:
    """(available, reason). Available means the Playwright package imports and its Chromium build is on disk.
    The answer is cached for the process; it does not start a browser."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False, "Playwright is not installed (install the browser extra, then run playwright install chromium)"
    try:
        with sync_playwright() as playwright:
            executable = Path(playwright.chromium.executable_path)
    except Exception as exc:   # the driver could not start: report it, never raise into a fetch
        return False, f"Playwright could not start ({type(exc).__name__})"
    if not executable.exists():
        return False, "Chromium is not installed (run python -m playwright install chromium)"
    return True, ""


def _desktop_user_agent(version: str) -> str:
    """A normal desktop Chrome User-Agent for the platform the agent runs on. It states the engine version, and
    nothing else is changed."""
    system = platform.system()
    if system == "Darwin":
        os_part = "Macintosh; Intel Mac OS X 10_15_7"
    elif system == "Windows":
        os_part = "Windows NT 10.0; Win64; x64"
    else:
        os_part = "X11; Linux x86_64"
    return f"Mozilla/5.0 ({os_part}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36"


class _Budget:
    """Time and byte budget for one render, plus the list of refused requests."""

    def __init__(self, deadline: float, max_bytes: int):
        self.deadline = deadline
        self.max_bytes = max_bytes
        self.bytes = 0
        self.blocked: list[tuple[str, str]] = []
        self.truncated = 0

    def spent(self) -> bool:
        return time.monotonic() >= self.deadline or self.bytes >= self.max_bytes

    def remaining_seconds(self) -> float:
        return max(self.deadline - time.monotonic(), 0.0)


def _default_address_check(url: str, deadline: float | None) -> tuple[SplitResult, list[str]]:
    return resolve_checked(url, deadline=deadline)


def _refuse(route, state: _Budget, url: str, reason: str) -> None:
    state.blocked.append((url, reason))
    route.abort("blockedbyclient")


def _serve(route, request, state: _Budget, http: httpx.Client, check: AddressCheck) -> None:
    """Answer one browser request: refuse it, or fetch it over a checked address and hand the body back.

    Motivation vs Logic
    Motivation: the browser would otherwise resolve and connect by itself, and a page could reach a router, a metadata
    service or a file on disk through a script it runs.
    Logic: the request is checked first (scheme, budget, then the address check). The body is fetched by this process
    over a checked address with the original name as Host header, as netguard.open_pinned does for fetch_page, and handed
    back to the browser. A refused request is aborted and recorded in state.blocked with its reason. Redirects are not
    followed here: a 3xx is returned to the browser, which asks again, and that request is checked again.
    """
    url = request.url
    scheme = urlsplit(url).scheme.lower()
    if scheme == "data":   # inline data needs no network
        route.continue_()
        return
    if scheme not in ("http", "https"):
        _refuse(route, state, url, "scheme is not http or https")
        return
    if request.resource_type in BLOCKED_TYPES:
        route.abort("blockedbyclient")   # not needed for a table; not counted as a refusal
        return
    if state.spent():
        _refuse(route, state, url, "the render's time or byte budget is used up")
        return
    try:
        check_syntax(url)
        parts, addresses = check(url, state.deadline)
    except UnsafeURL as exc:
        _refuse(route, state, url, str(exc))
        return
    except LookupTimeout:
        _refuse(route, state, url, "the DNS lookup did not finish before the deadline")
        return
    headers = {key: value for key, value in request.headers.items() if key.lower() not in DROP_REQUEST_HEADERS}
    body = request.post_data_buffer
    response = None
    for address in addresses:   # the connection goes to a checked address, as netguard.open_pinned does
        target = pinned(parts, address)
        outgoing = http.build_request(
            request.method, target.url, headers={**headers, "Host": target.host, "Connection": "close"},
            content=body, timeout=max(state.remaining_seconds(), 0.001),
            extensions={"sni_hostname": target.sni} if target.sni else None)
        try:
            response = http.send(outgoing, stream=True, follow_redirects=False)
            break
        except (httpx.ConnectError, httpx.ConnectTimeout):
            continue
    if response is None:
        _refuse(route, state, url, "no checked address accepted the connection")
        return
    try:
        data, stopped = read_capped(response, max(state.max_bytes - state.bytes, 1), state.deadline)
        status = response.status_code
        out_headers = {key: value for key, value in response.headers.multi_items()
                       if key.lower() not in DROP_RESPONSE_HEADERS}
    finally:
        response.close()
    if stopped == "encoding":
        state.blocked.append((url, "the response uses an encoding this reader does not decode"))
        route.abort("failed")
        return
    if stopped == "size" or stopped == "time":
        state.truncated += 1
    state.bytes += len(data)
    route.fulfill(status=status, headers=out_headers, body=data)


def _route(route, request, state: _Budget, http: httpx.Client, check: AddressCheck) -> None:
    try:
        _serve(route, request, state, http, check)
    except Exception as exc:   # a handler fault must not leave the browser waiting on a request
        state.blocked.append((request.url, f"request failed: {type(exc).__name__}"))
        try:
            route.abort("failed")
        except Exception:
            pass


def _settle(page, state: _Budget) -> None:
    """Give scripts a short time to build the page after the load event. The wait is capped by the render budget."""
    wait = min(SETTLE_SECONDS, state.remaining_seconds())
    if wait <= 0:
        return
    try:
        page.wait_for_load_state("networkidle", timeout=wait * 1000)
    except Exception:   # a page that never goes quiet is read as it stands
        pass


def render_page(url: str, *, address_check: AddressCheck | None = None, total_seconds: float = RENDER_SECONDS,
                max_bytes: int = RENDER_MAX_BYTES) -> Rendered:
    """Draw ``url`` in headless Chromium and return the final HTML. Never raises.

    Motivation vs Logic
    Motivation: fetch_page needs the HTML a reader would see, with the time and size caps a research tool must keep.
    Logic: the URL is syntax-checked first. Then one browser context is started with a route handler that checks and
    serves every request (see the module docstring). The page is read after the load event and a short settle wait.
    Any failure returns Rendered(ok=False) with a note, so the caller falls back to the plain fetch.
    ``address_check(url, deadline) -> (parts, addresses)`` defaults to netguard's check and is used only by tests.
    """
    started = time.monotonic()
    try:
        check_syntax(url)
    except UnsafeURL as exc:
        return Rendered(ok=False, note=f"refused: {exc}")
    available, reason = chromium_status()
    if not available:
        return Rendered(ok=False, note=f"render skipped: {reason}")
    from playwright.sync_api import sync_playwright

    check = address_check or _default_address_check
    state = _Budget(started + total_seconds, max_bytes)
    http = httpx.Client(follow_redirects=False, timeout=min(total_seconds, 15.0))
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                user_agent = _desktop_user_agent(browser.version)
                context = browser.new_context(user_agent=user_agent, service_workers="block", accept_downloads=False,
                                              viewport={"width": 1280, "height": 900})
                context.route("**/*", lambda route, request: _route(route, request, state, http, check))
                context.route_web_socket("**/*", lambda socket: socket.close())
                page = context.new_page()
                response = page.goto(url, wait_until="domcontentloaded",
                                     timeout=max(state.remaining_seconds(), 0.001) * 1000)
                _settle(page, state)
                html = page.content()
                title = " ".join(page.title().split())[:300]
                final_url = page.url
                status = response.status if response is not None else None
                context.close()
            finally:
                browser.close()
    except Exception as exc:   # a navigation refused by the route, a timeout or a crash: a failed render, not a raise
        return Rendered(ok=False, note=f"render failed: {type(exc).__name__}", blocked=state.blocked,
                        bytes=state.bytes, seconds=round(time.monotonic() - started, 2))
    finally:
        http.close()
    note = f"{state.truncated} response(s) cut at the render budget" if state.truncated else ""
    return Rendered(ok=bool(html.strip()), final_url=final_url, html=html, title=title, status=status,
                    seconds=round(time.monotonic() - started, 2), bytes=state.bytes, blocked=state.blocked, note=note)
