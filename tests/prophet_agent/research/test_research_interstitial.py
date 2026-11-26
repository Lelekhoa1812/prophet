"""interstitial: each wall kind is recognised from a fixture, a normal article is not, and fetch_page refuses a wall and never
caches it. Runs offline with httpx.MockTransport."""
import httpx
import pytest

from prophet_agent.research import fetch as fetch_mod
from prophet_agent.research.fetch import fetch_page, js_shell_reasons
from prophet_agent.research.interstitial import KINDS, classify_page
from prophet_agent.research.netguard import HostThrottle

PUBLIC = "93.184.216.34"

ADBLOCK_HTML = """<!doctype html><html><head><title>Adblocker detected! Please consider reading this notice.</title>
</head><body><h1>Adblocker detected</h1><p>We have detected that you are using an ad blocker. Please disable it
to continue reading this site.</p><script src="/ads.js"></script></body></html>"""

CLOUDFLARE_HTML = """<!doctype html><html><head><title>Just a moment...</title></head><body>
<div id="cf-challenge"><h1>Checking your browser before accessing example.com</h1>
<p>Verify you are human by completing the action below.</p></div>
<script src="/cdn-cgi/challenge-platform/h/b/orchestrate/chl_page/v1"></script></body></html>"""

CAPTCHA_HTML = """<!doctype html><html><head><title>Security check</title></head><body>
<p>Please complete the captcha to continue.</p>
<div class="h-captcha" data-sitekey="test-site-key"></div>
<script src="https://js.hcaptcha.com/1/api.js" async defer></script></body></html>"""

CONSENT_HTML = """<!doctype html><html><head><title>Before you continue to the site</title>
<script src="https://cdn.cookiebot.com/uc.js"></script></head><body>
<p>We use cookies to personalise content. Accept all Reject all Manage preferences</p></body></html>"""

LOGIN_HTML = """<!doctype html><html><head><title>Sign in</title></head><body>
<h2>Sign in to continue reading this report</h2>
<form><input type="email" name="email"><input type="password" name="pw"><button>Sign in</button></form>
</body></html>"""

PAYWALL_HTML = """<!doctype html><html><head><title>Market outlook | Example Daily</title></head><body><article>
<p>Copper prices rose again this week as supply tightened in the major mining regions and demand held steady across
the manufacturing sector, analysts said in a note published on Monday morning.</p>
<p>Subscribe to continue reading. Already a subscriber? Log in.</p></article></body></html>"""


def _long_article(extra: str = "") -> str:
    paragraph = ("Brent crude is forecast to average 78.5 dollars per barrel in 2027, the agency said in its monthly "
                 "report, citing lower inventories and slower growth in refining across the region. ")
    body = "".join(f"<p>{paragraph}</p>" for _ in range(14))
    return f"""<!doctype html><html><head><title>Crude outlook 2026</title></head><body><nav>Menu</nav>
<article><h1>Crude outlook 2026</h1>{body}<p>{extra}</p></article><footer>Copyright footer</footer></body></html>"""


COOKIE_MENTION = _long_article("This site uses cookies to remember your settings. See our privacy page for details.")
RECAPTCHA_FORM = _long_article('<form class="g-recaptcha" data-sitekey="x">Newsletter sign-up, protected by reCAPTCHA.</form>')


def _text_of(html: str) -> str:
    from bs4 import BeautifulSoup
    return " ".join(BeautifulSoup(html, "lxml").get_text(" ").split())


@pytest.mark.parametrize("kind,title,html,status", [
    ("adblock_wall", "Adblocker detected! Please consider reading this notice.", ADBLOCK_HTML, 200),
    ("bot_check", "Just a moment...", CLOUDFLARE_HTML, 403),
    ("captcha", "Security check", CAPTCHA_HTML, 200),
    ("consent_wall", "Before you continue to the site", CONSENT_HTML, 200),
    ("login_wall", "Sign in", LOGIN_HTML, 200),
    ("paywall", "Market outlook | Example Daily", PAYWALL_HTML, 200),
])
def test_each_wall_kind_is_recognised_from_its_fixture(kind, title, html, status):
    verdict = classify_page(title, _text_of(html), status, html)
    assert verdict.kind == kind
    assert verdict.blocked is True
    assert verdict.evidence, "the verdict names what matched"
    assert kind in KINDS


def test_adblock_notice_is_recognised_from_title_alone():
    verdict = classify_page("Adblocker detected! Please consider reading this notice.", "", 200, "")
    assert verdict.kind == "adblock_wall"
    assert any(item.startswith("title:") for item in verdict.evidence)


def test_cloudflare_challenge_is_a_bot_check_even_without_html():
    verdict = classify_page("Just a moment...", "Checking your browser before accessing the site.", 403, "")
    assert verdict.kind == "bot_check"


def test_normal_article_that_mentions_cookies_once_is_not_a_wall():
    verdict = classify_page("Crude outlook 2026", _text_of(COOKIE_MENTION), 200, COOKIE_MENTION)
    assert verdict.kind == "none" and verdict.evidence == ()


def test_long_article_with_a_recaptcha_form_is_not_a_captcha_page():
    verdict = classify_page("Crude outlook 2026", _text_of(RECAPTCHA_FORM), 200, RECAPTCHA_FORM)
    assert verdict.kind == "none"


def test_plain_csv_text_is_not_a_wall():
    verdict = classify_page("", "date,close\n2026-01-02,140.2\n2026-01-05,141.0", 200, "")
    assert verdict.kind == "none"


def test_a_password_form_on_a_long_page_is_not_a_login_wall():
    long_page = _long_article('<form><input type="password" name="pw"></form><p>Log in to comment below.</p>')
    verdict = classify_page("Crude outlook 2026", _text_of(long_page), 200, long_page)
    assert verdict.kind == "none"


def test_label_names_the_kind_for_a_person():
    assert classify_page("Adblocker detected!", "", 200, "").label == "adblock wall"
    assert classify_page("", "Please complete the captcha to continue.", 200, "").label == "CAPTCHA"


def test_js_shell_is_detected_and_a_table_page_is_not():
    shell = '<html><body><div id="root"></div><script>a()</script><script>b()</script><script>c()</script></body></html>'
    assert js_shell_reasons(shell, "") != []
    table_page = ("<html><body><table><tr><td>Arsenal</td><td>70</td></tr></table>"
                  "<script>a()</script><script>b()</script><script>c()</script></body></html>")
    assert js_shell_reasons(table_page, "Arsenal 70") == []
    assert js_shell_reasons(_long_article(), _text_of(_long_article())) == []


# fetch_page: a wall is returned as not ok with the kind named, and is never stored as a good page.

@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))
    return tmp_path / "cache"


def _client(body: str, status: int = 200, counter=None):
    def handler(request):
        if counter is not None:
            counter.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(status, headers={"content-type": "text/html; charset=utf-8"}, content=body.encode())
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def _fetch(url, client, cache_dir):
    return fetch_page(url, client=client, resolver=lambda host, port: [PUBLIC], throttle=HostThrottle(0.0),
                      cache_dir=cache_dir, render=False)


def test_fetch_refuses_an_adblock_notice_and_names_the_kind(tmp_path):
    cache = tmp_path / "cache"
    page = _fetch("https://footballdatabase.example/league", _client(ADBLOCK_HTML), cache)
    assert page.ok is False
    assert "adblock_wall" in page.note and "adblock wall" in page.note


def test_fetch_never_stores_a_wall_as_a_good_page(tmp_path):
    cache = tmp_path / "cache"
    counter: list[str] = []
    client = _client(ADBLOCK_HTML, counter=counter)
    for _ in range(3):
        page = _fetch("https://footballdatabase.example/league", client, cache)
        assert page.ok is False
    assert len(counter) == 3, "each attempt is fetched again, because nothing was cached"
    pages_dir = cache / "pages"
    assert not pages_dir.exists() or not list(pages_dir.glob("*.json"))


def test_fetch_names_a_challenge_page_served_with_an_error_status(tmp_path):
    page = _fetch("https://example.com/data", _client(CLOUDFLARE_HTML, status=403), tmp_path / "cache")
    assert page.ok is False
    assert "bot_check" in page.note and page.status == 403


def test_fetch_refuses_a_login_wall_and_a_paywall(tmp_path):
    for html, kind in ((LOGIN_HTML, "login_wall"), (PAYWALL_HTML, "paywall")):
        page = _fetch("https://example.com/story", _client(html), tmp_path / kind)
        assert page.ok is False and kind in page.note


def test_fetch_reads_a_normal_article_that_mentions_cookies(tmp_path):
    page = _fetch("https://example.com/article", _client(COOKIE_MENTION), tmp_path / "cache")
    assert page.ok is True
    assert "Brent crude" in page.text


def test_a_cached_wall_from_an_older_version_is_not_served(tmp_path, monkeypatch):
    from prophet_agent.research.search import cache_key
    cache = tmp_path / "cache"
    path = cache / "pages" / f"{cache_key('page', 'https://example.com/old')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    path.write_text(json.dumps({"url": "https://example.com/old", "final_url": "https://example.com/old",
                                "title": "Adblocker detected! Please consider reading this notice.",
                                "text": "We have detected that you are using an ad blocker.", "ok": True,
                                "media": "text/html", "file_links": [], "tables": [], "status": 200}))
    counter: list[str] = []
    page = _fetch("https://example.com/old", _client(_long_article(), counter=counter), cache)
    assert counter, "the stored wall is not used; the page is fetched again"
    assert page.ok is True and "Brent crude" in page.text


def test_fetch_module_still_exposes_the_page_fields():
    assert fetch_mod.Page(url="u", final_url="u").ok is False
