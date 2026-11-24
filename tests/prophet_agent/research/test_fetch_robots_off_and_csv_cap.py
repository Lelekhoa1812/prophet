"""Owner decisions of 2026-11-24, recorded in research/fetch.py:
- robots.txt is not honoured (HONOUR_ROBOTS is False): a path that robots.txt disallows is fetched, and robots.txt is not asked.
- a CSV body is kept up to CSV_MAX_TEXT_CHARS (180,000) characters; other plain text keeps the 60,000 cap.
No network: every request goes through httpx.MockTransport.
"""
import httpx

from prophet_agent.research import fetch as fetch_mod
from prophet_agent.research.fetch import CSV_MAX_TEXT_CHARS, MAX_TEXT_CHARS, fetch_page
from prophet_agent.research.netguard import HostThrottle, Robots

PUBLIC = "93.184.216.34"
ARTICLE = ("<html><head><title>Private note</title></head><body><article><p>"
           + "Freight rates rose again this quarter. " * 20 + "</p></article></body></html>")


def public_resolver(host, port):
    return [PUBLIC]


def _client(calls, body, content_type):
    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        return httpx.Response(200, headers={"content-type": content_type}, content=body)
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def test_robots_is_off_by_default():
    assert fetch_mod.HONOUR_ROBOTS is False


def test_a_path_robots_disallows_is_fetched_and_robots_txt_is_not_asked(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))
    calls = []
    client = _client(calls, ARTICLE.encode(), "text/html")
    page = fetch_page("https://news.example.com/private/x", client=client, resolver=public_resolver,
                      robots=Robots(client=client, resolver=public_resolver), throttle=HostThrottle(0.0),
                      use_cache=False)
    assert page.ok is True, page.note
    assert "/robots.txt" not in calls
    assert "Freight rates" in page.text


def test_a_csv_body_is_kept_up_to_the_csv_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))
    row = "2026-11-23,6850.40\n"
    body = ("DATE,SP500\n" + row * ((CSV_MAX_TEXT_CHARS + 5_000) // len(row))).encode()
    calls = []
    client = _client(calls, body, "text/csv")
    page = fetch_page("https://data.example.com/fredgraph.csv", client=client, resolver=public_resolver,
                      throttle=HostThrottle(0.0), use_cache=False)
    assert page.ok is True, page.note
    assert len(page.text) == CSV_MAX_TEXT_CHARS
    assert f"clipped to {CSV_MAX_TEXT_CHARS} characters" in page.note


def test_plain_text_is_still_capped_at_the_general_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))
    body = ("Copper held steady in the session. " * ((CSV_MAX_TEXT_CHARS + 5_000) // 34)).encode()
    calls = []
    client = _client(calls, body, "text/plain")
    page = fetch_page("https://data.example.com/notes.txt", client=client, resolver=public_resolver,
                      throttle=HostThrottle(0.0), use_cache=False)
    assert page.ok is True, page.note
    assert len(page.text) == MAX_TEXT_CHARS
    assert f"clipped to {MAX_TEXT_CHARS} characters" in page.note
