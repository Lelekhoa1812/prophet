"""search: backend cascade, diagnostics, cache TTL, URL normalising and validation. ddgs is replaced; no network."""
import json
import time

import pytest

import ddgs
from ddgs.exceptions import DDGSException
from prophet_agent.research import search
from prophet_agent.research.search import normalise_url, web_search


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("PROPHET_SEARCH_BACKENDS", raising=False)
    search.reset_backend_health()   # cooldowns are per process; one test must not cool a backend for the next
    return tmp_path / "cache"


class FakeDDGS:
    """Stands in for ddgs.DDGS. ``plan`` maps a backend name to a list of rows, or to an exception to raise."""
    calls = []
    plan = {}

    def __init__(self, *args, **kwargs):
        pass

    def _answer(self, kind, query, **kwargs):
        backend = kwargs["backend"]
        FakeDDGS.calls.append((kind, backend))
        outcome = FakeDDGS.plan.get(backend, [])
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def text(self, query, **kwargs):
        return self._answer("text", query, **kwargs)

    def news(self, query, **kwargs):
        return self._answer("news", query, **kwargs)


@pytest.fixture
def fake_ddgs(monkeypatch):
    FakeDDGS.calls = []
    FakeDDGS.plan = {}
    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    return FakeDDGS


ROWS = [
    {"title": "Oil outlook", "href": "https://www.eia.gov/outlook?utm_source=x", "body": "Brent to average 80."},
    {"title": "Oil news", "href": "https://www.reuters.com/markets/oil", "body": "Prices rose."},
]


def test_cascade_returns_first_non_empty_backend(fake_ddgs):
    fake_ddgs.plan = {"auto": [], "bing": RuntimeError("blocked"), "google": ROWS}
    result = web_search("WTI crude oil price outlook", backoff_s=0, use_cache=False)
    assert result.backend == "google"
    assert [h.url for h in result] == ["https://www.eia.gov/outlook", "https://www.reuters.com/markets/oil"]
    assert [h.rank for h in result] == [1, 2]
    assert any("auto: no results" in d for d in result.diagnostics)
    assert any("bing: RuntimeError" in d for d in result.diagnostics)
    assert [b for _, b in fake_ddgs.calls] == ["auto", "bing", "google"]


def test_cascade_survives_ddgs_exceptions(fake_ddgs):
    fake_ddgs.plan = {"auto": DDGSException("no backend"), "bing": TimeoutError("slow"), "google": ROWS}
    result = web_search("q", backoff_s=0, use_cache=False)
    assert result.backend == "google" and len(result) == 2


def test_all_backends_failing_returns_empty_with_diagnostics(fake_ddgs):
    fake_ddgs.plan = {name: DDGSException("nope") for name in search.DEFAULT_BACKENDS}
    result = web_search("q", backoff_s=0, use_cache=False)
    assert result == [] and result.backend is None
    assert len(result.diagnostics) == len(search.DEFAULT_BACKENDS) + 1
    assert "every backend failed" in result.diagnostics[-1]


def test_backend_list_can_come_from_environment(fake_ddgs, monkeypatch):
    monkeypatch.setenv("PROPHET_SEARCH_BACKENDS", "yahoo, mojeek")
    fake_ddgs.plan = {"yahoo": [], "mojeek": ROWS}
    result = web_search("q", backoff_s=0, use_cache=False)
    assert result.backend == "mojeek"
    assert [b for _, b in fake_ddgs.calls] == ["yahoo", "mojeek"]


def test_cache_hit_avoids_a_second_live_call(fake_ddgs):
    fake_ddgs.plan = {"bing": ROWS}
    first = web_search("oil", backends=["bing"], backoff_s=0)
    calls_after_first = len(fake_ddgs.calls)
    second = web_search("oil", backends=["bing"], backoff_s=0)
    assert first.cached is False and second.cached is True
    assert len(fake_ddgs.calls) == calls_after_first
    assert [h.url for h in second] == [h.url for h in first]


def test_empty_answers_are_not_cached(fake_ddgs):
    fake_ddgs.plan = {"bing": []}
    web_search("nothing here", backends=["bing"], backoff_s=0)
    fake_ddgs.plan = {"bing": ROWS}
    result = web_search("nothing here", backends=["bing"], backoff_s=0)
    assert result.cached is False and len(result) == 2


def test_news_cache_expires_after_six_hours_but_text_lasts_a_day(fake_ddgs, isolated_cache):
    fake_ddgs.plan = {"bing": ROWS}
    web_search("crude", kind="news", backends=["bing"], backoff_s=0)
    web_search("crude", kind="text", backends=["bing"], backoff_s=0)
    seven_hours_ago = time.time() - 7 * 3600
    for path in (isolated_cache / "search").glob("*.json"):
        data = json.loads(path.read_text())
        data["saved_at"] = seven_hours_ago
        path.write_text(json.dumps(data))
    news = web_search("crude", kind="news", backends=["bing"], backoff_s=0)
    text = web_search("crude", kind="text", backends=["bing"], backoff_s=0)
    assert news.cached is False
    assert text.cached is True


def test_news_hits_keep_date_and_source(fake_ddgs):
    fake_ddgs.plan = {"bing": [{"title": "Cut", "url": "https://www.bbc.co.uk/news/x", "body": "b",
                                "date": "2026-11-20T10:00:00+00:00", "source": "BBC"}]}
    result = web_search("rate cut", kind="news", backends=["bing"], backoff_s=0, use_cache=False)
    assert result[0].date == "2026-11-20T10:00:00+00:00"
    assert result[0].source == "BBC"


def test_normalise_url_drops_tracking_and_fragment():
    assert normalise_url("HTTPS://Example.com/a/?utm_source=x&b=2&utm_medium=y#frag") == "https://example.com/a?b=2"
    assert normalise_url("https://example.com/a?gclid=abc") == "https://example.com/a"


@pytest.mark.parametrize("kwargs", [
    {"query": "  "}, {"query": "x" * 301}, {"query": "ok", "kind": "video"}, {"query": "ok", "max_results": 0},
    {"query": "ok", "max_results": True}, {"query": "ok", "timelimit": "x"},
])
def test_invalid_arguments_are_refused(kwargs):
    with pytest.raises(ValueError):
        web_search(**kwargs)
