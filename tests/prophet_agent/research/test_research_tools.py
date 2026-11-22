"""research tools: argument validation, the three tools end to end with search and fetch mocked, and the observation cap."""
import json

import pytest

from prophet_agent.research.fetch import Page
from prophet_agent.research.search import Hit, SearchResult
from prophet_agent.tools import research_tools as rt

PAGE_TEXT = ("Brent crude is forecast to average 78.5 dollars per barrel in 2027.\n\n"
             "Inventories rose by 3.2 million barrels in the latest week.\n\n"
             "Unrelated remarks about shipping insurance and labour contracts.")


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))


def tools_by_name():
    return {tool.name: (tool, spec) for tool, spec in rt.tools()}


def hits_for(query):
    return SearchResult([
        Hit(title="EIA outlook", url="https://www.eia.gov/outlook", snippet="s", rank=1, backend="bing"),
        Hit(title="Reuters oil", url="https://www.reuters.com/oil?utm_campaign=x", snippet="s", rank=2,
            backend="bing"),
        Hit(title="Blog", url="https://blog.blogspot.com/best-oil", snippet="s", rank=3, backend="bing"),
    ], backend="bing")


def fake_fetch(url, **kwargs):
    if "reuters" in url:
        return Page(url=url, final_url=url, title="Reuters oil", text=PAGE_TEXT,
                    sha256="abc123", fetched_at="now", status=200, ok=True)
    if "blogspot" in url:
        return Page(url=url, final_url=url, note="HTTP 503", ok=False, status=503)
    return Page(url=url, final_url=url, title="EIA", date="2026-11-01", text=PAGE_TEXT,
                sha256="def456", fetched_at="now", status=200, ok=True)


def test_three_tools_are_declared_idempotent_with_specs():
    tools = tools_by_name()
    assert set(tools) == {"web_search", "web_fetch", "web_research"}
    for tool, spec in tools.values():
        assert tool.idempotent is True
        assert spec["name"] == tool.name and spec["title"] and spec["activity"] and spec["description"]
        assert spec["args"]


@pytest.mark.parametrize("name,args,allowed", [
    ("web_search", {"query": "oil", "depth": 3}, "allowed"),
    ("web_fetch", {"url": "https://example.com", "passages": 2}, "allowed"),
    ("web_research", {"queries": ["a"], "top": 2}, "allowed"),
])
def test_unknown_arguments_are_refused_and_named(name, args, allowed):
    tool, _ = tools_by_name()[name]
    with pytest.raises(ValueError, match="unknown arguments.*allowed"):
        tool.fn(args, "key")


@pytest.mark.parametrize("name,args", [
    ("web_search", {"query": ""}),
    ("web_search", {"query": "oil", "max_results": 11}),
    ("web_search", {"query": "oil", "kind": "video"}),
    ("web_search", {"query": "oil", "recency": "decade"}),
    ("web_fetch", {"url": ""}),
    ("web_fetch", {"url": "https://example.com", "queries": ["a"] * 9}),
    ("web_fetch", {"url": "https://example.com", "max_chars": 100}),
    ("web_research", {"queries": []}),
    ("web_research", {"queries": ["x"] * 9}),
    ("web_research", {"queries": ["ok"], "fetch_top": 9}),
    ("web_research", {"queries": ["ok"], "passages_per_page": 0}),
    ("web_research", {"queries": ["ok"], "kind": "video"}),
])
def test_bad_values_are_refused(name, args):
    tool, _ = tools_by_name()[name]
    with pytest.raises(ValueError):
        tool.fn(args, "key")


def test_web_search_tool_returns_ranked_hits_with_tiers(monkeypatch):
    monkeypatch.setattr(rt, "web_search", lambda q, **kw: hits_for(q))
    tool, _ = tools_by_name()["web_search"]
    out = tool.fn({"query": "oil outlook"}, "key")
    assert out["backend"] == "bing"
    assert [h["tier"] for h in out["hits"]] == [1, 2, 4]
    assert "diagnostics" not in out


def test_web_search_tool_reports_diagnostics_when_nothing_came_back(monkeypatch):
    empty = SearchResult([], backend=None, diagnostics=["auto: DDGSException: blocked"])
    monkeypatch.setattr(rt, "web_search", lambda q, **kw: empty)
    tool, _ = tools_by_name()["web_search"]
    out = tool.fn({"query": "oil"}, "key")
    assert out["hits"] == [] and out["diagnostics"] == ["auto: DDGSException: blocked"]


def test_web_fetch_returns_passages_for_the_queries(monkeypatch):
    monkeypatch.setattr(rt, "fetch_page", fake_fetch)
    tool, _ = tools_by_name()["web_fetch"]
    out = tool.fn({"url": "https://www.eia.gov/outlook", "queries": ["Brent crude 2027 forecast"]}, "key")
    assert out["ok"] and out["sha256"] == "def456" and out["tier"] == 1
    assert out["passages"][0]["text"].startswith("Brent crude")
    assert "text" not in out


def test_web_fetch_without_queries_returns_the_first_max_chars(monkeypatch):
    long_text = "word " * 400
    monkeypatch.setattr(rt, "fetch_page", lambda url, **kw: Page(url=url, final_url=url, text=long_text, ok=True))
    tool, _ = tools_by_name()["web_fetch"]
    out = tool.fn({"url": "https://www.eia.gov/outlook", "max_chars": 500}, "key")
    assert out["text"] == long_text[:500] and out["text_truncated"] is True
    full = tool.fn({"url": "https://www.eia.gov/outlook"}, "key")
    assert full["text"] == long_text and full["text_truncated"] is False   # 2,000 characters is under the 6,000 default


def test_web_fetch_failure_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(rt, "fetch_page", fake_fetch)
    tool, _ = tools_by_name()["web_fetch"]
    out = tool.fn({"url": "https://blog.blogspot.com/best-oil"}, "key")
    assert out["ok"] is False and "HTTP 503" in out["note"]


def test_web_research_end_to_end_with_everything_mocked(monkeypatch):
    seen_queries = []

    def fake_search(query, **kwargs):
        seen_queries.append(query)
        return hits_for(query)

    monkeypatch.setattr(rt, "web_search", fake_search)
    monkeypatch.setattr(rt, "fetch_page", fake_fetch)
    tool, _ = tools_by_name()["web_research"]
    out = tool.fn({"queries": ["WTI crude outlook", "Brent 2027 forecast"], "fetch_top": 3,
                   "passages_per_page": 2}, "key")
    assert sorted(seen_queries) == ["Brent 2027 forecast", "WTI crude outlook"]
    assert out["searched"] >= 3
    assert [s["id"] for s in out["sources"]] == ["S1", "S2"]
    eia = out["sources"][0]
    assert eia["url"] == "https://www.eia.gov/outlook" and eia["sha256"] == "def456"
    assert eia["tier"] == 1 and eia["date"] == "2026-11-01"
    assert all(p["text"] and p["score"] > 0 for p in eia["passages"])
    assert len(eia["passages"]) <= 2
    assert out["failed"] == [{"url": "https://blog.blogspot.com/best-oil", "why": "HTTP 503"}]


def test_web_research_records_failed_fetches_separately(monkeypatch):
    monkeypatch.setattr(rt, "web_search", lambda q, **kw: SearchResult([
        Hit(title="Blog", url="https://blog.blogspot.com/a", rank=1, backend="bing")], backend="bing"))
    monkeypatch.setattr(rt, "fetch_page", lambda url, **kw: Page(url=url, final_url=url, note="timed out", ok=False))
    tool, _ = tools_by_name()["web_research"]
    out = tool.fn({"queries": ["oil"]}, "key")
    assert out["sources"] == [] and out["failed"][0]["why"] == "timed out"


def test_web_research_passes_search_diagnostics_when_every_query_is_empty(monkeypatch):
    monkeypatch.setattr(rt, "web_search", lambda q, **kw: SearchResult([], diagnostics=["bing: no results"]))
    tool, _ = tools_by_name()["web_research"]
    out = tool.fn({"queries": ["nothing"]}, "key")
    assert out["sources"] == [] and out["searched"] == 0
    assert any("bing: no results" in d for d in out["search_diagnostics"])


def test_observe_clips_passages_to_500_characters():
    result = {"queries": ["q"], "failed": [], "searched": 1, "sources": [
        {"id": "S1", "url": "https://x.gov", "passages": [{"text": "a" * 900, "score": 2.0, "start": 0}]}]}
    seen = rt.observe(result)
    assert len(seen["sources"][0]["passages"][0]["text"]) <= 503
    assert "clipped to 500" in seen["observation_note"]


def test_observe_keeps_the_whole_observation_near_nine_thousand_characters():
    passages = [{"text": "b" * 480, "score": 1.0, "start": i} for i in range(60)]
    result = {"queries": ["q"], "failed": [], "searched": 9, "sources": [
        {"id": f"S{i}", "url": f"https://s{i}.gov", "passages": passages} for i in range(1, 6)]}
    seen = rt.observe(result)
    assert len(json.dumps(seen, ensure_ascii=False)) <= rt.OBSERVE_CHARS + 600
    assert "left out" in seen["observation_note"]


def test_observe_trims_a_long_fetched_text():
    seen = rt.observe({"url": "https://x.gov", "ok": True, "text": "c" * 20000})
    assert len(seen["text"]) < 9000 and "text_chars" in seen["observation_note"]


def test_small_observation_is_left_alone():
    result = {"url": "https://x.gov", "ok": True, "text": "short"}
    assert rt.observe(result) == result
