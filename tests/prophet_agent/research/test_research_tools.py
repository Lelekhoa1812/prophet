"""research tools: argument validation, the three tools end to end with search and fetch mocked, and the observation cap."""
import json

import pytest

from prophet_agent.research.fetch import Page
from prophet_agent.research.primary import registry_pages
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
    return Page(url=url, final_url=url, title="EIA", date="2026-11-01", date_source="meta", text=PAGE_TEXT,
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
    assert out["hits"][0]["source"] == "registry"
    assert out["hits"][0]["url"] in {page["url"] for page in registry_pages("oil")}
    assert [h["tier"] for h in out["hits"] if h.get("source") != "registry"] == [1, 2, 4]
    assert "diagnostics" not in out


def test_web_search_lists_the_noaa_daily_file_before_other_hits(monkeypatch):
    monkeypatch.setattr(rt, "web_search", lambda q, **kw: SearchResult([
        Hit(title="finder", url="https://gml.noaa.gov/ccgg/trends/data.html", snippet="s", rank=1, backend="bing")],
        backend="bing"))
    tool, _ = tools_by_name()["web_search"]
    out = tool.fn({"query": "Mauna Loa daily CO2"}, "key")
    assert out["hits"][0]["url"].endswith("/co2_daily_mlo.txt")
    assert out["hits"][0]["source"] == "registry"


def test_web_search_tool_reports_diagnostics_when_nothing_came_back(monkeypatch):
    empty = SearchResult([], backend=None, diagnostics=["auto: DDGSException: blocked"])
    monkeypatch.setattr(rt, "web_search", lambda q, **kw: empty)
    tool, _ = tools_by_name()["web_search"]
    out = tool.fn({"query": "history of the roman roads"}, "key")
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
    assert [s["id"] for s in out["sources"]] == [f"S{i}" for i in range(1, len(out["sources"]) + 1)]
    registry = [s for s in out["sources"] if s["origin"] == "registry"]
    search = [s for s in out["sources"] if s["origin"] == "search"]
    assert out["family"] == "oil" and len(registry) == len(registry_pages("oil"))
    assert [s["url"] for s in search] == ["https://www.eia.gov/outlook", "https://www.reuters.com/oil"]
    eia = search[0]
    assert eia["sha256"] == "def456" and eia["tier"] == 1 and eia["date"] == "2026-11-01"
    assert eia["date_source"] == "meta"
    assert all(p["text"] and p["score"] > 0 for p in eia["passages"])
    assert len(eia["passages"]) <= 2
    assert out["failed"] == [{"url": "https://blog.blogspot.com/best-oil", "why": "HTTP 503", "origin": "search"}]


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


def recording(monkeypatch, search=hits_for, page=fake_fetch):
    """Mock search and fetch, and return the list of URLs fetched in call order."""
    fetched = []

    def fetch(url, **kwargs):
        fetched.append(url)
        return page(url, **kwargs)

    monkeypatch.setattr(rt, "web_search", lambda q, **kw: search(q))
    monkeypatch.setattr(rt, "fetch_page", fetch)
    return fetched


def test_registry_pages_are_read_before_the_searches(monkeypatch):
    fetched = recording(monkeypatch)
    tool, _ = tools_by_name()["web_research"]
    out = tool.fn({"queries": ["WTI crude outlook"]}, "key")
    registry_urls = {p["url"] for p in registry_pages("oil")}
    assert out["family"] == "oil"
    assert set(fetched[:len(registry_urls)]) == registry_urls
    assert [s["origin"] for s in out["sources"][:len(registry_urls)]] == ["registry"] * len(registry_urls)


def test_no_registry_read_when_no_family_matches(monkeypatch):
    fetched = recording(monkeypatch)
    tool, _ = tools_by_name()["web_research"]
    out = tool.fn({"queries": ["history of the Roman road network"]}, "key")
    assert out["family"] is None
    assert all(s["origin"] == "search" for s in out["sources"])
    assert not set(fetched) & {p["url"] for p in registry_pages("oil") + registry_pages("fx")}


def test_family_argument_overrides_detection_and_unknown_family_is_refused(monkeypatch):
    recording(monkeypatch)
    tool, _ = tools_by_name()["web_research"]
    out = tool.fn({"queries": ["WTI crude outlook"], "family": "gold"}, "key")
    assert out["family"] == "gold"
    assert {s["name"] for s in out["sources"] if s["origin"] == "registry"} == \
        {p["name"] for p in registry_pages("gold")}
    with pytest.raises(ValueError, match="family must be one of"):
        tool.fn({"queries": ["WTI"], "family": "astrology"}, "key")


def test_a_failed_registry_page_is_listed_with_its_origin(monkeypatch):
    def page(url, **kwargs):
        if "opec" in url:
            return Page(url=url, final_url=url, note="HTTP 403", ok=False, status=403)
        return fake_fetch(url, **kwargs)
    recording(monkeypatch, page=page)
    tool, _ = tools_by_name()["web_research"]
    out = tool.fn({"queries": ["WTI crude outlook"]}, "key")
    assert {"url": "https://www.opec.org/opec_web/en/publications/338.htm", "why": "HTTP 403",
            "origin": "registry"} in out["failed"]


def test_exclude_urls_are_never_fetched_again_and_are_counted(monkeypatch):
    fetched = recording(monkeypatch)
    tool, _ = tools_by_name()["web_research"]
    out = tool.fn({"queries": ["WTI crude outlook"],
                   "exclude_urls": ["https://www.eia.gov/outlook/?utm_source=x",
                                    "https://www.eia.gov/petroleum/supply/weekly"]}, "key")
    assert "https://www.eia.gov/outlook" not in fetched
    assert "https://www.eia.gov/petroleum/supply/weekly/" not in fetched
    assert "https://www.reuters.com/oil" in fetched
    assert out["duplicate_sources"] == 2
    assert all(s["url"] not in ("https://www.eia.gov/outlook",) for s in out["sources"])


@pytest.mark.parametrize("value", ["https://www.eia.gov", ["not a url"], ["https://x.example"] * 201])
def test_bad_exclude_urls_are_refused(value):
    tool, _ = tools_by_name()["web_research"]
    with pytest.raises(ValueError, match="exclude_urls|http or https"):
        tool.fn({"queries": ["oil"], "exclude_urls": value}, "key")


def test_repeats_across_queries_are_counted_once(monkeypatch):
    recording(monkeypatch)
    tool, _ = tools_by_name()["web_research"]
    out = tool.fn({"queries": ["weather in Paris", "Paris rain forecast"], "region": "Paris, France"}, "key")
    assert out["duplicate_sources"] == 3
    assert len({s["url"] for s in out["sources"]}) == len(out["sources"])


def test_metrics_summarise_the_call(monkeypatch):
    recording(monkeypatch)
    tool, _ = tools_by_name()["web_research"]
    out = tool.fn({"queries": ["WTI crude outlook"]}, "key")
    metrics = out["metrics"]
    assert metrics == {"unique_domains": 4, "tier1_sources": 6, "dated_sources": 6,
                       "newest_date": "2026-11-01", "oldest_date": "2026-11-01", "new_sources": len(out["sources"])}


def test_search_news_date_is_used_only_when_the_page_has_none(monkeypatch):
    news_hit = Hit(title="Oil rises", url="https://www.cnbc.com/oil", snippet="s", date="2026-11-20T09:00:00+00:00",
                   rank=1, backend="bing")
    recording(monkeypatch, search=lambda q: SearchResult([news_hit], backend="bing"),
              page=lambda url, **kw: Page(url=url, final_url=url, text=PAGE_TEXT, ok=True))
    tool, _ = tools_by_name()["web_research"]
    out = tool.fn({"queries": ["cnbc oil"]}, "key")
    source = next(s for s in out["sources"] if s["url"] == "https://www.cnbc.com/oil")
    assert source["date"] == "2026-11-20T09:00:00+00:00" and source["date_source"] == "search"
    assert out["metrics"]["dated_sources"] == 1
