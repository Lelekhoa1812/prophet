"""A3-07 (round 3): web_research backfills failed or thin fetches from the next ranked results, keeps passages that carry
dates even when no query word appears, reads a stated publication date from the text, clamps max_chars to the schema,
and dedupes repeated queries.

Evidence (measured, round 3 audit A3): WTI rep4 research had fetch_top 6 and three failed search pages, so the cap was
spent; the OPEC+ page (title "OPEC+ Meeting 29 November 2026") was fetched but its passages scored 0 and were dropped;
the STEO passage said "released November 10" with no date; web_fetch was asked for 60000 characters against a 20000
schema cap; the electricity researcher repeated the same queries six times. All network calls are mocked.
"""
import pytest

from prophet_agent.research import passages as pz
from prophet_agent.research.fetch import Page
from prophet_agent.research.search import Hit, SearchResult
from prophet_agent.tools import research_tools as rt

GOOD = ("OPEC+ members meet on 29 November 2026 to decide output for the first quarter.\n\n"
        "Other remarks concern shipping insurance and labour contracts at length.")


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))


def research():
    return {tool.name: tool for tool, _ in rt.tools()}["web_research"]


def hit(url, rank):
    return Hit(title=f"Page {rank}", url=url, snippet="s", rank=rank, backend="bing")


@pytest.fixture
def identity_rank(monkeypatch):
    """Keep the hits in the order the test gives them, so the failing candidates are first."""
    monkeypatch.setattr(rt, "rank_report", lambda hits, **kw: (list(hits), 0))


def test_failed_fetches_are_backfilled_up_to_the_fetch_budget(monkeypatch, identity_rank):
    hits = [hit("https://www.eia.gov/a", 1), hit("https://www.eia.gov/b", 2),
            hit("https://www.reuters.com/c", 3), hit("https://www.reuters.com/d", 4)]
    monkeypatch.setattr(rt, "web_search", lambda q, **kw: SearchResult(hits, backend="bing"))
    fetched = []

    def fetch(url, **kwargs):
        fetched.append(url)
        if "eia.gov" in url:
            return Page(url=url, final_url=url, note="HTTP 403", ok=False, status=403)
        return Page(url=url, final_url=url, title="Reuters", text=GOOD, ok=True)

    monkeypatch.setattr(rt, "fetch_page", fetch)
    out = research().fn({"queries": ["OPEC+ meeting date"], "fetch_top": 2, "family": "oil"}, "key")
    search_sources = [s for s in out["sources"] if s["origin"] == "search"]
    assert [s["url"] for s in search_sources] == ["https://www.reuters.com/c", "https://www.reuters.com/d"]
    candidates = [h.url for h in hits]
    assert len([u for u in fetched if u in candidates]) == 4   # 2 failures, then 2 backfilled; the cap is 2 x fetch_top
    assert sum(1 for f in out["failed"] if f["origin"] == "search") == 2


def test_thin_page_counts_as_a_failed_fetch_and_is_backfilled(monkeypatch, identity_rank):
    hits = [hit("https://www.eia.gov/thin", 1), hit("https://www.reuters.com/full", 2)]
    monkeypatch.setattr(rt, "web_search", lambda q, **kw: SearchResult(hits, backend="bing"))

    def fetch(url, **kwargs):
        text = "Menu. Login." if "thin" in url else GOOD
        return Page(url=url, final_url=url, title="t", text=text, ok=True)

    monkeypatch.setattr(rt, "fetch_page", fetch)
    out = research().fn({"queries": ["OPEC+ meeting date"], "fetch_top": 1, "family": "oil"}, "key")
    assert [s["url"] for s in out["sources"] if s["origin"] == "search"] == ["https://www.reuters.com/full"]
    assert any(f["url"] == "https://www.eia.gov/thin" and "thin" in f["why"] for f in out["failed"])


def test_dated_passage_is_kept_when_no_query_word_appears():
    text = ("Ministers will gather on 29 November 2026 for the scheduled review.\n\n"
            "Unrelated commentary about harbour dredging schedules and pilotage fees.")
    best = pz.select_passages(text, ["Brent crude forecast"], k=3)
    assert best and best[0]["text"].startswith("Ministers will gather on 29 November 2026")
    assert best[0]["reason"] == "date"


def test_a_page_with_no_matching_passage_still_gives_its_opening():
    title_only = "Quarterly bulletin: summary of market conditions and outlook for the coming months."
    best = pz.select_passages(title_only, ["quantum chromodynamics lattice"], k=3)
    assert len(best) == 1 and best[0]["reason"] == "head"
    assert best[0]["text"] == title_only[:320]


def test_stated_publication_date_is_read_from_the_text(monkeypatch, identity_rank):
    text = ("The current STEO forecast was released November 10 and covers the next eighteen months.\n\n"
            "Brent is expected to average higher next year.")
    monkeypatch.setattr(rt, "web_search", lambda q, **kw: SearchResult([hit("https://www.eia.gov/steo", 1)],
                                                                        backend="bing"))
    monkeypatch.setattr(rt, "fetch_page", lambda url, **kw: Page(url=url, final_url=url, title="STEO", text=text,
                                                                 ok=True))
    out = research().fn({"queries": ["STEO forecast"], "family": "oil"}, "key")
    source = next(s for s in out["sources"] if s["url"] == "https://www.eia.gov/steo")
    assert source["date"] is not None and source["date"].endswith("-11-10")
    assert source["date_source"] in ("text", "text_year_inferred")


def test_repeated_queries_are_searched_once(monkeypatch, identity_rank):
    searched = []

    def search(query, **kwargs):
        searched.append(query)
        return SearchResult([hit("https://www.eia.gov/x", 1)], backend="bing")

    monkeypatch.setattr(rt, "web_search", search)
    monkeypatch.setattr(rt, "fetch_page", lambda url, **kw: Page(url=url, final_url=url, text=GOOD, ok=True))
    out = research().fn({"queries": ["Brent outlook", "brent  OUTLOOK", "Brent outlook"], "family": "oil"}, "key")
    assert searched == ["Brent outlook"]
    assert out["repeated_queries"] == 2


def test_a_data_table_is_a_preview_and_points_at_the_download(monkeypatch):
    # The NOAA daily file is this shape. web_fetch must not hand the body to the planner as the series.
    text = "# comment\n" + "".join(f"1974 5 19 1974.3781 {333.0 + i:.2f}\n" for i in range(40))
    monkeypatch.setattr(rt, "fetch_page", lambda url, **kw: Page(
        url=url, final_url=url, text=text, ok=True, media="text/plain",
        note="text clipped to 60000 characters from 578731"))
    tool = {t.name: t for t, _ in rt.tools()}["web_fetch"]
    out = tool.fn({"url": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_daily_mlo.txt", "queries": ["Mauna Loa"]}, "key")
    assert out["data_file"] is True and out["text_truncated"] is True
    assert "text" not in out and "passages" not in out
    assert "download_web_data" in out["note"] and "578731" in out["note"]
    assert out["preview"].count("\n") < 8
    assert len(out["preview"]) < len(text)


def test_a_cached_page_with_no_media_still_counts_as_a_data_file(monkeypatch):
    # The research cache written before the media field has media "". SciRes re-run 2026-11-24 took that path and
    # returned passages of the 1974 clip.
    text = "\n".join(f"1974 5 19 1974.3781 {333.0 + i:.2f}" for i in range(20))
    monkeypatch.setattr(rt, "fetch_page", lambda url, **kw: Page(
        url=url, final_url=url, text=text, ok=True, note="text clipped to 60000 characters"))
    tool = {t.name: t for t, _ in rt.tools()}["web_fetch"]
    out = tool.fn({"url": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_daily_mlo.txt", "queries": ["CO2"]}, "key")
    assert out["data_file"] is True and "passages" not in out and "text" not in out


def test_an_article_is_not_treated_as_a_data_file(monkeypatch):
    monkeypatch.setattr(rt, "fetch_page", lambda url, **kw: Page(url=url, final_url=url, text=GOOD, ok=True,
                                                                media="text/html"))
    tool = {t.name: t for t, _ in rt.tools()}["web_fetch"]
    out = tool.fn({"url": "https://www.eia.gov/x"}, "key")
    assert "data_file" not in out and out["text"].startswith("OPEC+")


def test_html_extract_that_looks_numeric_is_still_an_article(monkeypatch):
    text = "\n".join(f"1974 5 19 1974.3781 {333.0 + i:.2f}" for i in range(20))
    monkeypatch.setattr(rt, "fetch_page", lambda url, **kw: Page(url=url, final_url=url, text=text, ok=True,
                                                                media="text/html"))
    tool = {t.name: t for t, _ in rt.tools()}["web_fetch"]
    out = tool.fn({"url": "https://gml.noaa.gov/ccgg/trends/"}, "key")
    assert "data_file" not in out and "text" in out


def test_max_chars_above_the_schema_is_clamped_and_said_so(monkeypatch):
    monkeypatch.setattr(rt, "fetch_page", lambda url, **kw: Page(url=url, final_url=url, text="word " * 9000, ok=True))
    tool = {t.name: t for t, _ in rt.tools()}["web_fetch"]
    out = tool.fn({"url": "https://www.eia.gov/x", "max_chars": 60000}, "key")
    assert len(out["text"]) == 20000 and "clamped to 20000" in out["note"]
