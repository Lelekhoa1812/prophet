"""A3-08 and A4-08 (round 3): research returns a dated event list (date, event, source URL, how sure) that the planner can
match to the data's level shifts and forecast window, and a stated data end and horizon place each event.

Evidence (measured, round 3 audit A3): code search found no link from research to changepoints, levels or horizons. The
RWTCD print date (2026-11-18) was not compared with the last observed value; the 29 November OPEC+ meeting, inside the
20-day horizon, never reached the answer. All network calls are mocked.
"""
import pytest

from prophet_agent.research import events as ev
from prophet_agent.research.fetch import Page
from prophet_agent.research.search import Hit, SearchResult
from prophet_agent.tools import research_tools as rt


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))


def source(url, text, tier_url=None, date="2026-11-20", date_source="meta", sid="S1"):
    return {"id": sid, "url": url, "origin": "search", "date": date, "date_source": date_source,
            "tier": 1 if ".gov" in url or "eia.gov" in url else 3,
            "passages": [{"text": text, "score": 1.0, "start": 0, "reason": "query"}]}


def test_each_dated_sentence_becomes_an_event_with_its_source_and_how_sure():
    sources = [source("https://www.eia.gov/steo", "The OPEC+ meeting is scheduled for 29 November 2026. "
                                                 "Analysts expect no change.")]
    events = ev.dated_events(sources)
    assert len(events) == 1
    e = events[0]
    assert e["date"] == "2026-11-29" and "OPEC+ meeting" in e["event"]
    assert e["source_url"] == "https://www.eia.gov/steo" and e["source_id"] == "S1"
    assert e["how_sure"] == "high"           # tier 1 and the year is written in the text


def test_an_inferred_year_is_less_sure_and_a_non_official_source_is_low():
    eia = source("https://www.eia.gov/steo", "The forecast was released November 10 this year.", sid="S2")
    news = source("https://www.news.example/oil", "Traders expect a meeting on 4 December 2026.", sid="S3")
    by_url = {e["source_url"]: e for e in ev.dated_events([eia, news])}
    assert by_url["https://www.eia.gov/steo"]["how_sure"] == "medium"
    assert by_url["https://www.eia.gov/steo"]["date"].endswith("-11-10")
    assert by_url["https://www.news.example/oil"]["how_sure"] == "low"


def test_events_are_placed_against_the_data_end_and_the_horizon():
    sources = [source("https://www.eia.gov/a", "A review took place on 5 November 2026. A meeting is set for "
                                              "29 November 2026. A report follows on 20 December 2026.")]
    events = ev.dated_events(sources, data_end="2026-11-17", horizon_end="2026-12-07")
    position = {e["date"]: e["position"] for e in events}
    assert position["2026-11-05"] == "inside the data period"
    assert position["2026-11-29"] == "inside the forecast window"
    assert position["2026-12-20"] == "after the forecast window"


def test_position_is_none_when_the_data_end_is_not_given():
    events = ev.dated_events([source("https://www.eia.gov/a", "Meeting on 29 November 2026.")])
    assert events[0]["position"] is None


@pytest.mark.parametrize("value", ["17/11/2026", "2026-13-01", "Nov 17"])
def test_data_end_must_be_an_iso_date(value, monkeypatch):
    monkeypatch.setattr(rt, "web_search", lambda q, **kw: SearchResult([], backend="bing"))
    tool = {t.name: t for t, _ in rt.tools()}["web_research"]
    with pytest.raises(ValueError, match="data_end"):
        tool.fn({"queries": ["oil"], "family": "oil", "data_end": value}, "key")


def test_the_same_sentence_from_two_sources_is_one_event():
    text = "The meeting is scheduled for 29 November 2026."
    events = ev.dated_events([source("https://www.eia.gov/a", text, sid="S1"),
                              source("https://www.eia.gov/b", text, sid="S2")])
    assert len(events) == 1


def test_research_output_carries_events_for_the_planner(monkeypatch):
    text = ("OPEC+ ministers will meet on 29 November 2026 to decide output. Delegates expect a decision on "
            "the first-quarter production baseline, with storage and refinery runs also under review.")
    monkeypatch.setattr(rt, "web_search", lambda q, **kw: SearchResult(
        [Hit(title="OPEC+ meeting", url="https://www.iea.org/news/opec", snippet="s", rank=1, backend="bing")],
        backend="bing"))
    monkeypatch.setattr(rt, "fetch_page", lambda url, **kw: Page(
        url=url, final_url=url, title="OPEC+", ok=True, date="2026-11-20", date_source="meta",
        text=text if "news/opec" in url else "Quarterly remarks on storage and refinery runs."))
    tool = {t.name: t for t, _ in rt.tools()}["web_research"]
    out = tool.fn({"queries": ["OPEC+ meeting date"], "family": "oil", "data_end": "2026-11-17",
                   "horizon_end": "2026-12-07"}, "key")
    dated = [e for e in out["events"] if e["date"] == "2026-11-29"]
    assert dated and dated[0]["position"] == "inside the forecast window"
    assert dated[0]["source_url"] == "https://www.iea.org/news/opec"


def test_observation_keeps_the_event_list_when_the_result_is_small():
    result = {"queries": ["q"], "failed": [], "searched": 1, "sources": [], "events": [
        {"date": "2026-11-29", "event": "Meeting.", "source_id": "S1", "source_url": "https://x.gov",
         "how_sure": "high", "position": None}]}
    assert rt.observe(result)["events"] == result["events"]
