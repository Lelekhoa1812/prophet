"""A3-06 (round 3): the research family list must accept the domains the goal trigger uses (electricity, retail,
weather, prices, PM2.5), an invalid family must name the valid ones, regional families must not search without a
region, and a non-US region must not be sent US grid-operator queries.

Evidence (measured, round 3 audit A3): the electricity researcher's first call used family "electricity" and failed with
"family must be one of ..."; it then ran 12 calls aimed at ISO-NE, PJM, ERCOT and NYISO for a file of 60 Portuguese
clients with no region. All searches and fetches are mocked; nothing goes to the network.
"""
import pytest

from prophet_agent.research.fetch import Page
from prophet_agent.research.primary import detect_family
from prophet_agent.research.search import Hit, SearchResult
from prophet_agent.tools import research_tools as rt

TEXT = "Hourly demand was 4,250 MWh on 3 March 2026 in the reported series.\n\nFigures are published quarterly."


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))


@pytest.fixture
def calls(monkeypatch):
    seen = {"searched": [], "fetched": []}

    def search(query, **kwargs):
        seen["searched"].append(query)
        return SearchResult([Hit(title="Official statistics", url="https://www.ine.pt/demand", snippet="s", rank=1,
                                 backend="bing")], backend="bing")

    def fetch(url, **kwargs):
        seen["fetched"].append(url)
        return Page(url=url, final_url=url, title="Demand", text=TEXT, sha256="s1", ok=True)

    monkeypatch.setattr(rt, "web_search", search)
    monkeypatch.setattr(rt, "fetch_page", fetch)
    return seen


def research():
    return {tool.name: tool for tool, _ in rt.tools()}["web_research"]


@pytest.mark.parametrize("family", ["electricity", "retail", "weather", "prices", "pm25"])
def test_trigger_domains_are_accepted_as_a_family(calls, family):
    out = research().fn({"queries": ["hourly demand series"], "family": family, "region": "Portugal"}, "key")
    assert out["family"] == family


def test_invalid_family_error_names_the_valid_families():
    with pytest.raises(ValueError, match="family must be one of") as err:
        research().fn({"queries": ["x"], "family": "astrology"}, "key")
    message = str(err.value)
    for valid in ("oil", "electricity", "retail", "weather", "prices", "pm25"):
        assert valid in message


def test_trigger_wording_aliases_map_to_their_family(calls):
    out = research().fn({"queries": ["tariff changes"], "family": "tariffs", "region": "Ireland"}, "key")
    assert out["family"] == "prices"
    out = research().fn({"queries": ["PM2.5 readings"], "family": "PM2.5", "region": "Beijing"}, "key")
    assert out["family"] == "pm25"


def test_detection_knows_the_electricity_and_pm25_wording():
    assert detect_family("hourly electricity demand in kWh per client") == "electricity"
    assert detect_family("PM2.5 concentration and smog episodes") == "pm25"


def test_regional_family_without_a_region_runs_no_search(calls):
    out = research().fn({"queries": ["electricity demand Portugal"], "family": "electricity"}, "key")
    assert calls["searched"] == [] and calls["fetched"] == []
    assert out["region_unknown"] is True and out["sources"] == []
    assert "region" in out["not_obtained"]


def test_us_grid_operator_queries_are_dropped_for_a_non_us_region(calls):
    out = research().fn({"queries": ["ERCOT load forecast", "ISO-NE demand", "Portugal electricity demand"],
                         "family": "electricity", "region": "Portugal"}, "key")
    assert all("ERCOT" not in q and "ISO-NE" not in q for q in calls["searched"])
    assert len(out["dropped_queries"]) == 2
    assert calls["searched"]


def test_the_region_is_added_to_every_searched_query(calls):
    research().fn({"queries": ["retail sales Christmas closures", "shop footfall"], "family": "retail",
                   "region": "United Kingdom"}, "key")
    assert calls["searched"] and all("United Kingdom" in q for q in calls["searched"])


def test_us_region_keeps_the_us_grid_queries(calls):
    out = research().fn({"queries": ["ERCOT load forecast"], "family": "electricity", "region": "Texas, United States"},
                        "key")
    assert calls["searched"] == ["ERCOT load forecast Texas, United States"]
    assert out["dropped_queries"] == []


def test_spec_lists_every_accepted_family():
    family_text = rt.WEB_RESEARCH_SPEC["args"]["family"]
    for valid in ("electricity", "retail", "weather", "prices", "pm25", "oil"):
        assert valid in family_text
    assert "region" in rt.WEB_RESEARCH_SPEC["args"]
