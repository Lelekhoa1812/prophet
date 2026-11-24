"""primary: the registry of official pages per series family, and family detection from query words. No network."""
import pytest

from prophet_agent.research.primary import PRIMARY_SOURCES, detect_family, registry_pages
from prophet_agent.research.rank import credibility


@pytest.mark.parametrize("family", sorted(PRIMARY_SOURCES))
def test_every_registry_page_is_an_official_tier_one_source(family):
    pages = registry_pages(family)
    assert pages, family
    for page in pages:
        assert set(page) == {"name", "url", "what"}, page
        assert page["url"].startswith("https://"), page
        assert credibility(page["url"])[0] == 1, (family, page["url"])


def test_the_families_named_in_the_brief_are_all_present():
    assert set(PRIMARY_SOURCES) == {"oil", "fx", "rates", "equities", "inflation", "gas", "gold", "co2"}


def test_oil_registry_names_the_eia_opec_and_iea_pages():
    urls = " ".join(p["url"] for p in registry_pages("oil"))
    assert "eia.gov" in urls and "opec.org" in urls and "iea.org" in urls


def test_fx_registry_names_the_ecb_boj_and_fed_calendars():
    urls = " ".join(p["url"] for p in registry_pages("fx"))
    assert "ecb.europa.eu" in urls and "boj.or.jp" in urls and "federalreserve.gov" in urls


def test_inflation_registry_names_the_bls_schedule():
    assert any("bls.gov/schedule" in p["url"] for p in registry_pages("inflation"))


def test_co2_registry_names_the_noaa_daily_file():
    assert any(p["url"].endswith("/co2_daily_mlo.txt") for p in registry_pages("co2"))


def test_equities_registry_names_the_fred_daily_close():
    assert any("fredgraph.csv?id=SP500" in p["url"] for p in registry_pages("equities"))


def test_registry_returns_copies_so_callers_cannot_change_it():
    registry_pages("oil")[0]["url"] = "https://evil.example"
    assert registry_pages("oil")[0]["url"].startswith("https://www.eia.gov")


def test_unknown_or_missing_family_gives_no_pages():
    assert registry_pages("weather") == [] and registry_pages(None) == []


@pytest.mark.parametrize("query,family", [
    ("WTI crude outlook", "oil"),
    ("Brent crude 2027 forecast", "oil"),
    ("Henry Hub natural gas storage", "gas"),
    ("gold price outlook", "gold"),
    ("EUR/JPY ECB decision", "fx"),
    ("USD JPY exchange rate", "fx"),
    ("treasury yields and the rate path", "rates"),
    ("CPI inflation release date", "inflation"),
    ("S&P 500 earnings season", "equities"),
    ("Mauna Loa daily CO2", "co2"),
])
def test_family_is_detected_from_the_query_words(query, family):
    assert detect_family(query) == family


def test_no_keyword_means_no_family():
    assert detect_family("history of the Roman road network") is None
    assert detect_family("") is None
