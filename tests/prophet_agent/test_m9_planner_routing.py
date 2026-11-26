"""Mission 9 (planner.v27): the routing rules for outside data are in the planner prompt.

Live, 2026-11-26: the Nvidia run had no price tool and saved a sample file under its ticker, and the EPL run kept re-parsing one
passage because the table tools were not named. The rules below name the tool for each kind of source. Text checks only: the
tools download_market_prices and extract_page_charts and web_fetch's render argument are being added on another branch, so no
test here checks that they are registered.
"""
from __future__ import annotations

from prophet_agent import prompts


def _text() -> str:
    return " ".join(prompts.current("planner").text.split())


def test_planner_v27_is_the_current_planner_and_keeps_its_rules():
    planner = prompts.current("planner")
    assert planner.version == "planner.v31"
    assert "never generate, simulate" in planner.text.lower() and "A. The operating procedure" in planner.text


def test_daily_prices_are_routed_to_the_price_tool_with_the_clock_rule():
    text = _text()
    assert "Daily prices for a ticker or index: call download_market_prices with symbol, start and end" in text
    assert "a data end is never later than it" in text


def test_tables_and_charts_on_a_page_have_their_own_tools():
    text = _text()
    assert "call download_web_data on the page URL" in text and "`table` argument" in text
    assert "extract_page_tables when you need to choose the table" in text
    assert "extract_page_charts reads the chart's embedded data" in text


def test_javascript_pages_are_rendered_and_entities_keep_their_names():
    text = _text()
    assert "web_fetch with render true" in text
    assert "Never save one entity's data under another entity's name" in text


def test_blocked_interstitials_are_not_refetched():
    text = _text()
    assert "adblock, cookie consent, bot check, CAPTCHA, login wall" in text
    assert "Do not refetch it, and do not try to pass it" in text


def test_v28_routes_leagues_pdfs_cached_pages_and_indexes():
    """Batch 2 (2026-11-26): the planner is told the tools that now exist for these sources."""
    text = _text()
    assert "call simulate_season (sport football, or win_loss for no draws)" in text
    assert "never add ranges across teams" in text
    assert "download_web_data on the PDF URL" in text and "find_in_page searches it" in text
    assert "download_open_data with source `fred` (SP500" in text
    assert "table_kind league and expected_rows" in text
    assert "never offer invented, simulated or approximate numbers as an option" in text


def test_every_tool_v28_names_is_registered():
    from prophet_agent.tools import default_tools
    from prophet_agent.tools.opendata_tools import OpenDataContext
    names = set(default_tools(opendata=OpenDataContext(store=None, attach=None)))
    for tool in ("simulate_season", "find_in_page", "download_web_data", "download_market_prices", "download_open_data",
                 "extract_page_charts", "state_table_claim", "validate_table", "backtest_series"):
        assert tool in names, tool


def test_v29_routes_fred_full_season_results_and_copied_figures():
    """Panel round 2 (2026-11-26): CPI asked for an unknown source "fred"; EPL read a football.json that stopped on 20
    September; EPL passed the current season as history; NBA retyped standings that disagreed with the downloaded table."""
    from prophet_agent.opendata import sources
    text = _text()
    assert "download_open_data with source `fred` and params {\"id\": code}" in text and "CPIAUCSL" in text
    assert sources.get("fred").licence_key == "fred"
    assert "https://fixturedownload.com/download/epl-2026-GMTStandardTime.csv" in text
    assert "compare its last match date with today and prefer the fuller file" in text
    assert "`history` names earlier seasons' files only, never the current season's file" in text
    assert "or with `source` {run_id, file} for a results file a sandbox run wrote" in text
    assert "copied from a tool result, not typed from memory" in text


def test_v30_gives_sdmx_keys_without_guessing():
    """Panel AUUNEMP (2026-11-26): the model wrote an ABS key as "M.1.1.1..." until it was cut off, four times."""
    from prophet_agent.opendata import sources
    text = _text()
    assert "never build one by repeating a code" in text
    assert "flow ABS,LF,1.0.0, key M13.3.1599.20.AUS.M" in text
    url = sources.build_url("abs_sdmx", {"flow": "ABS,LF,1.0.0", "key": "M13.3.1599.20.AUS.M"})
    assert url.endswith("/ABS,LF,1.0.0/M13.3.1599.20.AUS.M?format=csv")
