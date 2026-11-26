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
    assert planner.version == "planner.v27"
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
