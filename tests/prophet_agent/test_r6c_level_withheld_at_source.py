"""Round 6c (2026-11-23): one level test for the contracts and the report tool.

Live Beijing d861b74e: forecast_report returned a 14-day panel total of PM2.5 (12,222 ug/m3), and the planner wrote it as a
daily average of 873. A level's total over series or days means nothing, so the tool withholds it. Every series is synthetic.
"""
from prophet_agent.runtime.contracts import is_level_name


def test_a_concentration_or_price_title_is_a_level():
    assert is_level_name("Beijing PM2.5 14-day forecast for 12 stations")
    assert is_level_name("Brent crude price, daily")


def test_a_flow_title_is_not_a_level_even_when_it_mentions_an_average():
    assert not is_level_name("Weekly net sales")
    assert not is_level_name("Average daily sales units")
    assert not is_level_name("")
