"""One season rule for every tool: business-day prices default to no season unless a weekly or monthly cycle is measured in
their one-step changes; daily, hourly, monthly, weekly and quarterly data test their own candidate lags."""
from __future__ import annotations

import numpy as np

from prophet_agent.forecast.season import choose_season, mase_scale_text


def _walk(n=600, seed=2):
    return 100 + np.cumsum(np.random.default_rng(seed).normal(0, 1, n))


def _weekly_changes(n=600, seed=2):
    """A price whose one-step changes follow a five-day pattern: the weekly cycle lives in the changes."""
    rng = np.random.default_rng(seed)
    steps = np.tile([2.0, -1.0, 0.5, -2.0, 0.0], n // 5 + 1)[:n] + rng.normal(0, 0.2, n)
    return 100 + np.cumsum(steps)


def test_a_business_day_price_with_no_cycle_has_season_one_and_says_why():
    out = choose_season(_walk(), freq_hint="B")
    assert out["season"] == 1 and out["kind"] == "business"
    assert "no weekly or monthly cycle" in out["reason"]


def test_a_business_day_price_with_a_weekly_cycle_in_its_changes_gets_season_five():
    out = choose_season(_weekly_changes(), freq_hint="B")
    assert out["season"] == 5 and "every 5 periods" in out["reason"]


def test_weekday_only_dates_are_read_as_business_days():
    dates = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-01") + 800)
    weekdays = dates[(((dates.astype("datetime64[D]").astype(np.int64)) + 3) % 7) < 5]
    out = choose_season(_walk(len(weekdays)), dates=weekdays)
    assert out["kind"] == "business" and out["season"] == 1


def test_daily_data_tests_seven_and_a_weekly_pattern_is_found():
    t = np.arange(700)
    x = 50 + 4 * np.sin(2 * np.pi * t / 7) + np.random.default_rng(4).normal(0, 1, 700)
    out = choose_season(x, freq_hint="D")
    assert out["season"] == 7 and set(out["strengths"]) == {7}


def test_monthly_data_finds_a_twelve_month_cycle_and_quarterly_tests_four():
    t = np.arange(240)
    x = 10 + 3 * np.sin(2 * np.pi * t / 12) + np.random.default_rng(5).normal(0, 0.5, 240)
    assert choose_season(x, freq_hint="MS")["season"] == 12
    assert set(choose_season(x, freq_hint="Q")["strengths"]) == {4}


def test_a_random_walk_without_a_hint_has_no_season_and_a_short_series_is_refused_as_one():
    assert choose_season(_walk())["season"] == 1
    assert choose_season(np.arange(10.0))["season"] == 1


def test_the_mase_scale_text_matches_the_season_used():
    assert mase_scale_text(1) == "mean absolute one-step change"
    assert mase_scale_text(7) == "mean absolute seasonal change at lag 7"
