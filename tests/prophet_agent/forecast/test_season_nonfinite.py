"""choose_season with missing or extreme values: no NaN is ever returned, and dates stay aligned with the values.

Motivation vs Logic
Motivation (GOAL.md, known defect): choose_season could return NaN and misalign dates. Values were filtered with isfinite while
the dates were not, so a missing value shifted every date after it, and a difference of two huge values overflowed to a NaN
strength that the picker then compared as if it were a number.
Logic: missing values keep their position (the changes and the level phases are measured on positions, so a gap never shifts a
cycle), dates that do not match the values are refused with a reason, and a series whose changes cannot be measured is refused
with a reason rather than returning NaN. All measurements here are on synthetic series.
"""
import warnings

import numpy as np

from prophet_agent.forecast.season import choose_season


def _annual(n=1612, amplitude=10.0, noise=3.0, seed=3):
    t = np.arange(n)
    return 50 + amplitude * np.sin(2 * np.pi * t / 365) + np.random.default_rng(seed).normal(0, noise, n)


def _weekly(n=700, seed=4):
    t = np.arange(n)
    return 50 + 4 * np.sin(2 * np.pi * t / 7) + np.random.default_rng(seed).normal(0, 1, n)


def _no_nan_in_numbers(out):
    numbers = [v for k, v in out.items() if isinstance(v, float)]
    numbers += list((out.get("strengths") or {}).values())
    numbers += list((out.get("level_strengths") or {}).values())
    return all(np.isfinite(v) for v in numbers)


def test_dates_that_do_not_match_the_values_are_refused_with_a_reason():
    # Before the fix a 699-date list against 700 values was accepted and read as business days (season 21).
    out = choose_season(_weekly(), freq_hint=None, dates=np.arange(699))
    assert out["season"] == 1
    assert "do not match" in out["reason"]


def test_a_missing_value_keeps_the_weekly_cycle_at_its_position():
    x = _weekly()
    x[300] = np.nan
    dates = (np.datetime64("2020-01-01") + np.arange(700)).astype(str)
    out = choose_season(x, dates=dates)
    assert out["season"] == 7
    assert _no_nan_in_numbers(out)


def test_an_annual_cycle_survives_scattered_missing_values_on_the_level_path():
    # The level measure reads phases by position; dropping the missing values would shift every later phase.
    x = _annual()
    x[np.random.default_rng(9).choice(len(x), 40, replace=False)] = np.nan
    out = choose_season(x, freq_hint="D")
    assert out["season"] == 365 and out["method"] == "levels"
    assert _no_nan_in_numbers(out)


def test_values_whose_changes_overflow_are_refused_without_a_nan_strength():
    # Before the fix: the changes overflowed to infinity, their standard deviation was NaN and the strength came out NaN.
    x = np.tile([1.7e308, -1.7e308], 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = choose_season(x, freq_hint="D")
    assert out["season"] == 1
    assert _no_nan_in_numbers(out)
    assert "too large" in out["reason"] or "no candidate" in out["reason"]


def test_a_series_with_too_few_finite_values_reports_the_finite_count():
    x = np.full(50, np.nan)
    x[:10] = np.arange(10.0)
    out = choose_season(x)
    assert out["season"] == 1
    assert "10 finite points" in out["reason"]
