"""Round 5: ps.change_test answers "is the change real, or the usual ups and downs?" against the series' own history.

All series here are synthetic (seeded), labelled as such.
"""
import numpy as np

from prophet_agent.sandbox import prophet_stats as ps


def _weekly(n=900, seed=1, drop_last=0, pct=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = 100 + 10 * np.sin(2 * np.pi * t / 7) + 8 * np.sin(2 * np.pi * t / 365.25) + rng.normal(0, 4, n)
    if drop_last:
        x[-drop_last:] *= 1 + pct / 100
    return x


def test_a_real_drop_is_called_a_decline():
    verdict, rep = ps.change_test(_weekly(drop_last=28, pct=-20))      # synthetic: a 20% drop in the last four weeks
    assert verdict == "decline" and rep["lag"] == 364 and rep["change_pct"] < -10 and rep["p_value"] < 0.1


def test_ordinary_noise_is_within_usual_variation():
    hits = sum(ps.change_test(_weekly(seed=s))[0] != "within usual variation" for s in range(20))
    assert hits <= 5                                                      # synthetic: about alpha = 10% by design


def test_a_short_history_says_so():
    verdict, rep = ps.change_test(np.arange(30.0), recent=28)
    assert verdict == "not enough history" and "need at least" in rep["note"]


def test_without_a_year_of_history_it_compares_with_the_window_before():
    verdict, rep = ps.change_test(_weekly(n=200, drop_last=28, pct=-30), recent=28)
    assert rep["lag"] == 28 and verdict == "decline"


def test_with_dates_a_trading_day_series_is_compared_a_calendar_year_back():
    import pandas as pd
    days = pd.bdate_range("2021-01-04", periods=800)               # synthetic trading days: no weekends
    x = _weekly(n=800, seed=3)
    verdict, rep = ps.change_test(x, recent=28, dates=days)
    assert rep["lag"] == 364 and rep["unit"] == "days" and "one year earlier" in rep["compared_with"]
    x2 = x.copy(); x2[-20:] *= 0.75                                  # the last four weeks of trading days, 25% lower
    assert ps.change_test(x2, recent=28, dates=days)[0] == "decline"
