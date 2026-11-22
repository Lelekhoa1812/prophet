"""Live Beijing, round 4 (measured): the level-shift text stated the ten-times unit-change rule without the measured ratio,
and the planner reported "median jumps of 10x or more" for breaks of 1.05 to 1.34 times, then stopped without forecasting.
The finding now leads with the measured ratio and says when no break is a unit change."""
import numpy as np
import pandas as pd

import prophet_scan as sc


def _finding(values):
    df = pd.DataFrame({"t": pd.date_range("2020-01-01", periods=len(values), freq="D"), "y": values})
    res = sc.scan(df, "t", "y")
    return next(f for f in res["findings"] if f["id"] == "changepoints.level_shift")


def test_a_small_shift_says_it_is_not_a_unit_change():
    rng = np.random.default_rng(1)
    x = np.r_[rng.normal(60, 5, 200), rng.normal(75, 5, 200)]
    f = _finding(x)
    text = f["decision_needed"] if "decision_needed" in f else f["message"]
    assert "none of these is one" in text and "1.2" in text
    assert f["severity"] != "blocker"


def test_a_tenfold_shift_is_still_a_blocker_and_says_so():
    rng = np.random.default_rng(2)
    x = np.r_[rng.normal(10, 1, 200), rng.normal(1000, 50, 200)]
    f = _finding(x)
    text = f["decision_needed"] if "decision_needed" in f else f["message"]
    assert f["severity"] == "blocker" and "unit-change size" in text
