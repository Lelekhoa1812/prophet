"""A6-08(c): explicit ignore_ranges are tested by the control lab, so a backtest that uses them has a corrected comparison.

Motivation vs Logic
Motivation: ignore_ranges (a keying error, a stock-out) reached backtest_series and timesfm_forecast, which score the choice on
the windows it was chosen on, while the lab (which applies Holm correction) had no candidate for it (audit A6-08c).
Logic: the lab takes the caller's ranges as an optional candidate. At each fold the ranges are cut at the fold's origin, so a
fold reads only the ranges that lie in its own history (Control.at_origin does the same cut). The range candidate then goes
through the same paired test and Holm step-down as the other treatments. The engine is stubbed.
"""
from __future__ import annotations

import numpy as np
import pytest

from prophet_agent.forecast.control import build_control
from prophet_agent.forecast.lab import CANDIDATE_NAMES, FoldContext, _grid, run_lab
from prophet_agent.forecast.types import Forecast


class _RecordingEngine:
    def __init__(self):
        self.calls = []   # (history length, control)

    def forecast_controlled(self, series, horizon, control):
        self.calls.append((len(series), control))
        point = np.full(horizon, float(np.asarray(series)[-1]))
        return Forecast(point, np.tile(np.linspace(0, 8, 9), (horizon, 1)))


def _ctx(origin: int) -> FoldContext:
    return FoldContext(hist=np.arange(origin, dtype=float), origin=origin, horizon=5, period=None, past={}, future={})


def test_the_range_candidate_exists_only_when_ranges_are_given():
    assert "ignore_ranges" in CANDIDATE_NAMES
    assert "ignore_ranges" not in [c.name for c in _grid(None, False)]
    assert "ignore_ranges" in [c.name for c in _grid(None, False, ranges=((10, 20),))]


def test_each_fold_sees_only_the_ranges_inside_its_own_history():
    cand = {c.name: c for c in _grid(None, False, ranges=((10, 20), (45, 70)))}["ignore_ranges"]
    assert cand.prepare(_ctx(50)).control.ignore == ((10, 20), (45, 50))   # the second range is cut at the origin
    assert cand.prepare(_ctx(15)).control.ignore == ((10, 15),)
    assert cand.prepare(_ctx(5)).control.is_default                        # no range inside this history: a no-op fold


def test_the_lab_runs_the_range_candidate_on_the_clipped_ranges():
    rng = np.random.default_rng(2)
    x = 50 + np.cumsum(rng.normal(0, 1, 200))
    eng = _RecordingEngine()
    out = run_lab(x, 5, eng, candidates=["ignore_ranges"], folds=6, ranges=[(20, 30)])
    assert "ignore_ranges" in out["decisions"]
    for n_hist, control in eng.calls:
        for s, e in control.ignore:
            assert 0 <= s < e <= n_hist                                  # no fold reads past its own origin
    assert any(control.ignore == ((20, 30),) for _, control in eng.calls)  # the folds whose history covers the range


def test_the_ranges_are_validated_like_the_backtest_ranges():
    with pytest.raises(ValueError):
        build_control({"ignore_ranges": [[30, 20]]}, 200, 5)
    with pytest.raises(ValueError):
        build_control({"ignore_ranges": [[190, 210]]}, 200, 5)
