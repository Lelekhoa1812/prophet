"""Round 5: the total's 80 percent range from the measured day-to-day error correlation.

Finding (live retail rep6, 2026-11-23): the pack gave two bounds for the horizon total, one if days miss independently and one
if they miss together, and no guidance between them. The backtest's stored per-origin predictions measure how the day errors
move together (variance ratio R: 1 independent, the horizon fully together). Every series in this file is synthetic.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pytest

from prophet_agent.forecast import rolling as R
from prophet_agent.forecast.types import Forecast

Z90 = 1.2815515655446004
LEVELS = [NormalDist().inv_cdf(i / 10) for i in range(1, 10)]


class BandEngine:
    """Synthetic fake engine: point and band come from a per-day sd function, the quantiles are normal at the nine levels."""

    def __init__(self, point_fn, sd_fn):
        self.point_fn, self.sd_fn = point_fn, sd_fn

    def forecast_batch(self, contexts, horizon):
        sd = np.asarray(self.sd_fn(horizon), dtype=float)
        pts = np.stack([np.full(horizon, float(self.point_fn(c))) for c in contexts])
        q = pts[..., None] + np.array(LEVELS)[None, None, :] * sd[None, :, None]
        return Forecast(pts, q)


def _random_walk_engine_series(n=400, seed=7):
    # synthetic random walk: unit steps, so the h-step error sd is sqrt(h)
    rng = np.random.default_rng(seed)
    return 100 + np.cumsum(rng.standard_normal(n))


def _iid_series(n=400, seed=5):
    return 10 + np.random.default_rng(seed).standard_normal(n)


def test_random_walk_with_right_width_band_measures_correlated_days_synthetic():
    # the band of the h-th day is the random walk's h-step sd: each day's error is the cumulative sum of steps
    eng = BandEngine(point_fn=lambda c: c[-1], sd_fn=lambda h: np.sqrt(np.arange(1, h + 1)))
    out = R.rolling_backtest(_random_walk_engine_series(), 8, n_origins=30, season=1, engine=eng)
    te = out["total_error"]
    assert te["origins"] == 30
    assert te["variance_ratio"] > 2.0          # correlated days: well above 1 (synthetic walk, seeds 7 to 9: R 5.4 to 5.8)
    assert te["variance_ratio"] <= 8.0         # never above the horizon


def test_iid_noise_around_a_constant_gives_r_near_one_synthetic():
    eng = BandEngine(point_fn=lambda c: 10.0, sd_fn=lambda h: np.ones(h))
    out = R.rolling_backtest(_iid_series(), 8, n_origins=30, season=1, engine=eng)
    te = out["total_error"]
    assert 0.5 <= te["variance_ratio"] <= 1.6


def test_total_error_only_when_enough_origins_and_not_lumpy():
    eng = BandEngine(point_fn=lambda c: 10.0, sd_fn=lambda h: np.ones(h))
    out = R.rolling_backtest(_iid_series(n=60), 8, n_origins=3, season=1, engine=eng)
    assert "total_error" not in out            # fewer than 4 origins
    lumpy = np.zeros(300)
    lumpy[::7] = 5.0
    out = R.rolling_backtest(lumpy, 8, n_origins=12, season=1, engine=eng)
    assert "total_error" not in out            # intermittent demand is not scored this way


def test_held_measured_leaves_the_judged_origin_out():
    # hand-built: origin 0 is the only one with a large total error; with it left out the rest say days are independent
    rng = np.random.default_rng(1)
    h, k = 6, 8
    sd = np.ones(h)
    bands = [np.column_stack([-Z90 * sd, np.zeros((h, 7)), Z90 * sd]) for _ in range(k)]     # (h, 9), unit day sd
    points = [np.zeros(h) for _ in range(k)]
    actuals = [rng.standard_normal(h) for _ in range(k)]
    actuals[0] = np.full(h, 4.0)                 # a big same-direction miss: total 24, centre 0, sd of total sqrt(6) ~ 2.45
    out = R._total_error(actuals, points, bands, h)
    # expected: the judged origin's R is computed without it, so the check reads the others only
    expected = 0
    for j in range(k):
        keep = [i for i in range(k) if i != j]
        a = sum(float(np.sum(actuals[i] - points[i])) ** 2 for i in keep)
        b = sum(float(np.sum((actuals[i] - points[i]) ** 2)) for i in keep)
        r_loo = float(min(max(a / b, 1 / h), h))
        total, centre = float(np.sum(actuals[j])), float(np.sum(points[j]))
        half = Z90 * (r_loo * h) ** 0.5
        expected += int(centre - half <= total <= centre + half)
    assert out["held_measured"] == expected
    # the judged origin 0 has a big total error; with R_loo from the other origins the range is narrow, so it is not held
    assert out["held_measured"] < k


def test_bounds_and_held_counts_follow_report_formulas():
    h, k = 4, 5
    sd = np.ones(h)
    bands = [np.column_stack([np.full(h, -Z90), *([np.zeros(h)] * 7), np.full(h, Z90)]) for _ in range(k)]
    points = [np.zeros(h) for _ in range(k)]
    actuals = [np.zeros(h) for _ in range(k)]
    out = R._total_error(actuals, points, bands, h)
    assert out["held_independent"] == k and out["held_together"] == k and out["held_measured"] == k
    assert out["variance_ratio"] == 1.0        # all errors zero: the fallback ratio


def test_a_different_backtest_horizon_is_extended_at_the_same_average_correlation(monkeypatch):
    # Changed in round 5 (live retail_biz rep3: backtest 20 steps, pack 43, no range applied and the answer mixed the bounds).
    from prophet_agent.runtime import contracts as C
    rows = [{"step": i + 1, "point": 10.0, "q10": 8.0, "q90": 12.0} for i in range(6)]
    report = {"table": {"rows": rows, "rows_omitted": 0}, "total_range_80": {"centre": 60.0}}
    te = {"origins": 12, "variance_ratio": 2.0, "held_measured": 10, "held_independent": 9, "held_together": 12}
    bt_other = {"horizon": 7, "total_error": te, "models": {}}
    hist = [{"kind": "tool", "tool": "backtest_series", "result": bt_other},
            {"kind": "tool", "tool": "forecast_report", "result": report}]
    r = C.measured_total_range(hist)
    # R = 2 over 7 steps: rho = 1/6; over 6 steps R' = 1 + 5/6. Per-day sd = 4 / (2 Z90); total sd = sqrt(R' * 6) * sd.
    sd = 4.0 / (2 * Z90)
    assert r["measured_horizon"] == 7
    assert r["q90"] - r["centre"] == pytest.approx(Z90 * ((1 + 5 / 6) * 6) ** 0.5 * sd)
    stmt = C.trust_statement(hist)
    assert "windows of 7 steps, extended to 6 steps" in stmt
    assert C.measured_total_range([{"kind": "tool", "tool": "forecast_report", "result": report}]) is None


def test_measured_total_range_lies_strictly_between_the_two_bounds():
    from prophet_agent.runtime import contracts as C
    h = 8
    rows = [{"step": i + 1, "point": 10.0, "q10": 8.0, "q90": 12.0} for i in range(h)]   # per-day sd 1.5/1.28 ~ 1.56
    report = {"table": {"rows": rows, "rows_omitted": 0}, "total_range_80": {"centre": 80.0}}
    te = {"origins": 12, "variance_ratio": 3.0, "held_measured": 10, "held_independent": 9, "held_together": 12}
    hist = [{"kind": "tool", "tool": "backtest_series", "result": {"horizon": h, "total_error": te, "models": {}}},
            {"kind": "tool", "tool": "forecast_report", "result": report}]
    mr = C.measured_total_range(hist)
    assert mr is not None and mr["variance_ratio"] == 3.0
    together_lo, together_hi = 8.0 * h, 12.0 * h                  # edges add: 64 to 96
    sd = 4.0 / (2 * Z90)
    half_ind = Z90 * (h * sd * sd) ** 0.5                          # independent half width: about 5.7
    assert together_lo < mr["q10"] < 80.0 - half_ind < 80.0 + half_ind < mr["q90"] < together_hi
    assert mr["held_measured"] == 10 and mr["origins"] == 12


def test_trust_statement_has_the_sentence_only_when_the_range_exists():
    from prophet_agent.runtime.contracts import trust_statement
    bt = {"models": {"timesfm": {"mase": 1.0}, "naive": {"mase": 1.2}}, "best_baseline": "naive", "origins": 12, "horizon": 8,
          "verdict": "timesfm_better", "timesfm_vs_best_baseline_pct": 16.7, "timesfm_range_coverage_80": 0.8}
    rows = [{"step": i + 1, "point": 10.0, "q10": 8.0, "q90": 12.0} for i in range(8)]
    report = {"table": {"rows": rows, "rows_omitted": 0}, "total_range_80": {"centre": 80.0}}
    plain = trust_statement([{"kind": "tool", "tool": "backtest_series", "result": bt}])
    assert "80% range of the" not in plain
    with_total = dict(bt, total_error={"origins": 12, "variance_ratio": 3.0, "held_measured": 10,
                                       "held_independent": 9, "held_together": 12})
    text = trust_statement([{"kind": "tool", "tool": "backtest_series", "result": with_total},
                            {"kind": "tool", "tool": "forecast_report", "result": report}])
    assert "80% range of the 8-step total:" in text
    assert "measured over 12 past windows" in text and "held the actual total in 10 of 12 windows" in text
    assert "random" not in text.lower() and "licen" not in text.lower()
    no_pack = trust_statement([{"kind": "tool", "tool": "backtest_series", "result": with_total}])
    assert "80% range of the" not in no_pack
