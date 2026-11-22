"""Round 3, A6-01 and A6-02: for intermittent demand the backtest scores the delivered object (TSB expected demand and the TSB
band), and calibration never puts a TimesFM band over it.

Findings (code-read, A6): the delivered point for lumpy demand is the TSB expected volume and its band is TSB's quantiles
(tools/timesfm_tool.py). The backtest scored TimesFM's median and TimesFM's band (A6-01), and calibrate() then rebuilt a band
around TimesFM's median (A6-02). The stub engines here return a near-zero median and a narrow band, as TimesFM does on
lumpy demand, so the two bands disagree on purpose.
"""
from __future__ import annotations

import numpy as np

from prophet_agent.forecast import rolling as R
from prophet_agent.forecast.tsb import expected_volume, tsb_band
from prophet_agent.forecast.types import Forecast

HORIZON = 4
N_ORIGINS = 10


def _lumpy(n=200, seed=21):
    rng = np.random.default_rng(seed)
    occurs = rng.random(n) < 0.2
    size = rng.lognormal(mean=np.log(10.0), sigma=0.3, size=n)
    return np.where(occurs, size, 0.0)


class NarrowTimesFM:
    """A model whose median is near zero and whose band is tight, as TimesFM gives on lumpy demand."""

    def __init__(self):
        self.calls = 0

    def forecast_batch(self, contexts, horizon):
        self.calls += 1
        point = np.zeros((len(contexts), horizon))
        q = np.repeat(np.linspace(-0.5, 0.5, 9)[None, None, :], len(contexts), axis=0)
        q = np.repeat(q, horizon, axis=1)
        return Forecast(point, q)

    def forecast(self, context, horizon):
        f = self.forecast_batch([context], horizon)
        return Forecast(f.point[0], f.q[0])


def test_backtest_scores_the_delivered_tsb_band_for_lumpy_demand():
    x = _lumpy()
    out = R.rolling_backtest(x, HORIZON, n_origins=N_ORIGINS, season=1, engine=NarrowTimesFM())
    assert out["point_basis"].startswith("expected demand")
    k = out["origins"]
    origins = [len(x) - HORIZON * (k - j) for j in range(k)]
    covered = []
    for o in origins:
        q = tsb_band(x[:o], HORIZON)
        act = x[o:o + HORIZON]
        covered.append(float(((act >= q[:, 0]) & (act <= q[:, 8])).mean()))
    assert abs(out["timesfm_range_coverage_80"] - round(float(np.mean(covered)), 3)) < 1e-9


def test_backtest_point_is_the_expected_volume_and_the_median_is_kept_separately():
    x = _lumpy()
    out = R.rolling_backtest(x, HORIZON, n_origins=N_ORIGINS, season=1, engine=NarrowTimesFM())
    tsb = out["models"]["tsb_expected"]["mase"]
    assert out["models"]["timesfm"]["mase"] == tsb
    assert "timesfm_median" in out["models"]
    assert out["models"]["timesfm_median"]["mase"] != out["models"]["timesfm"]["mase"]


def test_the_delivered_point_sits_inside_its_own_band_at_every_origin():
    x = _lumpy()
    k = 6
    for o in [len(x) - HORIZON * (k - j) for j in range(k)]:
        q = tsb_band(x[:o], HORIZON)
        point = expected_volume(x[:o], HORIZON)
        assert np.all((point >= q[:, 0]) & (point <= q[:, 8]))


def test_calibrate_keeps_the_tsb_band_for_lumpy_demand_and_never_calls_timesfm():
    from prophet_agent.forecast.calibrate import calibrate
    x = _lumpy()
    engine = NarrowTimesFM()
    res = calibrate(x, HORIZON, engine=engine)
    assert engine.calls == 0
    assert res["method"] == "tsb_delivered_band" and res["recommend"] == "keep_raw"
    lo, hi = np.asarray(res["calibrated"]["q10"]), np.asarray(res["calibrated"]["q90"])
    point = np.asarray(res["point"])
    assert np.all((point >= lo) & (point <= hi))
    assert np.allclose(point, expected_volume(x, HORIZON), atol=1e-3)


class ZeroMedian:
    """TimesFM's median on lumpy demand: zero at every step."""

    def forecast_batch(self, contexts, horizon):
        point = np.zeros((len(contexts), horizon))
        return Forecast(point, np.repeat(point[..., None], 9, axis=-1))


def test_mae_prefers_the_zero_median_but_mse_prefers_the_expected_volume():
    # Audit A2-08 (Gneiting 2011): a mean forecast scored by MAE loses to the median on lumpy demand, so the scores are
    # reported both ways. Selection is still by MASE (A2-08 leaves that to the user); this checks the two scores disagree.
    rng = np.random.default_rng(11)
    x = np.where(rng.random(400) < 0.3, rng.poisson(6, 400) + 1, 0).astype(float)
    m = R.rolling_backtest(x, 30, n_origins=8, season=1, engine=ZeroMedian())["models"]
    assert m["timesfm_median"]["mase"] < m["tsb_expected"]["mase"]
    assert m["tsb_expected"]["mse_scaled"] < m["timesfm_median"]["mse_scaled"]
