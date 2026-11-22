"""Round 3, A2-09: the backtest scores TimesFM's stored quantiles with the weighted interval score (WIS), next to MASE.

Finding A2-09 (inferred from code, reproduced by grep): the only proper rule was a truncated pinball average. WIS (Bracher,
Ray, Gneiting and Reich 2021, arXiv 2005.12881) uses the nine stored quantiles as four central intervals plus the median.
Note on A2-09's suggested closed form: for a zero-width band with the actual outside it, every interval term is
(2/alpha) * |y - m|, so WIS = (0.5 + 4) |y - m| / 4.5, not 0.5 |y - m| / 4.5 as the audit line says. These tests use the
formula from the paper.
"""
from __future__ import annotations

import numpy as np

from prophet_agent.forecast.types import Forecast


def _q_const(m, n=9):
    return np.full(n, float(m))


def test_wis_closed_form_for_a_zero_width_band_with_the_actual_above_it():
    from prophet_agent.forecast.metrics import wis
    # median 0, band collapsed at 0, actual 2: the median term is 0.5 * 2; each interval term is (2 / alpha) * 2 weighted by
    # alpha / 2, so 2 per level and 8 for four levels. Total 9, divided by K + 1/2 = 4.5.
    assert abs(wis(np.array(2.0), _q_const(0.0)) - 9 / 4.5) < 1e-12


def test_wis_is_zero_for_a_perfect_point_with_no_spread():
    from prophet_agent.forecast.metrics import wis
    assert wis(np.array(3.0), _q_const(3.0)) == 0.0


def test_wis_is_lower_for_the_true_quantiles_than_for_a_shifted_set():
    from prophet_agent.forecast.metrics import QS, wis
    from statistics import NormalDist
    rng = np.random.default_rng(4)
    y = rng.standard_normal(4000)
    true_q = np.array([NormalDist().inv_cdf(p) for p in QS])
    mean_true = np.mean([wis(v, true_q) for v in y[:400]])
    mean_shift = np.mean([wis(v, true_q + 0.5) for v in y[:400]])
    mean_wide = np.mean([wis(v, true_q * 2.0) for v in y[:400]])
    assert mean_true < mean_shift and mean_true < mean_wide


def test_backtest_reports_wis_scaled_alongside_mase():
    from prophet_agent.forecast import rolling as R

    class Engine:
        def forecast_batch(self, contexts, horizon):
            point = np.stack([np.full(horizon, float(c[-1])) for c in contexts])
            q = np.stack([p[:, None] + np.linspace(-2, 2, 9)[None, :] for p in point])
            return Forecast(point, q)

    x = 50 + np.cumsum(np.random.default_rng(9).standard_normal(120))
    out = R.rolling_backtest(x, 4, n_origins=8, season=1, engine=Engine())
    assert np.isfinite(out["models"]["timesfm"]["wis_scaled"])
    assert out["models"]["timesfm"]["wis_scaled"] > 0
    assert out["timesfm_wis_scaled"] == out["models"]["timesfm"]["wis_scaled"]
