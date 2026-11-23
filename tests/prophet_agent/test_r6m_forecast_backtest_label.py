"""R6m, audit item 1 (labelling): run_backtest must say what its coverage figures are measured against.

The per-model coverage is the measured share of actuals inside each model's 10 to 90 band. Its nominal level (80 percent)
was only in a docstring, so the output showed a number with no target beside it. The output now names the nominal level,
and it names the MASE denominator it actually uses (the one-step change, whatever the season), so no figure in the output
has an unstated reference. The rolling backtest uses a seasonal denominator and says so; the two are not the same measure.
"""
from __future__ import annotations

import numpy as np

from prophet_agent.forecast.backtest import BacktestData, run_backtest
from prophet_agent.forecast.baselines import moving_average
from prophet_agent.forecast.types import Forecast


def _ma_predictor(origin):
    points, bands = [], []
    for c in origin.contexts:
        p, q = moving_average(c, origin.horizon, 4)
        points.append(p)
        bands.append(q)
    return Forecast(np.stack(points), np.stack(bands))


def _demand():
    return np.abs(np.random.default_rng(3).normal(20.0, 3.0, size=(3, 60)))


def test_backtest_output_states_the_nominal_coverage_of_its_bands():
    out = run_backtest(lambda: BacktestData(demand=_demand()), {"ma_band": _ma_predictor}, horizon=4, n_origins=6)
    assert out["band_nominal_coverage"] == 0.8
    assert 0.0 <= out["models"]["ma_band"]["all"]["coverage"] <= 1.0


def test_backtest_names_the_mase_denominator_it_uses():
    out = run_backtest(lambda: BacktestData(demand=_demand()), {"ma_band": _ma_predictor}, horizon=4, n_origins=6)
    assert "one-step change" in out["mase_scale"]


def test_verdict_carries_the_nominal_level_beside_the_measured_coverage():
    def naive(origin):
        last = np.array([c[-1] for c in origin.contexts])
        return Forecast(np.repeat(last[:, None], origin.horizon, axis=1), np.repeat(last[:, None, None], 9, axis=-1)
                        .repeat(origin.horizon, axis=1))

    out = run_backtest(lambda: BacktestData(demand=_demand()), {"timesfm": _ma_predictor, "naive": naive}, horizon=4,
                       n_origins=6)
    assert out["verdict"]["band_coverage_nominal"] == 0.8
    assert out["verdict"]["band_coverage_head"] == out["models"]["timesfm"]["head"]["coverage"]
