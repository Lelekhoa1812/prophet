"""Offline walk-forward backtest checks (ported from src/lab/engine/backtest.py), using a tiny synthetic matrix and stubs."""
import numpy as np
import pytest

from prophet_agent.forecast.backtest import BacktestData, run_backtest
from prophet_agent.forecast.types import Forecast

HORIZON = 2
N_PERIODS = 30


@pytest.fixture
def demand():
    rng = np.random.default_rng(7)
    return rng.integers(1, 20, size=(3, N_PERIODS)).astype(float)


def _as_forecast(point):
    q = np.repeat(point[..., None], 9, axis=-1)
    return Forecast(point, q)


def _naive(origin):
    """Last observed value repeated over the horizon. Uses only the history it is given."""
    last = np.array([c[-1] for c in origin.contexts])
    return _as_forecast(np.repeat(last[:, None], origin.horizon, axis=1))


def _truth_blend(demand, weight):
    """Predictor that mixes the true future with the naive forecast. Cheats on purpose, to test scoring."""
    def predict(origin):
        truth = demand[origin.active, origin.index:origin.index + origin.horizon]
        naive = _naive(origin).point
        return _as_forecast(weight * truth + (1 - weight) * naive)
    return predict


def _loader(demand, weights=None, segments=None):
    return lambda: BacktestData(demand=demand, weights=weights, segments=segments)


def test_origins_are_the_last_n_positions_before_the_horizon(demand):
    out = run_backtest(_loader(demand), {"naive": _naive}, horizon=HORIZON, n_origins=3, min_history=8)
    assert out["origins"] == [26, 27, 28]
    assert out["horizon"] == HORIZON
    assert out["n_series"] == 3


def test_predictors_only_see_history_before_each_origin(demand):
    seen = []

    def spy(origin):
        assert all(len(c) == origin.index for c in origin.contexts)
        seen.append(origin.index)
        return _naive(origin)

    run_backtest(_loader(demand), {"spy": spy}, horizon=HORIZON, n_origins=4)
    assert seen == [25, 26, 27, 28]


def test_covariates_are_injected_and_reach_the_predictor(demand):
    calls = []

    def provider(o, active, h):
        calls.append((o, len(active), h))
        return [np.full(o + h, 1.0) for _ in active]

    def check(origin):
        assert origin.covariates is not None and len(origin.covariates) == len(origin.active)
        assert origin.covariates[0].shape == (origin.index + origin.horizon,)
        return _naive(origin)

    run_backtest(_loader(demand), {"naive": check}, horizon=HORIZON, n_origins=2, covariates=provider)
    assert [c[0] for c in calls] == [27, 28]
    assert all(c[2] == HORIZON for c in calls)


def test_verdict_reports_measured_improvement_without_a_pass_fail_flag(demand):
    out = run_backtest(_loader(demand), {"timesfm": _truth_blend(demand, 0.5), "naive": _naive},
                       horizon=HORIZON, n_origins=5)
    v = out["verdict"]
    assert v["best_baseline"] == "naive"
    assert v["challenger_head_wape"] == pytest.approx(0.5 * v["best_baseline_head_wape"])
    # halving the naive error halves the WAPE, so the measured improvement is exactly one half
    assert v["improvement_vs_best_baseline"] == pytest.approx(0.5)
    assert "passes_error_bar" not in v and "ship_bar" not in v


def test_no_verdict_without_the_challenger(demand):
    out = run_backtest(_loader(demand), {"naive": _naive}, horizon=HORIZON, n_origins=3)
    assert "verdict" not in out


def test_weights_define_the_head_set_and_segments_are_scored(demand):
    weights = np.array([100.0, 1.0, 1.0])
    segments = np.array(["a", "a", "b"])
    out = run_backtest(_loader(demand, weights, segments), {"naive": _naive}, horizon=HORIZON, n_origins=3,
                       head_share=0.8, min_segment_rows=1)
    assert out["head_share_series"] == 1
    assert out["models"]["naive"]["head"]["wape"] >= 0
    assert set(out["models"]["naive"]["by_segment"]) == {"a"}
