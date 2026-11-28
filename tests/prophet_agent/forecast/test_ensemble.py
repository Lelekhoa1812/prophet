"""Mission 10, M10-1: candidates scored on the same origins and combined with weights earned before each origin.

All series here are synthetic (seeded), built so the right answer is known.
"""
import numpy as np
import pytest

from prophet_agent.forecast import ensemble as en
from prophet_agent.forecast.metrics import QS
from scipy.stats import norm

Z = norm.ppf(QS)


def _band(centre, sd):
    centre = np.asarray(centre, dtype=float)
    sd = np.broadcast_to(np.asarray(sd, dtype=float), centre.shape)
    return centre[:, None] + sd[:, None] * Z[None, :]


def _walk(n=160, seed=0, drift=0.5):
    rng = np.random.default_rng(seed)
    return np.cumsum(drift + rng.normal(0, 1, n)) + 100


def _drift_cand(y, name="drift", sd=1.0):
    def f(t, h):
        x = y[:t]
        d = (x[-1] - x[0]) / (len(x) - 1)
        return _band(x[-1] + d * np.arange(1, h + 1), sd * np.sqrt(np.arange(1, h + 1)))
    return en.Candidate(name, "rule", f)


def _flat_cand(y, name="mean", sd=1.0):
    def f(t, h):
        return _band(np.full(h, y[:t].mean()), sd * np.ones(h))
    return en.Candidate(name, "rule", f)


def test_weights_favour_the_better_candidate_and_beat_the_worse():
    y = _walk()
    res = en.run(y, 6, [_drift_cand(y), _flat_cand(y)], n_origins=20, step=3)
    w = {r["name"]: r["weight"] for r in res["weights"]}
    assert w["drift"] > 0.6 > w["mean"]
    table = {r["name"]: r for r in res["candidates"]}
    assert table["drift"]["wis"] < table["mean"]["wis"]
    assert res["combination"]["wis"] < table["mean"]["wis"]
    assert res["shrinkage"]["lam"] == pytest.approx(20 / (20 + en.KAPPA), abs=1e-4)


def test_few_origins_stay_near_equal_weights():
    w, lam = en.weights(np.ones((2, 2, 4)), 2, 4)
    assert lam == 0.0 and np.allclose(w, 0.5)
    rows = np.stack([np.array([[1.0] * 4, [3.0] * 4])] * 3)        # 3 origins: first one where weights may move
    w, lam = en.weights(rows, 2, 4)
    assert lam == pytest.approx(3 / (3 + en.KAPPA))
    assert 0.5 < w[0, 0] < 0.75                                    # raw would be 0.75; shrinkage pulls it towards 0.5


def test_weights_never_see_their_own_window():
    """A candidate that is perfect only on the newest windows must not get weight at origins before those windows ended."""
    y = _walk(seed=3)
    seen = []

    def peek(t, h):
        seen.append(t)
        return _band(y[t:t + h] if t + h <= len(y) else np.full(h, y[t - 1]), 0.1 * np.ones(h))

    res = en.run(y, 4, [_drift_cand(y), en.Candidate("oracle", "model", peek)], n_origins=10, step=4)
    # The oracle is perfect at every scored origin, yet the first scored origin used only complete earlier windows.
    first = res["origins"][en.MIN_WEIGHT_ORIGINS]
    assert first - res["origins"][en.MIN_WEIGHT_ORIGINS - 1] >= 4
    assert res["scored_origins"] == len(res["origins"]) - en.MIN_WEIGHT_ORIGINS


def test_failing_candidate_is_left_out_not_fatal():
    y = _walk(seed=5)

    def broken(t, h):
        if t % 2:
            raise RuntimeError("no data at this origin")
        return _band(np.full(h, y[t - 1]), np.ones(h))

    res = en.run(y, 4, [_drift_cand(y), en.Candidate("broken", "market", broken)], n_origins=12, step=3)
    assert "broken" in res["failures"]
    assert np.isfinite(res["forecast"]["median"]).all()


def test_narrow_band_is_widened_by_conformal_scales():
    y = _walk(seed=7)
    res = en.run(y, 6, [_drift_cand(y, sd=0.05), _drift_cand(y, "drift2", sd=0.06)], n_origins=24, step=2)
    assert res["band"]["coverage_scored"] < 0.75
    assert res["band"]["widened"] and max(res["band"]["scales"]) > 1.0
    q = np.array(res["forecast"]["quantiles"])
    raw = np.array(res["members"]["drift"])
    assert (q[:, 8] - q[:, 0]).mean() > (raw[:, 8] - raw[:, 0]).mean()


def test_refuses_short_series_and_single_candidate():
    y = _walk(n=40)
    with pytest.raises(ValueError, match="at least two"):
        en.run(y, 4, [_drift_cand(y)])
    with pytest.raises(ValueError, match="too short"):
        en.run(y, 12, [_drift_cand(y), _flat_cand(y)], min_train=24)
