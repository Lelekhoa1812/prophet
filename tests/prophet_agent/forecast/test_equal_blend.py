"""Equal-weight blend of TimesFM's centre and the best simple rule: an exact mean, adopted only when the backtest on the same
windows says it beats both, and never fitted on the errors it is scored on.

The engines are fakes. On a series that alternates 90 and 110, the mean of the last season is 100 and misses by 10 on every
window. The mirror engine sets TimesFM's point to 2 * actual - 100 (it misses by 10 the other way), so the equal blend lands
exactly on the actual. The bias engine adds 30 to the actual. No number here is a measurement of TimesFM."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from prophet_agent.forecast import combine as C
from prophet_agent.forecast import rolling as R
from prophet_agent.forecast.types import Forecast

KW = dict(seed=20260, confidence=0.9, resamples=2000, win_share=0.6)


class _FakeEngine:
    """Forecasts each window as a function of its actual next values: mode 'mirror', 'perfect' or 'bias'."""

    def __init__(self, x, mode):
        self.x, self.mode = np.asarray(x, dtype=float), mode

    def forecast_batch(self, contexts, horizon):
        pts = []
        for ctx in contexts:
            y = self.x[len(ctx):len(ctx) + horizon]
            pts.append({"mirror": 2 * y - 100, "perfect": y.copy(), "bias": y + 30}[self.mode])
        point = np.stack(pts)
        return Forecast(point, np.repeat(point[..., None], 9, axis=-1))


def _alternating(n=60):
    return np.where(np.arange(n) % 2 == 0, 90.0, 110.0)


# (a) the blend is the mean of the two points, exactly

def test_the_equal_blend_is_the_exact_mean_of_the_two_points():
    a = np.array([1.0, 2.5, -3.0, 10.0])
    b = np.array([3.0, 0.5, 1.0, 4.0])
    assert np.array_equal(C.equal_blend(a, b), np.array([2.0, 1.5, -1.0, 7.0]))
    assert np.array_equal(C.equal_blend(b, a), C.equal_blend(a, b))


def test_the_two_points_must_cover_the_same_horizon():
    with pytest.raises(ValueError, match="same horizon"):
        C.equal_blend(np.ones(4), np.ones(5))


# (b) adopted only when it wins on the backtest windows, and not when it loses

def test_the_blend_is_adopted_only_when_it_beats_timesfm_and_the_rule():
    t, r = np.full(12, 10.0), np.full(12, 10.0)
    assert C.adopt_equal_blend(t, np.full(12, 4.0), r, **KW)["adopt"] is True
    assert C.adopt_equal_blend(t, np.full(12, 6.0), np.full(12, 1.0), **KW)["adopt"] is False   # beats TimesFM, not the rule
    assert C.adopt_equal_blend(np.full(12, 1.0), np.full(12, 6.0), np.full(12, 10.0), **KW)["adopt"] is False   # beats the rule, not TimesFM
    assert C.adopt_equal_blend(t, t, r, **KW)["adopt"] is False                                  # a tie is not a win


def test_a_big_win_on_a_few_windows_is_not_enough():
    t = np.full(12, 10.0)
    blend = np.concatenate([np.zeros(3), np.full(9, 10.5)])     # better by 10 on 3 windows, worse by 0.5 on 9
    out = C.adopt_equal_blend(t, blend, np.full(12, 12.0), **KW)
    assert out["vs_timesfm"]["wins"] == 3 and out["adopt"] is False


def test_the_backtest_adopts_the_blend_when_it_wins_on_every_window():
    x = _alternating()
    out = R.rolling_backtest(x, 1, n_origins=12, season=1, engine=_FakeEngine(x, "mirror"), blend_centre=True)
    ce = out["centre_choice"]
    assert ce["rule"] == "mean_last_season" and ce["adopt"] is True
    assert out["models"]["timesfm_equal_blend"]["mase"] == 0.0
    assert out["models"]["timesfm"]["mase"] == pytest.approx(0.5)
    assert out["models"]["mean_last_season"]["mase"] == pytest.approx(0.5)


def test_the_backtest_does_not_adopt_the_blend_when_timesfm_is_perfect():
    x = _alternating()
    out = R.rolling_backtest(x, 1, n_origins=12, season=1, engine=_FakeEngine(x, "perfect"), blend_centre=True)
    assert out["centre_choice"]["adopt"] is False
    assert out["models"]["timesfm"]["mase"] == 0.0


# (d) not fitted on the errors it is scored on, beyond the selection stated in (c)

def test_the_equal_blend_takes_no_errors_so_it_cannot_be_fitted():
    assert list(inspect.signature(C.equal_blend).parameters) == ["point_a", "point_b"]


def test_the_backtest_blend_uses_equal_weights_not_fitted_ones():
    # TimesFM is 30 too high and the rule is 10 low: inverse-MSE weights would lean on the rule. Equal weights do not.
    x = _alternating()
    out = R.rolling_backtest(x, 1, n_origins=12, season=1, engine=_FakeEngine(x, "bias"), blend_centre=True)
    origins = [len(x) - 12 + j for j in range(12)]
    expected = np.mean([abs(((x[o] + 30) + 100) / 2 - x[o]) / 20 for o in origins])
    assert out["models"]["timesfm_equal_blend"]["mase"] == pytest.approx(expected)
    assert expected == pytest.approx(0.75)


def test_the_choice_states_that_the_rule_and_the_adoption_were_selected_on_these_windows():
    x = _alternating()
    ce = R.rolling_backtest(x, 1, n_origins=12, season=1, engine=_FakeEngine(x, "mirror"), blend_centre=True)["centre_choice"]
    assert ce["selected_on_these_windows"] is True
    assert "not an unbiased" in ce["caveat"] and "optimistic" in ce["caveat"]


def test_without_the_flag_the_backtest_output_is_unchanged():
    x = _alternating()
    out = R.rolling_backtest(x, 1, n_origins=12, season=1, engine=_FakeEngine(x, "mirror"))
    assert "timesfm_equal_blend" not in out["models"] and "centre_choice" not in out
