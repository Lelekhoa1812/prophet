"""M9-1 price baselines: detection, the random-walk comparison on fixed origins, EWMA band coverage and timing numbers.

Every series here is simulated (labelled synthetic in each test). No number in these tests is an observed price.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from prophet_agent.forecast.price import (
    METHODS,
    RW,
    detect_price_series,
    method_forecast,
    price_comparison,
    score_forecasts,
    timing_numbers,
)


def _rw_prices(seed: int, n: int, mu: float = 0.0, sd: float = 0.01, start: float = 100.0) -> np.ndarray:
    """Synthetic geometric random walk (log returns iid normal with mean mu and sd sd)."""
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(mu, sd, n)))


def _garch_prices(seed: int, n: int) -> np.ndarray:
    """Synthetic GARCH(1,1) log returns (omega 2e-6, alpha 0.08, beta 0.90; unconditional sd 1 percent), priced from 100."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 2e-6, 0.08, 0.90
    var = omega / (1 - alpha - beta)
    r = np.empty(n)
    for t in range(n):
        r[t] = math.sqrt(var) * rng.standard_normal()
        var = omega + alpha * r[t] ** 2 + beta * var
    return 100.0 * np.exp(np.cumsum(r))


# detection

def test_detect_accepts_a_positive_persistent_random_walk():
    x = _rw_prices(1, 400)
    det = detect_price_series(x)
    assert det["is_price"] is True
    assert det["lag1_autocorrelation"] > 0.9


def test_detect_refuses_white_noise_around_a_level():
    x = 50.0 + np.random.default_rng(2).normal(0, 1, 400)          # synthetic: mean-reverting, not a price
    det = detect_price_series(np.abs(x))
    assert det["is_price"] is False
    assert any("autocorrelation" in reason for reason in det["reasons"])


def test_detect_refuses_negative_values_and_short_series():
    assert detect_price_series(_rw_prices(3, 400) - 200.0)["is_price"] is False
    assert detect_price_series(_rw_prices(3, 30))["is_price"] is False


def test_detect_uses_dates_when_given():
    x = _rw_prices(4, 200)
    business = np.arange("2024-01-01", "2024-12-31", dtype="datetime64[D]")
    business = business[np.is_busday(business)][:200]
    weekly = np.arange("2020-01-06", "2024-12-31", 7, dtype="datetime64[D]")[:200]
    assert detect_price_series(x, dates=business)["is_price"] is True
    assert detect_price_series(x, dates=weekly)["is_price"] is False
    assert detect_price_series(x, dates=weekly, force=True)["is_price"] is True


# comparison on fixed origins

def test_random_walk_no_method_beats_it_beyond_noise_synthetic():
    x = _rw_prices(11, 1500)                                       # synthetic zero-drift random walk
    res = price_comparison(x, horizon=20, origins_used=12, season=1)
    assert res["no_method_beat_random_walk"] is True
    assert res["beat_random_walk"] == []
    assert res["statement"].startswith("No method beat the random walk")
    assert res["best_method"] in METHODS
    assert set(res["methods"]) == set(METHODS)


def test_strong_drift_is_won_by_the_historical_mean_synthetic():
    x = _rw_prices(12, 1500, mu=0.004, sd=0.01)                    # synthetic: drift 0.4 percent a day, sd 1 percent
    res = price_comparison(x, horizon=20, origins_used=12, season=1)
    assert "hist_mean_drift" in res["beat_random_walk"]
    assert res["no_method_beat_random_walk"] is False
    assert res["best_method"] == "hist_mean_drift"
    assert "hist_mean_drift beat the random walk" in res["statement"]
    assert res["methods"]["hist_mean_drift"]["mase"] < res["methods"][RW]["mase"]


def test_comparison_uses_the_same_origins_for_every_method():
    x = _rw_prices(13, 600)
    res = price_comparison(x, horizon=10, origins_used=6, season=1)
    counts = {m: v["origins_scored"] for m, v in res["methods"].items()}
    assert set(counts.values()) == {6}
    assert res["origins"] == 6


def test_timesfm_points_are_scored_like_any_method_and_can_be_beaten_in_the_statement():
    x = _rw_prices(14, 600)
    h, k = 10, 5
    n = len(x)
    origins = [n - h * (k - j) for j in range(k)]
    # A "model" that returns the random walk's own point: it ties the random walk and cannot beat it.
    z = norm.ppf(np.arange(1, 10) / 10)
    pts = np.stack([np.full(h, x[o - 1]) for o in origins])
    qs = np.stack([x[o - 1] * np.exp(0.02 * np.sqrt(np.arange(1, h + 1)))[:, None] * np.exp(0.02 * z)[None, :]
                   for o in origins])
    res = price_comparison(x, horizon=h, origins_used=k, season=1, timesfm=(pts, qs))
    assert "timesfm" in res["methods"]
    assert "timesfm" not in res["beat_random_walk"]
    assert res["methods"]["timesfm"]["mase"] == res["methods"][RW]["mase"]


# bands

def test_ewma_band_coverage_is_near_80_percent_on_a_garch_like_series_synthetic():
    x = _garch_prices(21, 4000)                                    # synthetic GARCH-like returns; one-step bands
    origins = list(range(600, len(x)))
    fcs = {RW: [method_forecast(x[:o], 1, RW)[:2] for o in origins],
           "ewma_rw": [method_forecast(x[:o], 1, "ewma_rw")[:2] for o in origins]}
    scored = score_forecasts(x, origins, 1, 1, fcs)
    coverage = scored["ewma_rw"]["coverage_80"]
    assert abs(coverage - 0.8) < 0.05, coverage
    # The flat full-history band covers less than EWMA on a series whose volatility moves.
    assert scored[RW]["coverage_80"] < coverage + 0.01


def test_garch_fit_converges_on_garch_like_series_synthetic():
    x = _garch_prices(22, 800)
    med, q, info = method_forecast(x, 5, "garch_rw")
    assert info["fallback"] is None
    assert 0 < info["persistence"] < 1
    assert med.shape == (5,) and q.shape == (5, 9)
    assert np.all(np.diff(q, axis=1) >= 0)                          # quantiles do not cross


def test_method_forecast_quantiles_are_ordered_and_centred_on_the_median():
    x = _rw_prices(23, 300)
    med, q, _ = method_forecast(x, 8, "ewma_rw")
    assert np.all(np.diff(q, axis=1) >= 0)
    assert np.allclose(q[:, 4], med)


# timing

def test_timing_numbers_match_the_fitted_normal_in_log_price():
    last = 100.0
    mu, sd = 0.05, 0.10                                          # log-price drift and sd at one step (synthetic check)
    z = np.array([-1.2815515655, -0.8416212336, -0.5244005127, -0.2533471031, 0.0,
                  0.2533471031, 0.5244005127, 0.8416212336, 1.2815515655])
    steps = np.arange(1, 4)[:, None]                             # three steps: drift and sd grow with the horizon
    q = np.exp(math.log(last) + mu * steps + sd * np.sqrt(steps) * z[None, :])
    out = timing_numbers(q, last)
    expected = norm.cdf(-mu * steps[:, 0] / (sd * np.sqrt(steps[:, 0])))   # P(log price < log today) = Phi(-mu h / (sd sqrt h))
    assert np.allclose(out["prob_below_today_by_step"], expected, atol=1e-3)
    assert out["lowest_median_step"] == 1                        # the median rises with the drift, so the lowest is step 1
    assert out["median_flat"] is False


def test_zero_drift_has_no_lowest_median_step_and_even_odds():
    x = _rw_prices(24, 300)
    med, q, _ = method_forecast(x, 30, RW)
    out = timing_numbers(q, x[-1])
    assert out["median_flat"] is True
    assert out["lowest_median_step"] is None
    assert all(abs(p - 0.5) < 1e-6 for p in out["prob_below_today_by_step"])
    assert "not a prediction of a turning point" in out["basis"]
