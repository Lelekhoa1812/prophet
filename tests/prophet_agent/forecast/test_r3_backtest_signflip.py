"""Round 3, A2-06: the paired test that adopts a winner must hold its nominal level at the small origin counts the backtest uses.

Finding A2-06 (measured): the percentile bootstrap of the mean paired difference rejected a true null at 0.089 with 6 origins
(0.086 at 8, 0.068 at 12) against a nominal one-sided 0.05. An exact sign-flip test over the same kind of data holds the
level (3/64 = 0.047 at 6 origins). These tests reproduce the null rate through the verdict and the adoption rule.
"""
from __future__ import annotations

import numpy as np

from prophet_agent.forecast import rolling as R
from prophet_agent.forecast.combine import _beats


def _null_pairs(k, trials, seed):
    """Exchangeable errors: both forecasts have the same law, so the true gain is zero."""
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        yield np.abs(rng.standard_normal(k)) + 1.0, np.abs(rng.standard_normal(k)) + 1.0


def test_verdict_null_rate_at_six_origins_is_at_most_six_percent():
    hits = sum(R._verdict(b, c)[0]["verdict"] == "timesfm_better" for b, c in _null_pairs(6, 2000, seed=1))
    assert hits / 2000 <= 0.06


def test_adoption_null_rate_at_six_origins_is_at_most_six_percent():
    hits = sum(_beats(b, c, seed=1, confidence=0.9, resamples=2000, win_share=0.6)["beats"]
               for b, c in _null_pairs(6, 2000, seed=2))
    assert hits / 2000 <= 0.06


def test_exact_test_enumerates_every_sign_pattern():
    from prophet_agent.forecast.metrics import paired_signflip
    # diffs 1,2,3,4 (mean 2.5). Only the all-plus pattern reaches a total of 10, so the one-sided p is 1/16.
    out = paired_signflip(np.array([1.0, 2.0, 3.0, 4.0]), np.zeros(4))
    assert out["method"] == "signflip_exact"
    assert abs(out["p_cand_better"] - 1 / 16) < 1e-12
    assert out["p_base_better"] == 1.0          # every pattern gives a total at most the observed 10


def test_monte_carlo_is_used_above_sixteen_pairs():
    from prophet_agent.forecast.metrics import paired_signflip
    base = np.linspace(1.0, 2.0, 20)
    out = paired_signflip(base + 0.5, base)
    assert out["method"] == "signflip_montecarlo" and 0 < out["p_cand_better"] <= 1
