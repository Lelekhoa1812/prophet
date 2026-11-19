"""Focused tests: quantile distribution toolkit used by shortfall and freshness maths (business rules)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab.engine import distribution as D  # noqa: E402

Q = np.array([60, 70, 80, 90, 100, 112, 125, 145, 175], dtype=float)   # right-skewed weekly quantiles


def test_samples_reproduce_the_input_quantiles():
    d = D.draws(n=4000, weeks=2)
    s = D.week_samples(np.stack([Q, Q]), d)
    got = np.quantile(s[:, 0], D.LEVELS)
    assert np.allclose(got, Q, rtol=0.05)


def test_mean_sits_above_median_for_skewed_demand():
    d = D.draws(n=4000, weeks=1)
    s = D.week_samples(np.stack([Q]), d)[:, 0]
    assert s.mean() > np.median(s)


def test_weeks_are_lightly_correlated_so_a_two_week_sum_is_narrower_than_scaled_quantiles():
    d = D.draws(n=4000, weeks=2, rho=0.1)
    s = D.week_samples(np.stack([Q, Q]), d)
    one, two = s[:, 0].std(), (s[:, 0] + s[:, 1]).std()
    assert two < 1.6 * one            # perfectly correlated weeks would give 2.0
    assert two > 1.2 * one            # independent weeks would give 1.41


def test_daily_shares_sum_back_to_the_week():
    d = D.draws(n=500, weeks=1)
    share = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0])
    days = D.daily_samples(np.stack([Q]), np.zeros(7, dtype=int), share, d)
    assert np.allclose(days.sum(axis=1), D.week_samples(np.stack([Q]), d)[:, 0])
    assert days[:, 5].sum() == 0


def test_expected_shortfall_and_leftover_are_complements():
    d = D.draws(n=3000, weeks=1)
    total = D.week_samples(np.stack([Q]), d)[:, 0]
    x = 100.0
    assert abs((D.expected_shortfall(total, x) - D.expected_leftover(total, x)) - (total.mean() - x)) < 1e-6
    assert D.expected_shortfall(total, 1e9) == 0.0
    assert abs(D.expected_shortfall(total, 0.0) - total.mean()) < 1e-6
    assert 0.0 < D.prob_exceeds(total, x) < 1.0
