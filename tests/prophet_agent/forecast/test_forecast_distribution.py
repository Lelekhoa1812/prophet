"""Offline checks for prophet_agent.forecast.distribution (ported from src/lab/engine/distribution.py)."""
import numpy as np
import pytest

from prophet_agent.forecast import distribution as dist


def test_draws_are_uniform_marginals_with_expected_copula_correlation():
    d = dist.Draws.make(n=20000, periods=2, rho=0.1, seed=3)
    assert d.u.shape == (20000, 2)
    assert (d.u > 0).all() and (d.u < 1).all()
    # Gaussian copula with rho = 0.1 gives Spearman (Pearson on uniforms) of (6/pi) * arcsin(rho / 2), about 0.0955
    expected = 6 / np.pi * np.arcsin(0.05)
    assert abs(np.corrcoef(d.u[:, 0], d.u[:, 1])[0, 1] - expected) < 0.02


def test_draws_cache_returns_same_object_for_same_key():
    assert dist.draws(n=50, periods=3) is dist.draws(n=50, periods=3)


def test_constant_quantiles_give_constant_period_samples():
    d = dist.Draws.make(n=100, periods=3)
    qrows = np.full((3, 9), 10.0)
    s = dist.period_samples(qrows, d)
    assert s.shape == (100, 3)
    assert np.allclose(s, 10.0)


def test_subperiod_samples_split_each_period_by_share():
    d = dist.Draws.make(n=5, periods=2)
    qrows = np.full((2, 9), 10.0)
    out = dist.subperiod_samples(qrows, np.array([0, 0, 1]), np.array([0.2, 0.3, 0.5]), d)
    assert out.shape == (5, 3)
    assert np.allclose(out, [2.0, 3.0, 5.0])


def test_shortfall_leftover_and_exceedance_hand_values():
    total = np.array([0.0, 5.0, 10.0])
    assert dist.expected_shortfall(total, 4.0) == pytest.approx(7 / 3)
    assert dist.expected_leftover(total, 6.0) == pytest.approx(7 / 3)
    assert dist.prob_exceeds(total, 4.0) == pytest.approx(2 / 3)
