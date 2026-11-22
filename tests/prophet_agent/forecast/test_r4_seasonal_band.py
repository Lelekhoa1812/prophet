"""Round 4, A2-13: a seasonal naive band must not widen with sqrt(h) past one season.

Finding A2-13: seasonal naive repeats one season, so its h-step error does not accumulate. The band used Z * sd * sqrt(h)
(a random-walk band), which is too wide at 2 and 3 seasons out. The fix uses the residual sd of the lag-season difference
for each phase of the season, with no growth in h.
"""
from __future__ import annotations

import numpy as np

from prophet_agent.forecast import baselines


def _periodic_with_noise(seasons: int, season: int, sd: float, seed: int) -> np.ndarray:
    pattern = np.tile(np.arange(season, dtype=float) * 5.0 + 100.0, seasons)   # never near zero, so no floor applies
    return pattern + np.random.default_rng(seed).normal(0.0, sd, pattern.size)


def test_seasonal_band_constant_past_one_season():
    m = 4
    s = _periodic_with_noise(seasons=60, season=m, sd=2.0, seed=1)
    _, q = baselines.seasonal_naive(s, 3 * m + 1, season=m, k=4)
    width = q[:, 8] - q[:, 0]
    # Horizon 1 and horizon 3 seasons + 1 fall on the same phase: the band must be the same width.
    assert np.isclose(width[0], width[3 * m], rtol=1e-9)
    # And within a season the band does not grow with the horizon either.
    assert np.allclose(width[:m], width[0], rtol=0.2)


def test_seasonal_band_is_phase_specific_when_noise_differs_by_phase():
    m = 4
    rng = np.random.default_rng(2)
    base = np.tile(np.arange(m, dtype=float) * 5.0 + 100.0, 80)
    scale = np.tile(np.array([0.5, 4.0, 0.5, 4.0]), 80)    # the noisy phases are 1 and 3
    s = base + rng.normal(0.0, 1.0, base.size) * scale
    _, q = baselines.seasonal_naive(s, m, season=m, k=4)
    width = q[:, 8] - q[:, 0]
    assert width[1] > 3 * width[0] and width[3] > 3 * width[2]


def test_seasonal_band_keeps_the_old_sqrt_growth_when_asked():
    m = 4
    s = _periodic_with_noise(seasons=60, season=m, sd=2.0, seed=3)
    _, q_old = baselines.seasonal_naive(s, 3 * m + 1, season=m, k=4, band_mode="sqrt")
    width = q_old[:, 8] - q_old[:, 0]
    assert width[3 * m] > width[0] * 1.5
