"""Round 4, A2-13: a seasonal naive band must not widen with sqrt(h) past one season.

Finding A2-13: seasonal naive repeats one season, so its h-step error does not accumulate. The band used Z * sd * sqrt(h)
(a random-walk band), which is too wide at 2 and 3 seasons out. The phase band fixed that with no growth in h.

Changed by the measured-band default (synthetic data, fixed seeds): the default band is now the spread of the rule's own
past h-step errors at each lead (baselines._snaive_lead_sd). That spread is not exactly equal at leads that share a phase,
because each is estimated from its own past errors, so the width-equality assertion became a tolerance on the measured
width. The no-growth property is still asserted, and the phase and sqrt modes are still tested as their own behaviour.
"""
from __future__ import annotations

import numpy as np

from prophet_agent.forecast import baselines


def _periodic_with_noise(seasons: int, season: int, sd: float, seed: int) -> np.ndarray:
    pattern = np.tile(np.arange(season, dtype=float) * 5.0 + 100.0, seasons)   # never near zero, so no floor applies
    return pattern + np.random.default_rng(seed).normal(0.0, sd, pattern.size)


def _widths(q: np.ndarray) -> np.ndarray:
    return q[:, 8] - q[:, 0]


def test_seasonal_band_does_not_grow_past_one_season():
    """Changed: the measured band is estimated per lead, so leads that share a phase differ by sampling noise (measured,
    synthetic, period-4 with noise 2, 60 seasons: the lead-1 width is 5.48 and the lead-13 width 5.23). The test now asserts
    that the mean width of seasons 4 and 5 is within 15 percent of season 1. A sqrt(h) band would fail that: by the square
    root of the leads involved, it would grow by about 2.4 times over that range (a derived figure, not measured)."""
    m = 4
    s = _periodic_with_noise(seasons=60, season=m, sd=2.0, seed=1)
    _, q = baselines.seasonal_naive(s, 4 * m, season=m, k=4)
    w = _widths(q)
    first, last = w[:m].mean(), w[3 * m:4 * m].mean()
    assert abs(last / first - 1.0) < 0.15, (first, last)


def test_seasonal_band_is_phase_specific_when_noise_differs_by_phase():
    m = 4
    rng = np.random.default_rng(2)
    base = np.tile(np.arange(m, dtype=float) * 5.0 + 100.0, 80)
    scale = np.tile(np.array([0.5, 4.0, 0.5, 4.0]), 80)    # the noisy phases are 1 and 3
    s = base + rng.normal(0.0, 1.0, base.size) * scale
    _, q = baselines.seasonal_naive(s, m, season=m, k=4)
    width = _widths(q)
    assert width[1] > 3 * width[0] and width[3] > 3 * width[2]


def test_seasonal_band_keeps_the_old_sqrt_growth_when_asked():
    m = 4
    s = _periodic_with_noise(seasons=60, season=m, sd=2.0, seed=3)
    _, q_old = baselines.seasonal_naive(s, 3 * m + 1, season=m, k=4, band_mode="sqrt")
    width = _widths(q_old)
    assert width[3 * m] > width[0] * 1.5


def test_seasonal_band_keeps_the_phase_mode_when_asked():
    """The phase band (the default before the measured band) is still available by name, with its own sqrt(1 + 1/k) factor
    and no growth in h. Its exact per-lead widths are pinned by the phase-specific test above."""
    m = 4
    s = _periodic_with_noise(seasons=60, season=m, sd=2.0, seed=4)
    _, q = baselines.seasonal_naive(s, 3 * m + 1, season=m, k=4, band_mode="phase")
    width = _widths(q)
    assert np.isclose(width[0], width[3 * m], rtol=1e-9)


def test_measured_band_falls_back_to_the_phase_band_when_too_few_past_errors():
    """A history too short to give any lead MIN_PAIRS same-position errors uses the phase band for every lead, exactly.
    With season 4 and k 4, 36 points give at most five origins per lead, one short of MIN_PAIRS (6)."""
    m = 4
    s = _periodic_with_noise(seasons=9, season=m, sd=2.0, seed=5)[:36]
    _, q_measured = baselines.seasonal_naive(s, 4, season=m, k=4)
    _, q_phase = baselines.seasonal_naive(s, 4, season=m, k=4, band_mode="phase")
    assert np.array_equal(q_measured, q_phase)


def test_measured_band_covers_a_random_walk_and_a_period_7_series_near_80_percent():
    """Changed: the old phase band covered a random walk about half the time, so this asserts the measured band against the
    80 percent label (fixed seed, 600 replications of a 200-point history, horizon 14, season 7, k 4).

    Measured, synthetic, 1000 replications on each of seeds 7, 8 and 9: the measured band covers a random walk 0.74 to 0.77
    (phase band 0.48 to 0.50, sqrt band 0.83 to 0.85) and period 7 with noise sd 2 at 0.79 to 0.80. The random-walk
    coverage is therefore still about five points under its label, because the per-lead spread is estimated from about 24
    past errors. The window 0.70 to 0.85 records that measured shortfall rather than hiding it; it is not a claim that the
    band is calibrated on random walks."""
    n_hist, h, season, reps = 200, 14, 7, 600
    rng = np.random.default_rng(7)
    rw_hits = np.zeros(h)
    pz_hits = np.zeros(h)
    pattern = np.tile(100.0 + 5.0 * np.arange(season), (n_hist + h) // season + 1)[: n_hist + h]
    for _ in range(reps):
        walk = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n_hist + h))
        noisy = pattern + rng.normal(0.0, 2.0, n_hist + h)
        for s, hits in ((walk, rw_hits), (noisy, pz_hits)):
            _, q = baselines.seasonal_naive(s[:n_hist], h, season=season, k=4)
            fut = s[n_hist:]
            hits += (fut >= q[:, 0]) & (fut <= q[:, 8])
    rw_cov, pz_cov = rw_hits.mean() / reps, pz_hits.mean() / reps
    assert 0.70 < rw_cov < 0.85, rw_cov
    assert abs(pz_cov - 0.80) < 0.04, pz_cov
