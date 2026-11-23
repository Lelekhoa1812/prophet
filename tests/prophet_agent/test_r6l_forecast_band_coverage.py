"""R6l, audit item 1: the seasonal naive phase band must cover close to its stated 80 percent (the 10 to 90 quantiles).

The forecast is a mean of k same-phase values, so its error is sigma * sqrt(1 + 1/k), not sigma. The band used sigma alone
and covered about 70 percent on a synthetic period-4 series (measured, synthetic). The band now carries the factor, and its
estimate of sigma uses the unbiased (ddof=1) sd, which brings the coverage to about 78 percent at 48 points of history.
"""
from __future__ import annotations

import numpy as np

from prophet_agent.forecast.baselines import seasonal_naive
from prophet_agent.forecast.metrics import coverage


def test_seasonal_naive_phase_band_covers_close_to_its_nominal_80_percent():
    pattern = np.array([10.0, 14.0, 7.0, 12.0])
    rng = np.random.default_rng(20)
    hits = []
    for _ in range(1000):
        history = np.tile(pattern, 12) + rng.normal(0.0, 2.0, 48)
        future = np.tile(pattern, 2) + rng.normal(0.0, 2.0, 8)
        _, q = seasonal_naive(history, 8, season=4, k=4)
        hits.append(coverage(future, q))
    assert np.mean(hits) >= 0.75, np.mean(hits)


def test_band_width_follows_the_number_of_same_phase_values_averaged():
    # Same history, so the same residual sd per phase. Averaging 4 values has error sigma * sqrt(1.25); one value has
    # error sigma * sqrt(2). The width ratio at horizon 1 is therefore sqrt(2) / sqrt(1.25).
    rng = np.random.default_rng(3)
    s = np.tile(np.array([10.0, 14.0, 7.0, 12.0]), 12) + rng.normal(0.0, 2.0, 48)
    _, q1 = seasonal_naive(s, 1, season=4, k=1)
    _, q4 = seasonal_naive(s, 1, season=4, k=4)
    ratio = (q1[0, 8] - q1[0, 0]) / (q4[0, 8] - q4[0, 0])
    assert np.isclose(ratio, np.sqrt(2.0) / np.sqrt(1.25), rtol=1e-9)
