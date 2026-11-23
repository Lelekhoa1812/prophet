"""R6l, audit item 2: seasonal naive must not silently drop to the mean of one season past four seasons.

The point averaged the same-phase values of the last k seasons that lie in the history. Past k seasons ahead none of those
lags was in the history, so the point fell back to the mean of the last season, which has no phase at all. On a period-4
series a horizon of 20 gave 4.0 for steps 17 to 20 where the repeating pattern says 1, 5, 2, 8 (measured, synthetic).
"""
from __future__ import annotations

import numpy as np

from prophet_agent.forecast.baselines import moving_average, seasonal_naive


def test_point_keeps_its_phase_far_beyond_the_history():
    pattern = np.array([1.0, 5.0, 2.0, 8.0])
    s = np.tile(pattern, 10)                      # 40 points, five seasons of history
    point, _ = seasonal_naive(s, 20, season=4)    # horizon 20: five seasons ahead
    assert np.allclose(point, np.tile(pattern, 5))


def test_point_averages_the_most_recent_k_same_phase_values_at_every_horizon():
    s = np.arange(40, dtype=float)                # a ramp: the same-phase values differ by 4 per season
    point, _ = seasonal_naive(s, 12, season=4, k=4)
    # Step i (0-based) has phase i % 4 and the last four same-phase values are 4 seasons back from its own season.
    expected = np.array([np.mean([s[len(s) + i - 4 * j] for j in range(i // 4 + 1, i // 4 + 5)]) for i in range(12)])
    assert np.allclose(point, expected)


def test_a_history_shorter_than_one_season_uses_the_documented_moving_average():
    s = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    point, q = seasonal_naive(s, 3, season=7)
    p_ma, q_ma = moving_average(s, 3, 7)
    assert np.allclose(point, p_ma) and np.allclose(q, q_ma)
