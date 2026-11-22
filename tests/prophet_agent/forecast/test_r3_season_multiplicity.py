"""A2-11 (audit round 3): with no frequency hint the season detector takes the strongest of many lags against a fixed 0.3
bar, so short white-noise series got a false season about 8 percent of the time at n = 60 (measured). The bar now carries
a Bonferroni correction over the candidate lags, floored at 0.3."""
import numpy as np

from prophet_agent.forecast.season import choose_season


def test_season_false_positive_rate_small():
    rng = np.random.default_rng(11)
    hits = sum(choose_season(np.cumsum(rng.normal(size=60)))["season"] != 1 for _ in range(300))
    assert hits / 300 <= 0.05, hits / 300


def test_a_real_weekly_cycle_is_still_found_on_a_short_series():
    rng = np.random.default_rng(2)
    t = np.arange(84)
    x = 10 + 3 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.3, t.size)
    out = choose_season(x)
    assert out["season"] == 7, out
    assert "threshold" in out and out["threshold"] >= 0.3
