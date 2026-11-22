import numpy as np

from prophet_agent.forecast.randomwalk import random_walk_test


def _price(r):
    return 100 * np.exp(np.cumsum(r))


def test_a_random_walk_with_volatility_clusters_is_recognised():
    rng = np.random.default_rng(1)
    vol = np.repeat(rng.choice([0.005, 0.03], size=40), 50)        # calm and stormy regimes
    res = random_walk_test(_price(rng.normal(0, vol)))
    assert res["verdict"] == "consistent_with_random_walk" and res["changes"] == "log returns"
    assert abs(res["by_horizon"]["10"]["variance_ratio"] - 1) < 0.25


def test_reverting_and_persistent_changes_are_told_apart():
    rng = np.random.default_rng(2)
    e = rng.normal(0, 0.01, 2000)
    rev, per = np.zeros(2000), np.zeros(2000)
    for t in range(1, 2000):
        rev[t] = -0.4 * rev[t - 1] + e[t]
        per[t] = 0.4 * per[t - 1] + e[t]
    assert random_walk_test(_price(rev))["verdict"] == "mean_reverting"
    assert random_walk_test(_price(per))["verdict"] == "persistent"


def test_short_constant_or_signed_series():
    assert random_walk_test(np.arange(50.0)) is None
    assert random_walk_test(np.ones(300)) is None
    rng = np.random.default_rng(3)
    res = random_walk_test(np.cumsum(rng.normal(0, 1, 500)))       # crosses zero: plain differences
    assert res["changes"] == "differences" and res["verdict"] == "consistent_with_random_walk"


def test_the_trust_paragraph_says_so():
    from prophet_agent.runtime.contracts import trust_statement
    bt = {"models": {"timesfm": {"mase": 1.05}, "naive": {"mase": 1.0}}, "best_baseline": "naive", "origins": 12, "horizon": 20,
          "verdict": "baseline_better", "timesfm_range_coverage_80": 0.8, "random_walk": {"verdict": "consistent_with_random_walk"}}
    text = trust_statement([{"kind": "tool", "tool": "backtest_series", "result": bt}])
    assert "random walk" in text and "range, not the line" in text
