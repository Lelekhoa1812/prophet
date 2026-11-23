"""Round 6m, goal audit item 3(d): a benchmark claim naming a rule the backtest never ranked must not pass.

Synthetic backtests only.
"""
from __future__ import annotations

from prophet_agent.runtime import contracts


def _backtest(models, best="mean_last_season"):
    return {"kind": "tool", "tool": "backtest_series", "args": {"horizon": 14},
            "result": {"horizon": 14, "origins": 8, "models": models, "best_baseline": best}}


def _benchmark(checks):
    return [c for c in checks if c["name"] == "answer_benchmark_matches"]


def test_a_benchmark_claim_naming_a_rule_absent_from_the_backtest_blocks():
    models = {"mean_last_season": {"mase": 0.8}, "timesfm": {"mase": 0.9}}          # no naive
    bad = _benchmark(contracts.check_answer("Naive is the appropriate benchmark for this series.", [_backtest(models)]))
    assert len(bad) == 1 and bad[0]["ok"] is False and bad[0]["severity"] == "block"
    assert "naive" in bad[0]["detail"].lower()


def test_a_benchmark_claim_naming_a_backtested_best_rule_still_passes():
    models = {"naive": {"mase": 0.8}, "mean_last_season": {"mase": 1.0}}
    good = _benchmark(contracts.check_answer("Naive is the appropriate benchmark for this series.",
                                             [_backtest(models, best="naive")]))
    assert len(good) == 1 and good[0]["ok"] is True
