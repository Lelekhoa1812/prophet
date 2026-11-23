"""Round 6l (2026-11-23), audit item: a benchmark claim with no best baseline must not pass.

Before the fix, a backtest that stored no best_baseline (or one naming no usable model) produced a passing
answer_benchmark_matches check, and a benchmark claim with no backtest at all produced no check. Both now block.
Every series here is synthetic.
"""
from prophet_agent.runtime import contracts

CLAIM = "Naive is the appropriate benchmark for this series."


def _backtest(best_key=True, best="mean_last_season", models=None):
    result = {"horizon": 14, "origins": 8,
              "models": models if models is not None else {"naive": {"mase": 1.0}, "mean_last_season": {"mase": 0.8},
                                                            "timesfm": {"mase": 0.9}}}
    if best_key:
        result["best_baseline"] = best
    return {"kind": "tool", "tool": "backtest_series", "args": {"horizon": 14}, "result": result}


def _benchmark(checks):
    return [c for c in checks if c["name"] == "answer_benchmark_matches"]


def test_a_backtest_with_no_best_baseline_key_blocks_a_benchmark_claim():
    bad = _benchmark(contracts.check_answer(CLAIM, [_backtest(best_key=False)]))
    assert len(bad) == 1 and bad[0]["ok"] is False and bad[0]["severity"] == "block"
    assert "no usable best baseline" in bad[0]["detail"] and "naive" in bad[0]["detail"]


def test_a_backtest_with_a_none_best_baseline_blocks_a_benchmark_claim():
    bad = _benchmark(contracts.check_answer(CLAIM, [_backtest(best=None)]))
    assert len(bad) == 1 and bad[0]["ok"] is False


def test_a_best_baseline_that_is_not_a_backtested_model_blocks_a_benchmark_claim():
    bad = _benchmark(contracts.check_answer(CLAIM, [_backtest(best="ghost_model")]))
    assert len(bad) == 1 and bad[0]["ok"] is False


def test_a_best_baseline_with_no_finite_mase_blocks_a_benchmark_claim():
    models = {"naive": {"mase": 1.0}, "mean_last_season": {}}
    bad = _benchmark(contracts.check_answer(CLAIM, [_backtest(models=models)]))
    assert len(bad) == 1 and bad[0]["ok"] is False


def test_a_benchmark_claim_with_no_backtest_at_all_blocks_instead_of_passing_silently():
    bad = _benchmark(contracts.check_answer(CLAIM, []))
    assert len(bad) == 1 and bad[0]["ok"] is False and bad[0]["severity"] == "block"
    assert "no backtest is stored" in bad[0]["detail"]


def test_an_answer_with_no_benchmark_claim_adds_no_benchmark_check_without_a_backtest():
    assert _benchmark(contracts.check_answer("The forecast rises over the horizon.", [])) == []


def test_a_benchmark_claim_that_matches_the_backtest_best_still_passes():
    ok = _benchmark(contracts.check_answer("mean_last_season is the appropriate benchmark.", [_backtest()]))
    assert len(ok) == 1 and ok[0]["ok"] is True
