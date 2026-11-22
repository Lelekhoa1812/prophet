"""C3a, A6-09: narrative claims in the written answer are checked against the stored tool output.

Live evidence (measured, audit round 3, A6): retail rep4 answered "weekly (7-day) confirmed" while its health record measured
period 5; WTI answered "naive is the appropriate benchmark" while its backtest ranked mean_last_season ahead. Both claims reached
the person with no check. The period and the best-benchmark claims block a finish; the level-shift count and the uncited causes
are warnings.
"""
from prophet_agent.runtime import contracts


def _forecast(period=5, verdict="ok"):
    return {"kind": "tool", "tool": "timesfm_forecast", "args": {"values": [1.0] * 504, "horizon": 14},
            "result": {"point": [1.0] * 14, "config": {"horizon": 14, "history_points": 504},
                       "health": {"verdict": verdict, "period": period}}}


def _backtest(best="mean_last_season", naive=1.0, mls=0.8, tsf=0.9):
    return {"kind": "tool", "tool": "backtest_series", "args": {"horizon": 14},
            "result": {"horizon": 14, "origins": 8, "best_baseline": best,
                       "models": {"naive": {"mase": naive}, "mean_last_season": {"mase": mls}, "timesfm": {"mase": tsf}}}}


def _failing(checks, name):
    return [c for c in checks if c["name"] == name and not c["ok"]]


def test_a_cycle_the_answer_states_that_the_health_record_does_not_measure_blocks_the_finish():
    checks = contracts.check_answer("The series has a weekly (7-day) cycle, confirmed by the health check.",
                                    [_forecast(period=5), _backtest()])
    bad = _failing(checks, "answer_cycle_matches_health")
    assert len(bad) == 1 and bad[0]["severity"] == "block"
    assert "5" in bad[0]["detail"] and "7" in bad[0]["detail"]


def test_a_cycle_the_answer_states_that_matches_the_health_record_passes():
    checks = contracts.check_answer("The series has a weekly (7-day) cycle, confirmed by the health check.",
                                    [_forecast(period=7), _backtest()])
    assert _failing(checks, "answer_cycle_matches_health") == []


def test_naming_a_rule_the_backtest_ranked_below_the_best_blocks_the_finish():
    checks = contracts.check_answer("Naive is the appropriate benchmark for this series.", [_forecast(), _backtest()])
    bad = _failing(checks, "answer_benchmark_matches")
    assert len(bad) == 1 and bad[0]["severity"] == "block"
    assert "mean_last_season" in bad[0]["detail"] and "naive" in bad[0]["detail"]


def test_naming_the_best_rule_the_backtest_ranked_first_passes():
    checks = contracts.check_answer("mean_last_season is the appropriate benchmark for this series.",
                                    [_forecast(), _backtest()])
    assert _failing(checks, "answer_benchmark_matches") == []


def test_no_model_beats_naive_claims_naive_is_best_and_blocks_when_the_backtest_disagrees():
    checks = contracts.check_answer("No model beats naive over these windows.", [_forecast(), _backtest()])
    assert len(_failing(checks, "answer_benchmark_matches")) == 1


def test_a_negated_benchmark_claim_is_not_read_as_a_claim():
    checks = contracts.check_answer("Naive is not the best benchmark here.", [_forecast(), _backtest()])
    assert _failing(checks, "answer_benchmark_matches") == []


def test_a_benchmark_within_the_rule_tolerance_of_the_best_passes():
    near = _backtest(best="mean_last_season", naive=0.81, mls=0.8)     # naive 1.25% above the best: inside the 2% tolerance
    checks = contracts.check_answer("Naive is the appropriate benchmark.", [_forecast(), near])
    assert _failing(checks, "answer_benchmark_matches") == []


def test_an_answer_with_no_claim_and_no_stored_output_checks_nothing():
    assert contracts.check_answer("The forecast is ready.", []) == []


def test_a_level_shift_count_the_scan_does_not_hold_is_a_warning(tmp_path):
    scan = {"findings": [{"id": "changepoints.level_shift", "severity": "major", "scope": "WTI",
                          "evidence": {"count": 4, "breaks": []}}]}
    folder = tmp_path / "run_a" / "out"
    folder.mkdir(parents=True)
    (folder / "scan.json").write_text(__import__("json").dumps(scan))
    history = [{"kind": "tool", "tool": "sandbox_python", "args": {"code": "sc.write_scan(r)"},
                "result": {"ok": True, "run_id": "run_a", "outputs": ["out/scan.json (300 bytes)"]}}, _forecast()]
    checks = contracts.check_answer("The data holds five level shifts.", history, run_root=str(tmp_path))
    bad = _failing(checks, "answer_shift_count_matches")
    assert len(bad) == 1 and bad[0]["severity"] == "warn"


def test_a_causal_attribution_with_no_source_or_test_is_a_warning():
    checks = contracts.check_answer("The level shifts correspond to known oil-market regimes, such as the 2022 energy crisis.",
                                    [_forecast(), _backtest()])
    bad = _failing(checks, "answer_causes_cited")
    assert len(bad) == 1 and bad[0]["severity"] == "warn"


def test_a_causal_attribution_that_cites_a_research_source_passes():
    checks = contracts.check_answer("The level shift on 2020-04-20 corresponds to the demand collapse in S3.",
                                    [_forecast(), _backtest()])
    assert _failing(checks, "answer_causes_cited") == []
