"""C3a, A3-04: a ledger decision that attributes a shift to a cause must cite a research source id (S1, S2 ...) or a test; otherwise
the decision must say the shift is unexplained. A decision that promises a test must have a test run in the mission.

Live evidence (measured, audit round 3, A3): WTI rep4 ledger stage outliers_level_shifts: "Level shifts correspond to known oil
market regimes: COVID crash/recovery, post-COVID demand surge, 2022 energy crisis, 2024 demand concerns." No tool output supports
it, and 2022 is not among the scan's breaks. The same entry promised "Will test ignore_ranges ... in control_lab"; no control_lab
or ignore_ranges call appears. Both are warnings: the causal claim is a claim, not a number that reaches the person as data.
"""
import json

from prophet_agent.runtime import contracts


def _save(root, entries):
    folder = root / "run_a" / "out"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ledger.json").write_text(json.dumps(entries))


def _history():
    return [{"kind": "tool", "tool": "sandbox_python", "args": {"code": "ps.Ledger().save()"},
             "result": {"ok": True, "run_id": "run_a", "outputs": ["out/ledger.json (400 bytes)"]}},
            {"kind": "tool", "tool": "timesfm_forecast", "args": {"values": [1.0] * 50, "horizon": 5},
             "result": {"point": [1.0] * 5, "config": {"horizon": 5, "history_points": 50},
                        "health": {"verdict": "ok", "period": 7}}}]


def _entry(decision, stage="outliers_level_shifts", finding="changepoints.level_shift: 5 breaks"):
    return {"index": 1, "stage": stage, "finding": finding, "decision": decision}


def _check(checks, name):
    found = [c for c in checks if c["name"] == name]
    assert len(found) == 1, checks
    return found[0]


def test_a_causal_shift_decision_with_no_source_or_test_is_a_warning(tmp_path):
    _save(tmp_path, [_entry("Level shifts correspond to known oil market regimes: COVID crash/recovery, 2022 energy crisis.")])
    c = _check(contracts.check_evidence(_history(), run_root=str(tmp_path)), "causal_shift_cited")
    assert c["ok"] is False and c["severity"] == "warn"


def test_a_shift_recorded_as_unexplained_is_not_a_causal_claim(tmp_path):
    _save(tmp_path, [_entry("Level shifts are unexplained and kept as observed; no known event sits on those dates.")])
    assert _check(contracts.check_evidence(_history(), run_root=str(tmp_path)), "causal_shift_cited")["ok"] is True


def test_a_causal_shift_decision_that_cites_a_research_source_passes(tmp_path):
    _save(tmp_path, [_entry("The 2020-04-20 break corresponds to the demand collapse reported in S3.")])
    assert _check(contracts.check_evidence(_history(), run_root=str(tmp_path)), "causal_shift_cited")["ok"] is True


def test_a_causal_shift_decision_that_names_a_test_passes(tmp_path):
    _save(tmp_path, [_entry("The break corresponds to the regime change; control_lab kept it at 1.2% MASE gain.")])
    assert _check(contracts.check_evidence(_history(), run_root=str(tmp_path)), "causal_shift_cited")["ok"] is True


def test_a_causal_word_on_an_entry_that_is_not_about_shifts_is_not_read(tmp_path):
    _save(tmp_path, [_entry("Weekend zeros correspond to closed trading days.", stage="zeros", finding="zero share 0.4")])
    assert all(c["name"] != "causal_shift_cited" or c["ok"] for c in contracts.check_evidence(_history(), run_root=str(tmp_path)))


def test_a_promised_test_that_never_ran_is_a_warning(tmp_path):
    _save(tmp_path, [_entry("Will test ignore_ranges for 2020-04-20 in control_lab.")])
    c = _check(contracts.check_evidence(_history(), run_root=str(tmp_path)), "promised_tests_run")
    assert c["ok"] is False and c["severity"] == "warn"


def test_a_promised_test_that_ran_in_the_mission_passes(tmp_path):
    _save(tmp_path, [_entry("Will test ignore_ranges for 2020-04-20 in control_lab.")])
    history = _history() + [{"kind": "tool", "tool": "control_lab", "args": {"period": 7},
                             "result": {"ok": True, "horizon": 5, "points": 50, "measured": [], "table": []}}]
    assert _check(contracts.check_evidence(history, run_root=str(tmp_path)), "promised_tests_run")["ok"] is True


def test_the_causal_check_is_skipped_without_a_saved_ledger(tmp_path):
    names = {c["name"] for c in contracts.check_evidence(_history(), run_root=str(tmp_path))}
    assert "causal_shift_cited" not in names and "promised_tests_run" not in names
