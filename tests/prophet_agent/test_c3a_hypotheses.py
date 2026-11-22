"""C3a, A4-10: the mission records two or more falsifiable hypotheses and marks each confirmed or refuted. Warning level only, so no run
is blocked by it yet (the finding asks for that).

Live evidence (measured, audit round 3, A4): none of six ledgers held an expectation field; no run recorded a falsifiable expectation
such as "the level shifted in 2024", as planner step 4 asks. The Ledger may gain optional expected and outcome fields (another
slice). The check reads them when a ledger entry carries them, and otherwise falls back to entries staged "hypothesis".
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


def _check(checks):
    found = [c for c in checks if c["name"] == "hypotheses_tested"]
    assert len(found) == 1, checks
    return found[0]


def test_a_ledger_with_no_hypothesis_is_a_warning(tmp_path):
    _save(tmp_path, [{"index": 1, "stage": "coverage", "finding": "gaps 3%", "decision": "kept"}])
    c = _check(contracts.check_evidence(_history(), run_root=str(tmp_path)))
    assert c["ok"] is False and c["severity"] == "warn"


def test_two_hypotheses_with_an_expectation_and_a_confirmed_or_refuted_outcome_pass(tmp_path):
    _save(tmp_path, [
        {"index": 1, "stage": "seasonality", "finding": "weekly", "decision": "weekly cycle kept",
         "expected": "a weekly cycle of 7 days", "outcome": "refuted"},
        {"index": 2, "stage": "outliers", "finding": "spikes", "decision": "kept",
         "expected": "spikes are promotions", "outcome": "confirmed"}])
    assert _check(contracts.check_evidence(_history(), run_root=str(tmp_path)))["ok"] is True


def test_expectations_left_untested_do_not_count(tmp_path):
    _save(tmp_path, [
        {"index": 1, "stage": "seasonality", "finding": "weekly", "decision": "x", "expected": "a weekly cycle",
         "outcome": "not tested"},
        {"index": 2, "stage": "outliers", "finding": "spikes", "decision": "y", "expected": "spikes are promotions"}])
    assert _check(contracts.check_evidence(_history(), run_root=str(tmp_path)))["ok"] is False


def test_the_fallback_passes_with_two_hypothesis_entries_that_state_an_outcome(tmp_path):
    _save(tmp_path, [
        {"index": 1, "stage": "hypothesis", "finding": "H1: weekly cycle", "decision": "Confirmed: the period is 5 steps."},
        {"index": 2, "stage": "hypothesis", "finding": "H2: level shift in 2024", "decision": "Refuted: no break after 2024."}])
    assert _check(contracts.check_evidence(_history(), run_root=str(tmp_path)))["ok"] is True


def test_the_fallback_fails_with_one_hypothesis_entry(tmp_path):
    _save(tmp_path, [{"index": 1, "stage": "hypothesis", "finding": "H1", "decision": "Confirmed."}])
    assert _check(contracts.check_evidence(_history(), run_root=str(tmp_path)))["ok"] is False


def test_the_check_is_skipped_without_a_saved_ledger(tmp_path):
    assert all(c["name"] != "hypotheses_tested" for c in contracts.check_evidence(_history(), run_root=str(tmp_path)))
