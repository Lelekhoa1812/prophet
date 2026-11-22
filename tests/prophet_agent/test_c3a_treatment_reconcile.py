"""C3a, A4-04: the ledger's treatment decisions and the written answer must agree (negatives, outliers, gaps).

Live evidence (measured, audit round 3, A4): retail rep3 ledger "Keep negative values as genuine returns-heavy days" against the
answer "clipped to 0". Beijing ledger "Outliers kept as real pollution spikes" for all 12 stations. WTI rep3 answered "no gaps
filled" while the ledger said 109 gaps were filled by straight-line interpolation. Nothing reconciled them. A warning, as the
finding asks, until the Ledger's supersedes field exists on live runs.
"""
import json

from prophet_agent.runtime import contracts, grounding


def _save(root, run_id, entries):
    folder = root / run_id / "out"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ledger.json").write_text(json.dumps(entries))


def _history(run_id):
    return [{"kind": "tool", "tool": "sandbox_python", "args": {"code": "ps.Ledger().save()"},
             "result": {"ok": True, "run_id": run_id, "outputs": ["out/ledger.json (400 bytes)"]}},
            {"kind": "tool", "tool": "timesfm_forecast", "args": {"values": [1.0] * 50, "horizon": 5},
             "result": {"point": [1.0] * 5, "config": {"horizon": 5, "history_points": 50},
                        "health": {"verdict": "ok", "period": 7}}}]


def _entry(stage, decision):
    return {"index": 1, "stage": stage, "finding": "checked", "decision": decision}


def _failing(checks, name):
    return [c for c in checks if c["name"] == name and not c["ok"]]


def test_a_negatives_decision_the_answer_clips_is_a_conflict(tmp_path):
    _save(tmp_path, "run_a", [_entry("negatives", "Keep negative values as genuine returns-heavy days")])
    checks = contracts.check_answer("Negative sales were clipped to 0 before fitting.", _history("run_a"), run_root=str(tmp_path))
    bad = _failing(checks, "ledger_answer_consistent")
    assert len(bad) == 1 and bad[0]["severity"] == "warn"
    assert "negatives" in bad[0]["detail"] and "keep" in bad[0]["detail"] and "treat" in bad[0]["detail"]


def test_outliers_kept_in_the_ledger_and_winsorised_in_the_answer_is_a_conflict(tmp_path):
    _save(tmp_path, "run_a", [_entry("outliers", "Outliers kept as real pollution spikes")])
    checks = contracts.check_answer("The outliers were winsorised at the 1st and 99th percentile.", _history("run_a"),
                                    run_root=str(tmp_path))
    assert len(_failing(checks, "ledger_answer_consistent")) == 1


def test_gaps_the_ledger_filled_but_the_answer_says_none_were_filled_is_a_conflict(tmp_path):
    _save(tmp_path, "run_a", [_entry("coverage", "109 gaps in 102 runs filled by straight-line interpolation")])
    checks = contracts.check_answer("No gaps were filled; the holidays are not gaps.", _history("run_a"), run_root=str(tmp_path))
    assert len(_failing(checks, "ledger_answer_consistent")) == 1


def test_an_answer_that_agrees_with_the_ledger_passes(tmp_path):
    _save(tmp_path, "run_a", [_entry("negatives", "Keep negative values as genuine returns-heavy days")])
    checks = contracts.check_answer("Negative values are kept as genuine returns.", _history("run_a"), run_root=str(tmp_path))
    assert _failing(checks, "ledger_answer_consistent") == []


def test_a_negated_treatment_in_the_answer_agrees_with_a_keep_decision(tmp_path):
    _save(tmp_path, "run_a", [_entry("negatives", "Keep negative values as genuine returns-heavy days")])
    checks = contracts.check_answer("Negative values were not clipped.", _history("run_a"), run_root=str(tmp_path))
    assert _failing(checks, "ledger_answer_consistent") == []


def test_a_superseded_ledger_entry_does_not_count(tmp_path):
    old = dict(_entry("negatives", "Keep negative values as genuine returns"), superseded=True)
    _save(tmp_path, "run_a", [old, _entry("negatives", "Clip negative values to zero")])
    checks = contracts.check_answer("Negative sales were clipped to 0.", _history("run_a"), run_root=str(tmp_path))
    assert _failing(checks, "ledger_answer_consistent") == []


def test_the_reconciliation_is_skipped_without_a_saved_ledger_or_a_sandbox_folder(tmp_path):
    assert contracts.check_ledger_answer("Negative sales were clipped to 0.", _history("run_a"), None) is None
    assert contracts.check_ledger_answer("Negative sales were clipped to 0.", _history("run_a"), str(tmp_path)) is None


def test_stances_read_a_sentence_and_flip_on_negation():
    assert grounding.treatment_stances("Outliers were kept.") == {"outliers": {"keep"}}
    assert grounding.treatment_stances("Outliers were not kept.") == {"outliers": {"treat"}}
    assert grounding.treatment_stances("Nothing else.") == {}
