"""Round 3 gate B4, goal 3 (A4-05, reader side): the checks read the newest saved ledger of a mission, and a later ledger that
lost entries is reported.

Live evidence (measured): WTI rep3. Run 9de19bc9 saved five stage entries; a later cell (run d908f0a5) created an empty
ps.Ledger() and saved it over the record. The checks took the maximum over every saved ledger, so the empty file was invisible.
"""
import json

from prophet_agent.runtime import contracts

STAGES = ("time index", "coverage", "zeros", "outliers", "seasonality")


def _save(root, run_id, entries):
    folder = root / run_id / "out"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ledger.json").write_text(json.dumps(entries))


def _entries(stages):
    return [{"index": i + 1, "stage": st, "finding": f"{st} checked", "decision": "kept"} for i, st in enumerate(stages)]


def _sandbox(run_id):
    return {"kind": "tool", "tool": "sandbox_python", "args": {"code": "ps.Ledger().save()"},
            "result": {"ok": True, "run_id": run_id, "outputs": ["out/ledger.json (400 bytes)"]}}


def _history(*run_ids):
    return [_sandbox(r) for r in run_ids]


def test_the_newest_ledger_is_the_one_the_checks_read(tmp_path):
    # Earlier run holds five stages; the later run overwrote the record with an empty ledger. The newest is what counts.
    _save(tmp_path, "run_a", _entries(STAGES))
    _save(tmp_path, "run_b", [])
    assert contracts.ledger_stages(_history("run_a", "run_b"), str(tmp_path)) == 0


def test_a_later_ledger_that_lost_entries_is_reported_with_both_counts(tmp_path):
    _save(tmp_path, "run_a", _entries(STAGES))
    _save(tmp_path, "run_b", _entries(STAGES[:2]))
    check = contracts.check_ledger_kept(_history("run_a", "run_b"), str(tmp_path))
    assert check is not None and check["ok"] is False and check["severity"] == "warn"
    assert "5" in check["detail"] and "2" in check["detail"]


def test_a_growing_or_equal_record_is_not_reported(tmp_path):
    _save(tmp_path, "run_a", _entries(STAGES[:3]))
    _save(tmp_path, "run_b", _entries(STAGES))
    assert contracts.check_ledger_kept(_history("run_a", "run_b"), str(tmp_path))["ok"] is True
    assert contracts.check_ledger_kept(_history("run_a"), str(tmp_path))["ok"] is True


def test_the_shrink_check_is_skipped_when_the_folder_is_unknown():
    assert contracts.check_ledger_kept(_history("run_a"), None) is None


def test_the_evidence_check_carries_the_shrink_report_next_to_the_ledger_check(tmp_path):
    _save(tmp_path, "run_a", _entries(STAGES))
    _save(tmp_path, "run_b", [])
    obs = [{"kind": "tool", "tool": "inspect_data", "args": {}, "result": {"ok": True}},
           {"kind": "tool", "tool": "timesfm_forecast", "args": {"values": [1.0] * 30, "horizon": 3},
            "result": {"ok": True, "config": {"horizon": 3}, "health": {"verdict": "ok"}, "point": [1.0, 1.0, 1.0],
                       "q10": [0.5, 0.5, 0.5], "q90": [1.5, 1.5, 1.5]}}] + _history("run_a", "run_b")
    checks = {c["name"]: c for c in contracts.check_evidence(obs, run_root=tmp_path)}
    assert checks["analysis_ledger"]["ok"] is False
    assert checks["ledger_kept"]["ok"] is False and checks["ledger_kept"]["severity"] == "warn"
