"""Audit item 17: every blocker and major finding in the saved scan needs a ledger decision that names its id."""
import json

from prophet_agent.runtime import contracts


def _history(run_root, scan_findings, ledger_entries):
    run = run_root / "r1" / "out"
    run.mkdir(parents=True)
    (run / "scan.json").write_text(json.dumps({"findings": scan_findings, "summary": {}}), encoding="utf-8")
    (run / "ledger.json").write_text(json.dumps(ledger_entries), encoding="utf-8")
    result = {"run_id": "r1", "outputs": [{"path": "out/scan.json"}, {"path": "out/ledger.json"}]}
    return [{"kind": "tool", "tool": "sandbox_python", "result": result}]


FINDINGS = [{"id": "time.duplicate_timestamps", "severity": "blocker", "scope": "sku_a"},
            {"id": "gaps.long_run", "severity": "major", "scope": "all"},
            {"id": "info.grid", "severity": "info", "scope": "all"}]


def _entry(stage, finding, decision):
    return {"stage": stage, "finding": finding, "decision": decision}


def test_every_blocker_and_major_finding_named_in_the_ledger_passes(tmp_path):
    hist = _history(tmp_path, FINDINGS, [
        _entry("time index", "time.duplicate_timestamps in sku_a", "drop the repeated rows, keep the first"),
        _entry("coverage", "gaps.long_run across all series", "fill with NaN, keep open"),
    ])
    check = contracts.check_scan_decisions(hist, str(tmp_path))
    assert check["ok"] is True and check["severity"] == "block", check


def test_a_blocker_with_no_ledger_decision_fails_and_names_it(tmp_path):
    hist = _history(tmp_path, FINDINGS, [_entry("coverage", "gaps.long_run across all series", "keep open")])
    check = contracts.check_scan_decisions(hist, str(tmp_path))
    assert check["ok"] is False
    assert "time.duplicate_timestamps" in check["detail"] and "gaps.long_run" not in check["detail"]


def test_the_id_must_be_in_the_finding_or_decision_text_not_in_the_evidence(tmp_path):
    entry = {"stage": "time index", "finding": "repeated timestamps", "decision": "keep the first",
             "evidence": {"id": "time.duplicate_timestamps"}}
    hist = _history(tmp_path, [{"id": "time.duplicate_timestamps", "severity": "blocker", "scope": "all"}], [entry])
    assert contracts.check_scan_decisions(hist, str(tmp_path))["ok"] is False


def test_a_scan_with_only_minor_findings_needs_no_decision(tmp_path):
    hist = _history(tmp_path, [{"id": "info.grid", "severity": "info", "scope": "all"}], [])
    check = contracts.check_scan_decisions(hist, str(tmp_path))
    assert check["ok"] is True and "no blocker or major" in check["detail"]


def test_no_saved_scan_leaves_the_check_to_the_scan_check(tmp_path):
    assert contracts.check_scan_decisions([], str(tmp_path)) is None
    assert contracts.check_scan_decisions([], None) is None


def test_the_ledger_stage_count_is_unchanged_by_the_refactor(tmp_path):
    hist = _history(tmp_path, [], [_entry("time index", "f", "d"), _entry("coverage", "f", "d"), _entry("coverage", "f2", "d2"),
                                   _entry("zeros", "", "d")])
    assert contracts.ledger_stages(hist, str(tmp_path)) == 2
