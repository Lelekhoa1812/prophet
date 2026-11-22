"""C3a, A4-09 (contract side): a level shift the scan flags as a likely recording break (on a calendar boundary, or shared by several
entities on the same date) is kept only after a test of the treatment options (truncate, flag, none).

Live evidence (measured, audit round 3, A4): Beijing had shifts on 1 January 2014, 2015 and 2016 for three stations, and a
three-station step on 2016-11-25; the ledger said "real regime changes" with no test. Electricity had a 2012-01-01 shift in many
of 60 clients. Scan flags for these are added by the scan slice (C5). This check reads them when present and is a warning.
"""
import json

from prophet_agent.runtime import contracts


def _scan(root, evidence):
    folder = root / "run_a" / "out"
    folder.mkdir(parents=True, exist_ok=True)
    scan = {"findings": [{"id": "changepoints.level_shift", "severity": "major", "scope": "Beijing_1", "evidence": evidence}]}
    (folder / "scan.json").write_text(json.dumps(scan))
    (folder / "ledger.json").write_text(json.dumps([{"index": 1, "stage": "outliers_level_shifts", "finding": "level shift",
                                                    "decision": LEDGER_TEXT}]))


LEDGER_TEXT = "Level shifts on 2014-01-01 are real regime changes; kept."


def _history(*extra):
    return [{"kind": "tool", "tool": "sandbox_python", "args": {"code": "sc.write_scan(r); ps.Ledger().save()"},
             "result": {"ok": True, "run_id": "run_a", "outputs": ["out/scan.json (300 bytes)", "out/ledger.json (400 bytes)"]}},
            {"kind": "tool", "tool": "timesfm_forecast", "args": {"values": [1.0] * 50, "horizon": 5},
             "result": {"point": [1.0] * 5, "config": {"horizon": 5, "history_points": 50},
                        "health": {"verdict": "ok", "period": 7}}}, *extra]


def _check(checks):
    found = [c for c in checks if c["name"] == "shift_decision_tested"]
    return found[0] if found else None


def test_a_shift_on_a_calendar_boundary_kept_with_no_test_is_a_warning(tmp_path):
    _scan(tmp_path, {"count": 1, "breaks": [{"at": "2014-01-01", "on_calendar_boundary": True}]})
    c = _check(contracts.check_evidence(_history(), run_root=str(tmp_path)))
    assert c is not None and c["ok"] is False and c["severity"] == "warn"


def test_a_shared_step_across_entities_is_flagged_too(tmp_path):
    _scan(tmp_path, {"count": 1, "coincident_entities": 3, "breaks": [{"at": "2016-11-25"}]})
    c = _check(contracts.check_evidence(_history(), run_root=str(tmp_path)))
    assert c is not None and c["ok"] is False


def test_a_flagged_shift_with_a_recorded_and_run_truncate_versus_flag_test_passes(tmp_path):
    _scan(tmp_path, {"count": 1, "breaks": [{"at": "2014-01-01", "on_calendar_boundary": True}]})
    (tmp_path / "run_a" / "out" / "ledger.json").write_text(json.dumps([{"index": 1, "stage": "outliers_level_shifts",
        "finding": "level shift", "decision": "Recording hypothesis: truncate vs flag vs none, backtested; flag kept."}]))
    test = {"kind": "tool", "tool": "backtest_series", "args": {"horizon": 5, "ignore_ranges": [["2013-12-01", "2014-02-01"]]},
            "result": {"ok": True, "horizon": 5, "models": {}}}
    c = _check(contracts.check_evidence(_history(test), run_root=str(tmp_path)))
    assert c is not None and c["ok"] is True


def test_a_shift_with_no_flag_is_not_checked(tmp_path):
    _scan(tmp_path, {"count": 1, "breaks": [{"at": "2014-06-01", "on_calendar_boundary": False}]})
    assert _check(contracts.check_evidence(_history(), run_root=str(tmp_path))) is None
