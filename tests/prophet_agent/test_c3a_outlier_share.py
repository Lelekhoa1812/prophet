"""C3a, A4-11: the outlier share rule is applied. Above 2% of the observed steps flagged by Hampel, the ledger must state the share
and a test of a winsorised or robust copy. The scan already prints the share (outliers.share, evidence.share_by_method.hampel);
this check reads it and does not change the scan.

Live evidence (measured, audit round 3, A4): Beijing flagged 6.3% to 8.8% of rows per station and kept them all as real spikes
with one template sentence. Retail rep4 had 9.6% and wrote "use robust", then did not. The outliers skill says to stop above
about 2%. A warning.
"""
import json

from prophet_agent.runtime import contracts


def _scan_run(root, hampel_share, entity="Beijing_1"):
    folder = root / "run_a" / "out"
    folder.mkdir(parents=True, exist_ok=True)
    scan = {"findings": [{"id": "outliers.share", "severity": "minor", "scope": entity,
                          "evidence": {"share_by_method": {"hampel": hampel_share, "stl_remainder": 0.01},
                                       "severity_uses": "hampel", "observed_steps": 1461}}]}
    (folder / "scan.json").write_text(json.dumps(scan))
    (folder / "ledger.json").write_text(json.dumps(LEDGER))
    return root


LEDGER = [{"index": 1, "stage": "outliers", "finding": "Hampel flags outliers", "decision": "Outliers kept as real pollution spikes"}]


def _history():
    return [{"kind": "tool", "tool": "sandbox_python", "args": {"code": "sc.write_scan(r); ps.Ledger().save()"},
             "result": {"ok": True, "run_id": "run_a", "outputs": ["out/scan.json (300 bytes)", "out/ledger.json (400 bytes)"]}},
            {"kind": "tool", "tool": "timesfm_forecast", "args": {"values": [1.0] * 50, "horizon": 5},
             "result": {"point": [1.0] * 5, "config": {"horizon": 5, "history_points": 50},
                        "health": {"verdict": "ok", "period": 7}}}]


def _check(checks, name="outlier_share_decided"):
    found = [c for c in checks if c["name"] == name]
    assert len(found) == 1, checks
    return found[0]


def test_a_share_above_two_percent_with_a_bare_keep_decision_is_a_warning(tmp_path):
    _scan_run(tmp_path, 0.088)
    c = _check(contracts.check_evidence(_history(), run_root=str(tmp_path)))
    assert c["ok"] is False and c["severity"] == "warn"
    assert "Beijing_1" in c["detail"]


def test_a_share_above_two_percent_with_the_share_stated_and_a_winsorised_test_passes(tmp_path):
    _scan_run(tmp_path, 0.088)
    (tmp_path / "run_a" / "out" / "ledger.json").write_text(json.dumps([
        {"index": 1, "stage": "outliers", "finding": "Hampel flags 8.8% of rows",
         "decision": "Keep the 8.8% as real spikes; a winsorised copy (replace_outliers) moved the 80% range by 2%."}]))
    c = _check(contracts.check_evidence(_history(), run_root=str(tmp_path)))
    assert c["ok"] is True


def test_a_share_at_or_below_two_percent_is_not_checked(tmp_path):
    _scan_run(tmp_path, 0.015)
    assert all(c["name"] != "outlier_share_decided" for c in contracts.check_evidence(_history(), run_root=str(tmp_path)))


def test_the_check_is_skipped_when_the_sandbox_folder_is_unknown():
    assert all(c["name"] != "outlier_share_decided" for c in contracts.check_evidence(_history(), run_root=None))
