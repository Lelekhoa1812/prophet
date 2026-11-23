"""Round 6c (2026-11-23): the decision check reads the latest saved scan, not every scan the mission ever saved.

Live electricity c1154fd1 (build f84cce2): an earlier 15-entity scan's censoring.stockout_runs and schema.missing_column
blocked the 60-client answer, and the decision API could not record them because it reads only the current scan.
Every series here is synthetic.
"""
import json

from prophet_agent.runtime import contracts


def _run(run_root, run_id, findings):
    out = run_root / run_id / "out"
    out.mkdir(parents=True)
    (out / "scan.json").write_text(json.dumps({"findings": findings}), encoding="utf-8")
    return {"kind": "tool", "tool": "sandbox_python",
            "result": {"run_id": run_id, "outputs": [{"path": "out/scan.json"}]}}


OLD = [{"id": "schema.missing_column", "severity": "major", "scope": "3 series"}]
NEW = [{"id": "censoring.stockout_runs", "severity": "major", "scope": "MT_015"}]


def test_an_undecided_finding_from_an_earlier_scan_does_not_block_the_latest_one(tmp_path):
    hist = [_run(tmp_path, "r1", OLD), _run(tmp_path, "r2", NEW)]
    # The ledger decides the finding the current scan holds; the superseded scan's finding needs no decision.
    (tmp_path / "r3" / "out").mkdir(parents=True)
    (tmp_path / "r3" / "out" / "ledger.json").write_text(json.dumps(
        [{"stage": "zeros and censoring", "finding": "censoring.stockout_runs MT_015", "decision": "kept as observed"}]),
        encoding="utf-8")
    hist.append({"kind": "tool", "tool": "sandbox_python",
                 "result": {"run_id": "r3", "outputs": [{"path": "out/ledger.json"}]}})
    check = contracts.check_scan_decisions(hist, str(tmp_path))
    assert check is not None and check["ok"] is True, check


def test_the_latest_scan_still_blocks_when_its_finding_is_undecided(tmp_path):
    hist = [_run(tmp_path, "r1", OLD), _run(tmp_path, "r2", NEW)]
    check = contracts.check_scan_decisions(hist, str(tmp_path))
    assert check["ok"] is False
    assert "censoring.stockout_runs" in check["detail"] and "schema.missing_column" not in check["detail"]


def test_an_unreadable_latest_scan_falls_back_to_the_last_readable_one(tmp_path):
    hist = [_run(tmp_path, "r1", NEW)]
    hist.append({"kind": "tool", "tool": "sandbox_python",
                 "result": {"run_id": "r9", "outputs": [{"path": "out/scan.json"}]}})   # r9 was never written
    check = contracts.check_scan_decisions(hist, str(tmp_path))
    assert check["ok"] is False and "censoring.stockout_runs" in check["detail"]


# 97ab893f: the total bounds named with the wrong assumption (Beijing panel, 2026-11-23).

def _panel_pack_history():
    return [{"kind": "tool", "tool": "forecast_report", "args": {}, "result": {
        "n_series": 12,
        "panel_total": {"centre": 12221.65,
                        "if_entities_miss_together": {"q10": 4081.93, "q90": 26034.83},
                        "if_entities_miss_independently": {"q10": 11353.56, "q90": 13089.74}}}}]


def test_a_together_bound_named_as_independent_is_blocked_with_both_bounds_and_their_assumptions():
    text = "If stations vary independently: 4,081.9 to 26,034.8 µg/m³ for the 14-day panel total."
    check = contracts._misnamed_bound(text, _panel_pack_history())
    assert check is not None and check["ok"] is False and check["name"] == "answer_total_range_named"
    assert "11,353.56" not in check["detail"]                       # the stored figures are rounded in the text
    assert "11,354 to 13,090" in check["detail"] and "4,082 to 26,035" in check["detail"]
    assert "if the series miss independently: 11,354 to 13,090" in check["detail"]


def test_the_independent_bound_named_as_independent_passes():
    text = "If the stations miss independently, the 14-day panel total lies between 11,353.6 and 13,089.7 µg/m³."
    assert contracts._misnamed_bound(text, _panel_pack_history()) is None
