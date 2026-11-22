"""Round 5: an undecided major scan finding comes back with the cell that records it, in the mission's own names.

Live Beijing rep5 (measured, 2026-11-23): the check named changepoints.level_shift for Dongsi, Tiantan and Wanshouxigong. The
planner read the scan six times, read the ledger with code that expected a dict (the file is a list), and finished three more
times without an add.
"""
import json

from prophet_agent.runtime import contracts

LEDGER_RUN, SCAN_RUN = "a" * 32, "b" * 32


def _hist(tmp_path, ledger):
    for rid, name, body in ((SCAN_RUN, "scan.json", {"findings": [
            {"id": "changepoints.level_shift", "severity": "major", "scope": s} for s in ("Dongsi", "Tiantan")]
            + [{"id": "values.spikes", "severity": "minor", "scope": "Dongsi"}], "summary": {}}),
            (LEDGER_RUN, "ledger.json", ledger)):
        (tmp_path / rid / "out").mkdir(parents=True)
        (tmp_path / rid / "out" / name).write_text(json.dumps(body))
    return [{"kind": "tool", "tool": "sandbox_python", "result": {"run_id": SCAN_RUN, "outputs": ["out/scan.json (10 bytes)"]}},
            {"kind": "tool", "tool": "sandbox_python", "result": {"run_id": LEDGER_RUN, "outputs": ["out/ledger.json (10 bytes)"]}}]


def test_the_failing_check_gives_the_cell_with_the_ledger_to_carry_and_each_scope(tmp_path):
    c = contracts.check_scan_decisions(_hist(tmp_path, [{"stage": "time index", "finding": "ok", "decision": "kept"}]), tmp_path)
    assert not c["ok"]
    d = c["detail"]
    assert LEDGER_RUN in d and "Ledger.open('in/carried/ledger.json')" in d
    # Changed in round 5 (live electricity rep9): one add per finding id, its scopes listed together
    assert "# 2 series: Dongsi, Tiantan" in d and d.count("led.decide(") == 1
    assert "led.decide('changepoints.level_shift'" in d and "led.save()" in d and "no if around" in d
    assert SCAN_RUN in d                                  # the scan to carry, so decide can read it
    assert "values.spikes" not in d                       # minor findings are not asked for


def test_a_recorded_decision_passes(tmp_path):
    led = [{"stage": "outliers and level shifts", "finding": "changepoints.level_shift Dongsi and Tiantan 1.3x",
            "decision": "kept as observed, unexplained"}]
    assert contracts.check_scan_decisions(_hist(tmp_path, led), tmp_path)["ok"]


def test_a_wide_panel_gets_one_add_per_finding_id_and_every_id(tmp_path):
    # live electricity rep9 (measured, 2026-11-23): 8 ids over 60 clients; the recipe's first six rows were one id
    ids = ["censoring.plateau", "censoring.stockout_runs", "changepoints.level_shift", "coverage.dead_segment",
           "coverage.leading_zeros", "intermittency.class", "outliers.share", "stationarity.verdict"]
    findings = [{"id": i, "severity": "major", "scope": f"MT_{k:03d}"} for i in ids for k in range(1, 61)]
    (tmp_path / SCAN_RUN / "out").mkdir(parents=True)
    (tmp_path / SCAN_RUN / "out" / "scan.json").write_text(json.dumps({"findings": findings, "summary": {}}))
    hist = [{"kind": "tool", "tool": "sandbox_python", "result": {"run_id": SCAN_RUN, "outputs": ["out/scan.json (1 bytes)"]}}]
    d = contracts.check_scan_decisions(hist, tmp_path)["detail"]
    assert d.count("led.decide(") == 8
    for i in ids:
        assert f"led.decide('{i}', '<your decision and why>')  # 60 series: MT_001, MT_002, MT_003 and 57 more" in d
    assert "covers every series it names" in d


def test_a_stopped_run_without_a_draft_asks_for_the_results_not_the_gap_alone():
    from prophet_agent.runtime.goal import _gap_draft
    assert _gap_draft({"answer": "Forecast: 12 units.", "reason": "x"}) == "Forecast: 12 units."
    d = _gap_draft({"answer": "", "reason": "eight scan findings have no ledger decision"})
    assert "every result the analysis established" in d and "Not settled" in d
    assert d.endswith("eight scan findings have no ledger decision")


def test_the_open_ledger_item_gives_the_decide_cell_for_the_scan(tmp_path):
    # live retail_biz rep6 (measured, 2026-11-23): the scan was saved and the ledger item open; 25 cells printed the scan
    # findings and none recorded a decision, so the mission stalled with no forecast
    (tmp_path / SCAN_RUN / "out").mkdir(parents=True)
    (tmp_path / SCAN_RUN / "out" / "scan.json").write_text(json.dumps({"findings": [
        {"id": "calendar.closed_days", "severity": "major", "scope": "all"},
        {"id": "values.negative", "severity": "minor", "scope": "all"},
        {"id": "seasonality.strength", "severity": "info", "scope": "all"}], "summary": {}}))
    hist = [{"kind": "tool", "tool": "sandbox_python", "args": {"code": "x"},
             "result": {"run_id": SCAN_RUN, "outputs": ["out/scan.json (10 bytes)"]}}]
    item = next(i for i in contracts.progress(hist, run_root=tmp_path) if i["item"].startswith("analysis ledger"))
    d = item["detail"]
    assert item["status"] == "todo" and d.startswith("0 of")
    assert d.index("calendar.closed_days") < d.index("values.negative") < d.index("seasonality.strength")
    assert "led.decide('values.negative'" in d and SCAN_RUN in d and "only saved decisions" in d
