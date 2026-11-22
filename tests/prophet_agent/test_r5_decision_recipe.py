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
    assert "changepoints.level_shift (Dongsi)" in d and "changepoints.level_shift (Tiantan)" in d
    assert "outliers and level shifts" in d and "led.save()" in d and "is a list" in d
    assert "values.spikes" not in d                       # minor findings are not asked for


def test_a_recorded_decision_passes(tmp_path):
    led = [{"stage": "outliers and level shifts", "finding": "changepoints.level_shift Dongsi and Tiantan 1.3x",
            "decision": "kept as observed, unexplained"}]
    assert contracts.check_scan_decisions(_hist(tmp_path, led), tmp_path)["ok"]
