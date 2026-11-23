"""Round 6c (2026-11-23), live Beijing 3581d07d: a named ledger file counts, and a cell re-run with new carry ids is a repeat.

Seven decisions saved as out/ledger_full.json were read as zero stages, and the same ledger cell ran 57 times with a new carry
list each time, so the stall window never saw identical calls. Every series here is synthetic.
"""
import json

from prophet_agent.runtime import contracts
from prophet_agent.runtime import goal as G


def _entries(n):
    return [{"stage": f"stage {i}", "finding": f"finding {i}", "decision": f"decision {i}"} for i in range(n)]


def test_a_ledger_saved_under_a_variant_name_is_read(tmp_path):
    out = tmp_path / "r1" / "out"
    out.mkdir(parents=True)
    (out / "ledger_full.json").write_text(json.dumps(_entries(2)), encoding="utf-8")
    hist = [{"kind": "tool", "tool": "sandbox_python", "result": {"run_id": "r1", "outputs": [{"path": "out/ledger_full.json"}]}}]
    assert contracts.ledger_stages(hist, str(tmp_path)) == 2


def test_an_unreadable_ledger_is_not_counted_as_a_saved_empty_one(tmp_path):
    hist = [{"kind": "tool", "tool": "sandbox_python", "result": {"run_id": "r9", "outputs": [{"path": "out/ledger.json"}]}}]
    assert contracts.ledger_stages(hist, str(tmp_path)) == 0
    assert contracts._saved_ledgers(hist, str(tmp_path)) == []


def test_the_recipe_carries_the_file_that_was_saved(tmp_path):
    out = tmp_path / "r1" / "out"
    out.mkdir(parents=True)
    (out / "ledger_full.json").write_text("[]", encoding="utf-8")
    hist = [{"kind": "tool", "tool": "sandbox_python", "result": {"run_id": "r1", "outputs": [{"path": "out/ledger_full.json"}]}}]
    recipe = contracts._decision_recipe([{"id": "a.b", "severity": "major", "scope": "all"}], ["a.b"], hist, str(tmp_path))
    assert "file: 'out/ledger_full.json'" in recipe and "in/carried/ledger_full.json" in recipe


def _cell(run_id, carry_run):
    return {"kind": "tool", "tool": "sandbox_python",
            "args": {"code": "led = ps.Ledger(); led.save()", "carry": [{"file": "out/ledger_full.json", "run_id": carry_run}]},
            "result": {"run_id": run_id, "ok": True, "stdout": "Ledger saved with 7 entries\n",
                       "outputs": [{"path": "out/ledger_full.json", "sha256": "abc"}]}}


def test_the_same_cell_with_new_carry_ids_is_a_stall():
    hist = [_cell("r1", "c1"), _cell("r2", "c2"), _cell("r3", "c3")]
    assert G._stalled(hist, 3) is True


def test_different_code_with_the_same_output_is_a_stall():
    # Round 7 (GOAL.md open item, "409 identical-output sandbox cells with different code"): the same printed line and the same file
    # digest are no progress, whatever the code. This test said the opposite before round 7 and was changed with that item.
    a, b, c = _cell("r1", "c1"), _cell("r2", "c2"), _cell("r3", "c3")
    c["args"] = {**c["args"], "code": "print('something else')"}
    assert G._stalled([a, b, c], 3) is True
