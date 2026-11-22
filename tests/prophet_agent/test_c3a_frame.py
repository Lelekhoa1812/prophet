"""C3a, A3-13: the mission records a frame before it forecasts, meaning what the forecast is for, the horizon and the decision it feeds.
Warning level only.

The check reads the newest saved ledger for an entry staged "frame" with a non-empty decision. A ledger with no such entry warns.
Skipped when no ledger was saved.
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
    found = [c for c in checks if c["name"] == "frame_recorded"]
    assert len(found) == 1, checks
    return found[0]


def test_a_ledger_with_no_frame_entry_is_a_warning(tmp_path):
    _save(tmp_path, [{"index": 1, "stage": "coverage", "finding": "gaps 3%", "decision": "kept"}])
    c = _check(contracts.check_evidence(_history(), run_root=str(tmp_path)))
    assert c["ok"] is False and c["severity"] == "warn"


def test_a_frame_entry_with_a_decision_passes(tmp_path):
    _save(tmp_path, [{"index": 1, "stage": "frame", "finding": "purpose and horizon",
                      "decision": "Weekly replenishment for 5 steps ahead; feeds the stock order."}])
    assert _check(contracts.check_evidence(_history(), run_root=str(tmp_path)))["ok"] is True


def test_a_frame_entry_with_an_empty_decision_is_a_warning(tmp_path):
    _save(tmp_path, [{"index": 1, "stage": "frame", "finding": "purpose and horizon", "decision": "   "}])
    assert _check(contracts.check_evidence(_history(), run_root=str(tmp_path)))["ok"] is False


def test_a_superseded_frame_entry_does_not_count(tmp_path):
    _save(tmp_path, [{"index": 1, "stage": "frame", "finding": "old", "decision": "Old purpose.", "superseded": True}])
    assert _check(contracts.check_evidence(_history(), run_root=str(tmp_path)))["ok"] is False


def test_the_frame_check_is_skipped_without_a_saved_ledger(tmp_path):
    assert all(c["name"] != "frame_recorded" for c in contracts.check_evidence(_history(), run_root=str(tmp_path)))
