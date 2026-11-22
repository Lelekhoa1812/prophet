"""C3a, A6-07 (contract side): the health gate reads the forecast the person receives, which is the pack (forecast_report), and a
flagged band needs a control_lab at its period.

Defect (code-read, reproduced below): health_repaired reads the last forecast of any kind. A healthy timesfm_forecast run after a
flat pack cleared the gate, while the delivered pack still carried the flat band. The pack is what ships, so the gate follows it.
Live evidence (measured, audit round 3, A6): retail rep4 health period 5 with verdict flat_unexpected; no control_lab call and no
call with period=5 in that run.
"""
from prophet_agent.runtime import contracts


def _pack(verdict, period=5):
    return {"kind": "tool", "tool": "forecast_report", "args": {"centre": "naive"},
            "result": {"ok": True, "run_id": "run_a", "files": [],
                       "forecast": {"point": [1.0] * 14, "config": {"horizon": 14, "history_points": 504},
                                    "health": {"verdict": verdict, "period": period}}}}


def _timesfm(verdict, period=5):
    return {"kind": "tool", "tool": "timesfm_forecast", "args": {"values": [1.0] * 504, "horizon": 14},
            "result": {"point": [1.0] * 14, "config": {"horizon": 14, "history_points": 504},
                       "health": {"verdict": verdict, "period": period}}}


def _lab(period):
    return {"kind": "tool", "tool": "control_lab", "args": {"period": period, "horizon": 14},
            "result": {"ok": True, "horizon": 14, "points": 504, "measured": ["ctx_season"], "table": []}}


def _check(checks, name):
    found = [c for c in checks if c["name"] == name]
    assert len(found) == 1, checks
    return found[0]


def test_a_healthy_timesfm_run_after_a_flat_pack_does_not_clear_the_gate():
    checks = contracts.check_evidence([_pack("flat_unexpected"), _timesfm("ok", period=7)])
    gate = _check(checks, "health_repaired")
    assert gate["ok"] is False and gate["severity"] == "block"


def test_a_pack_rebuilt_on_a_healthy_forecast_clears_the_gate():
    checks = contracts.check_evidence([_pack("flat_unexpected"), _pack("ok")])
    assert _check(checks, "health_repaired")["ok"] is True


def test_a_flagged_pack_with_no_control_lab_at_its_period_is_a_warning():
    checks = contracts.check_evidence([_pack("flat_unexpected", period=5)])
    gate = _check(checks, "flagged_period_tested")
    assert gate["ok"] is False and gate["severity"] == "warn"
    assert "5" in gate["detail"]


def test_a_flagged_pack_with_a_control_lab_at_its_period_is_not_warned():
    checks = contracts.check_evidence([_pack("flat_unexpected", period=5), _lab(5)])
    assert _check(checks, "flagged_period_tested")["ok"] is True


def test_a_healthy_pack_has_no_period_check():
    checks = contracts.check_evidence([_pack("ok")])
    assert all(c["name"] != "flagged_period_tested" for c in checks)
