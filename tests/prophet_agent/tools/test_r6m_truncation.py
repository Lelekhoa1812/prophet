"""Round 6m: a planner view that cuts a list says so, with how many there were.

Probes (before the fixes): calibrate_intervals showed 30 points of a 50-point forecast with no note; control_lab showed the
first 5 skipped and failed candidates and the first 3 caveats with no count; sandbox_python cut the head of a long stderr
with no marker; drivers detect showed 40 candidates with no note; inspect_data showed 40 columns with no note; a panel
backtest dropped its per-series results with no note that they are kept in the ledger. The ledger keeps the full result
in every case, so the planner needs only to be told that something was left out.
"""
from __future__ import annotations

import importlib
import json

calibrate = importlib.import_module("prophet_agent.tools.calibrate_tool")
lab = importlib.import_module("prophet_agent.tools.lab_tool")
sandbox = importlib.import_module("prophet_agent.tools.sandbox_tool")
drivers = importlib.import_module("prophet_agent.tools.drivers_tool")
data = importlib.import_module("prophet_agent.tools.data_tools")


def _text(obj) -> str:
    return json.dumps(obj, default=str)


def test_calibrate_says_when_the_forecast_points_are_cut():
    result = {"horizon": 50, "target_coverage": 0.8, "point": [float(i) for i in range(50)],
              "calibrated": {"q10": [1.0] * 50, "q90": [2.0] * 50, "q50": [1.5] * 50},
              "raw": {"q10": [1.0] * 50, "q90": [2.0] * 50}}
    out = calibrate.observe(result)
    assert len(out["point"]) == 30
    assert "50" in _text({k: v for k, v in out.items() if k.endswith("note")}), "no note says 50 points were held"


def test_calibrate_shows_full_short_forecasts_without_a_note():
    result = {"horizon": 5, "point": [1.0] * 5, "calibrated": {"q10": [1.0] * 5, "q90": [2.0] * 5},
              "raw": {"q10": [1.0] * 5, "q90": [2.0] * 5}}
    out = calibrate.observe(result)
    assert len(out["point"]) == 5
    assert not any(k.endswith("note") for k in out)


def _lab_result(skipped: int, failed: int, caveats: int) -> dict:
    return {"horizon": 7, "points": 300, "period": 7, "fold_count": 3, "calls_spent": 10, "recommended": "default",
            "baseline_beats_default": False, "table": [],
            "skipped": [{"candidate": f"s{i}", "reason": "r"} for i in range(skipped)],
            "failed": [{"candidate": f"f{i}", "error": "e"} for i in range(failed)],
            "caveats": [f"caveat {i}" for i in range(caveats)]}


def test_control_lab_says_how_many_skipped_failed_and_caveats_were_left_out():
    out = lab.observe(_lab_result(skipped=8, failed=7, caveats=6))
    assert len(out["skipped"]) == 5 and len(out["failed"]) == 5 and len(out["caveats"]) == 3
    text = _text(out)
    assert "8 skipped" in text or "8 candidates skipped" in text or "of 8" in text
    assert "7 failed" in text or "of 7" in text
    assert "of 6" in text or "6 caveats" in text


def test_control_lab_says_nothing_when_nothing_is_left_out():
    out = lab.observe(_lab_result(skipped=2, failed=1, caveats=2))
    assert "of 2" not in _text(out) and "2 caveats" not in _text(out)


def _sandbox_result(stderr: str = "", stdout: str = "ok") -> dict:
    return {"ok": True, "run_id": "a" * 32, "stdout": stdout, "outputs": [], "stderr": stderr, "timed_out": False}


def test_sandbox_marks_a_cut_stderr_head():
    stderr = "Traceback line\n" * 300
    out = sandbox.observe(_sandbox_result(stderr=stderr))
    tail = out["stderr_tail"]
    assert tail.startswith("[... ") and "characters cut" in tail.split("\n", 1)[0]
    assert tail.endswith("Traceback line\n")
    assert len(tail) <= sandbox.SEEN_STDERR + 60


def test_sandbox_short_stderr_is_shown_whole():
    out = sandbox.observe(_sandbox_result(stderr="boom"))
    assert out["stderr_tail"] == "boom"


def test_detect_says_when_candidates_are_left_out():
    result = {"run_id": "b" * 32, "series": "x", "points": 500, "method": "m", "threshold": 6, "dated": True,
              "indices": list(range(60)), "candidates": [{"index": i} for i in range(60)], "ignore_ranges": []}
    out = drivers._observe_detect(result)
    assert len(out["candidates"]) == drivers.MAX_RANGES_SHOWN
    assert "60" in _text(out.get("note", "")) or "60" in _text({k: v for k, v in out.items() if "note" in k})


def test_inspect_says_when_columns_are_left_out():
    profile = {"rows": 10, "read": {"file": "a.csv"}, "issues": [], "layout_hints": [],
               "columns": [{"name": f"c{i}", "kind": "number", "sample": [1]} for i in range(50)]}
    out = data.observe_inspect({"file": "a.csv", "profile": profile, "run_id": "c" * 32})
    assert len(out["columns"]) == 40
    assert "50" in _text({k: v for k, v in out.items() if "note" in k or k.endswith("shown")})


def test_panel_backtest_says_its_per_series_results_are_in_the_ledger():
    result = {"series": {"a": {"models": {}}}, "summary": {"a": {"verdict": "timesfm_better"}}, "horizon": 3,
              "n_series": 1, "verdict_counts": {"timesfm_better": 1}, "reading": "r"}
    out = data.observe_backtest(result)
    assert "series" not in out
    assert "ledger" in _text(out.get("note", ""))
