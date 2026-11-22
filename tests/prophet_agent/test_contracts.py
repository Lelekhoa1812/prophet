"""Contract checks: each check passes and fails on a case it is meant to catch; evidence checks never raise."""
import hashlib
import json
import math

import pytest

from prophet_agent.runtime.contracts import (
    blocking, check_backtest, check_deliverables, check_evidence, check_forecast, summarise,
)

HISTORY = [float(v) for v in range(1, 13)]           # 1..12, never negative
POINT = [12.0, 13.0, 14.0, 15.0]
Q10 = [10.0, 11.0, 12.0, 13.0]
Q90 = [14.0, 15.0, 16.0, 17.0]
HORIZON = 4


def _by_name(checks, name):
    found = [c for c in checks if c["name"] == name]
    assert len(found) == 1, f"expected one {name} check, got {[c['name'] for c in checks]}"
    return found[0]


def _backtest(**overrides):
    result = {
        "points": 60, "horizon": 4, "origins": 8, "season_used": 1,
        "models": {
            "timesfm": {"mase": 0.8, "wape": 0.2, "volume_bias_pct": 1.0},
            "naive": {"mase": 1.0, "wape": 0.25, "volume_bias_pct": 0.0},
            "seasonal_naive": {"mase": 1.1, "wape": 0.3, "volume_bias_pct": 0.0},
            "drift": {"mase": 1.05, "wape": 0.26, "volume_bias_pct": 0.0},
        },
        "best_baseline": "naive",
    }
    result.update(overrides)
    return result


def _forecast_result(point=POINT, horizon=HORIZON, verdict="ok", q10=Q10, q90=Q90):
    return {"config": {"horizon": horizon, "checkpoint": "timesfm-3"}, "health": {"verdict": verdict},
            "point": list(point), "q10": list(q10), "q90": list(q90)}


def _tool(name, result=None, args=None, error=None):
    obs = {"kind": "tool", "tool": name, "args": args or {}}
    if error is not None:
        obs["error"] = error
    else:
        obs["result"] = result
    return obs


def _inspect():
    return _tool("sandbox_python", {"ok": True, "stdout": "rows 12", "run_id": "insp1", "outputs": ["out/ledger.json (400 bytes)", "out/scan.json (900 bytes)"]},
                 {"code": "print(1)"})


def _write_ledger(root, stages=("time index", "coverage", "zeros", "outliers", "seasonality"), run_id="insp1"):
    import json as _json
    folder = root / run_id / "out"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ledger.json").write_text(_json.dumps([{"index": i + 1, "stage": st, "finding": f"{st} checked", "decision": "kept"}
                                                     for i, st in enumerate(stages)]))


def _forecast(verdict="ok"):
    return _tool("timesfm_forecast", _forecast_result(verdict=verdict),
                 {"values": HISTORY, "horizon": HORIZON})


def _report(run_id="run1", files=("chart.png", "chart.svg", "forecast.xlsx", "report.docx", "report.pdf")):
    fc = _forecast_result()
    return _tool("forecast_report", {"forecast": fc, "run_id": run_id, "files": list(files),
                                     "manifest": f"/runs/{run_id}/manifest.json"},
                 {"values": HISTORY, "horizon": HORIZON, "title": "Test"})


def _good_history(run_id="run1"):
    return [_inspect(), _tool("backtest_series", _backtest()), _forecast(), _report(run_id)]


def _write_pack(root, run_id="run1"):
    run = root / run_id
    run.mkdir(parents=True, exist_ok=True)
    for name in ("chart.png", "chart.svg", "forecast.xlsx", "report.docx", "report.pdf"):
        (run / name).write_bytes(name.encode() * 10)
    return run


# ---------------------------------------------------------------- check_forecast

def test_forecast_length_passes_when_steps_match_horizon():
    assert _by_name(check_forecast(POINT, horizon=4), "forecast_length")["ok"] is True


def test_forecast_length_fails_when_steps_differ_from_horizon():
    c = _by_name(check_forecast(POINT, horizon=6), "forecast_length")
    assert c["ok"] is False and c["severity"] == "block"
    assert "4 steps" in c["detail"] and "6" in c["detail"]


def test_forecast_length_is_left_out_without_a_horizon():
    assert all(c["name"] != "forecast_length" for c in check_forecast(POINT))


def test_forecast_finite_passes_on_plain_numbers():
    assert _by_name(check_forecast(POINT), "forecast_finite")["ok"] is True


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_forecast_finite_fails_on_nan_or_infinity(bad):
    c = _by_name(check_forecast([12.0, bad, 14.0]), "forecast_finite")
    assert c["ok"] is False and c["severity"] == "block"


def test_interval_order_passes_when_band_brackets_point():
    c = _by_name(check_forecast(POINT, Q10, Q90), "interval_order")
    assert c["ok"] is True and c["severity"] == "warn"


def test_interval_order_warns_when_point_sits_outside_band():
    # Point above q90 at every step: 0 of 4 steps are inside the band.
    c = _by_name(check_forecast([20.0, 20.0, 20.0, 20.0], Q10, Q90), "interval_order")
    assert c["ok"] is False and c["severity"] == "warn"


def test_non_negative_fails_when_history_is_never_negative_but_forecast_dips_below_zero():
    c = _by_name(check_forecast([3.0, -1.0, 2.0], history=[0.0, 4.0, 2.0]), "non_negative_when_history_is")
    assert c["ok"] is False and c["severity"] == "block"


def test_non_negative_passes_and_is_skipped_when_history_has_negatives():
    ok = _by_name(check_forecast([3.0, 2.0], history=[0.0, 4.0]), "non_negative_when_history_is")
    assert ok["ok"] is True
    assert all(c["name"] != "non_negative_when_history_is"
               for c in check_forecast([-3.0, 2.0], history=[-5.0, 4.0]))


def test_plausible_scale_passes_for_a_forecast_near_the_history():
    c = _by_name(check_forecast(POINT, history=HISTORY), "plausible_scale")
    assert c["ok"] is True and c["severity"] == "warn"


def test_plausible_scale_warns_when_forecast_is_far_outside_history():
    # History spans 1..12 (range 11); a forecast of 500 is far beyond 12 + 3 * 11.
    c = _by_name(check_forecast([500.0, 510.0], history=HISTORY), "plausible_scale")
    assert c["ok"] is False and c["severity"] == "warn"


def test_check_forecast_values_are_json_safe_and_not_numpy():
    checks = check_forecast(POINT, Q10, Q90, horizon=4, history=HISTORY)
    assert len(checks) >= 4
    json.dumps(checks)                              # raises if a value is not plain JSON
    for c in checks:
        assert type(c["ok"]) is bool
        assert set(c) == {"name", "ok", "severity", "detail"}


def test_check_forecast_reports_a_point_that_is_not_a_list_of_numbers():
    checks = check_forecast("12, 13, 14")
    assert len(checks) == 1 and checks[0]["ok"] is False


# ---------------------------------------------------------------- check_backtest

def test_has_baselines_passes_with_a_naive_model_and_named_best_baseline():
    assert _by_name(check_backtest(_backtest()), "has_baselines")["ok"] is True


def test_has_baselines_fails_without_a_naive_model():
    result = _backtest(models={"timesfm": {"mase": 0.8}, "drift": {"mase": 1.0}}, best_baseline="drift")
    c = _by_name(check_backtest(result), "has_baselines")
    assert c["ok"] is False and c["severity"] == "block"


def test_metrics_finite_passes_when_every_mase_is_a_number():
    assert _by_name(check_backtest(_backtest()), "metrics_finite")["ok"] is True


def test_metrics_finite_fails_and_names_the_unscored_model():
    result = _backtest()
    result["models"]["drift"]["mase"] = None
    c = _by_name(check_backtest(result), "metrics_finite")
    assert c["ok"] is False and "drift" in c["detail"]


def test_beats_or_reports_passes_when_timesfm_beats_best_baseline():
    checks = check_backtest(_backtest())
    assert _by_name(checks, "beats_or_reports")["ok"] is True
    assert all(c["name"] != "model_does_not_beat_baseline" for c in checks)


def test_model_does_not_beat_baseline_warns_when_timesfm_loses_and_result_is_unflagged():
    result = _backtest()
    result["models"]["timesfm"]["mase"] = 1.2
    c = _by_name(check_backtest(result), "model_does_not_beat_baseline")
    assert c["ok"] is False and c["severity"] == "warn"


def test_flagged_result_that_does_not_beat_baseline_is_reported_not_warned():
    result = _backtest(flag="timesfm trails the baseline on these windows")
    result["models"]["timesfm"]["mase"] = 1.2
    checks = check_backtest(result)
    assert all(c["name"] != "model_does_not_beat_baseline" for c in checks)
    assert _by_name(checks, "beats_or_reports")["ok"] is True


@pytest.mark.parametrize("horizon, ok", [(4, True), (1, True), (0, False), (-2, False), (2.5, False), (True, False), (None, False)])
def test_horizon_present_needs_a_positive_whole_number(horizon, ok):
    assert _by_name(check_backtest(_backtest(horizon=horizon)), "horizon_present")["ok"] is ok


# ---------------------------------------------------------------- check_deliverables

def test_deliverable_kinds_warns_when_a_kind_is_missing_and_no_run_folder_is_given():
    files = ["chart.png", "chart.svg", "forecast.xlsx", "report.docx"]
    c = _by_name(check_deliverables(files), "deliverable_kinds")
    assert c["ok"] is False and c["severity"] == "warn" and "pdf" in c["detail"]


def test_deliverable_kinds_passes_with_all_five_kinds_listed_as_dicts():
    files = [{"path": f"out/x.{k}", "sha256": "0" * 64, "bytes": 1} for k in ("png", "svg", "xlsx", "docx", "pdf")]
    assert _by_name(check_deliverables(files), "deliverable_kinds")["ok"] is True


def test_deliverable_exists_passes_for_files_under_the_run_folder(tmp_path):
    run = _write_pack(tmp_path)
    checks = check_deliverables(["chart.png", "report.pdf"], run_dir=run)
    assert [c["name"] for c in checks] == ["deliverable_exists", "deliverable_exists"]
    assert all(c["ok"] for c in checks)


def test_deliverable_exists_fails_for_a_file_that_is_not_there(tmp_path):
    run = _write_pack(tmp_path)
    checks = check_deliverables(["chart.png", "missing.pdf"], run_dir=run)
    missing = [c for c in checks if "missing.pdf" in c["detail"]]
    assert missing and missing[0]["ok"] is False and missing[0]["severity"] == "block"


def test_deliverable_exists_fails_for_a_file_outside_the_run_folder(tmp_path):
    run = _write_pack(tmp_path / "root")
    outside = tmp_path / "elsewhere.pdf"
    outside.write_bytes(b"x")
    checks = check_deliverables([str(outside)], run_dir=run)
    assert checks[0]["ok"] is False and "outside" in checks[0]["detail"]


def test_deliverable_hash_matches_when_sha256_is_recorded(tmp_path):
    run = _write_pack(tmp_path)
    digest = hashlib.sha256((run / "chart.png").read_bytes()).hexdigest()
    checks = check_deliverables([{"path": "chart.png", "sha256": digest}], run_dir=run)
    assert _by_name(checks, "deliverable_hash")["ok"] is True


def test_deliverable_hash_mismatch_is_detected_when_the_file_has_changed(tmp_path):
    run = _write_pack(tmp_path)
    recorded = hashlib.sha256((run / "chart.png").read_bytes()).hexdigest()
    (run / "chart.png").write_bytes(b"changed after the manifest was written")
    checks = check_deliverables([{"path": "chart.png", "sha256": recorded}], run_dir=run)
    c = _by_name(checks, "deliverable_hash")
    assert c["ok"] is False and c["severity"] == "block" and "chart.png" in c["detail"]


def test_deliverable_with_no_files_fails_when_a_run_folder_is_given(tmp_path):
    c = check_deliverables([], run_dir=tmp_path)
    assert c[0]["name"] == "deliverable_exists" and c[0]["ok"] is False


# ---------------------------------------------------------------- reading the checks

def test_blocking_keeps_only_failed_block_checks():
    checks = [
        {"name": "a", "ok": False, "severity": "block", "detail": "x"},
        {"name": "b", "ok": False, "severity": "warn", "detail": "y"},
        {"name": "c", "ok": True, "severity": "block", "detail": "z"},
    ]
    assert [c["name"] for c in blocking(checks)] == ["a"]


def test_summarise_gives_blocking_sentences_first_then_warnings_and_respects_limit():
    checks = [
        {"name": "warn_one", "ok": False, "severity": "warn", "detail": "The curve is a little high."},
        {"name": "block_one", "ok": False, "severity": "block", "detail": "The forecast has 3 steps and the horizon is 4."},
        {"name": "block_two", "ok": False, "severity": "block", "detail": "A deliverable is missing"},
    ]
    text = summarise(checks)
    assert text == ("The forecast has 3 steps and the horizon is 4 [block_one]; A deliverable is missing [block_two]; "
                    "The curve is a little high [warn_one]")
    assert summarise(checks, limit=1) == "The forecast has 3 steps and the horizon is 4 [block_one]; and 2 more"
    assert summarise([{"name": "ok", "ok": True, "severity": "block", "detail": "fine"}]) == "No contract checks failed."


# ---------------------------------------------------------------- check_evidence

def test_good_mission_passes_with_no_blocking_failures(tmp_path):
    _write_pack(tmp_path)
    _write_ledger(tmp_path)
    checks = check_evidence(_good_history(), run_root=tmp_path)
    assert blocking(checks) == []
    names = {c["name"] for c in checks}
    assert {"data_inspected", "forecast_exists", "baseline_compared", "health_repaired", "forecast_length",
            "has_baselines", "deliverable_exists"} <= names


def test_mission_without_a_backtest_blocks_on_baseline_compared():
    history = [_inspect(), _forecast(), _report()]
    blocked = blocking(check_evidence(history))
    assert [c["name"] for c in blocked] == ["baseline_compared"]


def test_last_forecast_with_flat_unexpected_health_blocks():
    history = [_inspect(), _tool("backtest_series", _backtest()), _forecast("flat_unexpected")]
    c = _by_name(check_evidence(history), "health_repaired")
    assert c["ok"] is False and c["severity"] == "block"


def test_a_later_ok_forecast_clears_an_earlier_flat_unexpected_health():
    history = [_inspect(), _tool("backtest_series", _backtest()), _forecast("flat_unexpected"), _forecast("ok")]
    assert _by_name(check_evidence(history), "health_repaired")["ok"] is True


def test_forecast_before_any_data_inspection_blocks_on_data_inspected():
    history = [_tool("backtest_series", _backtest()), _forecast(), _inspect()]
    c = _by_name(check_evidence(history), "data_inspected")
    assert c["ok"] is False and c["severity"] == "block"


def test_failed_inspection_does_not_count_as_data_inspected():
    history = [_tool("sandbox_python", error="SyntaxError"), _tool("backtest_series", _backtest()), _forecast()]
    assert _by_name(check_evidence(history), "data_inspected")["ok"] is False


def test_no_forecast_yet_warns_forecast_exists_and_blocks_nothing():
    checks = check_evidence([_inspect(), _tool("backtest_series", _backtest())])
    c = _by_name(checks, "forecast_exists")
    assert c["ok"] is False and c["severity"] == "warn"
    assert blocking(checks) == []


def test_forecast_length_is_skipped_for_a_thinned_compact_forecast():
    thinned = _forecast_result(point=[1.0, 2.0, 3.0], q10=[0.5, 1.5, 2.5], q90=[1.5, 2.5, 3.5], horizon=200)
    thinned["note"] = "200 periods; every ~6th shown"
    history = [_inspect(), _tool("backtest_series", _backtest()),
               _tool("timesfm_forecast", thinned, {"values": HISTORY})]
    assert all(c["name"] != "forecast_length" for c in check_evidence(history))


def test_forecast_length_blocks_when_the_forecast_is_short_of_its_horizon():
    short = _forecast_result(point=[1.0, 2.0], q10=[0.5, 1.5], q90=[1.5, 2.5], horizon=4)
    history = [_inspect(), _tool("backtest_series", _backtest()), _tool("timesfm_forecast", short, {"values": HISTORY})]
    c = _by_name(check_evidence(history), "forecast_length")
    assert c["ok"] is False and c["severity"] == "block"


def test_deliverable_hash_mismatch_inside_a_mission_blocks(tmp_path):
    run = _write_pack(tmp_path)
    manifest_like = {"path": "chart.png", "sha256": "0" * 64}
    report = _tool("forecast_report", {"forecast": _forecast_result(), "run_id": "run1",
                                       "files": [manifest_like]}, {"values": HISTORY})
    history = [_inspect(), _tool("backtest_series", _backtest()), _forecast(), report]
    c = [x for x in check_evidence(history, run_root=tmp_path) if x["name"] == "deliverable_hash"]
    assert c and c[0]["ok"] is False and c[0]["severity"] == "block"


def test_run_id_with_path_separators_is_not_used_as_a_folder(tmp_path):
    run = _write_pack(tmp_path)
    checks = check_evidence(_good_history(run_id="../run1"), run_root=tmp_path)
    kinds = [c for c in checks if c["name"] == "deliverable_kinds"]
    assert kinds and kinds[0]["ok"] is True        # no run folder used, so only the kinds are read


@pytest.mark.parametrize("history", [
    None, 0, "text", [], [None], [1, "x"], [{}], [{"kind": None}], [{"kind": "tool"}],
    [{"kind": "tool", "tool": "timesfm_forecast"}],
    [{"kind": "tool", "tool": "timesfm_forecast", "result": "bad"}],
    [{"kind": "tool", "tool": "timesfm_forecast", "result": {"point": "abc", "config": 5, "health": None}}],
    [{"kind": "tool", "tool": "timesfm_forecast",
      "result": {"point": [math.nan], "config": {"horizon": "4"}, "health": "flat_unexpected"}}],
    [{"kind": "tool", "tool": "timesfm_forecast", "result": {"point": [10 ** 400], "config": {"horizon": 4}}}],
    [{"kind": "tool", "tool": "backtest_series", "result": {"models": None}}],
    [{"kind": "tool", "tool": "backtest_series", "result": {"models": {"naive": None}, "best_baseline": ["x"]}}],
    [{"kind": "tool", "tool": "forecast_report", "result": {"forecast": "x", "files": None}}],
    [{"kind": "tool", "tool": "forecast_report",
      "result": {"forecast": {"point": [1, 2]}, "files": [None, 3, {"path": None}], "run_id": None}}],
    [{"kind": "tool", "tool": "forecast_report", "result": {"forecast": {"point": [1]}, "files": ["x.png"], "run_id": 7}}],
    [{"kind": "tool", "tool": "sandbox_python", "error": "boom"}],
    [{"kind": "tool", "tool": "timesfm_forecast", "error": "boom", "args": None}],
    [{"kind": "tool", "tool": "sandbox_python", "result": {"ok": False}}],
    [{"kind": "ask_user", "answers": {"x": 1}}],
])
def test_check_evidence_never_raises_on_garbage(history, tmp_path):
    checks = check_evidence(history, run_root=tmp_path)
    assert isinstance(checks, list)
    for c in checks:
        assert set(c) == {"name", "ok", "severity", "detail"}
        assert isinstance(c["ok"], bool) and c["severity"] in ("block", "warn")
    assert isinstance(summarise(checks), str)
    json.dumps(checks)


def test_error_only_history_blocks_nothing_but_reports_no_forecast():
    history = [_tool("timesfm_forecast", error="model not loaded"), _tool("sandbox_python", error="boom")]
    checks = check_evidence(history)
    assert blocking(checks) == []
    assert _by_name(checks, "forecast_exists")["ok"] is False


def test_a_forecast_with_no_analysis_ledger_blocks_when_the_sandbox_folder_is_known(tmp_path):
    _write_pack(tmp_path)
    blocked = blocking(check_evidence(_good_history(), run_root=tmp_path))
    assert [c["name"] for c in blocked] == ["analysis_ledger"]


def test_a_ledger_with_too_few_stages_blocks_and_one_with_enough_passes(tmp_path):
    _write_pack(tmp_path)
    _write_ledger(tmp_path, stages=("time index", "coverage", "coverage"))
    assert [c["name"] for c in blocking(check_evidence(_good_history(), run_root=tmp_path))] == ["analysis_ledger"]
    _write_ledger(tmp_path)
    assert blocking(check_evidence(_good_history(), run_root=tmp_path)) == []


def test_ledger_entries_without_a_decision_do_not_count(tmp_path):
    import json as _json
    _write_pack(tmp_path)
    folder = tmp_path / "insp1" / "out"
    folder.mkdir(parents=True)
    (folder / "ledger.json").write_text(_json.dumps([{"stage": f"s{i}", "finding": "x", "decision": ""} for i in range(6)]))
    assert [c["name"] for c in blocking(check_evidence(_good_history(), run_root=tmp_path))] == ["analysis_ledger"]


def test_a_ledger_saved_by_a_step_that_failed_later_in_the_cell_still_counts(tmp_path):
    _write_pack(tmp_path)
    _write_ledger(tmp_path)
    history = _good_history()
    history[0]["result"]["ok"] = False          # the cell raised after ps.Ledger().save()
    history.insert(0, _inspect())
    blocked = [c["name"] for c in blocking(check_evidence(history, run_root=tmp_path))]
    assert "analysis_ledger" not in blocked


def _lab_obs(tool, result, args=None):
    return {"kind": "tool", "tool": tool, "args": args or {}, "result": result}


def test_a_hand_tuned_forecast_needs_a_control_lab_result_first():
    from prophet_agent.runtime.contracts import check_controls
    tuned = {"values": HISTORY, "znorm": True}
    assert check_controls([], {"values": HISTORY}) is None, "the default needs no lab"
    assert check_controls([], {"values": HISTORY, "znorm": False}) is None
    missing = check_controls([], tuned)
    assert missing["name"] == "controls_tested" and not missing["ok"] and missing["severity"] == "block"
    # Audit item 18: a lab counts only when it measured this forecast: same horizon and length, and the treatment in its table.
    lab = {"horizon": 10, "points": 120, "table": [{"name": "default"}, {"name": "znorm"}],
           "recommended": {"candidate": "znorm"}}
    record = {"config": {"horizon": 10, "history_points": 120}}
    ran = check_controls([_lab_obs("control_lab", lab, {"horizon": 10})], tuned, record)
    assert ran["ok"], ran
    default_lab = {"horizon": 10, "points": 120, "table": [{"name": "default"}, {"name": "znorm"}],
                   "recommended": "keep the default"}
    default = check_controls([_lab_obs("control_lab", default_lab)], tuned, record)
    assert not default["ok"] and default["severity"] == "warn"
    assert check_controls([], {"future_covariates": ["holidays"]}) is None, "drivers are inputs, not tuning"


def test_a_control_lab_that_measured_another_series_or_horizon_does_not_vouch_for_this_forecast():
    from prophet_agent.runtime.contracts import check_controls
    tuned = {"values": HISTORY, "znorm": True}
    table = [{"name": "default"}, {"name": "znorm"}]
    record = {"config": {"horizon": 10, "history_points": 120}}
    other_horizon = {"horizon": 30, "points": 120, "table": table, "recommended": {"candidate": "znorm"}}
    check = check_controls([_lab_obs("control_lab", other_horizon)], tuned, record)
    assert not check["ok"] and check["severity"] == "block" and "did not test this forecast" in check["detail"]
    other_series = {"horizon": 10, "points": 400, "table": table, "recommended": {"candidate": "znorm"}}
    assert not check_controls([_lab_obs("control_lab", other_series)], tuned, record)["ok"]


def test_the_treatment_must_appear_in_the_labs_own_table():
    from prophet_agent.runtime.contracts import check_controls
    tuned = {"values": HISTORY, "znorm": True}
    record = {"config": {"horizon": 10, "history_points": 120}}
    skipped = {"horizon": 10, "points": 120, "table": [{"name": "default"}, {"name": "ctx_4h"}],
               "recommended": "keep the default"}
    check = check_controls([_lab_obs("control_lab", skipped)], tuned, record)
    assert not check["ok"] and check["severity"] == "block" and "did not measure znorm" in check["detail"]


def test_a_lab_that_recommends_a_different_treatment_warns():
    from prophet_agent.runtime.contracts import check_controls
    tuned = {"values": HISTORY, "znorm": True}
    record = {"config": {"horizon": 10, "history_points": 120}}
    lab = {"horizon": 10, "points": 120, "table": [{"name": "znorm"}, {"name": "symmetric_averaging"}],
           "recommended": {"candidate": "symmetric_averaging"}}
    check = check_controls([_lab_obs("control_lab", lab)], tuned, record)
    assert not check["ok"] and check["severity"] == "warn" and "symmetric_averaging" in check["detail"]



def test_the_treatment_gate_accepts_a_lab_that_went_through_the_planners_observation():
    # Audit item 18: the stored control_lab result is the planner's observation (lab_tool.observe), not the raw lab dict.
    # The gate must still see the series length, the horizon and every measured candidate, or it blocks every hand-tuned forecast.
    from prophet_agent.runtime.contracts import check_controls
    from prophet_agent.tools.lab_tool import observe
    tuned = {"values": HISTORY, "znorm": True}
    record = {"config": {"horizon": 10, "history_points": 120}}
    row = lambda name, kind: {"name": name, "kind": kind, "gain_pct": 1.5, "pinball_gain_pct": 0.5, "wins": 5,
                              "interval_pct": [-1.0, 2.0], "decision": "adopt" if name == "znorm" else "reject"}
    raw = {"horizon": 10, "period": None, "points": 120, "fold_count": 8, "calls_spent": 20,
           "recommended": {"candidate": "znorm"}, "baseline_beats_default": None,
           "table": [row("default", "reference"), row("znorm", "treatment"), row("symmetric_averaging", "treatment")],
           "skipped": [], "failed": [], "caveats": []}
    stored = observe(raw)
    assert stored["points"] == 120 and stored["horizon"] == 10
    assert {"default", "znorm", "symmetric_averaging"} <= set(stored["measured"])
    check = check_controls([_lab_obs("control_lab", stored, {"horizon": 10})], tuned, record)
    assert check["ok"], check
