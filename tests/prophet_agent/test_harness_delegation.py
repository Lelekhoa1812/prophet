"""The harness starts sub-agents itself in three situations the planner left alone live: market research, panels, repeat failures."""
from prophet_agent.runtime import contracts
from prophet_agent.runtime.goal import _harness_plan

TOOLS = {"web_research": object(), "sandbox_python": object()}
INSPECT = {"kind": "tool", "tool": "inspect_data", "args": {}, "result": {"file": "a.csv", "profile": {"columns": [
    {"kind": "text", "name": "station", "unique": 12},
    {"kind": "time", "name": "date", "duplicate_timestamps": 16071},
    {"kind": "number", "name": "pm25"}]}}}
FLAT = {"kind": "tool", "tool": "inspect_data", "args": {}, "result": {"file": "wti.csv", "profile": {"columns": [
    {"kind": "time", "name": "date", "duplicate_timestamps": 0}, {"kind": "number", "name": "price"}]}}}


def test_nothing_happens_before_the_data_is_looked_at():
    assert _harness_plan("Forecast the WTI spot price", [], TOOLS) is None


def test_a_market_goal_gets_a_researcher_once():
    plan = _harness_plan("Forecast the WTI spot price for 20 trading days", [FLAT], TOOLS)
    assert plan["action"] == "delegate" and plan["agent_type"] == "researcher" and plan["auto"] == "research"
    done = [FLAT, {"kind": "subagent", "agent_type": "researcher", "auto": "research"}]
    assert _harness_plan("Forecast the WTI spot price", done, TOOLS) is None
    assert _harness_plan("Forecast daily sales", [FLAT], TOOLS) is None          # not a market series


def test_a_panel_gets_parallel_analysts_on_slices_before_any_forecast():
    plan = _harness_plan("Forecast PM2.5 for every station", [INSPECT], TOOLS)
    assert plan["auto"] == "panel" and len(plan["tasks"]) == 4
    assert "ranked 1 to 3" in plan["tasks"][0]["task"] and "ranked 10 to 12" in plan["tasks"][3]["task"]
    assert all(t["agent_type"] == "analyst" for t in plan["tasks"])
    done = [INSPECT] + [{"kind": "subagent", "agent_type": "analyst", "auto": "panel"}]
    assert _harness_plan("Forecast PM2.5 for every station", done, TOOLS) is None


def test_five_identical_failures_hand_the_cell_to_a_fresh_analyst_once_per_error():
    bad = {"kind": "tool", "tool": "sandbox_python", "args": {"code": "x = df['nope']"},
           "result": {"ok": False, "run_id": "a" * 32, "stderr_tail": "Traceback\nKeyError: 'nope'"}}
    plan = _harness_plan("Forecast daily sales", [FLAT] + [bad] * 5, TOOLS)
    assert plan["agent_type"] == "analyst" and "x = df['nope']" in plan["task"] and plan["auto"].startswith("repeat:")
    seen = [FLAT] + [bad] * 5 + [{"kind": "subagent", "agent_type": "analyst", "auto": plan["auto"]}] + [bad] * 5
    assert _harness_plan("Forecast daily sales", seen, TOOLS) is None
    assert _harness_plan("Forecast daily sales", [FLAT] + [bad] * 4, TOOLS) is None


def test_the_accuracy_paragraph_says_when_a_simple_rule_wins():
    h = [{"kind": "tool", "tool": "backtest_series", "args": {}, "result": {
        "best_baseline": "naive", "horizon": 30, "origins": 8, "models": {"naive": {"mase": 0.2038}, "timesfm": {"mase": 0.2294}},
        "timesfm_range_coverage_80": 0.8, "timesfm_vs_best_baseline_pct": -12.6}},
         {"kind": "tool", "tool": "calibrate_intervals", "args": {}, "result": {
             "heldout_coverage_raw": 0.692, "heldout_coverage_calibrated": 0.708, "recommend": "keep_raw"}}]
    text = contracts.trust_statement(h)
    assert "0.229" in text and "0.204" in text and "simple rule was more accurate" in text
    assert "69% raw" in text and "71% after calibration" in text
    assert contracts.trust_statement([]) == ""


def test_the_state_card_names_the_newest_file_of_each_kind_and_the_sub_agent_reports():
    from prophet_agent.runtime.goal import _state_card
    h = [{"kind": "tool", "tool": "sandbox_python", "args": {}, "result": {"ok": True, "run_id": "r1", "outputs": [{"path": "out/series.json"}]}},
         {"kind": "tool", "tool": "sandbox_python", "args": {}, "result": {"ok": True, "run_id": "r2", "outputs": [{"path": "out/series.json"}, {"path": "out/scan.json"}]}},
         {"kind": "subagent", "agent_id": "a1", "agent_type": "researcher", "status": "met", "answer": "OPEC meets 29 Nov", "files": []}]
    card = _state_card(h)
    assert "out/series.json from run r2" in card and "out/scan.json from run r2" in card and "r1" not in card
    assert "Sub-agent a1 (researcher, met): OPEC meets 29 Nov" in card
    assert _state_card([]) == ""


def test_outside_claims_whose_figures_no_page_contains_are_removed(tmp_path):
    from prophet_agent.runtime.goal import _check_outside
    from prophet_agent.runtime.store import Store

    class C:
        pass
    store = Store(tmp_path / "s.sqlite")
    mid = store.create_mission("goal", {})
    c = C(); c.store, c.mission_id = store, mid
    store.tool_start(mid, 0, "web_research", {"queries": ["x"]}, "k0")
    store.tool_set("k0", "done", {"sources": [{"passages": [{"text": "The IMF rate is JPY 159.47 per USD on 20 Nov 2026."}]}]})
    answer = ("The forecast is flat.\n\n## Outside context\n- IMF (20 Nov 2026): JPY 159.47 per USD.\n"
              "- IMF (20 Nov 2026): EUR/JPY 162.34.\n\n## Files\n- out/forecast.png")
    out, removed = _check_outside(c, answer)
    assert removed == ["162.34"] and "159.47" in out and "162.34 ." not in out.split("_Some")[0]
    assert "EUR/JPY 162.34" not in out and "## Files" in out and "left out" in out
    assert _check_outside(c, "No outside section.") == ("No outside section.", [])
