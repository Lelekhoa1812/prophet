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
    # A goal the research trigger does not match, so the panel rule is the first to fire (PM2.5 now triggers research first,
    # round 3 follow-up: the Beijing mission was never researched).
    plan = _harness_plan("Forecast sensor readings for every station", [INSPECT], TOOLS)
    assert plan["auto"] == "panel" and len(plan["tasks"]) == 4
    assert "ranked 1 to 3" in plan["tasks"][0]["task"] and "ranked 10 to 12" in plan["tasks"][3]["task"]
    assert all(t["agent_type"] == "analyst" for t in plan["tasks"])
    done = [INSPECT] + [{"kind": "subagent", "agent_type": "analyst", "auto": "panel"}]
    assert _harness_plan("Forecast sensor readings for every station", done, TOOLS) is None


def test_a_pollution_panel_is_researched_first_then_sliced():
    assert _harness_plan("Forecast PM2.5 for every station", [INSPECT], TOOLS)["auto"] == "research"
    researched = [INSPECT, {"kind": "subagent", "agent_type": "researcher", "auto": "research"}]
    assert _harness_plan("Forecast PM2.5 for every station", researched, TOOLS)["auto"] == "panel"


def test_five_identical_failures_hand_the_cell_to_a_fresh_analyst_once_per_error():
    bad = {"kind": "tool", "tool": "sandbox_python", "args": {"code": "x = df['nope']"},
           "result": {"ok": False, "run_id": "a" * 32, "stderr_tail": "Traceback\nKeyError: 'nope'"}}
    plan = _harness_plan("Forecast daily sales", [FLAT] + [bad] * 5, TOOLS)
    assert plan["agent_type"] == "analyst" and "x = df['nope']" in plan["task"] and plan["auto"].startswith("repeat:")
    seen = [FLAT] + [bad] * 5 + [{"kind": "subagent", "agent_type": "analyst", "auto": plan["auto"]}] + [bad] * 5
    assert _harness_plan("Forecast daily sales", seen, TOOLS) is None
    assert _harness_plan("Forecast daily sales", [FLAT] + [bad] * 4, TOOLS) is None
    good = {"kind": "tool", "tool": "sandbox_python", "args": {"code": "print(1)"}, "result": {"ok": True, "run_id": "b" * 32}}
    read = {"kind": "tool", "tool": "analyst_skill", "args": {}, "result": {"text": "..."}}
    mixed = [FLAT] + [bad, good, read, bad, good, bad, read, bad, good, bad]      # live WTI: the same line, between other calls
    assert _harness_plan("Forecast daily sales", mixed, TOOLS)["auto"].startswith("repeat:")
    assert _harness_plan("Forecast daily sales", mixed[:-1], TOOLS) is None       # newest call succeeded: nothing to hand over


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
    # Composer v7 (round 10): each line quotes its passage; the figure check still runs on a quoted line.
    answer = ("The forecast is flat.\n\n## Outside context\n- IMF (20 Nov 2026): \"The IMF rate is JPY 159.47 per USD\".\n"
              "- IMF (20 Nov 2026): \"The IMF rate is JPY\" and EUR/JPY 162.34.\n\n## Files\n- out/forecast.png")
    out, removed = _check_outside(c, answer)
    assert removed == ["162.34"] and "159.47" in out and "162.34 ." not in out.split("_Some")[0]
    assert "EUR/JPY 162.34" not in out and "## Files" in out and "1 outside claim(s) were left out" in out
    assert "162.34" not in out          # the note gives a count, never the unsourced figure itself
    assert _check_outside(c, "No outside section.") == ("No outside section.", [])


def test_the_panel_trigger_reads_the_full_profile_from_the_ledger_when_history_holds_the_compact_view():
    compact = {"kind": "tool", "tool": "inspect_data", "args": {}, "result": {"file": "a.csv", "columns": ["text unique=12"]}}
    assert _harness_plan("Forecast sensor readings for every station", [compact], TOOLS) is None
    plan = _harness_plan("Forecast sensor readings for every station", [compact], TOOLS, lambda t: [INSPECT["result"]] if t == "inspect_data" else [])
    assert plan["auto"] == "panel"


def test_newest_data_outputs_are_carried_into_every_sandbox_run():
    from prophet_agent.runtime.goal import _auto_carry
    h = [{"kind": "tool", "tool": "sandbox_python", "args": {}, "result": {"ok": True, "run_id": "a" * 32,
          "outputs": ["out/scan.json (10 bytes)", "out/chart.png (99 bytes)"]}},
         {"kind": "tool", "tool": "sandbox_python", "args": {}, "result": {"ok": True, "run_id": "b" * 32,
          "outputs": ["out/scan.json (12 bytes)", "out/series.json (50 bytes)"]}}]
    out = _auto_carry({"code": "x"}, h)
    assert {"run_id": "b" * 32, "file": "out/scan.json"} in out["carry"] and len(out["carry"]) == 2   # png not carried
    mine = _auto_carry({"code": "x", "carry": [{"run_id": "a" * 32, "file": "out/scan.json"}]}, h)
    assert mine["carry"][0]["run_id"] == "a" * 32 and len(mine["carry"]) == 2      # the planner's choice wins
    assert _auto_carry({"code": "x"}, []) == {"code": "x"}


def test_a_stuck_checklist_item_goes_to_a_fresh_analyst_once_and_its_ledger_counts(tmp_path):
    import json
    from prophet_agent.runtime import contracts
    plan = _harness_plan("Forecast daily sales", [FLAT], TOOLS, stuck=("analysis ledger: at least 4 stages", 16))
    assert plan["auto"] == "stuck:analysis ledger: at least 4 stages" and "Do exactly this step" in plan["task"]
    assert _harness_plan("Forecast daily sales", [FLAT], TOOLS, stuck=("analysis ledger: at least 4 stages", 15)) is None
    done = [FLAT, {"kind": "subagent", "agent_type": "analyst", "auto": plan["auto"]}]
    assert _harness_plan("Forecast daily sales", done, TOOLS, stuck=("analysis ledger: at least 4 stages", 30)) is None
    run = "c" * 32
    (tmp_path / run / "out").mkdir(parents=True)
    (tmp_path / run / "out" / "ledger.json").write_text(json.dumps(
        [{"stage": s, "finding": "f", "decision": "d"} for s in ("time index", "coverage", "zeros", "outliers")]))
    sub = {"kind": "subagent", "agent_type": "analyst", "evidence": [{"kind": "tool", "tool": "sandbox_python",
           "result": {"ok": True, "run_id": run, "outputs": ["out/ledger.json (200 bytes)", "out/scan_A.json (9 bytes)"]}}]}
    assert contracts.ledger_stages([sub], tmp_path) == 4 and contracts.scan_ran([sub])
    assert contracts.ledger_stages([], tmp_path) == 0


def test_the_rescue_brief_carries_the_items_recipe():
    # live retail_biz rep6 (2026-11-23): the analyst got the ledger item's name only and printed the scan 14 times
    recipe = "0 of 4 recorded; One cell does it: led = ps.Ledger(); led.decide('calendar.closed_days', '<your decision>')"
    plan = _harness_plan("Forecast daily sales", [FLAT], TOOLS, stuck=("analysis ledger: at least 4 stages", 16, recipe))
    assert plan["auto"] == "stuck:analysis ledger: at least 4 stages"          # the dedup key is the item alone
    assert "led.decide('calendar.closed_days'" in plan["task"]


def test_repeated_download_failures_are_not_handed_to_an_analyst_who_cannot_download():
    # VN Unemployment (2026-11-27): five HTTP 404s from guessed series codes went to an analyst with no download tools.
    bad = {"kind": "tool", "tool": "download_open_data", "args": {"source": "fred", "params": {"id": "VNUNR"}},
           "error": "HTTP 404; only a 200 response is used"}
    plan = _harness_plan("Forecast daily sales", [FLAT] + [bad] * 5, TOOLS)
    assert plan is None or not str(plan.get("auto", "")).startswith("repeat:")


def test_files_a_sub_agent_wrote_are_auto_carried_for_the_lead():
    from prophet_agent.runtime.goal import _runs_with_outputs
    sub_step = {"kind": "tool", "tool": "sandbox_python", "args": {}, "result": {"ok": True, "run_id": "f" * 32,
                                                                                    "outputs": ["out/vn_annual.csv (120 bytes)"]}}
    history = [FLAT, {"kind": "subagent", "agent_type": "analyst", "evidence": [sub_step]}]
    assert ("f" * 32, {"out/vn_annual.csv"}) in _runs_with_outputs(history)


def test_a_question_option_with_a_figure_no_tool_printed_is_refused():
    # Round 20 (EPL): "Liverpool (28 points)" against a downloaded table that said 21.
    from prophet_agent.runtime.goal import _option_figure_problems
    h = [{"kind": "tool", "tool": "download_web_data", "args": {}, "result": {"columns": ["team", "played", "pts"],
                                                                                   "first_rows": [["Liverpool", "12", "21"]]}}]
    qs = [{"question": "Which clubs?", "header": "Clubs", "multiSelect": True, "options": [
        {"label": "Liverpool (28 points)", "description": "leader"}, {"label": "Top 6 clubs", "description": "keep six"},
        {"label": "Liverpool (21 pts)", "description": "from the table"}]}]
    problems = _option_figure_problems(qs, h, "Research the EPL 2026/27")
    assert len(problems) == 1 and "options[0] states 28" in problems[0]
