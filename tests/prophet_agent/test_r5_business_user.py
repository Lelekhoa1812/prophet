"""Round 5: defects found by a business-user mission (live retail_biz, measured 2026-11-23).

The person asked "Sales feel like they've been slipping lately. Is that real...? And what should I expect for next month?"
- The answer named "seasonal naïve" as the best simple rule; the backtest ranked the last-season mean best. The accent hid it
  from the benchmark check.
- The measured range of the 20-day total, appended by code (777,152 to 1,137,370), was flagged as unverified when the answer
  repeated it, so the run ended partial.
"""
from prophet_agent.runtime import contracts, grounding


def test_an_accented_rule_name_is_read():
    assert grounding.baselines_named("best simple rule (seasonal naïve)") == ["seasonal_naive"]
    bt = {"kind": "tool", "tool": "backtest_series", "args": {}, "result": {
        "horizon": 20, "origins": 12, "best_baseline": "mean_last_season",
        "models": {"seasonal_naive": {"mase": 1.2}, "mean_last_season": {"mase": 1.03}, "timesfm": {"mase": 0.81}}}}
    bad = [c for c in contracts.check_answer("MASE 0.81 versus 1.2 for the best simple rule (seasonal naïve).", [bt])
           if c["name"] == "answer_benchmark_matches" and not c["ok"]]
    assert bad


def _measured_history():
    rows = [{"step": i + 1, "point": 100.0, "q10": 80.0, "q90": 120.0} for i in range(4)]
    pack = {"kind": "tool", "tool": "forecast_report", "args": {},
            "result": {"table": {"rows": rows, "rows_omitted": 0}, "total_range_80": {"centre": 400.0}}}
    bt = {"kind": "tool", "tool": "backtest_series", "args": {}, "result": {
        "horizon": 4, "origins": 12, "total_error": {"variance_ratio": 2.0, "origins": 12, "held_measured": 9}}}
    return [pack, bt]


def test_the_appended_measured_total_range_is_evidence():
    hist = _measured_history()
    rng = contracts.measured_total_range(hist)
    text = f"80% range of the total: {rng['q10']:,.0f} to {rng['q90']:,.0f}."
    assert grounding.unverified(text, hist, "goal") == []


def test_the_questions_in_the_goal_reach_the_evaluator_and_the_composer():
    from prophet_agent.runtime.goal import Memo, _compose_messages, _eval_messages, asked_questions
    goal = ("Sales feel like they've been slipping lately. Is that real, or just the usual ups and downs? "
            "And what should I expect for next month?")
    assert asked_questions(goal) == ["Is that real, or just the usual ups and downs?",
                                     "And what should I expect for next month?"]
    assert asked_questions("Forecast net sales for the next 14 days.") == []
    ev = _eval_messages(goal, [], Memo())[-1]["content"]
    assert "met only when the evidence answers each one" in ev and "1. Is that real" in ev
    comp = _compose_messages(goal, [], Memo(), "", "draft")[-1]["content"]
    assert "Open the answer by answering each one directly" in comp and "2. And what should I expect" in comp


def _wti_history(answer="All of December 2026 (calendar month)", last_row="2026-12-16"):
    insp = {"kind": "tool", "tool": "inspect_data", "args": {}, "result": {"columns": ["date: time first=2016-11-21 last=2026-11-17"]}}
    ask = {"kind": "ask_user", "answers": {"Period": answer}}
    rows = [{"step": 1, "date": "2026-11-18", "point": 90.0, "q10": 80.0, "q90": 100.0},
            {"step": 21, "date": last_row, "point": 90.0, "q10": 78.0, "q90": 103.0}]
    pack = {"kind": "tool", "tool": "forecast_report", "args": {}, "result": {"table": {"rows": rows, "rows_omitted": 0}}}
    return [insp, ask, pack]


def test_a_named_month_from_the_persons_answer_is_the_span_the_pack_must_reach():
    from prophet_agent.runtime.goal import horizon_checks, requested_window
    import datetime as dt
    goal = "I need a number for next month's fuel budget. What will crude cost on average next month?"
    hist = _wti_history()
    assert requested_window(goal, hist) == ("December 2026", dt.date(2026, 12, 31))
    c = horizon_checks(goal, hist)
    assert c and c[0]["name"] == "window_matches_request" and not c[0]["ok"] and c[0]["severity"] == "block"
    assert "2026-12-16" in c[0]["detail"]
    assert horizon_checks(goal, _wti_history(last_row="2026-12-31"))[0]["ok"]
    assert horizon_checks(goal, _wti_history(last_row="2026-12-29"))[0]["ok"]      # last trading day before year end


def test_next_month_with_no_answer_is_the_month_after_the_data():
    from prophet_agent.runtime.goal import requested_window
    hist = _wti_history()[:1]
    assert requested_window("What will it cost on average next month?", hist)[0] == "next month (December 2026)"
    assert requested_window("Sales since January fell.", hist) is None


def test_the_pack_and_the_measured_range_give_the_average_with_its_range():
    from prophet_agent.tools.report_tool import total_range
    tr = total_range({"point": [10.0, 20.0], "quantiles": {"0.1": [8.0, 16.0], "0.9": [12.0, 24.0]}})
    assert tr["mean_per_step"] == 15.0 and tr["if_days_miss_together"]["mean_q10"] == 12.0
    hist = _measured_history()
    rng = contracts.measured_total_range(hist)
    assert rng["mean"] == 100.0
    text = f"Average {rng['mean']:,.2f}, range {rng['mean_q10']:,.2f} to {rng['mean_q90']:,.2f}."
    assert grounding.unverified(text, hist, "goal") == []
