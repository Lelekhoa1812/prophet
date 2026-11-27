"""Round 21 repairs (2026-11-27). "The next N units" runs from today, not from the data's last point, and a forecast from data far
older than a fresh release is blocked. GOLD forecast 2017-10 to 2018-09 from a stopped series and was judged met; VNUNEMP and
AURATE forecast from 2024. AURATE's month-end dates gave a 31-day step, so 5 years never matched the 60 months it ran."""
from __future__ import annotations

from prophet_agent.runtime import goal as g

TODAY = "2026-11-27"


def _history(horizon, last_date, step=31.0):
    return [{"kind": "tool", "tool": "timesfm_forecast", "args": {"horizon": horizon},
             "result": {"point": [1.0] * horizon, "config": {"horizon": horizon, "series_step": step},
                        "last_observed": {"date": last_date, "value": 1.0}}}]


def _named(checks, name):
    found = [c for c in checks if c["name"] == name]
    return found[0] if found else None


def test_a_monthly_step_of_31_days_is_a_calendar_month():
    checks = g.horizon_checks("Forecast it over the next 5 years.", _history(60, "2026-10-31"), TODAY)
    assert _named(checks, "horizon_matches_request")["ok"] is True
    assert _named(checks, "data_is_current") is None, "October data at the end of November is current"


def test_old_data_needs_the_gap_plus_the_request_and_is_blocked_as_stale():
    checks = g.horizon_checks("Forecast it month by month over the next 2 years.", _history(24, "2024-12-31"), TODAY)
    span = _named(checks, "horizon_matches_request")
    assert span["ok"] is False and "from today (2026-11-27)" in span["detail"] and "2024-12-31" in span["detail"]
    stale = _named(checks, "data_is_current")
    assert stale["ok"] is False and stale["severity"] == "block" and "696 days" in stale["detail"]
    longer = g.horizon_checks("Forecast it month by month over the next 2 years.", _history(46, "2024-12-31"), TODAY)
    assert _named(longer, "horizon_matches_request")["ok"] is True, "22 months to today plus 24 asked"


def test_without_a_run_date_the_check_is_as_before():
    checks = g.horizon_checks("Forecast it month by month over the next 2 years.", _history(24, "2024-12-31"))
    assert _named(checks, "horizon_matches_request")["ok"] is True and _named(checks, "data_is_current") is None


def test_quarterly_data_has_a_longer_allowance():
    fresh = g.horizon_checks("Forecast over the next 2 years.", _history(9, "2026-06-30", step=91.0), TODAY)
    assert _named(fresh, "data_is_current") is None
    old = g.horizon_checks("Forecast over the next 2 years.", _history(8, "2024-12-31", step=92.0), TODAY)
    assert _named(old, "data_is_current")["ok"] is False


def test_the_end_of_data_falls_back_to_the_forecast_window():
    hist = [{"kind": "tool", "tool": "timesfm_forecast", "args": {"horizon": 12},
             "result": {"point": [1.0] * 12, "config": {"horizon": 12, "series_step": "monthly"},
                        "window": {"first": "2017-10-01", "last": "2018-09-01", "steps": 12}}}]
    checks = g.horizon_checks("Forecast it month by month over the next year.", hist, TODAY)
    assert _named(checks, "data_is_current")["ok"] is False
    assert _named(checks, "horizon_matches_request")["ok"] is False


def test_the_next_year_without_a_digit_is_one_year():
    assert g.requested_horizon("forecast it month by month over the next year with a range") == (1, "year")
    assert g.requested_horizon("within the coming quarter") == (1, "quarter")
    assert g.requested_horizon("what will rates be next year") is None
    assert g.requested_horizon("over the next 2 years, then the next year after") == (2, "year")


def test_total_as_a_category_label_is_not_a_headline_total():
    """Round 21 (AUUNEMP): "ages 15+, total" was read as a stated total and kept the answer partial."""
    from prophet_agent.runtime import grounding as gr
    for label in ("The unemployment rate (ages 15+, total) was 4.9% in October 2026.",
                  "Both sexes, total: 4.3% in September.",
                  "The total unemployment rate is forecast at 5.1% by 2028.",
                  "Persons, total (both sexes) stood at 4.1% in 2025."):
        assert gr.claimed_totals(label) == [], label
    for claim in ("The total over the next 24 months is 118.2 percentage points.", "In all, sales reach 12,400 units."):
        assert gr.claimed_totals(claim) == [claim], claim


def test_the_loop_blocks_a_forecast_from_stale_data_with_the_run_date(tmp_path):
    from prophet_agent.runtime.goal import GoalConfig, goal_program
    from prophet_agent.runtime.runner import Runtime, Tool
    from prophet_agent.runtime.store import Store

    stale = {"ok": True, "point": [1.0] * 24, "config": {"horizon": 24, "series_step": 31.0},
             "last_observed": {"date": "2017-09-30", "value": 1.0}}

    class Brain:
        def decide(self, messages, schema, name):
            if name == "plan":
                return ({"action": "call_tool", "tool": "timesfm_forecast", "args": {"horizon": 24}, "rationale": "x"}
                        if not getattr(self, "called", False) and not setattr(self, "called", True)
                        else {"action": "finish", "rationale": "done"})
            return {"verdict": "met", "reason": "ok", "answer": "The gold price forecast is 1,300."}

    store = Store(tmp_path / "s.sqlite")
    goal = "Forecast the gold price month by month over the next 2 years."
    tools = {"timesfm_forecast": Tool("timesfm_forecast", lambda a, k: stale, idempotent=True)}
    out = Runtime(store).run(store.create_mission(goal, {}), goal_program(goal, tools, Brain(), GoalConfig(max_iterations=14)))
    store.close()
    assert out["status"] != "failed", out
    assert out["result"]["status"] != "met"
    assert "data_is_current" in (out["result"].get("contract_failures") or []) or "data_is_current" in str(out["result"])


def test_options_that_name_a_league_position_need_computed_standings():
    """Round 21 (EPL): "2nd after 12 rounds" was offered before any table existed, and was wrong."""
    qs = [{"question": "Which clubs?", "header": "Clubs", "multiSelect": True,
           "options": [{"label": "Arsenal", "description": "Current leader after 12 rounds"},
                       {"label": "Liverpool", "description": "2nd after 12 rounds"}]}]
    fixtures = [{"kind": "tool", "tool": "sandbox_python", "result": {"ok": True, "stdout": "Home Team Away Team Result"}}]
    problems = g._option_figure_problems(qs, fixtures, "Who can win the league?")
    assert any("no tool has computed the standings" in p for p in problems)
    table = fixtures + [{"kind": "tool", "tool": "sandbox_python",
                         "result": {"ok": True, "stdout": "position team played pts\n1 Arsenal 11 28\n2 Brighton 11 24"}}]
    assert not any("standings" in p for p in g._option_figure_problems(qs, table, "Who can win the league?"))
    plain = [{"question": "Which grain?", "header": "Grain", "multiSelect": False,
              "options": [{"label": "Quarterly", "description": "As published"}, {"label": "Monthly", "description": "Derived"}]}]
    assert g._option_figure_problems(plain, fixtures, "Forecast unemployment") == []
