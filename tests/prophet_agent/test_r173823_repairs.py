"""Repairs from the audit of panel round 20261128-173823: superseded runs, stale figures, combination claims.

Every history here is hand-built; no figure is an observed result."""
from prophet_agent.runtime import goal, grounding
from prophet_agent.runtime import prediction_quality as pq


def _fr(h, centre, values, gains=None):
    res = {"table": values, "horizon": h}
    if gains:
        res["combination"] = {"weights": {"timesfm": 0.5, "naive": 0.5}, "gains": gains}
    return {"kind": "tool", "tool": "forecast_report", "args": {"horizon": h, "centre": centre}, "result": res}


def test_a_rerun_supersedes_the_earlier_call_with_the_same_centre_only():
    hist = [_fr(24, "combination", [4.31, 4.4]), _fr(12, "drift", [7143.0]), _fr(25, "combination", [4.52, 4.6])]
    assert goal.superseded_by(hist, start=1) == {1: 3}
    text = goal._evidence_lines(hist)
    assert "1. [SUPERSEDED by step 3" in text and "2. [SUPERSEDED" not in text
    assert "4.52" in text


def test_a_figure_only_in_a_superseded_run_is_unverified():
    hist = [_fr(24, "combination", [4.31]), _fr(25, "combination", [4.52])]
    assert grounding.unverified("The rate ends at 4.52%.", hist, "g") == []
    assert grounding.unverified("The rate ends at 4.31%.", hist, "g") == ["4.31%"]


def test_combination_claims_follow_the_sign_of_the_backtest():
    gains = {"vs_equal_wis_pct": -0.41, "vs_best_so_far_wis_pct": 9.1, "vs_best_single_wis_pct": -12.95,
             "best_single": "timesfm"}
    hist = [_fr(60, "combination", [4.6], gains)]
    bad = pq.check_combination_claims("The combination beats TimesFM by 9.1%. It also beats equal weights by 0.4%.", hist)
    assert bad["ok"] is False and "-12.95" in bad["detail"] and "-0.41" in bad["detail"]
    good = pq.check_combination_claims("It beats the best candidate so far by 9.1% but trails TimesFM by 12.95% and "
                                       "equal weights by 0.41%.", hist)
    assert good["ok"] is True
    assert pq.check_combination_claims("", hist) is None
    assert pq.check_combination_claims("x", [_fr(60, "timesfm", [1.0])]) is None


def test_a_reworded_lane_brief_that_already_reported_is_left_out():
    from prophet_agent.runtime import lanes
    hist = [{"kind": "subagent", "lane": True, "status": "not_met", "name": "GSO labour releases",
             "task": "Find Vietnam's General Statistics Office (GSO) quarterly Labour Force Survey releases for 2025-Q1 "
                     "through 2026-Q2. Search 'Vietnam GSO labour force survey 2025 Q1 Q2 Q3 Q4'. Return each finding "
                     "with its period, unemployment rate value, unit, exact quote and URL."}]
    again = {"name": "GSO 2025-26 LFS",
             "task": "Find Vietnam's General Statistics Office (GSO) quarterly Labour Force Survey releases for 2025-Q1 "
                     "through 2026-Q4. Search 'Vietnam GSO labour force survey 2026 Q1 Q2 Q3 Q4'. Return each finding "
                     "with its period, unemployment rate value, unit, exact quote and URL."}
    other = {"name": "Vietnam GDP", "task": "Find Vietnam's latest quarterly GDP growth and the manufacturing PMI for 2025 "
                                            "and 2026 from the statistics office and S&P Global."}
    keep, skipped = lanes._not_run_before([again, other], hist)
    assert keep == [other] and "same task as the earlier lane" in skipped[0]
