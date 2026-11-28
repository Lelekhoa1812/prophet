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
    hist = [_fr(25, "combination", [4.31, 4.4]), _fr(12, "drift", [7143.0]), _fr(25, "combination", [4.52, 4.6]),
            _fr(24, "combination", [4.1])]
    assert goal.superseded_by(hist, start=1) == {1: 3}           # another horizon is another forecast (round 201606)
    text = goal._evidence_lines(hist)
    assert "1. [SUPERSEDED by step 3" in text and "2. [SUPERSEDED" not in text
    assert "4.52" in text


def test_a_figure_only_in_a_superseded_run_is_unverified():
    hist = [_fr(25, "combination", [4.31]), _fr(25, "combination", [4.52])]
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


def test_every_team_of_the_simulation_is_appended_in_order():
    from prophet_agent.runtime import league_table
    teams = [{"team": f"Club {k}", "current": 30 - k, "expected_final": 80.0 - 2.5 * k, "expected_position": k + 1.2,
              "final": {"p10": 70 - 2 * k, "median": 80 - 2 * k, "p90": 90 - 2 * k}, "title": max(0.0, 0.4 - 0.05 * k),
              "top_places": 0.9 - 0.04 * k, "relegation": 0.01 * k} for k in range(20)]
    teams.reverse()
    hist = [{"kind": "tool", "tool": "simulate_season", "result": {"teams": teams[:6]},
             "full": {"teams": teams, "settings": {"n_sims": 10000, "n_played": 110, "n_remaining": 270,
                                                   "signals": {"history_matches": 0, "market": 0, "xg": 0}}}}]
    md = league_table.table_markdown(hist)
    rows = [ln for ln in md.splitlines() if ln.startswith("| ") and "Pos" not in ln]
    assert len(rows) == 20 and rows[0].startswith("| 1 | Club 0 | 30 | 80.0 | 70 to 90 |") and "40.0%" in rows[0]
    assert "signals counted: history_matches 0, market 0, xg 0" in md
    assert league_table.table_markdown([{"kind": "tool", "tool": "simulate_season", "full": {"teams": teams[:2]}}]) == ""
    msgs = goal._compose_messages("g", hist, goal.Memo(), "", "draft")
    assert "appended after your answer by code" in msgs[1]["content"]


def test_a_table_cell_matches_a_backtest_keyed_by_model_name():
    """Round 20261128-201606 (AUUNEMP): TimesFM 0.879 and drift 0.814 came from backtest_series, keyed by model name."""
    from prophet_agent.runtime import table_cells
    hist = [{"kind": "tool", "tool": "backtest_series", "result": {"models": {"timesfm": {"mase": 0.8792},
                                                                           "drift": {"mase": 0.8141}}}},
            {"kind": "tool", "tool": "forecast_report", "result": {"combination": {"candidates": [
                {"name": "timesfm", "mase": 4.4171}, {"name": "drift", "mase": 5.3959}]}}}]
    answer = "| Model | MASE |\n|---|---|\n| TimesFM | 0.879 |\n| Drift | 0.814 |\n"
    assert table_cells.mismatches(answer, hist) == []
    assert table_cells.mismatches(answer.replace("0.814", "0.700"), hist)[0]["stated"] == "0.700"



def test_a_backtest_of_a_saved_series_starts_the_combination_when_no_forecast_exists():
    """Round 20261128-201606 (GOLD): a backtest, a refused driver scan, a stall, and no forecast at all."""
    tools = {"forecast_report": object()}
    g = "Find the gold price (US dollars per troy ounce) and forecast it month by month over the next year with a range"
    src = {"run_id": "b36c", "file": "out/series.json"}
    hist = [{"kind": "tool", "tool": "backtest_series", "args": {"source": src, "horizon": 12, "price": True},
             "result": {"best_baseline": "drift"}}]
    step = goal._combination_step(g, hist, tools)
    assert step["args"]["source"] == src and step["args"]["horizon"] == 12 and step["args"]["centre"] == "combination"
    failed = [{**hist[0], "error": "boom"}]
    failed[0].pop("result")
    assert goal._combination_step(g, failed, tools) is None


def test_a_series_built_from_a_carried_official_series_keeps_its_source(tmp_path, monkeypatch):
    """Round 20261128-201606 (VNUNEMP): backtest_series refused the derived series four times ("a source needs id")."""
    import json
    from prophet_agent.tools import series_io
    monkeypatch.setattr(series_io, "SANDBOX_ROOT", tmp_path)
    monkeypatch.setattr(series_io, "_fetch_record_exists", lambda url: url.startswith("https://www.nso.gov.vn/"))
    rid = "a" * 32
    carried = tmp_path / rid / "in" / "carried"
    carried.mkdir(parents=True)
    good = {"origin": "retrieved", "source_name": "Vietnam statistics office",
            "sources": [{"id": "S1", "url": "https://www.nso.gov.vn/release-q3-2026/"}]}
    (carried / "series_official.json").write_text(json.dumps({"series": {"u": [2.2]}, "provenance": good}))
    made_up = {"origin": "retrieved", "source_name": "GSO", "sources": [{"id": "official_series", "url": "https://gso.gov.vn"}]}
    got = series_io.traced_provenance(made_up, "out/series.json", rid)
    assert got["sources"][0]["url"].startswith("https://www.nso.gov.vn/") and got["inherited_from"] == "series_official.json"
    (carried / "series_official.json").write_text(json.dumps({"series": {"u": [2.2]}, "provenance": made_up}))
    import pytest
    with pytest.raises(ValueError, match="needs id"):
        series_io.traced_provenance(made_up, "out/series.json", rid)


def test_a_quoted_source_headline_is_not_a_claimed_total():
    """Round 20261128-201606 (AURATE): a quoted ABS headline blocked a cash-rate answer as a stated total."""
    quoted = '- **ABS (abs.gov.au, 3 Nov 2026):** "Total dwelling approvals rise 5.1% in September" - housing is picking up.'
    assert grounding.claimed_totals(quoted) == []
    assert grounding.claimed_totals("The total over the horizon is 52.3 percentage points.")
