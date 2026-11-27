"""A prediction is charted, and a flat centre line is held until several ways of finding its shape were tried (owner, 2026-11-27)."""
from __future__ import annotations

from prophet_agent.runtime import prediction_quality as pq


def pack(point, files=("out/forecast.png (1 KB)",), centre="timesfm"):
    q10 = [v - 5 for v in point]
    q90 = [v + 5 for v in point]
    return {"kind": "tool", "tool": "forecast_report", "args": {"centre": centre},
            "result": {"ok": True, "files": list(files), "forecast": {"point": point}},
            "full": {"forecast": {"point": point, "quantiles": {"0.1": q10, "0.9": q90}}}}


def call(tool, args=None, **result):
    return {"kind": "tool", "tool": tool, "args": args or {}, "result": {"ok": True, **result}}


FLAT = [10.0] * 12
MOVING = [10.0 + 2 * i for i in range(12)]


def test_forecast_with_a_chart_passes():
    c = pq.check_prediction_chart([pack(MOVING)])
    assert c["ok"]


def test_numbers_only_forecast_is_blocked_with_how_to_chart():
    c = pq.check_prediction_chart([call("timesfm_forecast", point=MOVING)])
    assert not c["ok"] and c["severity"] == "block" and "forecast_report" in c["detail"]


def test_league_simulation_without_a_picture_asks_for_a_club_compared_chart():
    c = pq.check_prediction_chart([call("simulate_season", teams=[{"team": "A"}])])
    assert not c["ok"] and "club-compared" in c["detail"] and "finishing place" in c["detail"]


def test_league_chart_written_after_the_simulation_passes():
    hist = [call("simulate_season"), call("sandbox_python", outputs=["out/clubs.png (20 KB)"])]
    assert pq.check_prediction_chart(hist)["ok"]


def test_a_chart_made_before_the_prediction_does_not_count():
    hist = [call("sandbox_python", outputs=["out/eda.png"]), call("simulate_season")]
    assert not pq.check_prediction_chart(hist)["ok"]


def test_no_prediction_no_chart_check():
    assert pq.check_prediction_chart([call("sandbox_python", outputs=[])]) is None


def test_chart_made_by_a_sub_agent_counts():
    hist = [call("simulate_season"),
            {"kind": "subagent", "evidence": [call("sandbox_python", outputs=[{"path": "out/clubs.svg"}])]}]
    assert pq.check_prediction_chart(hist)["ok"]


def test_a_line_that_moves_is_not_held():
    assert pq.check_flat_centre([pack(MOVING)]) is None


def test_flat_line_is_blocked_and_names_the_untried_strategies():
    c = pq.check_flat_centre([pack(FLAT)])
    assert not c["ok"] and c["severity"] == "block"
    assert "0 of 4" in c["detail"] and "steps" in c["detail"] and "drivers" in c["detail"]


def test_flat_line_passes_after_four_strategies():
    hist = [call("sandbox_python", {"code": "path = step_path(meeting_dates, moves)"}),
            call("sandbox_python", {"code": "res = STL(y, period=12).fit()"}),
            call("control_lab", {"with_covariates": True, "past_covariates": [[1, 2]]}),
            call("sandbox_python", {"code": "y = hampel(y); level = UnobservedComponents(y, 'local level').fit()"}),
            pack(FLAT)]
    tried = pq.strategies_tried(hist)
    assert {"steps", "seasonal", "drivers", "cleaning"} <= set(tried)
    c = pq.check_flat_centre(hist)
    assert c["ok"]


def test_three_strategies_is_not_enough():
    hist = [call("sandbox_python", {"code": "p = step_path(d)"}), call("sandbox_python", {"code": "STL(y).fit()"}),
            call("control_lab", {"with_covariates": True, "past_covariates": [[1.0]]}), pack(FLAT)]
    assert not pq.check_flat_centre(hist)["ok"]


def test_prediction_checks_collects_both_and_never_raises():
    out = pq.prediction_checks([pack(FLAT, files=())])
    assert {c["name"] for c in out} == {"prediction_charted", "centre_not_flat", "signals_gathered", "fundamentals_covered"}
    assert pq.prediction_checks([None, "x", {"kind": "tool"}]) == []


def test_signals_gathered_warns_without_blocking_and_clears_with_a_driver_test():
    warn = pq.check_signals_gathered([pack(MOVING)])
    assert not warn["ok"] and warn["severity"] == "warn"
    with_driver = [call("control_lab", {"with_covariates": True, "past_covariates": [[1.0]]}), pack(MOVING)]
    assert pq.check_signals_gathered(with_driver) is None
    assert pq.check_signals_gathered([call("sandbox_python")]) is None


def test_words_in_a_scan_a_default_argument_or_a_data_search_are_not_strategies():
    # Round 22 (AURATE, AUUNEMP): "trend" and "Level shift" printed by a scan cell, backtest_series' default combine and
    # open-data searches passed every flat line.
    hist = [call("sandbox_python", {"code": "print('trend', 'Level shift', 'Outlier', 'Seasonal', 'covariate')"}),
            call("backtest_series", {"combine": True}), call("search_open_data", {"query": "open_data rates"}),
            {"kind": "tool", "tool": "web_research", "args": {}, "error": "timeout"}, pack(FLAT)]
    assert pq.strategies_tried(hist) == []
    assert not pq.check_flat_centre(hist)["ok"]


def test_a_daily_forecast_for_a_month_by_month_request_is_blocked():
    # Round 22 (SPX): "month by month over the next 2 years" delivered as 504 business days.
    daily = call("timesfm_forecast", point=[1.0] * 504, config={"horizon": 504, "series_step": 1.0})
    c = pq.check_grain("Forecast it month by month over the next 2 years", [daily])
    assert not c["ok"] and c["severity"] == "block" and "Aggregate" in c["detail"]
    monthly = call("timesfm_forecast", point=[1.0] * 24, config={"horizon": 24, "series_step": "monthly"})
    assert pq.check_grain("monthly unemployment, next 2 years", [monthly])["ok"]
    quarterly = call("timesfm_forecast", point=[1.0] * 8, config={"horizon": 8, "series_step": "quarterly"})
    assert pq.check_grain("monthly unemployment", [quarterly])["severity"] == "warn"
    assert pq.check_grain("forecast the next 2 years", [daily]) is None


def test_an_answer_range_that_is_not_the_delivered_one_is_blocked():
    # Round 22 (AUUNEMP): the pack ended at 2.63 to 8.23 and the answer said 4.3% to 5.5%.
    p = pack([4.86] * 24)
    full = p["full"]["forecast"]["quantiles"]
    full["0.1"][-1], full["0.9"][-1] = 2.6318, 8.2283
    wrong = "The likely range is 4.3% to 5.5% by October 2028."
    c = pq.check_answer_band(wrong, [p])
    assert not c["ok"] and "2.632 to 8.228" in c["detail"]
    assert pq.check_answer_band("The 10-90 range by October 2028 is 2.6% to 8.2%.", [p])["ok"]
    assert pq.check_answer_band("The centre stays near 4.9%.", [p]) is None


def test_a_goal_that_asks_for_drivers_blocks_until_one_is_tested():
    # Round 22 (GOLD, AURATE): "Search for the signals that move it" and no driver was ever tested.
    goal = "Forecast gold monthly over the next year. Search for the signals that move it."
    researched = [call("web_research", {"query": "gold drivers"}), pack(MOVING)]
    c = pq.check_signals_gathered(researched, goal)
    assert not c["ok"] and c["severity"] == "block" and "control_lab" in c["detail"]
    tested = [call("control_lab", {"with_covariates": True, "past_covariates": ["dxy"]}), pack(MOVING)]
    assert pq.check_signals_gathered(tested, goal)["ok"]


# ---------------------------------------------------------------- owner review 2026-11-27: league signals, fundamentals

def sim(signals, sport="football"):
    return {"kind": "tool", "tool": "simulate_season", "args": {},
            "result": {"ok": True, "sport": sport, "settings": {"signals": signals}, "files": ["out/league_forecast.png"]}}


def test_a_league_simulated_from_this_seasons_goals_only_is_blocked():
    # EPL 6e19a3b6: Brighton 45% for the title from 11 matches of goals.
    c = pq.check_league_signals([sim({"goals": 110, "xg": 0, "market": 0, "history_matches": 0})])
    assert not c["ok"] and "football-data.co.uk" in c["detail"] and "history" in c["detail"]
    c = pq.check_league_signals([sim({"goals": 622, "xg": 0, "market": 0, "history_matches": 512})])
    assert not c["ok"] and "expected goals or bookmaker prices" in c["detail"]
    ok = pq.check_league_signals([sim({"goals": 622, "xg": 110, "market": 622, "history_matches": 512})])
    assert ok["ok"]
    assert pq.check_league_signals([sim({}, sport="win_loss")]) is None
    assert pq.check_league_signals([pack(MOVING)]) is None


GOAL = "Find Australia's official interest rate (the RBA cash rate target) and forecast it over the next 5 years"


def record(findings, benchmarks=(), target=GOAL):
    return call("driver_map", {"action": "record", "target": target, "findings": list(findings),
                               "benchmarks": list(benchmarks)}, coverage={})


def F(family, kind, evidence, signal=None):
    return {"family": family, "signal": signal or f"{family} signal", "kind": kind, "direction": "up",
            "effect": "e", "evidence": evidence}


def test_fundamentals_block_without_a_driver_map_and_name_the_families():
    c = pq.check_fundamentals([pack(MOVING)], GOAL)
    assert not c["ok"] and "no driver map" in c["detail"] and "monetary_policy" in c["detail"]
    assert pq.check_fundamentals([call("web_search", {"query": "x"})], GOAL) is None


def test_fundamentals_pass_when_every_finding_is_traced_and_scenarios_applied():
    sources = [call("web_fetch", {"url": f"https://www.rba.gov.au/{k}"}, text="x") for k in
               ("smp", "cpi", "labour", "gdp", "asx-tracker")]
    test1 = call("control_lab", {"with_covariates": True, "past_covariates": [[1.0]]}, run_id="a" * 32)
    test2 = call("sandbox_python", {"code": "import statsmodels.api as sm; m = sm.OLS(y, X).fit(); print(m.params)"},
                 run_id="b" * 32)
    findings = [F("monetary_policy", "researched", "https://www.rba.gov.au/smp"),
                F("inflation", "tested", "a" * 32), F("labour_market", "tested", "b" * 32),
                F("activity", "researched", "rba.gov.au/gdp"),
                F("market_pricing", "researched", "https://www.rba.gov.au/asx-tracker")]
    bench = [{"source": "RBA SMP", "value": "3.85", "what": "cash rate 2027", "evidence": "https://www.rba.gov.au/smp"}]
    scen = call("scenario_forecast", {"pack": "c" * 32, "scenarios": [{"name": "a"}, {"name": "b"}]},
                end={"q10": 3.1, "q90": 4.6}, files=["out/scenario_forecast.png"])
    hist = sources + [test1, test2, pack(MOVING), record(findings, bench), scen]
    c = pq.check_fundamentals(hist, GOAL)
    assert c["ok"], c["detail"]


def test_fundamentals_drop_evidence_no_call_produced_and_tests_that_measured_nothing():
    src = call("web_fetch", {"url": "https://www.rba.gov.au/smp"})
    plain = call("sandbox_python", {"code": "print(df.tail())"}, run_id="d" * 32)
    findings = [F("monetary_policy", "researched", "https://www.rba.gov.au/smp"),
                F("inflation", "tested", "d" * 32),                            # a print is not a test
                F("geopolitics", "researched", "https://made-up.example.com/war")]
    c = pq.check_fundamentals([src, plain, pack(MOVING), record(findings)], GOAL)
    assert not c["ok"]
    assert "made-up.example.com" in c["detail"] and "tested against the target: 0 of 2" in c["detail"]
    assert "researched with traced evidence: 2 of 5" in c["detail"]


def test_scenario_forecast_end_is_the_delivered_band():
    scen = call("scenario_forecast", {"pack": "c" * 32}, end={"q10": 3.12, "q90": 4.61})
    hist = [pack([4.0] * 12), scen]
    assert not pq.check_answer_band("The range by 2030 is 3.5% to 4.2%.", hist)["ok"]
    assert pq.check_answer_band("The 10-90 range by 2030 is 3.1% to 4.6%.", hist)["ok"]


def test_fundamentals_warn_by_default_and_block_when_switched(monkeypatch):
    # Owner choice (2026-11-27): warn on the first live rounds, block once PROPHET_FUNDAMENTALS_GATE=block.
    from prophet_agent.runtime import contracts
    c = pq.check_fundamentals([pack(MOVING)], GOAL)
    assert not c["ok"] and c["severity"] == pq.FUNDAMENTALS_SEVERITY
    monkeypatch.setattr(pq, "FUNDAMENTALS_SEVERITY", "warn")
    assert contracts.blocking([pq.check_fundamentals([pack(MOVING)], GOAL)]) == []
    monkeypatch.setattr(pq, "FUNDAMENTALS_SEVERITY", "block")
    assert contracts.blocking([pq.check_fundamentals([pack(MOVING)], GOAL)])


def rate_pack(first, last=4.6, end="2026-10-31"):
    p = pack([first] * 6)
    p["full"]["forecast"]["last_observed"] = {"date": end, "value": last}
    return p


def news(date, text, url="https://www.example-news.com.au/rba"):
    return {"kind": "tool", "tool": "web_research", "args": {"queries": ["RBA cash rate"]}, "result": {"ok": True},
            "full": {"sources": [{"date": date, "url": url, "passages": [{"text": text}]}]}}


HIKE = "At its November meeting the RBA lifted the cash rate to an 18-year high of 4.85 per cent. Markets reacted."


def test_a_level_published_after_the_data_ends_must_start_the_forecast():
    # Round 22 (AURATE): the series ended at 4.60% on 2026-10-31 and sources dated in November said 4.85%.
    c = pq.check_newest_level([news("2026-11-24", HIKE), rate_pack(4.6)], GOAL)
    assert not c["ok"] and c["severity"] == "block" and "4.85" in c["detail"] and "2026-11-24" in c["detail"]
    assert pq.check_newest_level([news("2026-11-24", HIKE), rate_pack(4.85)], GOAL)["ok"]


def test_forecasts_old_sources_and_other_targets_do_not_count():
    later = "Westpac expects the RBA will cut the cash rate to 4.35 per cent by August."
    assert pq.check_newest_level([news("2026-11-24", later), rate_pack(4.6)], GOAL) is None
    assert pq.check_newest_level([news("2026-10-02", HIKE), rate_pack(4.6)], GOAL) is None     # before the data ends
    assert pq.check_newest_level([news("2026-11-24", HIKE), rate_pack(4.6)], "Forecast the S&P 500 monthly") is None
    youth = "The youth unemployment rate rose to 10.2 per cent in October."
    goal = "Forecast Australia's unemployment rate for the next 2 years"
    assert pq.check_newest_level([news("2026-11-19", youth), rate_pack(4.3, 4.3)], goal) is None
    adult = "The seasonally adjusted unemployment rate rose to 4.9 per cent in October."
    c = pq.check_newest_level([news("2026-11-19", adult), rate_pack(4.3, 4.3)], goal)
    assert not c["ok"] and c["severity"] == "warn"


def band_pack(lo, hi, h=12):
    p = pack([5.0] * h)
    p["full"]["forecast"]["quantiles"] = {"0.1": [lo] * h, "0.9": [hi] * h}
    p["full"]["run_id"] = "e" * 32
    return p


def test_band_width_is_judged_against_the_series_own_moves(monkeypatch):
    import math
    hist = [5.0 + math.sin(i / 6.0) for i in range(240)]        # 12-step changes span about 3.2 between 10th and 90th
    monkeypatch.setattr(pq, "_pack_input", lambda run_id: {"history": hist, "point": [5.0] * 12})
    spread = pq.change_spread(hist, None, 12, False)
    assert 2.5 < spread < 4.0
    ok = pq.check_band_width([band_pack(5 - spread / 2, 5 + spread / 2)], "Forecast the unemployment rate")
    assert ok["ok"] and "ratio 1.00" in ok["detail"]
    narrow = pq.check_band_width([band_pack(4.9, 5.1)], "Forecast the unemployment rate")
    assert not narrow["ok"] and "far narrower" in narrow["detail"] and narrow["severity"] == pq.BAND_SEVERITY
    wide = pq.check_band_width([band_pack(0.5, 15.0)], "Forecast the unemployment rate")
    assert not wide["ok"] and "far wider" in wide["detail"]
    monkeypatch.setattr(pq, "_pack_input", lambda run_id: None)
    assert pq.check_band_width([band_pack(4.9, 5.1)], "x") is None
