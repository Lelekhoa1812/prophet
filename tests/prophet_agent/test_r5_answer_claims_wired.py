"""Round 5: the answer-claim checks run in the goal loop, and a withheld headline total is not stated as one.

Finding (measured on the code, 2026-11-23): contracts.check_answer (C3a, A6-09) was merged with its own tests, but nothing in
goal.py called it, so a wrong cycle or a wrong best rule in the written answer reached the person unchecked in every live run.
Follow-up to A7-03: forecast_report withholds its headline total when the health verdict is not ok or a day was floored at
zero. The answer must then give the reason, not a total.
"""
from __future__ import annotations

import pytest

from prophet_agent.runtime import contracts
from prophet_agent.runtime.goal import goal_program
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import Store


# ---------------------------------------------------------------- the withheld-total check

def _pack(withheld=None):
    return {"kind": "tool", "tool": "forecast_report", "args": {},
            "result": {"forecast": {"point": [1.0] * 14, "health": {"verdict": "check" if withheld else "ok"}},
                       "headline": {"verdict": "check" if withheld else "ok", "total": None if withheld else 14.0,
                                    "withheld": withheld}}}


def _panel(withheld_for=()):
    series = {n: {"health": "check" if n in withheld_for else "ok",
                  **({"total_withheld": "the health check verdict is check, not ok"} if n in withheld_for else {})}
              for n in ("North", "South")}
    return {"kind": "tool", "tool": "forecast_report", "args": {}, "result": {"n_series": 2, "series": series}}


def _named(checks, name="answer_total_withheld"):
    return [c for c in checks if c["name"] == name]


REASON = "the health check verdict is check, not ok"


def test_a_total_stated_when_the_pack_withheld_it_blocks():
    c = _named(contracts.check_answer("Total forecast: 707,188 units over 14 days.", [_pack(REASON)]))
    assert len(c) == 1 and not c[0]["ok"] and c[0]["severity"] == "block"
    assert "check, not ok" in c[0]["detail"]


def test_a_sentence_that_explains_the_withheld_total_is_not_a_total_claim():
    text = "The total of 707,188 is withheld because the health verdict is check, not ok. See the daily table."
    c = _named(contracts.check_answer(text, [_pack(REASON)]))
    assert len(c) == 1 and c[0]["ok"]


def test_no_total_and_no_reason_is_a_warning():
    c = _named(contracts.check_answer("Daily values are in the table.", [_pack(REASON)]))
    assert len(c) == 1 and not c[0]["ok"] and c[0]["severity"] == "warn"


def test_a_total_the_pack_did_not_withhold_adds_no_check():
    assert _named(contracts.check_answer("Total forecast: 14 units.", [_pack(None)])) == []


def test_a_panel_blocks_only_a_total_for_an_entity_whose_total_was_withheld():
    hist = [_panel(withheld_for=("South",))]
    ok = _named(contracts.check_answer("North total: 1,200 units.", hist))
    bad = _named(contracts.check_answer("South total: 900 units.", hist))
    assert ok and ok[0]["severity"] == "warn"             # no South total, but no reason either
    assert bad and not bad[0]["ok"] and bad[0]["severity"] == "block"


def test_the_newest_pack_decides():
    assert _named(contracts.check_answer("Total: 14 units.", [_pack(REASON), _pack(None)])) == []


# ---------------------------------------------------------------- wired into the goal loop

class Brain:
    def __init__(self, plans, evals, writes):
        self.plans, self.evals, self.writes = list(plans), list(evals), list(writes)
        self.prompts = []

    def decide(self, messages, schema, name):
        return (self.plans if name == "plan" else self.evals).pop(0)

    def write(self, messages):
        self.prompts.append(messages[-1]["content"])
        return self.writes.pop(0)


BACKTEST = {"horizon": 14, "origins": 8, "best_baseline": "mean_last_season",
            "models": {"naive": {"mase": 1.0}, "mean_last_season": {"mase": 0.8}}}


def _run(store, writes):
    brain = Brain(plans=[{"action": "call_tool", "tool": "backtest_series", "args": {}, "rationale": "compare"},
                         {"action": "finish", "rationale": "done"}],
                  evals=[{"verdict": "met", "reason": "enough", "answer": "draft"}], writes=writes)
    tools = {"backtest_series": Tool("backtest_series", lambda args, key: BACKTEST, idempotent=True)}
    mid = store.create_mission("g", {})
    return brain, Runtime(store, owner="r1").run(mid, goal_program("which rule is the benchmark", tools, brain))


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "claims.sqlite")
    yield s
    s.close()


def test_a_wrong_benchmark_claim_is_rewritten_once_and_then_passes(store):
    brain, out = _run(store, ["Naive is the appropriate benchmark.", "mean_last_season is the appropriate benchmark."])
    assert out["result"]["status"] == "met", out["result"].get("reason")
    assert out["result"]["answer"] == "mean_last_season is the appropriate benchmark."
    assert "disagree with the stored analysis output" in brain.prompts[1]
    assert "answer_benchmark_matches" in brain.prompts[1]


def test_a_claim_still_wrong_after_the_rewrite_leaves_partial(store):
    _, out = _run(store, ["Naive is the appropriate benchmark.", "Naive is still the appropriate benchmark."])
    assert out["result"]["status"] == "partial"
    assert out["result"]["contract_failures"] == ["answer_benchmark_matches"]


# ---------------------------------------------------------------- false positives found by replaying 169 stored live answers

def _health(period):
    return {"kind": "tool", "tool": "timesfm_forecast", "args": {}, "result": {"point": [1.0] * 14,
                                                                             "health": {"verdict": "ok", "period": period}}}


def test_an_annual_cycle_within_one_percent_of_the_measured_period_passes():
    # live CO2 6258bd3d: "period 365 days" against a measured 364 was blocked
    checks = contracts.check_answer("Strong annual cycle (period 365 days, strength 1.0).", [_health(364)])
    assert not [c for c in checks if c["name"] == "answer_cycle_matches_health" and not c["ok"]]


def test_a_short_cycle_has_no_tolerance():
    checks = contracts.check_answer("A weekly (7-day) cycle.", [_health(6)])
    assert [c for c in checks if c["name"] == "answer_cycle_matches_health" and not c["ok"]]


def test_a_horizon_followed_by_a_comma_is_not_a_cycle_claim():
    # live retail 61e96f55: "(horizon 14 days, weekly seasonality)" was read as a 14-step cycle
    from prophet_agent.runtime.grounding import claimed_periods
    assert claimed_periods("A backtest over 12 past windows (horizon 14 days, weekly seasonality) gave MASE 0.88.") == []
    assert claimed_periods("A 7-day seasonal pattern.") == [(7, "A 7-day seasonal pattern.")]
    assert claimed_periods("a (7-day) cycle") == [(7, "a (7-day) cycle")]


def test_live_benchmark_misreads_are_not_claims():
    from prophet_agent.runtime.grounding import claimed_best_baselines
    for text in ("TimesFM is about 9.5% more accurate than naive (lower MASE), but 1.7% less accurate than the best simple baseline.",
                 "MASE 7.13 versus 7.38 for the best naive baseline (mean of last season).",
                 "- Naive rule comparison: the best simple rule (mean of the last 5 business days) scored MASE 3.13;",
                 "- Versus naive rule: TimesFM scored MASE 7.26 against 6.66 for the best simple rule (mean of last season)."):
        assert [n for n, _ in claimed_best_baselines(text) if n == "naive"] == [], text
    assert claimed_best_baselines("No model tested beats the naive forecast, and naive is the appropriate benchmark.")


def test_a_count_of_cycles_is_not_a_cycle_length():
    from prophet_agent.runtime.grounding import claimed_periods
    assert claimed_periods("Health is ok (strong weekly cycle, 105 cycles in history).") == []
    assert claimed_periods("Strong annual cycle (period 365 days).") == [(365, "Strong annual cycle (period 365 days).")]


def test_a_quoted_tool_argument_is_not_a_cycle_claim():
    from prophet_agent.runtime.grounding import claimed_periods
    assert claimed_periods("Backtest (8 origins, horizon 30, season=252): TimesFM MASE 0.229.") == []
    assert claimed_periods("Seasonality: period: 7.") == [(7, "Seasonality: period: 7.")]


# ---------------------------------------------------------------- a total's bound given as its range (live retail_biz rep7)

def _single_pack():
    rows = [{"step": i + 1, "point": 10.0, "q10": 8.0, "q90": 12.0} for i in range(6)]
    tr = {"centre": 60.0, "if_days_miss_independently": {"q10": 55.83, "q90": 64.17},
          "if_days_miss_together": {"q10": 48.0, "q90": 72.0}}
    te = {"origins": 12, "variance_ratio": 2.0, "held_measured": 10, "held_independent": 9, "held_together": 12}
    return [{"kind": "tool", "tool": "backtest_series", "result": {"horizon": 6, "total_error": te, "models": {}}},
            {"kind": "tool", "tool": "forecast_report", "result": {"table": {"rows": rows, "rows_omitted": 0},
                                                                  "total_range_80": tr}}]


def test_the_together_bound_given_as_the_range_blocks_and_names_the_measured_range():
    c = _named(contracts.check_answer("Expect 60 units, with an 80% range of 48 to 72.", _single_pack()),
               "answer_total_range_named")
    assert len(c) == 1 and not c[0]["ok"] and c[0]["severity"] == "block"
    assert "misses the same way at once" in c[0]["detail"] and "the measured 80% range is" in c[0]["detail"]


def test_a_bound_named_for_what_it_assumes_passes_and_scaled_words_are_read():
    ok = "If every day misses the same way at once, the total could reach 48 to 72."
    assert _named(contracts.check_answer(ok, _single_pack()), "answer_total_range_named") == []
    from prophet_agent.runtime.grounding import bound_misnamed
    assert bound_misnamed("The range is £1.26 million to £2.61 million.", 1_257_168, 2_610_558, True)
    assert bound_misnamed("The 80% range is 1,358,450 to 2,142,805.", 1_257_168, 2_610_558, True) is None


def test_a_panel_sum_of_bounds_called_the_likely_range_blocks():
    # live electricity rep10: "The likely range, summing each client's 10-90% interval, is 3,871,341 - 5,089,679 kWh"
    pt = {"centre": 4375772.9, "steps": 7, "if_entities_miss_independently": {"q10": 4320869.3, "q90": 4430676.6},
          "if_entities_miss_together": {"q10": 3871340.6, "q90": 5089679.0}}
    hist = [{"kind": "tool", "tool": "forecast_report", "result": {"n_series": 2, "series": {}, "panel_total": pt}}]
    text = "The likely range, summing each client's 10–90% interval, is **3,871,341 – 5,089,679 kWh**."
    c = _named(contracts.check_answer(text, hist), "answer_total_range_named")
    assert c and not c[0]["ok"] and "give both bounds" in c[0]["detail"]


# ---------------------------------------------------------------- the average's own range (live wti_budget reps 2 and 3)

AVG_GOAL = "I need a number for next month's fuel budget. What will crude cost on average next month?"


def test_an_average_goal_needs_the_averages_measured_range():
    hist = _single_pack()                 # measured range of the 6-step average: about 8.5 to 11.5 around 10
    mr = contracts.measured_total_range(hist)
    day_band = "Next month's average: 10 per unit (80% range: 8 to 12 by month-end)."
    c = _named(contracts.check_answer(day_band, hist, goal=AVG_GOAL), "answer_average_range")
    assert len(c) == 1 and not c[0]["ok"] and c[0]["severity"] == "block" and f"{mr['mean_q10']:,.2f}" in c[0]["detail"]
    good = f"Next month's average: 10 per unit, 80% range {mr['mean_q10']:.2f} to {mr['mean_q90']:.2f}."
    assert _named(contracts.check_answer(good, hist, goal=AVG_GOAL), "answer_average_range")[0]["ok"]
    assert _named(contracts.check_answer(day_band, hist, goal="Forecast next month."), "answer_average_range") == []


def test_mase_scores_called_mae_block_unless_a_cell_printed_an_mae():
    # live electricity rep12, electricity rep4 and Beijing (2026-11-23): panel_backtest's MASE table headed "Overall MAE"
    bt = {"kind": "tool", "tool": "panel_backtest", "result": {"overall": {"timesfm": 1.47}, "best_overall": "timesfm"}}
    text = "| Model | Overall MAE |\n|---|---|\n| TimesFM | 1.47 |"
    c = _named(contracts.check_answer(text, [bt]), "answer_metric_named")
    assert len(c) == 1 and not c[0]["ok"] and "name the metric MASE" in c[0]["detail"]
    assert _named(contracts.check_answer(text.replace("MAE", "MASE"), [bt]), "answer_metric_named") == []
    own = {"kind": "tool", "tool": "sandbox_python", "result": {"stdout": "MAE per model: timesfm 812.4"}}
    assert _named(contracts.check_answer(text, [bt, own]), "answer_metric_named") == []


def test_a_series_given_a_health_verdict_the_pack_did_not_give_it_blocks():
    # live Beijing rep9 (2026-11-23): "Two stations (Nongzhanguan, Huairou) received a flat_expected verdict"; Huairou was ok
    pack = {"kind": "tool", "tool": "forecast_report", "result": {"n_series": 2, "series": {
        "Nongzhanguan": {"health": "flat_expected"}, "Huairou": {"health": "ok"}}}}
    bad = "Two stations (Nongzhanguan, Huairou) received a *flat_expected* verdict."
    c = _named(contracts.check_answer(bad, [pack]), "answer_series_health_matches")
    assert len(c) == 1 and not c[0]["ok"] and "Huairou: ok" in c[0]["detail"]
    for fine in ("Nongzhanguan received a flat_expected verdict.",
                 "Huairou is ok, while Nongzhanguan is flat_expected.",
                 "| Huairou | 44 | flat_expected |"):
        assert _named(contracts.check_answer(fine, [pack]), "answer_series_health_matches") == [], fine


# ---------------------------------------------------------------- a level is not summed (live WTI rep9, Beijing reps 8 and 9)

def _level(name):
    return {"kind": "tool", "tool": "inspect_data", "args": {"file": "x.csv"},
            "result": {"columns": ["date: time, unique=100", f"{name}: number, max=1, unique=90"]}}


def test_level_and_flow_measures_are_told_apart_from_their_names():
    for name in ("wti_cushing_spot_price_usd_per_barrel", "pm25_daily_mean", "usd_jpy_rate", "temperature_c"):
        assert contracts.measure_is_level([_level(name)]), name
    for name in ("net_sales_gbp", "kwh_daily_sum", "units_sold", "orders"):
        assert not contracts.measure_is_level([_level(name)]), name


def test_a_total_of_a_level_blocks_and_the_accuracy_facts_give_the_average():
    hist = [_level("wti_cushing_spot_price_usd_per_barrel")] + _single_pack()
    c = _named(contracts.check_answer("The 6-day total is 60 with an 80% range of 48 to 72.", hist), "answer_level_not_summed")
    assert len(c) == 1 and not c[0]["ok"] and "average over the horizon" in c[0]["detail"]
    assert _named(contracts.check_answer("The average price is 10 a day.", hist), "answer_level_not_summed") == []
    ts = contracts.trust_statement(hist)
    assert "is a level, so it is not summed" in ts and "80% range of its average over the 6 steps" in ts
    assert "-step total" not in ts
    flow = [_level("net_sales_gbp")] + _single_pack()
    assert "80% range of the 6-step total" in contracts.trust_statement(flow)
    assert _named(contracts.check_answer("The 6-day total is 60.", flow), "answer_level_not_summed") == []
