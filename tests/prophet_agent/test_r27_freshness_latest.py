"""Round 20261128-125459: stale data presented as a forecast (VNUNEMP) and a latest value the series does not hold (AUUNEMP)."""
from __future__ import annotations

from prophet_agent.runtime import contracts, prediction_quality as pq


def _fc(last: str, future: list[str], value: float = 1.452) -> list[dict]:
    return [{"kind": "tool", "tool": "forecast_report", "args": {},
             "result": {"forecast": {"last_observed": {"date": last, "value": value}, "point": [value] * len(future),
                                     "future_dates": future}}}]


QUARTERS = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30", "2026-09-30", "2026-12-31"]


def test_a_forecast_whose_first_periods_are_past_must_say_when_the_data_ends():
    hist = _fc("2024-12-31", QUARTERS)
    c = pq.check_data_fresh("The quarterly forecast for 2023-Q4 to 2025-Q3 is about 1.5%.", hist, today="2026-11-28")
    assert c and not c["ok"] and "7 of the 8 forecast periods" in c["detail"] and "newer data" in c["detail"]
    ok = pq.check_data_fresh("The latest quarter in the data is Q4 2024; later quarters are estimates.", hist, today="2026-11-28")
    assert ok and ok["ok"]
    assert pq.check_data_fresh("x", _fc("2026-10-01", ["2026-11-01", "2026-12-01"]), today="2026-11-28") is None


def test_the_check_runs_with_the_other_prediction_checks():
    names = [c["name"] for c in pq.prediction_checks(_fc("2024-12-31", QUARTERS), "", "a forecast")]
    assert "data_end_stated" in names


def test_a_latest_official_figure_after_the_number_is_read():
    obs = _fc("2026-10-01", ["2026-11-01"], value=4.865)
    bad = contracts._latest_and_end("The rate was **4.1% in October 2026** (the latest official figure).", obs)
    assert bad and bad[0]["name"] == "answer_latest_value" and "gives 4.1 as" in bad[0]["detail"]
    assert contracts._latest_and_end("The rate was **4.9% in October 2026** (the latest official figure).", obs) == []
    assert contracts._latest_and_end("The latest official figure is 4.1% for October 2026.", obs)


def test_an_answer_written_after_refused_finishes_is_checked_and_rewritten_once():
    """Round 20261128-125459: AUUNEMP and EPL ended through the refused-finish path, where no answer check ran."""
    from prophet_agent.runtime import goal
    hist = _fc("2026-10-01", ["2026-11-01"], value=4.865)
    bad = "Unemployment was **4.1% in October 2026** (the latest official figure)."
    assert "answer_latest_value" in [c["name"] for c in goal._gap_answer_claims(bad, hist, "")]
    seen = {}

    def compose(ctx, brain, step, g, history, memo, context, draft, bad=None, fixes=None):
        seen.update(step=step, fixes=fixes)
        return "Unemployment was 4.9% in October 2026 (the latest official figure)."

    real = goal._compose
    goal._compose = compose
    try:
        text, wrong = goal._checked_gap_answer(None, None, "d0", 7, "", hist, None, "", bad)
    finally:
        goal._compose = real
    assert seen["step"] == "gapclaims:d0:7" and "4.865" in seen["fixes"]
    assert wrong == [] and "4.9%" in text
