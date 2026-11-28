"""The harness driver and market step (owner decision 2026-11-28, after round 20261128-110827).

Live AUUNEMP refused 17 finishes and SPX 3 without scoring an outside driver or reading the market's pricing, although the
blocking checks named both each time. The harness now calls market_benchmark and driver_scan itself; their results count
for signals_gathered and fundamentals_covered; FRED is reached with a User-Agent it answers.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pytest

from prophet_agent.forecast.types import Forecast
from prophet_agent.opendata import driver_series as ds
from prophet_agent.runtime import contracts, goal, prediction_quality
import importlib

from prophet_agent.tools import series_io

driver_scan_tool = importlib.import_module("prophet_agent.tools.driver_scan_tool")

RUN = "a" * 32


class _Engine:
    def __init__(self):
        self.past_seen = 0

    def forecast_controlled(self, series, horizon, control):
        if control.past:
            self.past_seen += 1
        point = np.full(horizon, float(np.asarray(series)[-1]))
        return Forecast(point, np.tile(np.linspace(-1, 1, 9), (horizon, 1)) + point[:, None])


def _months(n: int, start: date = date(2000, 1, 1)) -> list[str]:
    out, y, m = [], start.year, start.month
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}-01")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def _fred(rows: list[tuple[str, float]], sid: str) -> bytes:
    return ("observation_date," + sid + "\n" + "\n".join(f"{d},{v}" for d, v in rows)).encode()


def test_a_driver_is_the_lagged_mean_of_each_target_period():
    dates = _months(4, date(2026, 1, 1))
    rows = [("2026-01-05", 1.0), ("2026-01-20", 3.0), ("2026-02-10", 5.0), ("2026-04-02", 9.0)]
    # January mean 2, February 5, March none (carried: 5), April 9; one period later: None, 2, 5, 5
    assert ds.aligned(dates, rows) == [None, 2.0, 5.0, 5.0]
    assert ds.aligned(dates, rows, lag=0) == [2.0, 5.0, 5.0, 9.0]


def test_the_candidates_follow_the_domain_and_the_place():
    au = [c["id"] for c in ds.candidates("labour_market", "Australia")]
    vn = [c["id"] for c in ds.candidates("labour_market", "Vietnam")]
    assert "IRSTCI01AUM156N" in au and "IRSTCI01AUM156N" not in vn and "UNRATE" in vn
    assert ds.candidates("sports_league", None) == []


@pytest.fixture()
def saved(tmp_path, monkeypatch):
    monkeypatch.setattr(series_io, "SANDBOX_ROOT", tmp_path)
    n = 150
    dates = _months(n)
    rng = np.random.default_rng(0)
    values = list(np.round(5 + np.cumsum(rng.normal(0, 0.1, n)), 3))
    (tmp_path / RUN / "out").mkdir(parents=True)
    (tmp_path / RUN / "out" / "series.json").write_text(json.dumps({"series": {"unemp": values}, "dates": {"unemp": dates}}))
    return {"run_id": RUN, "file": "out/series.json"}, dates


def test_the_scan_scores_each_fetched_driver_and_says_why_others_were_not(saved):
    source, dates = saved
    long_rows = [(d, 3.0 + 0.01 * k) for k, d in enumerate(dates)]
    short_rows = [(d, 1.0) for d in dates[-10:]]

    def get(url):
        if "IRSTCI01AUM156N" in url:
            return _fred(long_rows, "IRSTCI01AUM156N")
        if "IRLTLT01AUM156N" in url:
            return _fred(short_rows, "IRLTLT01AUM156N")
        if "UNRATE" in url:
            raise driver_scan_tool.market.MarketError("transport error: ReadTimeout")
        return _fred([(d, 50.0 + k) for k, d in enumerate(dates)], "X")

    eng = _Engine()
    out = driver_scan_tool._run({"source": source, "goal": "Australia monthly unemployment rate, next 2 years",
                                 "horizon": 24}, "k", engine_factory=lambda: eng, get=get)
    assert out["domain"] == "labour_market" and out["place"] == "Australia"
    ids = [d["id"] for d in out["drivers"]]
    assert "IRSTCI01AUM156N" in ids and "DCOILBRENTEU" in ids
    why = {n["id"]: n["reason"] for n in out["not_scored"]}
    assert "ReadTimeout" in why["UNRATE"] and "overlaps the target for 9 periods" in why["IRLTLT01AUM156N"]
    row = next(d for d in out["drivers"] if d["id"] == "IRSTCI01AUM156N")
    assert row["lag_periods"] == 1 and row["points"] == len(dates) - 1 and row["verdict"]["decision"]
    assert row["horizon_scored"] <= 24 and eng.past_seen > 0
    seen = driver_scan_tool.observe(out)
    assert {s["family"] for s in seen["scored"]} >= {"monetary_policy", "commodities_energy"}


def test_the_scan_refuses_a_series_without_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(series_io, "SANDBOX_ROOT", tmp_path)
    (tmp_path / RUN / "out").mkdir(parents=True)
    (tmp_path / RUN / "out" / "series.json").write_text(json.dumps({"series": {"x": list(range(60))}}))
    with pytest.raises(ValueError, match="holds no dates"):
        driver_scan_tool._run({"source": {"run_id": RUN}, "goal": "gold price"}, "k", engine_factory=_Engine, get=lambda u: b"")


def _backtest(source: dict, horizon: int = 24) -> dict:
    return {"kind": "tool", "tool": "backtest_series", "args": {"source": source, "horizon": horizon}, "result": {"ok": True}}


TOOLS = {"market_benchmark": object(), "driver_scan": object()}


def test_the_harness_reads_the_market_then_scans_the_drivers_once_each(saved):
    source, _ = saved
    g = "Forecast the S&P 500 daily close over the next 2 years"
    assert goal._market_driver_step(g, [], TOOLS) is None                       # nothing saved yet, so no level
    hist = [_backtest(source)]
    step = goal._market_driver_step(g, hist, TOOLS)
    assert step["tool"] == "market_benchmark" and step["args"]["kind"] == "vix" and step["args"]["level"] > 0
    hist.append({"kind": "tool", "tool": "market_benchmark", "args": step["args"], "error": "did not answer"})
    step = goal._market_driver_step(g, hist, TOOLS)                             # a failed call is not made again
    assert step["tool"] == "driver_scan" and step["args"] == {"source": source, "goal": g, "horizon": 24}
    hist.append({"kind": "tool", "tool": "driver_scan", "args": step["args"], "result": {"drivers": []}})
    assert goal._market_driver_step(g, hist, TOOLS) is None


def test_the_cash_rate_path_needs_no_saved_series_and_sport_gets_nothing():
    step = goal._market_driver_step("Forecast the RBA cash rate in Australia over 5 years", [], TOOLS)
    assert step["args"] == {"kind": "au_cash_rate"}
    assert goal._market_driver_step("Predict the 2026/27 English Premier League table", [], TOOLS) is None
    assert goal._market_driver_step("Australia unemployment rate monthly forecast", [], TOOLS) is None


def test_a_scan_counts_as_drivers_tested_and_a_matching_benchmark_counts():
    scan = {"kind": "tool", "tool": "driver_scan", "args": {},
            "result": {"drivers": ["..."], "scored": [{"family": "real_rates", "id": "DFII10", "url": "u", "decision": "no evidence"},
                                                     {"family": "us_dollar", "id": "DTWEXBGS", "url": "u", "decision": "adopt"}]}}
    assert contracts._signal_searched([scan])
    assert prediction_quality._scanned_families([scan]) == {"real_rates", "us_dollar"}
    fc = {"kind": "tool", "tool": "timesfm_forecast", "args": {}, "result": {"forecast": [1.0]}}
    gvz = {"kind": "tool", "tool": "market_benchmark", "args": {"kind": "gvz"}, "result": {"implied_vol_pct": 30}}
    vix = {**gvz, "args": {"kind": "vix"}}
    g = "Forecast the gold price monthly"
    with_gvz = prediction_quality.check_fundamentals([fc, scan, gvz], g)
    with_vix = prediction_quality.check_fundamentals([fc, scan, vix], g)
    assert "tested: 2 of 2" not in with_gvz["detail"] and "benchmark forecast" not in with_gvz["detail"]
    assert "benchmark forecast" in with_vix["detail"]


def test_fred_is_sent_the_library_name_with_the_application():
    from prophet_agent.research import netguard
    h = netguard.agent_headers("https://fred.stlouisfed.org/graph/fredgraph.csv?id=GVZCLS",
                               {"user-agent": "Mozilla/5.0", "Accept": "text/csv"})
    assert h["User-Agent"].startswith("python-httpx/") and "ProphetOpenData" in h["User-Agent"]
    assert "user-agent" not in h and h["Accept"] == "text/csv"
    assert netguard.agent_headers("https://www.rba.gov.au/x.csv", {"User-Agent": "A"}) == {"User-Agent": "A"}


def test_two_refusals_for_a_driver_check_focus_the_planner_on_the_tools_that_close_it():
    tools = {n: object() for n in ("web_research", "web_fetch", "driver_scan", "control_lab", "driver_map", "forecast_report",
                                   "scenario_forecast", "read_observation", "market_benchmark", "inspect_data")}
    once = [{"kind": "evaluation", "verdict": "not_met", "contract": ["signals_gathered"]}]
    assert goal._focus_tools(once, tools) == (tools, "")
    twice = once + [{"kind": "tool", "tool": "web_research"},
                    {"kind": "evaluation", "verdict": "not_met", "contract": ["fundamentals_covered", "answer_claims"]}]
    offered, note = goal._focus_tools(twice, tools)
    assert set(offered) == {"driver_map", "scenario_forecast", "market_benchmark", "driver_scan", "web_research", "web_fetch",
                            "forecast_report", "read_observation"}
    assert "refused 2 times for fundamentals_covered" in note
    lifted = twice + [{"kind": "evaluation", "verdict": "not_met", "missing": "figures"}]
    assert goal._focus_tools(lifted, tools) == (tools, "")
