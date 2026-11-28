"""Mission 10, M10-1 and M10-2: candidate forecasts (rules, market paths, structural models) and the combination centre.

Every series here is synthetic and seeded; outside data comes from fake fetchers. No figure is an observed result.
"""
import datetime as dt
import importlib

import numpy as np
import pytest

from prophet_agent.forecast import candidates as cd
from prophet_agent.forecast import ensemble as en
from prophet_agent.forecast.types import Forecast
from prophet_agent.runtime import goal

rt = importlib.import_module("prophet_agent.tools.report_tool")


def _months(n, start=(2000, 1)):
    y, m = start
    out = []
    for _ in range(n):
        out.append(f"{y}-{m:02d}-01")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def test_month_labels_cover_the_whole_month_and_other_dates_are_kept():
    assert cd.cutoffs(["2026-10-01", "2026-11-01"]) == [dt.date(2026, 10, 31), dt.date(2026, 11, 30)]
    assert cd.cutoffs(["2026-10-31", "2026-11-30"]) == [dt.date(2026, 10, 31), dt.date(2026, 11, 30)]
    assert cd.step_years(_months(24)) == pytest.approx(1 / 12, abs=0.005)


def test_market_path_turns_average_rates_into_forwards():
    path = cd.market_path([(1, 4.0), (3, 4.5), (6, 5.0), (24, 4.5)], 30)
    assert path[0] == pytest.approx(4.0)
    assert path[2] == pytest.approx((3 * 4.5 - 1 * 4.0) / 2)
    assert path[5] == pytest.approx((6 * 5.0 - 3 * 4.5) / 3)
    assert path[20] == pytest.approx((24 * 4.5 - 6 * 5.0) / 18)
    assert path[29] == path[23]                                   # flat after the last knot


def test_band_reads_only_errors_from_windows_that_ended_before_the_origin():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0, 1, 200)) + 50
    cand = cd.rule_candidates(y, log=False, season=None, mean_reverting=False)[0]
    before = cand.predict(150, 6)
    y2 = y.copy()
    y2[150:] += 40                                               # the future changes; the forecast from 150 must not
    cand2 = cd.rule_candidates(y2, log=False, season=None, mean_reverting=False)[0]
    assert np.allclose(before, cand2.predict(150, 6))


def test_carry_candidate_uses_only_values_known_at_the_cutoff():
    dates = _months(60, (2020, 1))
    y = np.linspace(1800, 2600, 60)
    cut = cd.cutoffs(dates)
    days = [(dt.date(2020, 1, 1) + dt.timedelta(days=k)).isoformat() for k in range(0, 1900)]
    iv = cd.Dated([(d, 20.0 if d <= "2022-12-31" else 80.0) for d in days], "GVZ")
    bill = cd.Dated([(d, 4.0) for d in days], "DTB3")
    c = cd.carry_candidate(y, cut, 1 / 12, iv, bill, "GVZ")
    q = c.predict(36, 12)                                        # origin after December 2022: the 20 per cent volatility
    spread = np.log(q[-1, 8] / q[-1, 0])
    assert spread == pytest.approx(2 * 1.2816 * 0.20, rel=0.02)
    assert q[-1, 4] == pytest.approx(y[35] * np.exp(0.04), rel=1e-6)


def _fake_fetch(urls):
    days = [(dt.date(1995, 1, 1) + dt.timedelta(days=k)).isoformat() for k in range(0, 11700)]
    csv_iv = "observation_date,GVZCLS\n" + "\n".join(f"{d},25" for d in days)
    csv_rf = "observation_date,DTB3\n" + "\n".join(f"{d},3.0" for d in days)

    def fetch(url):
        urls.append(url)
        if "GVZCLS" in url:
            return csv_iv.encode()
        if "DTB3" in url:
            return csv_rf.encode()
        raise RuntimeError("HTTP 404")
    return fetch


def test_build_adds_the_gold_forward_and_scores_it_with_the_rules():
    rng = np.random.default_rng(1)
    y = 1500 * np.exp(np.cumsum(rng.normal(0.004, 0.04, 220)))
    urls = []
    cands, notes, rel = cd.build(y, _months(220, (2005, 1)), domain="precious_metal", place=None, target="gold",
                                 fetch=_fake_fetch(urls))
    assert rel and [c.name for c in cands] == ["naive", "drift", "market_forward"] and not notes
    res = en.run(y, 12, cands, n_origins=60, step=1, min_train=80, relative=True)
    assert {w["name"] for w in res["weights"]} == {"naive", "drift", "market_forward"}
    assert all(c["origins"] == res["scored_origins"] for c in res["candidates"])


class _Engine:
    def forecast_batch(self, series_list, h):
        q = np.stack([np.asarray(s, float)[-1] + np.outer(np.ones(h), np.linspace(-1, 1, 9)) for s in series_list])
        return Forecast(q[:, :, 4], q)


def test_timesfm_candidate_batches_all_origins_once():
    calls = []

    class Counting(_Engine):
        def forecast_batch(self, series_list, h):
            calls.append(len(series_list))
            return super().forecast_batch(series_list, h)

    y = np.arange(100, dtype=float)
    c = cd.timesfm_candidate(y, Counting())
    c.prepare([40, 50, 60], 4)
    assert calls == [3]
    assert c.predict(50, 4).shape == (4, 9) and calls == [3]


def test_report_draws_the_combination_and_says_what_it_weighed(monkeypatch):
    rng = np.random.default_rng(2)
    values = list(4 + np.cumsum(rng.normal(0, 0.1, 140)))
    monkeypatch.setattr(rt, "_engine", lambda: _Engine())
    monkeypatch.setattr(rt, "_fetch", lambda url: (_ for _ in ()).throw(RuntimeError("offline")))
    fc = {"point": [values[-1]] * 6, "quantiles": {k: [values[-1]] * 6 for k in
                                                   ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9")},
          "config": {}}
    monkeypatch.setattr(rt, "run_dates", lambda args, name: _months(140, (2010, 1)))
    out = rt._combination_centre(fc, values, {"goal": "Forecast Australia's unemployment rate", "source": {}}, "u")
    centre = out["centre"]
    assert centre["rule"] == "combination" and out["combination_check"]["adopt"]
    assert {w["name"] for w in centre["weights"]} >= {"naive", "drift", "mean_reversion", "timesfm"}
    assert any("could not be built" in n for n in centre["notes"])          # the vacancy data was offline, and it says so
    note = rt._method_note({}, centre)
    assert "weighted average of the quantiles" in note and "equal weights" in note
    assert len(out["point"]) == 6 and out["quantiles"]["0.1"][0] <= out["point"][0] <= out["quantiles"]["0.9"][0]
    short = rt._combination_centre(fc, values[:30], {"goal": "Forecast Australia's unemployment rate"}, "u")
    assert short["combination_check"]["adopt"] is False


def test_harness_draws_the_combination_once_per_source_and_horizon():
    tools = {"forecast_report": object()}
    g = "Find the gold price (US dollars per troy ounce) and forecast it month by month over the next year"
    src = {"run_id": "r1", "file": "out/series.json"}
    hist = [{"kind": "tool", "tool": "forecast_report", "args": {"source": src, "horizon": 12, "centre": "drift"}}]
    step = goal._combination_step(g, hist, tools)
    assert step["args"]["centre"] == "combination" and step["args"]["goal"] == g and step["args"]["horizon"] == 12
    done = hist + [{"kind": "tool", "tool": "forecast_report", "args": step["args"]}]
    assert goal._combination_step(g, done, tools) is None
    assert goal._combination_step("Predict the final Premier League table", hist, tools) is None


def test_adopted_drivers_become_a_candidate_and_never_see_the_future():
    seen = []

    class Controlled(_Engine):
        def forecast_controlled(self, series, h, control):
            seen.append({k: len(v) for k, v in control.past.items()} | {"n": len(series)})
            f = self.forecast_batch([series], h)
            return Forecast(f.point[0], f.q[0])

    y = np.linspace(1, 2, 120)
    cands, notes, _ = cd.build(y, [], domain="equity_index", place=None, engine=Controlled(),
                               drivers={"T10Y2Y": list(np.linspace(0, 1, 120)), "short": [1.0, 2.0]})
    drv = next(c for c in cands if c.name == "timesfm_drivers")
    assert drv.kind == "driver" and "T10Y2Y" in drv.note
    assert any("short" in n for n in notes)
    assert drv.predict(80, 6).shape == (6, 9)
    assert seen[-1] == {"T10Y2Y": 80, "n": 80}


def test_harness_runs_the_combination_on_the_driver_scan_file():
    tools = {"forecast_report": object()}
    g = "Find the S&P 500 index and forecast it month by month over the next 2 years"
    src = {"run_id": "r1", "file": "out/series.json"}
    adopted = {"run_id": "r9", "file": "out/series_drivers.json", "name": "spx"}
    hist = [{"kind": "tool", "tool": "driver_scan", "result": {"use": {"source": adopted, "points": 300}}},
            {"kind": "tool", "tool": "forecast_report", "args": {"source": src, "horizon": 24}}]
    step = goal._combination_step(g, hist, tools)
    assert step["args"]["source"] == adopted
    done = hist + [{"kind": "tool", "tool": "forecast_report", "args": step["args"]}]
    assert goal._combination_step(g, done, tools) is None
