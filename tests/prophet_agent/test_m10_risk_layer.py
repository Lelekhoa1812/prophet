"""Mission 10 batch 2: the policy and geopolitical risk layer (indices, regimes, event studies, the risk tool, the answer).

Every series and workbook here is synthetic and seeded; no figure is an observed result.
"""
import datetime as dt
import importlib
import io

import numpy as np
import openpyxl
import pytest

from prophet_agent.forecast import candidates as cd
from prophet_agent.forecast import ensemble as en
from prophet_agent.forecast import regimes as rg
from prophet_agent.forecast.metrics import compare_losses, diebold_mariano
from prophet_agent.opendata import risk_indices as ri
from prophet_agent.runtime import goal
from prophet_agent.runtime import prediction_quality as pq

rrt = importlib.import_module("prophet_agent.tools.risk_regime_tool")


def _months(n, start=(1990, 1)):
    y, m = start
    out = []
    for _ in range(n):
        out.append(dt.date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _gpr_book(n=300, spikes=(100, 200)):
    rng = np.random.default_rng(3)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["month", "GPR", "GPRT", "GPRA", "GPRC_AUS", "GPRC_VNM", "GPRC_USA", "GPRC_CHN"])
    for k, d in enumerate(_months(n, (1985, 1))):
        base = 100 + rng.normal(0, 5)
        if any(s <= k < s + 3 for s in spikes):
            base += 120
        ws.append([dt.datetime(d.year, d.month, 1)] + [base] * 7)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _epu_book(n=300):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["year", "month", "Australia News-Based Policy Uncertainty Index"])
    for k, d in enumerate(_months(n, (1998, 1))):
        ws.append([d.year, d.month, 100 + 10 * np.sin(k / 7)])
    ws.append(["Source: notes row", None, None])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_index_parsers_date_each_value_at_the_end_of_its_month():
    rows = ri.parse_gpr(_gpr_book(), "GPRC_AUS")
    assert rows[0][0] == "1985-01-31" and len(rows) == 300
    epu = ri.parse_epu(_epu_book())
    assert epu[1][0] == "1998-02-28" and len(epu) == 300                 # the notes row is skipped
    fred = ri.parse_fred_monthly(b"observation_date,USEPUINDXM\n2026-09-01,168.5\n2026-10-01,157.0\n", "USEPUINDXM")
    assert fred[-1] == ("2026-10-31", 157.0)
    with pytest.raises(ValueError, match="no 'GPRX'"):
        ri.parse_gpr(_gpr_book(), "GPRX")
    assert ri.domain_index("policy_rate", "Australia") == "epu_au" and ri.domain_index("precious_metal", None) == "gpr"
    assert ri.dashboard_keys("labour_market", "Australia")[0] == "epu_au"


def test_regime_reads_only_what_was_known_at_the_cutoff():
    days = np.array([dt.date(2000, 1, 31).toordinal() + 30 * k for k in range(200)])
    vals = np.arange(200, dtype=float)
    cut = dt.date.fromordinal(int(days[100]))
    lab, pct = rg.regime_at(days, vals, cut)
    vals2 = vals.copy()
    vals2[101:] = -1000                                                 # the future changes; the label must not
    assert rg.regime_at(days, vals2, cut) == (lab, pct) and lab == "high"
    assert rg.regime_at(days, vals, dt.date.fromordinal(int(days[10])))[0] is None          # too little history
    assert rg.regime_at(days, vals, dt.date.fromordinal(int(days[-1]) + 200))[0] is None     # stale


def test_event_study_finds_a_planted_response_and_refuses_noise():
    n = 360
    dates = _months(n)
    cut = [dt.date(d.year, d.month, 28) for d in dates]
    days = np.array([c.toordinal() for c in cut])
    rng = np.random.default_rng(5)
    vals = np.empty(n)
    vals[0] = 100.0
    for k in range(1, n):                                               # a smooth index, as the real ones are
        vals[k] = 100 + 0.97 * (vals[k - 1] - 100) + rng.normal(0, 0.5)
    onsets = list(range(60, 340, 30))
    for s in onsets:
        vals[s] += 80
    y = np.cumsum(rng.normal(0, 0.05, n)) + 5
    for s in onsets:                                                    # after each spike the target rises 1 point
        y[s + 1:] += 1.0
    ev = rg.event_study(y, cut, days, vals, 6, relative=False)
    assert ev["n"] >= 5 and ev["measured"] and ev["shift"] > 0.5
    flat = rg.event_study(np.cumsum(rng.normal(0, 0.05, n)) + 5, cut, days, vals, 6, relative=False)
    assert flat["n"] == ev["n"] and not flat["measured"]


def test_band_is_conditioned_on_the_regime_its_windows_shared():
    rng = np.random.default_rng(7)
    n = 300
    labels = [None] + ["high" if (t // 25) % 2 else "low" for t in range(n)]
    shocks = np.array([rng.normal(0, 3.0 if labels[t + 1] == "high" else 0.5) for t in range(n)])
    y = 50 + np.cumsum(shocks)
    cands = cd.rule_candidates(y, log=False, season=None, mean_reverting=False)
    labels[-1] = "high"
    res = en.run(y, 6, cands, n_origins=200, step=1, min_train=60, regimes=labels, regime_name="test index")
    reg = res["band"]["regime"]
    assert reg["current"] == "high" and reg["error_ratio_by_regime"]["high"] > reg["error_ratio_by_regime"]["low"]
    assert reg["coverage_by_regime"]["high"] < reg["coverage_by_regime"]["low"]
    assert "significance" in res and "equal_weights" in res["significance"]
    with pytest.raises(ValueError, match="one label per number"):
        en.run(y, 6, cands, n_origins=200, step=1, min_train=60, regimes=labels[:-1])


def test_loss_comparison_needs_both_tests_and_enough_independent_windows():
    rng = np.random.default_rng(2)
    a, b = 1.3 + rng.normal(0, 0.2, 240), 1.0 + rng.normal(0, 0.2, 240)
    good = compare_losses(a, b, 12)
    assert good["significant"] and good["independent"]["n"] == 20
    assert not compare_losses(a, b, 60)["significant"]                  # four independent windows can never reach 0.05
    assert not compare_losses(b, a, 12)["significant"]
    assert np.isnan(diebold_mariano(a[:5], b[:5], 1)["p_cand_better"])


def _risk_fetch():
    gpr, epu = _gpr_book(), _epu_book()

    def fetch(url):
        if "gpr" in url:
            return gpr
        if "policyuncertainty" in url:
            return epu
        if "USEPUINDXM" in url:
            return b"observation_date,USEPUINDXM\n" + "\n".join(
                f"{d.isoformat()},{100 + k % 17}" for k, d in enumerate(_months(300, (1985, 1)))).encode()
        raise RuntimeError("HTTP 404")
    return fetch


def test_risk_tool_reports_today_and_counts_only_the_registered_index(monkeypatch):
    dates = [d.isoformat() for d in _months(200, (2005, 1))]
    y = np.linspace(4, 5, 200)
    monkeypatch.setattr(rrt, "resolve_series", lambda args: [("u", y)])
    monkeypatch.setattr(rrt, "run_dates", lambda args, name: dates)
    out = rrt._run({"source": {"run_id": "r"}, "goal": "Forecast Australia's unemployment rate", "horizon": 12}, "k",
                   fetch=_risk_fetch())
    assert out["registered_index"] == "epu_au" and out["today"]["index"] == "epu_au"
    assert {s["index"] for s in out["event_studies"]} == {"epu_au", "gpr"}
    assert all(not s["measured"] for s in out["event_studies"] if s["index"] == "gpr")
    view = rrt.observe(out)
    assert "judgement" in view["state"] or "measured response" in view["state"]
    with pytest.raises(ValueError, match="horizon"):
        rrt._run({"source": {"run_id": "r"}, "goal": "g", "horizon": 0}, "k", fetch=_risk_fetch())


def test_harness_reads_risk_once_after_the_combination():
    tools = {"risk_regime": object(), "forecast_report": object()}
    g = "Find Australia's monthly unemployment rate and forecast it month by month over the next 2 years"
    src = {"run_id": "r1", "file": "out/series.json"}
    hist = [{"kind": "tool", "tool": "forecast_report", "args": {"source": src, "horizon": 24, "centre": "combination"}}]
    step = goal._risk_regime_step(g, hist, tools)
    assert step["tool"] == "risk_regime" and step["args"] == {"source": src, "horizon": 24, "goal": g}
    assert goal._risk_regime_step(g, hist + [{"kind": "tool", "tool": "risk_regime"}], tools) is None
    assert goal._risk_regime_step("Predict the final Premier League table", hist, tools) is None


def test_answer_states_the_risk_reading():
    hist = [{"kind": "tool", "tool": "risk_regime", "result": {"today": {
        "label": "economic policy uncertainty, Australia", "value": 211.58, "month": "2026-10", "percentile_10y": 0.85,
        "regime": "high"}}}]
    assert pq.check_risk_stated("Policy uncertainty in Australia is at its 85th percentile.", hist)["ok"] is True
    bad = pq.check_risk_stated("The rate rises.", hist)
    assert bad["ok"] is False and "85%" in bad["detail"]
    assert pq.check_risk_stated("x", []) is None
