"""risk_regime: where geopolitical and policy risk stand today, and what the target did after past risk spikes.

Motivation vs Logic
Motivation (Mission 10 batch 2, owner review 2026-11-28): the forecasts did not account for political, geopolitical and policy
risk; scenarios carried judgement shifts with nothing measured behind them. The research behind the batch (reports/Geopolitical
risk and structural drivers.md) found no measured out-of-sample gain from these indices in a point forecast, so the effects
have to be measured on each target's own history. Measured 2026-11-28 on real staging data (240 monthly origins each): the
combination's 10-90 bands held 61 to 74 per cent of outcomes when the domain's index was in its high regime against 75 to 89
per cent when it was low, for the cash rate, Australian unemployment, gold and the S&P 500; no spike of the registered index
moved the centre by an amount whose 80 per cent interval excluded zero.
Logic: the indices come from opendata.risk_indices (GPR global, threats, acts and by country; EPU for the United States,
Australia and global). For each index shown for the target: its latest month, value, percentile against its last ten years
and regime. For the index registered for the target's domain (and for global GPR beside it), forecast.regimes.event_study
gives the target's median change over the horizon after past spike onsets against its median change from every origin, with
an 80 per cent interval; only the registered index's result can be called measured (testing several indices and keeping the
one that "works" would manufacture an effect). The output says how to use it: the regime conditions the band (forecast_report
centre 'combination' does this on its own); a scenario shift may cite this call; a shift that is not measured is judgement.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from prophet_agent.forecast import candidates as cd
from prophet_agent.forecast import regimes as rg
from prophet_agent.opendata import risk_indices as ri
from prophet_agent.runtime.runner import Tool
from prophet_agent.tools.series_io import resolve_series, run_dates

ALLOWED = {"source", "goal", "horizon"}


def _fetch(url: str) -> bytes:
    from prophet_agent.opendata import market
    return market._get(url, timeout_s=60.0, resolver=None)[0]


def _run(args: dict, _key: str, fetch: Callable[[str], bytes] | None = None) -> dict[str, Any]:
    from prophet_agent.research import driver_families as dfam
    extra = sorted(set(args) - ALLOWED)
    if extra:
        raise ValueError(f"unknown arguments {extra}; allowed: {sorted(ALLOWED)}")
    goal = str(args.get("goal") or "")
    h = args.get("horizon")
    if not isinstance(h, int) or isinstance(h, bool) or not 1 <= h <= 120:
        raise ValueError("horizon must be the forecast's number of steps, 1 to 120")
    named = resolve_series({"source": args.get("source")})
    name, y = named[0]
    dates = run_dates({"source": args.get("source")}, name)
    if not dates or len(dates) != len(y):
        raise ValueError("the series needs its dates to be lined up with the monthly indices; save it with dates")
    domain, place = dfam.detect(goal), dfam.place_of(goal)
    fetch = fetch or _fetch
    registered = ri.domain_index(domain, place)
    cache: dict[str, bytes] = {}
    y = np.asarray(y, dtype=float)
    relative = domain in cd.LEVEL_LOG_DOMAINS and bool((y > 0).all())
    cut = cd.cutoffs(dates)
    board, studies, failures = [], [], []
    for key in ri.dashboard_keys(domain, place):
        try:
            idx = ri.fetch_index(key, fetch, cache)
        except Exception as exc:          # one unreadable file never hides the others
            failures.append(f"{key}: {str(exc)[:160]}")
            continue
        d = cd.Dated(idx["rows"], idx["label"])
        recent = d.vals[-rg.WINDOW:]
        pct = rg.percentile_of_last(recent)
        regime = "low" if pct < rg.LOW else "high" if pct >= rg.HIGH else "mid"
        row = {"index": key, "label": idx["label"], "url": idx["url"], "month": idx["last"][:7],
               "value": round(float(d.vals[-1]), 2), "percentile_10y": round(pct, 2), "regime": regime,
               "registered": key == registered}
        if len(d.vals) >= 4:
            row["change_3m"] = round(float(d.vals[-1] - d.vals[-4]), 2)
        board.append(row)
        if key in (registered, "gpr"):
            ev = rg.event_study(y, cut, d.days, d.vals, h, relative=relative)
            ev = {k: v for k, v in ev.items() if k != "after_each"}
            if key != registered:
                ev["measured"] = False
                ev["note"] = "shown beside the registered index; not counted as measured"
            studies.append({"index": key, "label": idx["label"], "url": idx["url"], **_rounded(ev),
                            "in_units": _units(ev, relative)})
    if not board:
        raise ValueError("no risk index could be read: " + "; ".join(failures[:3]))
    today = next((b for b in board if b["registered"]), None)
    return {"series": name, "domain": domain, "place": place, "horizon": h, "registered_index": registered,
            "today": today, "indices": board, "event_studies": studies, "failures": failures,
            "figure_label": "measured on the index files and the series (real staging data)",
            "method": ("each index ranked against its last ten years (regime low below the 1/3 point, high from the 2/3 "
                       "point); spike onsets are months at or above the trailing 90th percentile after six months below; "
                       "the target's median change over the horizon after onsets against its median change from every "
                       "origin, with an 80 per cent interval from resampling onsets and blocks of origins")}


def _rounded(ev: dict) -> dict:
    out = {}
    for k, v in ev.items():
        if isinstance(v, float):
            out[k] = round(v, 6)
        elif isinstance(v, list) and v and isinstance(v[0], float):
            out[k] = [round(x, 6) for x in v]
        else:
            out[k] = v
    return out


def _units(ev: dict, relative: bool) -> str | None:
    if "shift" not in ev:
        return None
    lo, hi = ev["ci80"]
    if relative:
        pct = lambda x: f"{100 * (np.exp(x) - 1):+.1f}%"
        return f"{pct(ev['shift'])} over {ev['h']} steps (80% interval {pct(lo)} to {pct(hi)})"
    return f"{ev['shift']:+.3g} over {ev['h']} steps (80% interval {lo:+.3g} to {hi:+.3g})"


def observe(result: dict) -> dict:
    if "indices" not in result:
        return result
    today = result.get("today") or {}
    measured = [s for s in result.get("event_studies", []) if s.get("measured")]
    return {**{k: result[k] for k in ("series", "horizon", "registered_index", "today", "indices", "event_studies",
                                       "figure_label", "method") if k in result},
            "failures": result.get("failures", [])[:3],
            "state": (f"state today's {today.get('label', 'registered index')} reading ({today.get('value')} in "
                      f"{today.get('month')}, {today.get('percentile_10y', 0):.0%} of its last ten years, regime "
                      f"{today.get('regime')}) and what it means for the band (the combination's coverage by regime); "
                      + ("the measured response after past spikes may set a scenario shift, citing this call"
                         if measured else
                         "no spike response is measured for this target, so any risk scenario shift is judgement and must "
                         "be labelled as such"))}


def risk_regime_tool(fetch: Callable[[str], bytes] | None = None) -> Tool:
    return Tool("risk_regime", lambda args, key: _run(args, key, fetch), idempotent=True,
                doc="risk_regime: geopolitical and policy risk today, and the target's moves after past risk spikes",
                title="Risk regime", activity="Reading geopolitical and policy risk", observe=observe)


RISK_REGIME_SPEC = {
    "name": "risk_regime",
    "title": "Risk regime",
    "activity": "Reading geopolitical and policy risk",
    "description": "Read the geopolitical-risk indices (Caldara-Iacoviello GPR: global, threats, acts, and the place's own) "
                   "and the economic-policy-uncertainty indices (Baker-Bloom-Davis EPU: the place's own and global): each "
                   "index's latest month, value, percentile against its last ten years and regime (low, mid, high). For the "
                   "index registered for the target's domain, and for global GPR beside it, measure how the target moved "
                   "over the horizon after past spikes against its moves from every origin, with an 80 per cent interval; "
                   "only the registered index's result can count as measured. Use it to state the risk regime beside the "
                   "forecast and to set or label scenario shifts (cite this call as evidence).",
    "args": {"source": "object {run_id, file, name}: the dated series being forecast",
             "goal": "the forecast asked for in words (names the domain and place)",
             "horizon": "the forecast's number of steps"},
}
