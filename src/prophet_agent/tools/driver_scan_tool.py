"""driver_scan: score the domain's named outside drivers against the target series with the control lab.

Motivation vs Logic
Motivation (round 20261128-110827; owner decision 2026-11-28, "Both"): AUUNEMP and SPX never tested a driver although the
blocking checks named the step on every refused finish. The harness now runs this tool itself once the target series is
saved, and the model is also offered it. The model interprets the verdicts; it does not have to assemble them.
Logic: the domain and place are read from the goal (driver_families). Each candidate (opendata.driver_series) is fetched
through market._get (netguard, size and time caps), aligned to the target's own dates with a one-period lag, and the target
is cut to the periods where the driver has a value. control_lab's run_lab then scores TimesFM with that driver as a past
covariate against TimesFM without it, on the same rolling origins (MASE gain, pinball gain, wins, Holm-adjusted decision).
The scoring horizon is the requested one, halved until the overlap holds enough folds. A driver that cannot be fetched or is
too short is listed with its reason. Every number is measured on the fetched series; nothing is assumed.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from prophet_agent.forecast.lab import MAX_HORIZON, run_lab
from prophet_agent.opendata import driver_series as ds
from prophet_agent.opendata import market
from prophet_agent.opendata.market_benchmark import parse_close
from prophet_agent.research import driver_families as dfam
from prophet_agent.runtime.runner import Tool
from prophet_agent.tools.lab_tool import _LazyEngine, _timesfm
from prophet_agent.tools.series_io import SOURCE_HELP, resolve_series, run_dates

ALLOWED = {"source", "goal", "horizon"}
MIN_OVERLAP = 24
MAX_DRIVERS = 5


def _fetch(url: str) -> bytes:
    return market._get(url, timeout_s=market.DEFAULT_TIMEOUT_S, resolver=None)[0]


def _verdict(res: dict) -> dict:
    row = next((r for r in res.get("table", []) if r.get("name") == "with_covariates"), None)
    if row is None:
        why = next((s.get("reason") for s in res.get("skipped", []) if s.get("candidate") == "with_covariates"), None) or \
            next((f.get("error") for f in res.get("failed", []) if f.get("candidate") == "with_covariates"), "not run")
        return {"decision": "not scored", "reason": str(why)[:200]}
    return {"mase_gain_pct": row.get("gain_pct"), "pinball_gain_pct": row.get("pinball_gain_pct"), "wins": row.get("wins"),
            "folds": res.get("fold_count"), "decision": row.get("decision"), "interval_pct": row.get("interval_pct")}


def _score(values: list[float], cov: list[float], name: str, horizon: int, engine: Any) -> tuple[dict, int]:
    h = horizon
    last_err = None
    while h >= 1:
        try:
            res = run_lab(values, h, engine, ["with_covariates"], past_covariates={name: cov})
            return res, h
        except ValueError as exc:
            last_err = exc
            if "needs at least" not in str(exc) and "folds" not in str(exc):
                raise
            h //= 2
    raise ValueError(str(last_err))


def _run(args: dict, _key: str, engine_factory: Callable[[], Any] | None = None,
         get: Callable[[str], bytes] | None = None) -> dict[str, Any]:
    extra = sorted(set(args) - ALLOWED)
    if extra:
        raise ValueError(f"unknown arguments {extra}; allowed: {sorted(ALLOWED)}")
    goal = args.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal must be the forecast asked for, in words (it names the target and the place)")
    horizon = args.get("horizon", 12)
    if not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"horizon must be an integer from 1 to {MAX_HORIZON}, in the series' own periods")
    named = resolve_series({"source": args.get("source")})
    if len(named) != 1:
        raise ValueError(f"score one series at a time; choose with source.name from {[n for n, _ in named][:20]}")
    name, arr = named[0]
    dates = run_dates({"source": args["source"]}, name)
    if not dates or len(dates) != len(arr):
        raise ValueError("the series file holds no dates for this series (or not one per value); save it with its dates "
                         "(prophet_lib.save_series of a dated pandas Series) so drivers can be aligned by date")
    domain, place = dfam.detect(goal), dfam.place_of(goal)
    cands = ds.candidates(domain, place)[:MAX_DRIVERS]
    if not cands:
        raise ValueError(f"no named driver series for the domain {domain!r}; test drivers by hand (download them, save "
                         "them as covariates with the series, and score them with control_lab with_covariates)")
    fetch = get or _fetch
    engine = _LazyEngine(engine_factory or _timesfm)
    target = [float(v) for v in arr]
    scored, not_scored = [], []
    for c in cands:
        try:
            rows = parse_close(fetch(c["url"]), c["id"])
        except (market.MarketError, ValueError) as exc:
            not_scored.append({**c, "reason": f"not fetched: {str(exc)[:150]}"})
            continue
        cov = ds.aligned(dates, rows)
        start = next((k for k, v in enumerate(cov) if v is not None), None)
        if start is None or len(cov) - start < MIN_OVERLAP:
            not_scored.append({**c, "reason": f"overlaps the target for {0 if start is None else len(cov) - start} "
                                              f"periods; {MIN_OVERLAP} are needed"})
            continue
        y, x = target[start:], [float(v) for v in cov[start:]]
        try:
            res, h = _score(y, x, c["id"], horizon, engine)
        except (ValueError, RuntimeError) as exc:
            not_scored.append({**c, "reason": f"not scored: {str(exc)[:200]}"})
            continue
        corr = ds.change_corr(y, x)
        direction = "none" if corr is None or abs(corr) < 0.1 else ("up" if corr > 0 else "down")
        scored.append({**c, "from": dates[start][:10], "to": dates[-1][:10], "points": len(y), "horizon_scored": h,
                       "lag_periods": ds.LAG_PERIODS, "change_corr": corr, "direction": direction,
                       "data_last": rows[-1][0], "verdict": _verdict(res)})
    won = [s["id"] for s in scored if s["verdict"].get("decision") == "adopt"]
    return {"target": name, "domain": domain, "place": place, "points": len(target), "drivers": scored,
            "not_scored": not_scored, "adopted": won, "candidates": "with_covariates",
            "method": ("each driver is the mean of its observations in each target period, shifted one period later (known "
                       "by the end of the period before); the target is cut to the periods the driver covers; TimesFM with the "
                       "driver as a past covariate is compared with TimesFM alone on the same rolling origins (control lab, "
                       "Holm-adjusted). change_corr is the correlation of period changes of the target and the lagged "
                       "driver; direction is its sign where |corr| >= 0.1."),
            "next": ("record each scored driver with driver_map action record (kind 'tested', its family, the verdict and "
                     "change_corr as the effect, and its url as evidence); keep an adopted driver in the forecast "
                     "(save it with the series as a covariate); a driver with no evidence of gain is reported as tested and "
                     "dropped")}


def observe(result: dict) -> dict:
    lines = []
    for s in result.get("drivers", []):
        v = s["verdict"]
        gain = v.get("mase_gain_pct")
        lines.append(f"{s['family']}: {s['label']} [{s['id']}], {s['points']} periods {s['from']} to {s['to']}, horizon "
                     f"{s['horizon_scored']}: MASE gain {'n/a' if gain is None else f'{gain:+.1f}%'}, pinball "
                     f"{v.get('pinball_gain_pct')}, wins {v.get('wins')}/{v.get('folds')}, {v.get('decision')}; change "
                     f"corr {s['change_corr']} ({s['direction']}); evidence {s['url']}")
    return {"target": result.get("target"), "domain": result.get("domain"), "place": result.get("place"),
            "drivers": lines, "adopted": result.get("adopted"),
            "scored": [{"family": s["family"], "id": s["id"], "url": s["url"], "decision": s["verdict"].get("decision"),
                        "direction": s["direction"]} for s in result.get("drivers", [])],
            "not_scored": [f"{n['id']}: {n['reason']}" for n in result.get("not_scored", [])],
            "method": result.get("method"), "next": result.get("next")}


def driver_scan_tool(engine_factory: Callable[[], Any] | None = None,
                     get: Callable[[str], bytes] | None = None) -> Tool:
    return Tool("driver_scan", lambda args, key: _run(args, key, engine_factory, get), idempotent=True,
                doc="driver_scan: fetch the named outside drivers for the target's domain and score each with the control "
                    "lab (with_covariates) against the target",
                title="Driver scan", activity="Testing outside drivers against the target", observe=observe)


DRIVER_SCAN_SPEC = {
    "name": "driver_scan",
    "title": "Driver scan",
    "activity": "Testing outside drivers against the target",
    "description": "Score the outside drivers of the target's domain against the saved target series: up to five named "
                   "series (rates, yields, the dollar, oil, volatility, the place's own labour or rate series) are fetched, "
                   "aligned to the target's dates one period behind (no look-ahead), and each is scored as a past covariate "
                   "of TimesFM against TimesFM alone on the same rolling origins (MASE and pinball gain, wins, decision). "
                   "Each row also gives the correlation of period changes and its direction. Use the rows as tested driver "
                   "findings (driver_map record, kind 'tested', the row's url as evidence).",
    "args": {"source": SOURCE_HELP + " The file must hold the series' dates.",
             "goal": "the forecast asked for, in words; it names the target and the place (domain and place are read from it)",
             "horizon": f"optional integer 1 to {MAX_HORIZON}, the periods to score at (halved until the history holds "
                        "enough folds); default 12"},
}
