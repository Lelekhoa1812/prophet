"""Known-ahead drivers and suspicious points as typed tools, so the planner states each choice as arguments, not code.

Motivation vs Logic
Motivation: a promotion, a public holiday or a stock-out changes the numbers, and TimesFM 3 only uses such a driver when it
is handed to it as a covariate of the right length. Writing that by hand in sandbox code is where lengths, date alignment
and event windows go wrong. A spike is also not automatically an error. The planner needs a measured candidate list, then a
decision it can state.
Logic: ``build_drivers`` takes named event windows and/or a public-holiday calendar, aligns them to the dates saved with a
series, and writes a new series file (same values, dates, and covariates as long as history plus horizon) in a fresh
sandbox run. ``timesfm_forecast`` and ``backtest_series`` then take it by reference with future_covariates. ``detect_
anomalies`` reports candidate ignore_ranges with their scores; it does not decide which are errors. Both run in the sandbox,
so a malformed file cannot reach the server process, and both refuse bad arguments with the fix.

Dated events (M9-4): ``dated_events`` takes the events a research call found in news text. They are normalised into an event
table (every source URL kept), written as a CSV into the run, and turned into event features (an indicator, optional leads,
lags and counts per type). Each feature is kept only when a rolling backtest on this series shows it beats the simple rules
(forecast/event_check.py); the verdicts are returned with the file. ``event_features.check`` false keeps them unchecked, and
the result says so.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from prophet_agent.forecast.event_check import N_ORIGINS, check_features, guess_season
from prophet_agent.research.event_table import CERTAINTY, EVENT_TYPES, event_table_csv, normalise_events, select_events
from prophet_agent.runtime.runner import Tool
from prophet_agent.sandbox import prophet_events as pe
from prophet_agent.sandbox.prophet_events import MAX_LEAD
from prophet_agent.sandbox.runner import OUT_DIR, SANDBOX_ROOT, run_python
from prophet_agent.tools.series_io import SOURCE_HELP, source_path

MAX_EVENTS = 50
MAX_DATED = 200
EVENT_KEYS = {"types", "leads", "lags", "counts", "check", "min_certainty"}
EVENT_TABLE_IN = "in/event_table.csv"
EVENT_TABLE_OUT = f"{OUT_DIR}/event_table.csv"
MAX_HORIZON = 512
NAME = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
COUNTRY = re.compile(r"^[A-Z]{2}$")
SUBDIV = re.compile(r"^[A-Za-z0-9_-]{1,10}$")
DRIVERS_OUT = f"{OUT_DIR}/series_drivers.json"
ANOMALY_OUT = f"{OUT_DIR}/anomalies.json"
MAX_RANGES_SHOWN = 40


def _date(raw: Any, what: str) -> str:
    # A number is not a date: pandas reads 2026 as 2026 nanoseconds after 1970, so the window fell in 1970 without a word.
    if isinstance(raw, (bool, int, float)):
        raise ValueError(f"{what} must be a date such as 2026-03-14 (text), got {raw!r}")
    try:
        ts = pd.Timestamp(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a date such as 2026-03-14, got {raw!r}")
    if pd.isna(ts):
        raise ValueError(f"{what} must be a date such as 2026-03-14, got {raw!r}")
    return ts.strftime("%Y-%m-%d")


def _events(raw: Any) -> list[tuple[str, str, str]]:
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > MAX_EVENTS:
        raise ValueError(f"events must be a list of at most {MAX_EVENTS} objects {{name, start, end}}")
    out = []
    for i, e in enumerate(raw):
        if not isinstance(e, dict) or not {"name", "start"} <= set(e) or set(e) - {"name", "start", "end"}:
            raise ValueError(f"events[{i}] needs name and start (end optional), nothing else")
        name = str(e["name"])
        if not NAME.match(name):
            raise ValueError(f"events[{i}].name must be lowercase letters, digits and underscores, starting with a letter, at most 40")
        start = _date(e["start"], f"events[{i}].start")
        end = _date(e.get("end", e["start"]), f"events[{i}].end")
        if end < start:
            raise ValueError(f"events[{i}] ends before it starts")
        out.append((name, start, end))
    return out


def _holidays(raw: Any) -> dict | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) - {"country", "subdiv", "name"} or "country" not in raw:
        raise ValueError("holidays must be {country: 'AU', subdiv: optional state code, name: optional covariate name}")
    country = str(raw["country"])
    if not COUNTRY.match(country):
        raise ValueError("holidays.country must be a two-letter code such as AU or US")
    subdiv = raw.get("subdiv")
    if subdiv is not None and not SUBDIV.match(str(subdiv)):
        raise ValueError("holidays.subdiv must be a short state code such as VIC or CA")
    name = str(raw.get("name", "public_holiday"))
    if not NAME.match(name):
        raise ValueError("holidays.name must be a plain covariate name")
    return {"country": country, "subdiv": None if subdiv is None else str(subdiv), "name": name}


def _series_name(args: dict) -> str | None:
    src = args.get("source")
    name = src.get("name") if isinstance(src, dict) else None
    return None if name is None else str(name)


# The program below is built from validated values only (names, ISO dates, codes, ints) through repr(), never from raw text.
DRIVERS_CODE = '''
import json
import pandas as pd
import prophet_lib as pl

doc = json.load(open("in/series.json", encoding="utf-8"))
raw = doc["series"] if isinstance(doc, dict) and "series" in doc else doc
named = raw if isinstance(raw, dict) else {{"series": raw}}
want = {want!r}
if want is None and len(named) != 1:
    raise SystemExit("the file holds %d series; choose one with source.name: %s" % (len(named), sorted(named)[:20]))
key = want or next(iter(named))
if key not in named:
    raise SystemExit("series %r not in the file" % key)
dates = doc.get("dates") if isinstance(doc, dict) else None
dates = dates.get(key) if isinstance(dates, dict) else (dates if len(named) == 1 else None)
if not dates or len(dates) != len(named[key]):
    raise SystemExit("this series has no saved dates; save it with pl.save_series(pd.Series(values, index=dates)) so events can be aligned")
values = [float(v) for v in named[key]]
n, h = len(values), {horizon!r}
idx = pd.DatetimeIndex(pd.to_datetime(dates))
freq = doc.get("freq")
covs = {{}}
for cname, spans in {spans!r}.items():
    covs[cname] = pl.event_covariate(idx, spans, h, freq)
hol = {hol!r}
if hol:
    covs[hol["name"]] = pl.holiday_covariate(idx, hol["country"], hol["subdiv"], h, freq)
ev_opt = {ev!r}
if ev_opt is not None:
    import prophet_events as pe
    ev_table = pd.read_csv("in/event_table.csv", dtype=str, keep_default_na=False)
    feats = pe.event_features(idx, ev_table, h, freq, types=ev_opt["types"], leads=ev_opt["leads"], lags=ev_opt["lags"],
                              counts=ev_opt["counts"])
    for cname in ev_opt["keep"]:
        if cname not in feats:
            raise SystemExit("event feature %r is not built from this table and series" % cname)
        if cname in covs:
            raise SystemExit("event feature %r has the same name as another driver; rename the driver" % cname)
        covs[cname] = feats[cname]
    open("out/event_table.csv", "wb").write(open("in/event_table.csv", "rb").read())
if not covs:
    raise SystemExit("no driver was built")
pl.save_series(pd.Series(values, index=idx), {out!r}, freq=freq, covariates=covs)
'''

ANOMALY_CODE = '''
import json
import pandas as pd
import prophet_lib as pl

doc = json.load(open("in/series.json", encoding="utf-8"))
raw = doc["series"] if isinstance(doc, dict) and "series" in doc else doc
named = raw if isinstance(raw, dict) else {{"series": raw}}
want = {want!r}
if want is None and len(named) != 1:
    raise SystemExit("the file holds %d series; choose one with source.name: %s" % (len(named), sorted(named)[:20]))
key = want or next(iter(named))
if key not in named:
    raise SystemExit("series %r not in the file" % key)
dates = doc.get("dates") if isinstance(doc, dict) else None
dates = dates.get(key) if isinstance(dates, dict) else (dates if len(named) == 1 else None)
values = [float(v) for v in named[key]]
dated = bool(dates) and len(dates) == len(values)
res = pl.detect_anomalies(values, season={season!r}, threshold={threshold!r}, dates=dates if dated else None)
res["series"] = key
res["dated"] = dated
json.dump(res, open({out!r}, "w"))
print(json.dumps({{"points": len(values), "candidates": len(res["indices"]), "ranges": len(res["ignore_ranges"])}}))
'''


def _run_code(code: str, source: Path, files: dict[str, bytes] | None = None) -> Any:
    return run_python(code, timeout_s=60, mem_mb=1536, inputs={"series.json": source}, files=files)


def _periods(raw: Any, what: str) -> list[int]:
    if raw is None:
        return []
    if (not isinstance(raw, list) or len(raw) > 10
            or not all(isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= MAX_LEAD for k in raw)):
        raise ValueError(f"{what} must be a list of up to 10 whole periods from 1 to {MAX_LEAD}")
    return sorted(set(raw))


def _event_options(raw: Any) -> dict:
    """event_features: which types, leads, lags and counts to build, whether to backtest each, and the certainty floor."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or set(raw) - EVENT_KEYS:
        raise ValueError(f"event_features must be an object with any of: {', '.join(sorted(EVENT_KEYS))}")
    types = raw.get("types")
    if types is not None and (not isinstance(types, list) or not types or not all(t in EVENT_TYPES for t in types)):
        raise ValueError(f"event_features.types must be a non-empty list from: {', '.join(EVENT_TYPES)}")
    counts, check = raw.get("counts", False), raw.get("check", True)
    if not isinstance(counts, bool) or not isinstance(check, bool):
        raise ValueError("event_features.counts and event_features.check must be true or false")
    min_certainty = raw.get("min_certainty", "medium")
    if min_certainty not in CERTAINTY:
        raise ValueError(f"event_features.min_certainty must be one of {', '.join(CERTAINTY)}")
    return {"types": None if types is None else sorted(set(types)), "leads": _periods(raw.get("leads"), "event_features.leads"),
            "lags": _periods(raw.get("lags"), "event_features.lags"), "counts": counts, "check": check,
            "min_certainty": min_certainty}


def _event_plan(raw: Any, opts: Any) -> dict | None:
    """The dated events as an event table: normalised, deduplicated and filtered by certainty. None when none were given."""
    if raw is None:
        if opts is not None:
            raise ValueError("event_features needs dated_events: pass the events list of a research call")
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError("dated_events must be a non-empty list of event objects (the events list of a research call)")
    if len(raw) > MAX_DATED:
        raise ValueError(f"dated_events takes at most {MAX_DATED} events; pass the most relevant")
    options = _event_options(opts)
    rows, refused = normalise_events(raw)
    kept_rows = select_events(rows, options["min_certainty"])
    if not kept_rows:
        raise ValueError(f"no dated event is at least {options['min_certainty']} certain ({len(refused)} refused, "
                         f"{len(rows)} read); set event_features.min_certainty to low to use them, or give dated events "
                         "with a year written in the text")
    return {"rows": kept_rows, "csv": event_table_csv(kept_rows), "refused": refused, "opts": options}


def _saved_series(src: Path, want: str | None) -> tuple[list[float], list[str], str | None]:
    """The values, dates and freq of the chosen series in a saved series file, read the way the sandbox reads it."""
    doc = json.loads(src.read_text(encoding="utf-8"))
    raw = doc["series"] if isinstance(doc, dict) and "series" in doc else doc
    named = raw if isinstance(raw, dict) else {"series": raw}
    if want is None and len(named) != 1:
        raise ValueError(f"the file holds {len(named)} series; choose one with source.name: {sorted(named)[:20]}")
    key = want or next(iter(named))
    if key not in named:
        raise ValueError(f"series {key!r} not in the file")
    dates = doc.get("dates") if isinstance(doc, dict) else None
    dates = dates.get(key) if isinstance(dates, dict) else (dates if len(named) == 1 else None)
    if not dates or len(dates) != len(named[key]):
        raise ValueError("this series has no saved dates; save it with pl.save_series(pd.Series(values, index=dates)) so events can be aligned")
    return [float(v) for v in named[key]], list(dates), doc.get("freq")


def _event_features_in_process(plan: dict, src: Path, want: str | None, horizon: int) -> tuple[dict, dict | None, list[str]]:
    """Build the candidate event features on the series calendar, backtest each (unless check is false), and return
    (features, verdicts, kept names)."""
    values, dates, freq = _saved_series(src, want)
    table = pd.read_csv(io.BytesIO(plan["csv"]), dtype=str, keep_default_na=False)
    opts = plan["opts"]
    feats = pe.event_features(dates, table, horizon, freq, types=opts["types"], leads=opts["leads"], lags=opts["lags"],
                              counts=opts["counts"])
    if not feats:
        raise ValueError("none of the dated events falls inside the series or its horizon, so no event feature was built; "
                         "check the dates against the series and the horizon")
    if not opts["check"]:
        return feats, None, sorted(feats)
    try:
        verdict = check_features(values, feats, horizon, season=guess_season(dates), n_origins=N_ORIGINS)
    except ValueError as exc:
        raise ValueError(f"the event features could not be backtested ({exc}); give a longer series, or set "
                         "event_features.check to false to keep them unchecked") from None
    return feats, verdict["verdicts"], verdict["kept"]


def _read_out(run_id: str, rel: str) -> dict:
    return json.loads((SANDBOX_ROOT / run_id / rel).read_text(encoding="utf-8"))


def _build(args: dict, _key: str) -> dict[str, Any]:
    extra = sorted(set(args) - {"source", "events", "holidays", "horizon", "dated_events", "event_features"})
    if extra:
        raise ValueError(f"unknown arguments {extra}; allowed: source, events, holidays, horizon, dated_events, event_features")
    horizon = args.get("horizon")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"horizon must be an integer from 1 to {MAX_HORIZON}: the periods you will forecast")
    events = _events(args.get("events"))
    hol = _holidays(args.get("holidays"))
    plan = _event_plan(args.get("dated_events"), args.get("event_features"))
    if not events and hol is None and plan is None:
        raise ValueError("give events (named date windows), holidays (a public-holiday calendar) and/or dated_events "
                         "(the events of a research call)")
    names = [n for n, _, _ in events] + ([hol["name"]] if hol else [])
    if len(set(names)) != len(names):
        raise ValueError(f"driver names must be unique; got {names}")
    src = source_path(args)
    spans: dict[str, list[tuple[str, str]]] = {}
    for name, a, b in events:
        spans.setdefault(name, []).append((a, b))
    event_block, ev_opt, files = None, None, None
    if plan is not None:
        _feats, verdicts, keep = _event_features_in_process(plan, src, _series_name(args), horizon)
        if not keep and not events and hol is None:
            raise ValueError("no event feature beat the simple rules in the rolling backtest, so no driver was built: "
                             + "; ".join(f"{v['feature']} improvement {v['improvement_vs_best_rule']}"
                                         for v in (verdicts or {}).values()) + ". Measured; see event_features.check")
        opts = plan["opts"]
        ev_opt = {"types": opts["types"], "leads": opts["leads"], "lags": opts["lags"], "counts": opts["counts"], "keep": keep}
        files = {EVENT_TABLE_IN: plan["csv"]}
        event_block = {"rows": len(plan["rows"]), "refused": len(plan["refused"]),
                       "refused_reasons": plan["refused"][:5], "features_built": sorted(_feats), "kept": keep,
                       "checked": opts["check"], "min_certainty": opts["min_certainty"],
                       "verdicts": verdicts, "table_file": EVENT_TABLE_OUT}
    out = DRIVERS_OUT
    res = _run_code(DRIVERS_CODE.format(want=_series_name(args), horizon=horizon, spans=spans, hol=hol, out=out, ev=ev_opt),
                    src, files)
    if not res.ok:
        raise ValueError("the drivers could not be built: " + (res.stderr.strip()[-400:] or "timed out"))
    doc = _read_out(res.run_id, out)
    n = len(doc["series"]) if isinstance(doc["series"], list) else 0
    covs = doc.get("covariates", {})
    summary = {c: {"history_periods_active": sum(1 for v in vs[:-horizon] if v > 0) if horizon < len(vs) else 0,
                   "horizon_periods_active": sum(1 for v in vs[-horizon:] if v > 0)} for c, vs in covs.items()}
    result = {"run_id": res.run_id, "file": out, "history_points": n, "horizon": horizon,
              "covariates": summary, "covariate_names": sorted(covs)}
    if event_block is not None:
        result["event_features"] = event_block
    return result


def _observe_build(result: dict) -> dict:
    out = {k: result[k] for k in ("run_id", "file", "history_points", "horizon", "covariates")} | {
        "use": "pass source {run_id, file} and future_covariates with the names above to timesfm_forecast or backtest_series"}
    if "event_features" in result:
        out["event_features"] = {**result["event_features"], "reading": (
            "Event features are kept only where the rolling backtest measured a gain over the simple rules (verdicts); "
            "the events are source claims, not measurements.")}
    return out


def _detect(args: dict, _key: str) -> dict[str, Any]:
    extra = sorted(set(args) - {"source", "season", "threshold"})
    if extra:
        raise ValueError(f"unknown arguments {extra}; allowed: source, season, threshold")
    season = args.get("season")
    if season is not None and (not isinstance(season, int) or isinstance(season, bool) or not 1 <= season <= 1000):
        raise ValueError("season must be an integer from 1 to 1000 (1 for none), or left out")
    threshold = args.get("threshold", 6.0)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 3 <= threshold <= 20:
        raise ValueError("threshold must be a number from 3 to 20 (default 6; lower finds more candidates)")
    src = source_path(args)
    res = _run_code(ANOMALY_CODE.format(want=_series_name(args), season=season, threshold=float(threshold), out=ANOMALY_OUT), src)
    if not res.ok:
        raise ValueError("the series could not be checked: " + (res.stderr.strip()[-400:] or "timed out"))
    doc = _read_out(res.run_id, ANOMALY_OUT)
    return {"run_id": res.run_id, **doc}


def _observe_detect(result: dict) -> dict:
    ranges = result.get("ignore_ranges", [])
    out = {k: result[k] for k in ("run_id", "series", "points", "method", "threshold", "dated") if k in result}
    candidates = result.get("candidates", [])
    out["candidate_count"] = len(result.get("indices", []))
    out["candidates"] = candidates[:MAX_RANGES_SHOWN]
    out["ignore_ranges"] = ranges[:MAX_RANGES_SHOWN]
    # Round 6m: the candidate list was cut with no note; only the ranges had one.
    notes = [f"{len(seq)} {label}; the first {MAX_RANGES_SHOWN} shown" for seq, label in
             ((candidates, "candidates"), (ranges, "ranges")) if len(seq) > MAX_RANGES_SHOWN]
    if notes:
        out["note"] = "; ".join(notes) + ". The full lists are kept in the ledger"
    out["reading"] = ("These are candidates, not verdicts. Compare each against known events and holidays before passing "
                      "any range as ignore_ranges; keep only the unexplained ones and check with backtest_series.")
    return out


def build_drivers_tool() -> Tool:
    return Tool("build_drivers", _build, idempotent=True, observe=_observe_build)


def detect_anomalies_tool() -> Tool:
    return Tool("detect_anomalies", _detect, idempotent=True, observe=_observe_detect)


BUILD_DRIVERS_SPEC = {
    "name": "build_drivers",
    "title": "Drivers",
    "activity": "Building the known events and holidays",
    "description": "Build known-ahead drivers (promotions, campaigns, public holidays) for a series that has saved dates. "
                   "Each driver is a 0 to 1 share of each period inside the event, as long as the history plus the horizon. "
                   "Writes a new series file that carries the same values and these driver columns; forecast that file with "
                   "timesfm_forecast (source = the returned run_id and file, future_covariates = the returned names) and check "
                   "with backtest_series. Use it only for drivers the person told you about or that a public calendar gives. "
                   "dated_events (the events list of a web_research call) become event features; each is kept only if a rolling "
                   "backtest on this series beats the simple rules, and the verdicts are returned.",
    "args": {"source": SOURCE_HELP,
             "events": "optional list of {name: lowercase covariate name, start: first day, end: last day (optional, default start)}",
             "holidays": "optional {country: two-letter code such as AU, subdiv: optional state code, name: optional covariate name "
                         "(default public_holiday)}",
             "horizon": "integer, the periods you will forecast (1 to 512); drivers cover history plus this",
             "dated_events": f"optional list of up to {MAX_DATED} event objects from a research call: each needs date (YYYY-MM-DD), "
                             "event (text) and source_url; end_date, type, entity, how_sure and date_found are optional",
             "event_features": "optional {types: list of event types, leads: list of periods, lags: list of periods (1 to 30), "
                               "counts: true or false, check: true (default; backtest each feature), min_certainty: high, medium "
                               "(default) or low}"},
}

DETECT_ANOMALIES_SPEC = {
    "name": "detect_anomalies",
    "title": "Spike check",
    "activity": "Looking for spikes and errors in the series",
    "description": "Measured candidates for points far from their neighbours or the usual cycle: index, date, value, score, and "
                   "ignore_ranges in the form timesfm_forecast takes. It reports facts only; a promotion or holiday looks like a "
                   "spike too. Explain each range you intend to ignore, and ignore only the unexplained ones.",
    "args": {"source": SOURCE_HELP,
             "season": "length of the repeating cycle in periods, 1 for none (optional; use the value from find_season or the data)",
             "threshold": "robust score cut-off from 3 to 20 (optional, default 6)"},
}
