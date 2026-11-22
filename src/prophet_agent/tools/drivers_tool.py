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
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from prophet_agent.runtime.runner import Tool
from prophet_agent.sandbox.runner import OUT_DIR, SANDBOX_ROOT, run_python
from prophet_agent.tools.series_io import SOURCE_HELP, source_path

MAX_EVENTS = 50
MAX_HORIZON = 512
NAME = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
COUNTRY = re.compile(r"^[A-Z]{2}$")
SUBDIV = re.compile(r"^[A-Za-z0-9_-]{1,10}$")
DRIVERS_OUT = f"{OUT_DIR}/series_drivers.json"
ANOMALY_OUT = f"{OUT_DIR}/anomalies.json"
MAX_RANGES_SHOWN = 40


def _date(raw: Any, what: str) -> str:
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


def _run_code(code: str, source: Path) -> Any:
    return run_python(code, timeout_s=60, mem_mb=1536, inputs={"series.json": source})


def _read_out(run_id: str, rel: str) -> dict:
    return json.loads((SANDBOX_ROOT / run_id / rel).read_text(encoding="utf-8"))


def _build(args: dict, _key: str) -> dict[str, Any]:
    extra = sorted(set(args) - {"source", "events", "holidays", "horizon"})
    if extra:
        raise ValueError(f"unknown arguments {extra}; allowed: source, events, holidays, horizon")
    horizon = args.get("horizon")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"horizon must be an integer from 1 to {MAX_HORIZON}: the periods you will forecast")
    events = _events(args.get("events"))
    hol = _holidays(args.get("holidays"))
    if not events and hol is None:
        raise ValueError("give events (named date windows) and/or holidays (a public-holiday calendar)")
    names = [n for n, _, _ in events] + ([hol["name"]] if hol else [])
    if len(set(names)) != len(names):
        raise ValueError(f"driver names must be unique; got {names}")
    src = source_path(args)
    spans: dict[str, list[tuple[str, str]]] = {}
    for name, a, b in events:
        spans.setdefault(name, []).append((a, b))
    out = DRIVERS_OUT
    res = _run_code(DRIVERS_CODE.format(want=_series_name(args), horizon=horizon, spans=spans, hol=hol, out=out), src)
    if not res.ok:
        raise ValueError("the drivers could not be built: " + (res.stderr.strip()[-400:] or "timed out"))
    doc = _read_out(res.run_id, out)
    n = len(doc["series"]) if isinstance(doc["series"], list) else 0
    covs = doc.get("covariates", {})
    summary = {c: {"history_periods_active": sum(1 for v in vs[:-horizon] if v > 0) if horizon < len(vs) else 0,
                   "horizon_periods_active": sum(1 for v in vs[-horizon:] if v > 0)} for c, vs in covs.items()}
    return {"run_id": res.run_id, "file": out, "history_points": n, "horizon": horizon,
            "covariates": summary, "covariate_names": sorted(covs)}


def _observe_build(result: dict) -> dict:
    return {k: result[k] for k in ("run_id", "file", "history_points", "horizon", "covariates")} | {
        "use": "pass source {run_id, file} and future_covariates with the names above to timesfm_forecast or backtest_series"}


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
    out["candidate_count"] = len(result.get("indices", []))
    out["candidates"] = result.get("candidates", [])[:MAX_RANGES_SHOWN]
    out["ignore_ranges"] = ranges[:MAX_RANGES_SHOWN]
    if len(ranges) > MAX_RANGES_SHOWN:
        out["note"] = f"{len(ranges)} ranges; the first {MAX_RANGES_SHOWN} shown"
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
                   "with backtest_series. Use it only for drivers the person told you about or that a public calendar gives.",
    "args": {"source": SOURCE_HELP,
             "events": "optional list of {name: lowercase covariate name, start: first day, end: last day (optional, default start)}",
             "holidays": "optional {country: two-letter code such as AU, subdiv: optional state code, name: optional covariate name "
                         "(default public_holiday)}",
             "horizon": "integer, the periods you will forecast (1 to 512); drivers cover history plus this"},
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
