"""Deterministic contract checks on the evidence a mission gathered.

Motivation vs Logic
Motivation: a model that says it is finished is a claim. A check that runs on the evidence is not. A forecast can have the
wrong number of steps, a band that does not bracket its own point, a deliverable that changed after it was listed, or no
baseline to be judged against, and the planner will still say "met". These checks catch that without a model call, so the
evaluator and the loop have something concrete to point at.
Logic: every check is a plain dict {name, ok, severity, detail}. A failing check with severity "block" stops a finish from
being accepted; "warn" is reported and does not stop it. The functions are pure: they read the observations and, for
deliverables only, the files named in them. Malformed input never raises; a check whose inputs are missing is skipped.
"""
from __future__ import annotations

import re

import hashlib
import math
import numbers
import json
from pathlib import Path
from typing import Any

from prophet_agent.forecast.timesfm import nonnegative_domain
from prophet_agent.runtime import grounding
from prophet_agent.runtime.questions import SYNTHETIC_GOAL, invented_data_phrase
from prophet_agent.source_table.diagnose import diagnose_failure

EXPECTED_KINDS = ("png", "svg", "xlsx", "docx", "pdf")
INTERVAL_SHARE = 0.9      # share of steps that must sit inside q10 <= point <= q90
SCALE_MARGIN = 3.0        # forecast may run this many history ranges beyond the history's own range
FORECAST_TOOLS = ("timesfm_forecast", "forecast_report")
INSPECTION_TOOLS = ("sandbox_python", "inspect_data")


def _check(name: str, ok: bool, detail: str, severity: str = "block") -> dict:
    return {"name": name, "ok": bool(ok), "severity": severity, "detail": str(detail)}


def _numbers(seq: Any) -> list[float] | None:
    """The values as floats (non-finite allowed), or None when the value is not a list of real numbers."""
    if not isinstance(seq, (list, tuple)):
        return None
    out: list[float] = []
    for v in seq:
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            return None
        try:
            out.append(float(v))
        except (OverflowError, ValueError, TypeError):
            return None
    return out


def _finite(v: Any) -> float | None:
    """A finite real number as a float, or None."""
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        return None
    try:
        f = float(v)
    except (OverflowError, ValueError, TypeError):
        return None
    return f if math.isfinite(f) else None


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


# ---------------------------------------------------------------- forecast

def check_forecast(point, q10=None, q90=None, *, horizon=None, history=None) -> list[dict]:
    """Checks on one forecast. Checks whose inputs are not given are left out, not reported as passing."""
    p = _numbers(point)
    if p is None:
        return [_check("forecast_finite", False, "the forecast is not a list of numbers")]
    checks: list[dict] = []
    if horizon is not None:
        checks.append(_check("forecast_length", len(p) == horizon,
                             f"the forecast has {len(p)} steps and the horizon is {horizon}"))
    bad = sum(1 for v in p if not math.isfinite(v))
    checks.append(_check("forecast_finite", bad == 0,
                         "every forecast value is a finite number" if bad == 0
                         else f"{bad} forecast values are missing or infinite"))
    if q10 is not None and q90 is not None and p:
        lo, hi = _numbers(q10), _numbers(q90)
        if lo is None or hi is None or len(lo) != len(p) or len(hi) != len(p):
            checks.append(_check("interval_order", False,
                                 "the interval bounds are not lists of numbers matching the forecast length", "warn"))
        else:
            inside = sum(1 for a, m, b in zip(lo, p, hi) if a <= m <= b)
            checks.append(_check("interval_order", inside / len(p) >= INTERVAL_SHARE,
                                 f"q10 <= point <= q90 holds at {inside} of {len(p)} steps; at least "
                                 f"{round(INTERVAL_SHARE * 100)} per cent are needed", "warn"))
    hist = _numbers(history) if history is not None else None
    if hist and all(math.isfinite(v) for v in hist) and p and nonnegative_domain(hist):
        # Live retail rep3 (measured): -3,606 on a Saturday, while the history held only two refund days below zero. The
        # domain is non-negative when a small share of points is below zero (forecast.timesfm.NEG_SHARE_MAX), so those
        # days no longer switch the check off. The band counts too: a negative q10 is an impossible sales value as well.
        neg = sum(1 for v in p if not v >= 0)
        band_lo = _numbers(q10) if q10 is not None else None
        band_hi = _numbers(q90) if q90 is not None else None
        band_neg = sum(1 for v in (band_lo or []) + (band_hi or []) if math.isfinite(v) and v < 0)
        ok = neg == 0 and band_neg == 0
        if ok:
            detail = "the history is essentially non-negative and the forecast and its band are never negative"
        else:
            parts = []
            if neg:
                parts.append(f"{neg} forecast steps are below zero")
            if band_neg:
                parts.append(f"{band_neg} band values are below zero")
            detail = "the history is essentially non-negative but " + " and ".join(parts)
        checks.append(_check("non_negative_when_history_is", ok, detail))
    if hist and all(math.isfinite(v) for v in hist) and p and all(math.isfinite(v) for v in p):
        lo_h, hi_h = min(hist), max(hist)
        span = hi_h - lo_h
        floor, ceil = lo_h - SCALE_MARGIN * span, hi_h + SCALE_MARGIN * span
        out = sum(1 for v in p if v < floor or v > ceil)
        checks.append(_check("plausible_scale", out == 0,
                             "the forecast stays within the history range, widened by the history's own span" if out == 0
                             else f"{out} forecast steps lie outside the history range widened by {SCALE_MARGIN:g} times "
                                  "its span", "warn"))
    return checks


# ---------------------------------------------------------------- backtest

def _flagged(result: dict) -> bool:
    return bool(result.get("flag") or result.get("flags"))


def check_backtest(result) -> list[dict]:
    """Checks on a backtest_series result: baselines present, metrics finite, TimesFM judged against the best baseline."""
    if not isinstance(result, dict):
        return [_check("has_baselines", False, "the backtest result is not a mapping of models")]
    if isinstance(result.get("series"), dict):
        # A panel backtest: the checks apply to each series that could be backtested.
        out: list[dict] = []
        for r in result["series"].values():
            if isinstance(r, dict) and "error" not in r:
                out.extend(check_backtest(r))
        return out or [_check("has_baselines", False, "no series of the panel could be backtested")]
    models = result.get("models")
    checks: list[dict] = []
    if not isinstance(models, dict) or not models:
        checks.append(_check("has_baselines", False, "the backtest names no models"))
    else:
        best = result.get("best_baseline")
        has_naive = any("naive" in str(m) for m in models)
        ok = has_naive and isinstance(best, str) and best in models
        checks.append(_check("has_baselines", ok,
                             f"a naive baseline is present and the best baseline is {best}" if ok
                             else "the backtest lacks a naive baseline or does not name the best baseline"))
        unscored = sorted(str(name) for name, m in models.items()
                          if not isinstance(m, dict) or _finite(m.get("mase")) is None)
        checks.append(_check("metrics_finite", not unscored,
                             "every model has a finite MASE" if not unscored
                             else "no finite MASE for: " + ", ".join(unscored)))
        tm = models.get("timesfm")
        tmase = _finite(tm.get("mase")) if isinstance(tm, dict) else None
        bmase = _finite(models[best].get("mase")) if isinstance(best, str) and best in models and isinstance(
            models[best], dict) else None
        if tmase is not None and bmase is not None:
            if tmase < bmase:
                checks.append(_check("beats_or_reports", True,
                                     f"TimesFM MASE {tmase:.3f} is below {best} at {bmase:.3f}"))
            elif _flagged(result):
                checks.append(_check("beats_or_reports", True,
                                     f"TimesFM MASE {tmase:.3f} does not beat {best} at {bmase:.3f}; the result is flagged"))
            else:
                checks.append(_check("model_does_not_beat_baseline", False,
                                     f"TimesFM MASE {tmase:.3f} does not beat {best} at {bmase:.3f}", "warn"))
    h = result.get("horizon")
    checks.append(_check("horizon_present", _is_int(h) and h >= 1,
                         f"the backtest horizon is {h}" if _is_int(h) and h >= 1
                         else "the backtest does not record a whole-number horizon"))
    return checks


# ---------------------------------------------------------------- deliverables

def _path_of(entry: Any) -> str | None:
    """The file path of a deliverable entry: a path string (any " (annotation)" dropped) or a {"path": ...} mapping."""
    if isinstance(entry, str):
        raw = entry.split(" (")[0].strip()
    elif isinstance(entry, dict) and isinstance(entry.get("path"), str):
        raw = entry["path"].strip()
    else:
        return None
    return raw or None


def _suffix(raw: str | None) -> str:
    return Path(raw).suffix.lower().lstrip(".") if raw else ""


def _inside(root: Path, raw: str) -> Path | None:
    """The resolved path of a deliverable if it lies under the run folder, else None."""
    p = Path(raw)
    try:
        resolved = (p if p.is_absolute() else root / p).resolve()
        resolved.relative_to(root.resolve())
    except (ValueError, OSError, RuntimeError):
        return None
    return resolved


def _sha256(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def check_deliverables(files, run_dir=None) -> list[dict]:
    """Without run_dir: the list must name each expected kind (png, svg, xlsx, docx, pdf), as a warning. With run_dir:
    every listed file must exist under it, and where a sha256 is listed it must match the file on disk."""
    items = list(files) if isinstance(files, (list, tuple)) else []
    if run_dir is None:
        kinds = {_suffix(_path_of(f)) for f in items}
        missing = [k for k in EXPECTED_KINDS if k not in kinds]
        return [_check("deliverable_kinds", not missing,
                       "all five deliverable kinds are listed" if not missing
                       else "missing deliverable kinds: " + ", ".join(missing), "warn")]
    if not items:
        return [_check("deliverable_exists", False, "no deliverables are listed")]
    root = Path(run_dir)
    checks: list[dict] = []
    for entry in items:
        raw = _path_of(entry)
        if raw is None:
            checks.append(_check("deliverable_exists", False, "a deliverable entry does not name a file"))
            continue
        name = Path(raw).name
        path = _inside(root, raw)
        if path is None:
            checks.append(_check("deliverable_exists", False, f"{name} lies outside the run folder"))
            continue
        if not path.is_file():
            checks.append(_check("deliverable_exists", False, f"{name} is missing from the run folder"))
            continue
        checks.append(_check("deliverable_exists", True, f"{name} is in the run folder"))
        expected = entry.get("sha256") if isinstance(entry, dict) else None
        if isinstance(expected, str) and expected.strip():
            actual = _sha256(path)
            ok = actual is not None and actual == expected.strip().lower()
            checks.append(_check("deliverable_hash", ok,
                                 f"{name} matches its recorded sha256" if ok
                                 else f"{name} does not match its recorded sha256 (the file changed or cannot be read)"))
    return checks


# ---------------------------------------------------------------- evidence

def _ok_result(h: Any, tool: str) -> Any:
    """The result of a successful call to ``tool`` in one observation, or None."""
    if not isinstance(h, dict) or h.get("kind") != "tool" or h.get("tool") != tool:
        return None
    if h.get("error") or "result" not in h:
        return None
    res = h["result"]
    if isinstance(res, dict) and res.get("ok") is False:
        return None
    return res


def _forecast_record(h: dict) -> dict | None:
    """The forecast carried by a successful timesfm_forecast or forecast_report observation, or None."""
    res = _ok_result(h, "timesfm_forecast")
    if isinstance(res, dict):
        return res
    res = _ok_result(h, "forecast_report")
    if isinstance(res, dict) and isinstance(res.get("forecast"), dict):
        return res["forecast"]
    if isinstance(res, dict) and "n_series" in res and isinstance(res.get("series"), dict):
        # A panel pack, seen as one line per series. It is held when a series' band lost its cycle, whatever its centre line:
        # the band is TimesFM's either way (A6-03, live retail rep4: a naive centre on a flat_unexpected band was delivered).
        lost = [n for n, ln in res["series"].items() if isinstance(ln, dict) and ln.get("health") == "flat_unexpected"]
        return {"panel": True, "n_series": res["n_series"],
                "health": {"verdict": "flat_unexpected" if lost else "ok", "causes": [f"cycle lost: {', '.join(lost[:6])}"] if lost else []}}
    return None


def _quantile(rec: dict, level: str, key: str) -> Any:
    if key in rec:
        return rec[key]
    q = rec.get("quantiles")
    return q.get(level) if isinstance(q, dict) else None


MIN_LEDGER_STAGES = 4   # distinct analysis stages a forecast's ledger must record (proposed default, to calibrate on live runs)


def with_delegated(history) -> list:
    """The observations plus the tool calls inside each sub-agent's report, so work a sub-agent did (a ledger, a scan) counts
    for the mission. Only for checks on saved files, which say nothing about who wrote them."""
    out = []
    for h in history if isinstance(history, list) else []:
        out.append(h)
        if isinstance(h, dict) and h.get("kind") == "subagent" and isinstance(h.get("evidence"), list):
            out.extend(e for e in h["evidence"] if isinstance(e, dict) and e.get("kind") == "tool")
    return out


# The ledger a cell saves: out/ledger.json, or a named variant such as out/ledger_full.json (live Beijing 3581d07d, 2026-11-23:
# seven decisions were saved as ledger_full.json, which this check did not read, so the mission reported zero stages).
LEDGER_FILE = re.compile(r"out/ledger[^/]*\.json")


def _saved_ledgers(history, run_root) -> list[list]:
    """The entries of every saved ledger in this mission (newest first). A step that saved the ledger and then failed later in
    the cell still saved it: the file is the evidence. Unreadable or unsafe paths are skipped."""
    out: list[list] = []
    for h in reversed(with_delegated(history)):
        res = h.get("result") if isinstance(h, dict) and h.get("kind") == "tool" and h.get("tool") == "sandbox_python" else None
        if not isinstance(res, dict) or not isinstance(res.get("run_id"), str) or "/" in res["run_id"] or ".." in res["run_id"]:
            continue
        entries: list = []
        read = False
        for path in [_output_path(o) for o in (res.get("outputs") or [])]:
            if not LEDGER_FILE.fullmatch(path):
                continue
            try:
                doc = json.loads((Path(run_root) / res["run_id"] / path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(doc, list):
                entries.extend(doc)
                read = True
        if read:
            out.append(entries)
    return out


def ledger_stages(history, run_root) -> int | None:
    """The distinct stages (with a finding and a decision each) in the newest saved ledger of this mission. None when
    ``run_root`` is unknown, so a caller can tell "zero" from "cannot look". The newest, not the best: a later cell that saved
    an empty ps.Ledger() over the record must be seen (live WTI rep3, A4-05, measured: the file of run d908f0a5 is [])."""
    if run_root is None:
        return None
    ledgers = _saved_ledgers(history, run_root)
    if not ledgers:
        return 0
    stages = {str(e.get("stage", "")).strip().lower() for e in ledgers[0]
              if isinstance(e, dict) and str(e.get("finding", "")).strip() and str(e.get("decision", "")).strip()}
    return len(stages - {""})


def check_ledger_kept(history, run_root) -> dict | None:
    """Warns when a later saved ledger of the mission holds fewer entries than an earlier one: a record was overwritten (A4-05,
    live WTI rep3). None when no ledger was saved or ``run_root`` is unknown."""
    if run_root is None:
        return None
    ledgers = list(reversed(_saved_ledgers(history, run_root)))      # oldest first
    if not ledgers:
        return None
    peak = 0
    for entries in ledgers:
        n = len(entries)
        if n < peak:
            return _check("ledger_kept", False,
                          f"a later saved ledger holds {n} entries, fewer than the {peak} an earlier ledger of this mission "
                          "holds: a record was overwritten. Load the earlier ledger with ps.Ledger.open(path) before adding "
                          "to it, then save", "warn")
        peak = max(peak, n)
    return _check("ledger_kept", True, f"the saved ledgers never shrink (the largest holds {peak} entries)", "warn")


def _saved_scan_findings(history, run_root) -> list[dict] | None:
    """The findings of the latest scan the mission saved (out/scan*.json in the last run that wrote one and can be read).
    None when no scan was saved or the files cannot be read.

    Live electricity c1154fd1 (2026-11-23): the check read every scan the mission had ever saved, 838 findings from several
    scans, so an earlier 15-entity scan's censoring.stockout_runs and schema.missing_column blocked a 60-client answer. The
    decision API reads only the current scan, so those ids could never be decided. A later scan replaces an earlier one.
    """
    if run_root is None:
        return None
    latest: list[dict] | None = None
    seen = False
    for h in with_delegated(history):
        res = h.get("result") if isinstance(h, dict) and h.get("kind") == "tool" and h.get("tool") == "sandbox_python" else None
        if not isinstance(res, dict) or not isinstance(res.get("run_id"), str) or "/" in res["run_id"] or ".." in res["run_id"]:
            continue
        for o in res.get("outputs") or []:
            path = _output_path(o)
            if not SCAN_FILE.fullmatch(path):
                continue
            seen = True
            try:
                doc = json.loads((Path(run_root) / res["run_id"] / path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(doc, dict) and isinstance(doc.get("findings"), list):
                latest = [f for f in doc["findings"] if isinstance(f, dict)]
    return latest if seen and latest is not None else None


def check_scan_decisions(history, run_root) -> dict | None:
    """Every blocker and major finding in the saved scan has a ledger decision that names its id (audit item 17: the scan's
    blockers were not enforced; a forecast could be built on a table the scan said was broken). A finding is recorded when a
    ledger entry's stage, finding or decision text contains its id, such as ``time.duplicate_timestamps``. None when no scan
    was saved or the sandbox folder cannot be read; the scan check reports the missing scan."""
    findings = _saved_scan_findings(history, run_root)
    if findings is None:
        return None
    required = sorted({f["id"] for f in findings if f.get("severity") in ("blocker", "major") and isinstance(f.get("id"), str)})
    if not required:
        return _check("scan_decisions_recorded", True, "the scan raised no blocker or major finding to decide on")
    texts = [" ".join(str(e.get(k, "")) for k in ("stage", "finding", "decision")) for entries in _saved_ledgers(history, run_root)
             for e in entries if isinstance(e, dict)]
    missing = [i for i in required if not any(i in t for t in texts)]
    if not missing:
        return _check("scan_decisions_recorded", True,
                      f"the ledger records a decision for each of the {len(required)} blocker and major findings in the scan")
    return _check("scan_decisions_recorded", False,
                  f"the scan has blocker or major findings with no ledger decision: {', '.join(missing[:8])}"
                  + (f" and {len(missing) - 8} more" if len(missing) > 8 else "")
                  + "; record each one with ps.Ledger().add(stage, finding, decision) and put the finding id in the text"
                  + _decision_recipe(findings, missing, history, run_root))


STAGE_OF = {"time": "time index", "coverage": "coverage", "values": "values", "zeros": "zeros and censoring",
            "censoring": "zeros and censoring", "changepoints": "outliers and level shifts",
            "outliers": "outliers and level shifts", "season": "seasonality"}


def _scope_text(names: list[str], show: int = 3) -> str:
    """"all", one name, or "14 series: MT_001, MT_004, MT_009 and 11 more"."""
    if len(names) == 1:
        return names[0]
    head = ", ".join(names[:show])
    return f"{len(names)} series: {head}" + (f" and {len(names) - show} more" if len(names) > show else "")


def _decision_recipe(findings: list[dict], missing: list[str], history, run_root) -> str:
    """The cell that closes the check, in this mission's own names: the ledger to carry, and one add per undecided major
    finding with its scope. The decision text is left for the planner to write.

    Live Beijing rep5 (2026-11-23): the check named changepoints.level_shift; the planner read the scan six times, read the
    ledger with code that expected {'entries': ...} (the file is a list), saw nothing, and finished three more times."""
    # One add per finding id, its scopes folded into one list (live electricity rep9, 60 clients: the first six rows were all
    # censoring.plateau for six clients, so the recipe covered one of the eight ids, and 480 adds were never going to be written).
    scopes: dict[str, list[str]] = {}
    for f in findings:
        if f.get("id") in missing:
            got = scopes.setdefault(f["id"], [])
            if str(f.get("scope")) not in got:
                got.append(str(f.get("scope")))
    rows = [(i, _scope_text(scopes[i])) for i in missing if i in scopes]
    newest = next((h["result"]["run_id"] for h in reversed(with_delegated(history))
                   if isinstance(h, dict) and h.get("tool") == "sandbox_python" and isinstance(h.get("result"), dict)
                   and any(LEDGER_FILE.fullmatch(_output_path(o)) for o in (h["result"].get("outputs") or []))), None)
    ledger_path = next((_output_path(o) for h in reversed(with_delegated(history))
                        if isinstance(h, dict) and h.get("tool") == "sandbox_python" and isinstance(h.get("result"), dict)
                        and h["result"].get("run_id") == newest
                        for o in (h["result"].get("outputs") or []) if LEDGER_FILE.fullmatch(_output_path(o))), "out/ledger.json")
    carry = (f"carry=[{{run_id: '{newest}', file: '{ledger_path}'}}], then led = ps.Ledger.open('in/carried/{ledger_path.split('/')[-1]}')"
             if newest else "led = ps.Ledger()")
    # led.decide reads the stage and the finding text from the saved scan, so the cell has no scan parsing to get wrong
    # (live retail_biz rep5: guards on scan keys that do not exist skipped both adds, and the save changed nothing).
    adds = "; ".join(f"led.decide('{i}', '<your decision and why>')  # {scope}" for i, scope in rows)
    scan_at = next(((h["result"]["run_id"], p) for h in reversed(with_delegated(history))
                    if isinstance(h, dict) and h.get("tool") == "sandbox_python" and isinstance(h.get("result"), dict)
                    for p in [_output_path(o) for o in (h["result"].get("outputs") or [])] if SCAN_FILE.fullmatch(p)), None)
    scan_carry = (f" Carry the scan too ({{run_id: '{scan_at[0]}', file: '{scan_at[1]}'}}) so decide can read it."
                  if scan_at else "")
    return (f". One cell does it: {carry}; {adds}; led.save(). Write no code that reads the scan file and no if around the "
            f"calls: decide finds each finding itself, and one call covers every series it names.{scan_carry}")


def scan_ran(history) -> bool:
    """True when an analysis step saved the dataset scan (``prophet_scan.write_scan`` writes ``out/scan.json``). A step that
    saved it and failed later in the cell still saved it: the listed output is the evidence."""
    for h in with_delegated(history):
        res = h.get("result") if isinstance(h, dict) and h.get("kind") == "tool" and h.get("tool") == "sandbox_python" else None
        # out/scan.json, or one scan per entity (out/scan_<name>.json): live, a 12-station panel saved per-station scans and
        # re-ran its profiling 9 times because this item never ticked.
        if isinstance(res, dict) and any(SCAN_FILE.fullmatch(_output_path(o)) for o in (res.get("outputs") or [])):
            return True
    return False


SCAN_FILE = re.compile(r"out/scan[^/]*\.json")


def _output_path(o: Any) -> str:
    return str(o.get("path", "")) if isinstance(o, dict) else str(o).split(" (")[0]


def check_scan(history, run_root) -> dict | None:
    """The forecast's data was scanned: duplicates, grid, gaps, scale, censoring, breaks and the rest ran as one battery and
    its findings were saved. None when ``run_root`` is unknown, as for the ledger."""
    if run_root is None:
        return None
    ok = scan_ran(history)
    return _check("data_scanned", ok,
                  "the dataset scan was run and saved" if ok else
                  "the dataset scan was never saved; run `import prophet_scan as sc; r = sc.scan(df, time_col, value_col); "
                  "sc.write_scan(r)` in a sandbox step, read every blocker and major finding, and record a ledger decision for each")


CONTROL_OPTIONS = ("context_points", "ignore_ranges", "znorm", "symmetric_averaging")   # drivers (covariates) are inputs, not tuning


SELECTION_LIMIT = 3       # configurations a planner may try on one series before the choice is reported as an optimistic selection


def used_controls(args: dict, n_points: int | None = None) -> list[str]:
    """The TimesFM tuning options a forecast call switched on (a false or empty value is the default). A context window that
    covers the whole history is a no-op, not a treatment: the default already uses every point (A6-08; live retail rep4 used
    context_points=504 on 504 points, measured configured_vs_default_pct 0.0, and counted it as tested). The history length is
    ``n_points`` when known, else the length of the call's ``values``."""
    if n_points is None and isinstance(args.get("values"), list):
        n_points = len(args["values"])
    out: list[str] = []
    for k in CONTROL_OPTIONS:
        v = args.get(k)
        if not v:
            continue
        if k == "context_points" and _is_int(v) and _is_int(n_points) and v >= n_points:
            continue
        out.append(k)
    return out


def check_selection(history) -> dict | None:
    """Warns when more than SELECTION_LIMIT distinct treatment configurations were backtested on the mission's series. Each
    configured backtest is scored on the windows it was chosen on, with no multiplicity control, so the best of many is
    optimistic (A6-08). A warning. None when no configured backtest ran."""
    obs = [h for h in history if isinstance(h, dict)] if isinstance(history, list) else []
    configs = set()
    for h in obs:
        if not isinstance(h.get("args"), dict) or _ok_result(h, "backtest_series") is None:
            continue
        used = used_controls(h["args"])
        if used:
            configs.add(tuple((k, str(h["args"].get(k))) for k in used))
    if not configs:
        return None
    if len(configs) > SELECTION_LIMIT:
        return _check("controls_selection", False,
                      f"{len(configs)} treatment configurations were backtested on the same series; the best of them is optimistic "
                      f"(more than {SELECTION_LIMIT} tried). Report the number tried, or confirm the choice with control_lab",
                      "warn")
    return _check("controls_selection", True,
                  f"{len(configs)} treatment configuration(s) backtested, within the {SELECTION_LIMIT} allowed", "warn")


# The lab candidates that measure each hand-set treatment. The lab's own candidates are close to the planner's options, not
# identical: ctx_* are context windows (not an exact point count), ignore/replace_outliers are automatic (not the planner's
# ranges). The check accepts the closest measured candidate and says so in its detail.
TREATMENT_CANDIDATES = {
    "context_points": ("ctx_2h", "ctx_4h", "ctx_8h", "ctx_season"),
    "ignore_ranges": ("ignore_outliers", "replace_outliers"),
    "znorm": ("znorm",),
    "symmetric_averaging": ("symmetric_averaging",),
    "past_covariates": ("with_covariates",),
    "future_covariates": ("with_covariates",),
}


def check_controls(history, args: dict, record: dict | None = None) -> dict | None:
    """A forecast that tunes TimesFM by hand (shorter context, ignored points, znorm, symmetric averaging) needs a
    ``control_lab`` result on the history: measured evidence that the treatment helps, because measured evidence so far says
    most do not. Blocking when no lab measured it. Audit item 18: a lab counts only when it measured THIS forecast, so it must
    have run on the same horizon and series length (``record`` is the forecast's own record), and its table must hold a
    candidate for each treatment the forecast uses. A warning when the lab recommended the default or another treatment."""
    rec_cfg = record.get("config") if isinstance(record, dict) and isinstance(record.get("config"), dict) else {}
    used = used_controls(args, rec_cfg.get("history_points"))
    if not used:
        return None
    labs = [r for h in history if isinstance(r := _ok_result(h, "control_lab"), dict)]
    # backtest_series with the same options also scores default TimesFM on the same windows, which is the same measurement
    # (live WTI: control_lab cannot take explicit ignore_ranges, so the planner could not satisfy this check and stalled).
    configured = [r for h in history if isinstance(r := _ok_result(h, "backtest_series"), dict)
                  and isinstance(r.get("configured_vs_default_pct"), (int, float))
                  and set(used) <= set(used_controls(h.get("args") or {}))]
    if configured and not labs:
        gain = configured[-1]["configured_vs_default_pct"]
        if gain > 0:
            return _check("controls_tested", True, f"backtest_series measured the configuration {gain:+.1f}% against default TimesFM")
        return _check("controls_tested", False, f"backtest_series measured the configuration {gain:+.1f}% against default "
                      f"TimesFM, yet the forecast uses {', '.join(used)}", "warn")
    if not labs:
        return _check("controls_tested", False,
                      f"the forecast uses {', '.join(used)} but control_lab never ran; test the treatment on a leak-free "
                      "walk-forward comparison, or forecast with the default")
    cfg = record.get("config") if isinstance(record, dict) and isinstance(record.get("config"), dict) else {}
    want_h = cfg.get("horizon", args.get("horizon"))
    want_n = cfg.get("history_points")
    matching = [r for r in labs if (not _is_int(want_h) or r.get("horizon") == want_h)
                and (not _is_int(want_n) or r.get("points") == want_n)]
    if not matching:
        return _check("controls_tested", False,
                      f"control_lab ran on {labs[-1].get('points')} points at horizon {labs[-1].get('horizon')}, but the forecast "
                      f"is {want_n} points at horizon {want_h}: the lab did not test this forecast. Run control_lab on the same "
                      "series and horizon")
    lab = matching[-1]
    # The planner sees the lab through its compact observation, which lists every measured candidate in ``measured``.
    measured = {str(n) for n in (lab.get("measured") or [])} | {str(row.get("name")) for row in (lab.get("table") or [])
                                                             if isinstance(row, dict)}
    untested = [t for t in used if not set(TREATMENT_CANDIDATES.get(t, ())) & measured]
    if untested:
        return _check("controls_tested", False,
                      f"control_lab did not measure {', '.join(untested)} on this series (its table has {sorted(measured)[:12]}); "
                      "name the matching candidate in control_lab's candidates and run it again")
    rec = lab.get("recommended")
    if rec == "keep the default":
        return _check("controls_tested", False,
                      f"control_lab recommended keeping the default, yet the forecast uses {', '.join(used)}", "warn")
    chosen = rec.get("candidate") if isinstance(rec, dict) else None
    tested_treatments = {t for t in used if set(TREATMENT_CANDIDATES.get(t, ())) & {chosen}}
    if chosen is not None and not tested_treatments:
        return _check("controls_tested", False,
                      f"control_lab recommended {chosen}, not the treatment the forecast uses ({', '.join(used)})", "warn")
    return _check("controls_tested", True, "control_lab measured the forecast's own series and horizon and the treatment was tested")


def delivered_verdict(obs: list) -> str | None:
    """The health verdict of the pack the person receives.

    A later timesfm_forecast is a diagnostic. It does not replace the newest forecast_report.
    With no pack, the last forecast is what the answer delivers.
    """
    pack = last = None
    for h in obs if isinstance(obs, list) else []:
        if not isinstance(h, dict):
            continue
        rec = _forecast_record(h)
        verdict = delivered_health(rec) if isinstance(rec, dict) else None
        if not verdict:
            continue
        last = str(verdict)
        if h.get("tool") == "forecast_report":
            pack = last
    return pack or last


def delivered_health(rec: dict) -> str | None:
    """The health verdict of the band the pack delivers. The band is TimesFM's whatever centre line is drawn (forecast_report
    shifts it, it does not redraw it), so a simple rule as the centre does not clear a flat band. Live retail rep4 (measured):
    centre naive, health flat_unexpected, and the answer quoted "80% range 810k to 1.57m" built on that band. An earlier
    version returned "centre_rule:<rule>" here and let such a band through (A6-03)."""
    health = rec.get("health") if isinstance(rec, dict) else None
    return health.get("verdict") if isinstance(health, dict) else None


def check_ledger(history, run_root) -> dict | None:
    """The analyst's ledger: a sandbox step saved ``out/ledger.json`` (``ps.Ledger.save``) with a finding and a decision for at
    least ``MIN_LEDGER_STAGES`` distinct stages. A forecast whose data was never broken down and decided on leaves no such
    record. None when ``run_root`` is unknown, so the check is skipped where the sandbox folder cannot be read."""
    best = ledger_stages(history, run_root)
    if best is None:
        return None
    ok = best >= MIN_LEDGER_STAGES
    return _check("analysis_ledger", ok,
                  f"the analysis ledger records {best} stages with a finding and a decision" if ok else
                  f"the analysis ledger records {best} stages; a forecast needs at least {MIN_LEDGER_STAGES} (time index, coverage, "
                  "zeros and censoring, outliers and level shifts, seasonality) with a finding and a decision each, saved with "
                  "ps.Ledger().save() in a sandbox step")


RESEARCH_TOOLS = ("web_research", "web_search", "web_fetch")


def _briefing(res: Any) -> bool:
    """True when a tool result holds something a reader can use: a non-empty body, and not marked ok False."""
    if isinstance(res, dict):
        if res.get("ok") is False:
            return False
        return any(v not in (None, "", [], {}) for k, v in res.items() if k not in ("ok", "error", "status"))
    if isinstance(res, (list, tuple)):
        return len(res) > 0
    return isinstance(res, str) and bool(res.strip())


def researched(history) -> bool:
    """True once a usable outside-context briefing exists: a web call returned content, or a researcher sub-agent gave a
    non-empty answer or its own web calls returned content. A failed or empty attempt does not tick the item, so the planner
    must research again or say plainly why research was not possible (A3-02; live WTI rep4: the researcher returned answer ""
    with "research briefing is incomplete", and the item ticked anyway)."""
    for h in history if isinstance(history, list) else []:
        if not isinstance(h, dict):
            continue
        if h.get("kind") == "tool" and h.get("tool") in RESEARCH_TOOLS:
            if not h.get("error") and _briefing(h.get("result")):
                return True
        if h.get("kind") == "subagent" and h.get("agent_type") == "researcher":
            if isinstance(h.get("answer"), str) and h["answer"].strip():
                return True
            for e in h.get("evidence") if isinstance(h.get("evidence"), list) else []:
                if (isinstance(e, dict) and e.get("kind") == "tool" and e.get("tool") in RESEARCH_TOOLS
                        and not e.get("error") and _briefing(e.get("result"))):
                    return True
    return False


def data_entity_count(history) -> tuple[int, str] | None:
    """How many entities the uploaded data holds, read from inspect_data: the unique count of its one grouping column, with
    the column's name. None when no inspection found exactly one grouping column with two or more values.

    Live Beijing rep4 (2026-11-23): the lead built every backtest and the pack from one analyst's file of 3 stations. The
    panel check read the expected count from those same results, so 3 of 3 looked complete; the data held 12."""
    obs = [h for h in history if isinstance(h, dict)] if isinstance(history, list) else []
    best: tuple[int, str] | None = None
    for h in obs:
        if _ok_result(h, "inspect_data") is None:
            continue
        # The full result when it was kept (round 4), else the planner's compact view, whose columns are lines such as
        # "entity: text, unique=60; e.g. [...]" and whose layout hints sit at the top (live rep6: only that view was read).
        r = h["full"] if isinstance(h.get("full"), dict) else h["result"]
        prof = r.get("profile") if isinstance(r.get("profile"), dict) else r
        groups = [g for hint in prof.get("layout_hints") or [] if isinstance(hint, str)
                  for g in re.findall(r"grouping column\(s\) \[([^\]]*)\]", hint)]
        names = [n.strip().strip("'\"") for g in groups for n in g.split(",") if n.strip()]
        if len(names) != 1:
            continue
        k = None
        for col in prof.get("columns") or []:
            if isinstance(col, dict) and col.get("name") == names[0]:
                k = col.get("unique")
            elif isinstance(col, str) and col.split(":", 1)[0].strip() == names[0]:
                m = re.search(r"\bunique=(\d+)", col)
                k = int(m.group(1)) if m else None
        if _is_int(k) and k >= 2 and (best is None or k > best[0]):
            best = (k, names[0])
    return best


def panel_build_cell(history) -> str | None:
    """The one cell that saves every entity of the uploaded long table as a panel file, in this data's own names: the file,
    the grouping, time and value columns read from inspect_data. None unless exactly one time column and one grouping column
    are found; with several numeric columns the value column is left for the planner to choose.

    Live electricity rep12 (measured, 2026-11-23): the only panel backtest covered one analyst's 15 clients; the coverage note
    said to build all 60 "with pl.panel(df, entity='entity', ...)", and the lead re-ran one cell that listed the entities three
    times, then stalled with no forecast."""
    counted = data_entity_count(history)
    if counted is None:
        return None
    for h in reversed([h for h in history if isinstance(h, dict)]):
        if _ok_result(h, "inspect_data") is None:
            continue
        r = h["full"] if isinstance(h.get("full"), dict) else h["result"]
        prof = r.get("profile") if isinstance(r.get("profile"), dict) else r
        cols = []
        for col in prof.get("columns") or []:
            if isinstance(col, dict):
                cols.append((str(col.get("name")), str(col.get("kind"))))
            elif isinstance(col, str) and ":" in col:
                name, rest = col.split(":", 1)
                cols.append((name.strip(), rest.strip().split(",")[0].strip()))
        times = [n for n, k in cols if k == "time"]
        values = [n for n, k in cols if k == "number" and n != counted[1]]
        file = (h.get("args") or {}).get("file") or r.get("file")
        if len(times) != 1 or not values or not isinstance(file, str):
            return None
        value = values[0] if len(values) == 1 else "<value column>"
        return (f"df = pd.read_csv('in/{file}'); series, report = pl.panel(df, entity='{counted[1]}', time='{times[0]}', "
                f"value='{value}'); print(len(series), 'series'); pl.save_panel(series)  # writes out/series.json; then "
                f"panel_backtest and forecast_report with source {{run_id of this cell, file: 'out/series.json'}}")
    return None


def result_entities(result) -> int | None:
    """How many series a forecast or backtest result covers: n_series, entities (panel_backtest) or its series table."""
    if not isinstance(result, dict):
        return None
    for key in ("n_series", "entities"):
        if _is_int(result.get(key)):
            return result[key]
    return None


def panel_entities(history) -> dict | None:
    """For a panel mission (a result naming more than one series): how many entities the mission has, how many the newest
    forecast delivered, and which are missing when the names are known. None when no panel result was seen (A3-12; live
    Beijing: 12 stations in the panel, the forecast held 10, and the answer said "all 12")."""
    obs = [h for h in history if isinstance(h, dict)] if isinstance(history, list) else []
    expected = None
    names = None
    delivered = None
    delivered_names = None
    not_shown = 0
    for h in obs:
        for tool in ("backtest_series", "forecast_report", "timesfm_forecast"):
            r = _ok_result(h, tool)
            if not isinstance(r, dict):
                continue
            if _is_int(r.get("n_series")) and r["n_series"] > 1:
                expected = r["n_series"]
                if tool == "backtest_series" and isinstance(r.get("series"), dict):
                    names = list(r["series"])
            if tool != "backtest_series" and isinstance(r.get("series"), dict):
                delivered_names = list(r["series"])
                not_shown = r.get("not_shown") if _is_int(r.get("not_shown")) else 0
                delivered = len(delivered_names) + not_shown
            elif tool != "backtest_series" and _is_int(r.get("series_count")):
                delivered, delivered_names, not_shown = r["series_count"], None, 0
    data = data_entity_count(obs)
    if data is not None and delivered is not None and data[0] > (expected or 0):
        expected, names = data[0], None      # the data holds more entities than any result covered
    if expected is None or expected < 2:
        return None
    missing = None
    if names is not None and delivered_names is not None and not not_shown:
        missing = [n for n in names if n not in delivered_names]
    return {"expected": expected, "delivered": delivered or 0, "missing": missing}




def _ledger_recipe(history, run_root) -> str:
    """The cell that fills the ledger from the saved scan: one led.decide per finding id, blockers and majors first.

    Live retail_biz rep6 (measured, 2026-11-23): with the scan saved and this item open, the lead printed the scan's findings
    in eleven cells and its rescue analyst in fourteen more, and no decision was recorded; the mission stalled with no
    forecast. The item said what was missing, not how to close it."""
    findings = _saved_scan_findings(history, run_root) or []
    rank = {"blocker": 0, "major": 1, "minor": 2, "info": 3}
    ids: list[str] = []
    for f in sorted((f for f in findings if isinstance(f.get("id"), str)), key=lambda f: rank.get(f.get("severity"), 4)):
        if f["id"] not in ids:
            ids.append(f["id"])
    if not ids:
        return ""
    texts = [" ".join(str(e.get(k, "")) for k in ("stage", "finding", "decision")) for entries in _saved_ledgers(history, run_root)
             for e in entries if isinstance(e, dict)]
    open_ids = [i for i in ids if not any(i in t for t in texts)][:8]
    if not open_ids:
        return ""
    return ("; " + _decision_recipe(findings, open_ids, history, run_root).lstrip(". ")
            + ". Printing the scan again does not count: only saved decisions do")


SERIES_FILE = re.compile(r"out/series(?:_[^/]+)?\.json\Z")
SERIES_ITEM = ("save a numeric series (out/series.json from save_series or save_panel, "
               "or bind_series accepted)")


def _observations(history) -> list:
    return [h for h in history if isinstance(h, dict)] if isinstance(history, list) else []


def stored_web_fetch_urls(history) -> list[str]:
    """Urls of pages web_fetch stored. A failed call, including one given search queries, is not a stored page."""
    urls: list[str] = []
    for h in _observations(history):
        if h.get("kind") != "tool" or h.get("tool") != "web_fetch" or h.get("error"):
            continue
        res = h.get("result")
        if not isinstance(res, dict) or res.get("ok") is False:
            continue
        args = h.get("args") if isinstance(h.get("args"), dict) else {}
        url = args.get("url")
        if isinstance(url, str) and url and url not in urls:
            urls.append(url)
    return urls


def html_download_urls(history) -> list[str]:
    """Download urls whose error is an HTML page. The class comes from diagnose_failure."""
    urls: list[str] = []
    for h in _observations(history):
        if h.get("kind") != "tool" or h.get("tool") != "download_web_data" or not h.get("error"):
            continue
        if diagnose_failure(str(h.get("error"))).get("signal") != "html_page":
            continue
        args = h.get("args") if isinstance(h.get("args"), dict) else {}
        url = args.get("url")
        if isinstance(url, str) and url and url not in urls:
            urls.append(url)
    return urls


def extract_target_urls(history) -> list[str]:
    """The stored page to extract. Prefer the url the HTML download already tried, and keep at most three."""
    stored = stored_web_fetch_urls(history)
    matched = [url for url in html_download_urls(history) if url in stored]
    return (matched or stored)[:3]


def _column_kinds(res: Any) -> set[str]:
    if not isinstance(res, dict):
        return set()
    prof = res.get("profile") if isinstance(res.get("profile"), dict) else {}
    cols = prof.get("columns") if isinstance(prof.get("columns"), list) else res.get("columns")
    if not isinstance(cols, list):
        return set()
    kinds: set[str] = set()
    for col in cols:
        if isinstance(col, dict):
            kind = col.get("kind")
        elif isinstance(col, str) and ":" in col:
            kind = col.split(":", 1)[1].strip().split(",", 1)[0].strip()
        else:
            kind = ""
        if isinstance(kind, str) and kind:
            kinds.add(kind)
    return kinds


def _profile_has_time_and_number(res: Any) -> bool:
    kinds = _column_kinds(res)
    return "time" in kinds and "number" in kinds


def _table_has_numbers_and_no_time(history) -> bool:
    """An inspected table has a number column and no time column. One shared date is not stored as a time column."""
    saw_number = saw_time = False
    for h in _observations(history):
        kinds = _column_kinds(_ok_result(h, "inspect_data"))
        saw_number = saw_number or "number" in kinds
        saw_time = saw_time or "time" in kinds
    return saw_number and not saw_time


def _saved_series_file(h: Any) -> bool:
    res = _ok_result(h, "sandbox_python")
    if not isinstance(res, dict):
        return False
    return any(SERIES_FILE.fullmatch(_output_path(o)) for o in (res.get("outputs") or []))


def _series_documents(history, run_root) -> list:
    """Parsed series files. None means the name was saved and the bytes could not be read."""
    docs: list = []
    for h in _observations(history):
        res = _ok_result(h, "sandbox_python")
        if not isinstance(res, dict) or not isinstance(res.get("run_id"), str):
            continue
        run_id = res["run_id"]
        if "/" in run_id or ".." in run_id:
            continue
        for output in res.get("outputs") or []:
            path = _output_path(output)
            if not SERIES_FILE.fullmatch(path):
                continue
            if run_root is None:
                docs.append(None)
                continue
            try:
                doc = json.loads((Path(run_root) / run_id / path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                docs.append(None)
                continue
            docs.append(doc if isinstance(doc, dict) else None)
    return docs


def _has_two_distinct_dates(doc: dict) -> bool:
    series, dates = doc.get("series"), doc.get("dates")
    pairs: list[tuple] = []
    if isinstance(series, list):
        pairs.append((series, dates if isinstance(dates, list) else []))
    elif isinstance(series, dict) and isinstance(dates, dict):
        pairs.extend((vals, dates.get(name) or []) for name, vals in series.items() if isinstance(vals, list))
    for vals, stamps in pairs:
        if isinstance(stamps, list) and len(stamps) == len(vals) and len({str(item) for item in stamps}) > 1:
            return True
    return False


def _undated_list_is_not_a_path(history, run_root) -> bool:
    """A number table with no time column, saved as a list with no dates, is not a path through time."""
    if not _table_has_numbers_and_no_time(history):
        return False
    docs = [doc for doc in _series_documents(history, run_root) if isinstance(doc, dict)]
    return bool(docs) and not any(_has_two_distinct_dates(doc) for doc in docs)


def numeric_series_ready(history, run_root=None) -> bool:
    """True when the evidence holds a numeric series the model tools can read.

    A scan or a ledger does not count. A profile with a time column and a number column can pass on one row.
    A table with numbers and no time column does not make an undated list a path."""
    skip_undated = _undated_list_is_not_a_path(history, run_root)
    for h in _observations(history):
        if _saved_series_file(h) and not skip_undated:
            return True
        bound = _ok_result(h, "bind_series")
        if isinstance(bound, dict) and bound.get("status") == "accepted" and not _one_time_many_rows(bound):
            return True
        if _profile_has_time_and_number(_ok_result(h, "inspect_data")):
            return True
        if any(isinstance(_ok_result(h, tool), dict)
               for tool in ("timesfm_forecast", "backtest_series", "panel_backtest", "forecast_report")):
            return True
    return False


def _extract_found_a_table(history) -> bool:
    for h in _observations(history):
        res = _ok_result(h, "extract_page_tables")
        if isinstance(res, dict) and isinstance(res.get("candidates"), list) and res["candidates"]:
            return True
    return False


def _one_time_many_rows(result: dict) -> bool:
    """Many rows that share one time are a cross-section. One row can still be a short series."""
    times = result.get("distinct_times")
    rows = result.get("row_count")
    return times == 1 and isinstance(rows, int) and not isinstance(rows, bool) and rows > 1


def _bound_cross_section(history) -> bool:
    return any(_one_time_many_rows(res) for h in _observations(history)
               if isinstance(res := _ok_result(h, "bind_series"), dict) and res.get("status") == "accepted")


def series_item_detail(history, run_root=None) -> str:
    """What to do before a model comparison. The url is one web_fetch already stored."""
    if _undated_list_is_not_a_path(history, run_root) or _bound_cross_section(history):
        return ("the table has one time and many rows, so a list of its numbers is not a path. "
                "Save one entity across dates, or say the path is not in the table.")
    urls = extract_target_urls(history)
    if html_download_urls(history) and urls and not _extract_found_a_table(history):
        named = ", ".join(f"'{url}'" for url in urls)
        return f"call extract_page_tables with url {named}. The page is already stored."
    if _extract_found_a_table(history):
        return "map the candidate and call bind_series, or write out/series.json with save_series."
    return ("write out/series.json with save_series or save_panel, or accept a table with bind_series. "
            "A scan or a ledger is not a series.")


_HOME = re.compile(r"^(?:home(?:[ _]?team)?|hometeam|team ?1|team1)$", re.I)
_AWAY = re.compile(r"^(?:away(?:[ _]?team)?|awayteam|visitor(?:[ _]?team)?|team ?2|team2)$", re.I)


def match_results_landed(obs: list) -> bool:
    """True when a downloaded or inspected table has a home-team column and an away-team column: a match results file."""
    for h in obs:
        for tool in ("download_web_data", "download_open_data", "inspect_data"):
            res = _ok_result(h, tool)
            if not isinstance(res, dict):
                continue
            cols = res.get("columns")
            if not isinstance(cols, list):
                prof = res.get("profile") if isinstance(res.get("profile"), dict) else {}
                cols = prof.get("columns") if isinstance(prof.get("columns"), list) else []
            names = [str(c.get("name") if isinstance(c, dict) else str(c).split(":", 1)[0]).strip() for c in cols]
            if any(_HOME.match(n) for n in names) and any(_AWAY.match(n) for n in names):
                return True
    return False


def progress(history, *, run_root=None, wants_files: bool = False, external: bool = False) -> list[dict]:
    """The mission checklist: what a finished forecast needs, each item done or still to do, from the evidence alone.

    Motivation: a model that sees only what it has done tends to finish when it has done something. Showing the whole list on
    every step, with the next undone item named, is the same device as a feature list a long-running agent works down. The
    list is derived, never stored: it cannot drift from the evidence, and a resumed run reads the same list.
    Logic: items are ordered as an analyst would do them. ``status`` is "done" or "todo"; ``next`` marks the first todo."""
    obs = [h for h in history if isinstance(h, dict)] if isinstance(history, list) else []
    forecasts = [(i, r) for i, h in enumerate(obs) if (r := _forecast_record(h)) is not None]
    inspected = any(_ok_result(h, t) is not None for h in obs for t in INSPECTION_TOOLS)
    # A panel backtest compares each entity with its own simple rules too (live electricity rep5: with only panel_backtest
    # run, this item stayed todo and the same panel_backtest was sent twice more until the run stalled).
    backtested = any(isinstance(_ok_result(h, t), dict) for h in obs for t in ("backtest_series", "panel_backtest"))
    stages = ledger_stages(obs, run_root)
    items: list[dict] = [{"item": "inspect the data (sandbox_python or inspect_data)", "status": "done" if inspected else "todo"}]
    simulated = any(isinstance(_ok_result(h, "simulate_season"), dict) for h in obs)
    if simulated or match_results_landed(obs):
        # A league is forecast by simulating its remaining fixtures from match results, not as a time series. Panel round 3
        # (2026-11-26): EPL simulated the season and the list still asked for out/series.json and TimesFM, so 16 calls
        # re-saved panels; NBA downloaded all 1,207 games and spent 31 cells on the scan and ledger without simulating.
        items.append({"item": "simulate the rest of the season from the match results (simulate_season, results naming the "
                              "downloaded file; its backtest compares the model with equal odds, the home rate and Elo)",
                      "status": "done" if simulated else "todo"})
        first = next((it for it in items if it["status"] == "todo"), None)
        if first is not None:
            first["next"] = True
        return items
    if run_root is not None:
        items.append({"item": "scan the data and save it (prophet_scan: sc.scan(...) then sc.write_scan(r))",
                      "status": "done" if scan_ran(obs) else "todo"})
    if stages is not None:
        items.append({"item": f"analysis ledger: at least {MIN_LEDGER_STAGES} stages with a finding and a decision "
                              "(time index, coverage, zeros and censoring, outliers and level shifts, seasonality)",
                      "status": "done" if stages >= MIN_LEDGER_STAGES else "todo",
                      "detail": f"{stages} of {MIN_LEDGER_STAGES} recorded" + (_ledger_recipe(obs, run_root)
                                                                                if stages < MIN_LEDGER_STAGES else "")})
    if external:
        items.append({"item": "outside context: the series is driven by things its history cannot show, so delegate a researcher "
                              "(agent_type researcher) for the regime, scheduled events in the horizon and the main risks, or say "
                              "plainly why web research could not be used",
                      "status": "done" if researched(obs) else "todo"})
    ready = numeric_series_ready(obs, run_root)
    items.append({"item": SERIES_ITEM, "status": "done" if ready else "todo",
                  **({} if ready else {"detail": series_item_detail(obs, run_root)})})
    if not ready:
        # A scan and a ledger are not the series the model tools read. Asking for a backtest here sends a modeler
        # at a file that does not exist (measured mission 56465803efa547009cc7e054312ea7ab).
        first = next((it for it in items if it["status"] == "todo"), None)
        if first is not None:
            first["next"] = True
        return items
    items.append({"item": "compare the forecast method with the simple baselines (backtest_series)",
                  "status": "done" if backtested else "todo"})
    items.append({"item": "forecast the series (timesfm_forecast)", "status": "done" if forecasts else "todo"})
    entities = panel_entities(obs)
    if entities is not None:
        # A panel goal is not finished while an entity has no delivered forecast. Live Beijing (measured): 12 stations, 10 in
        # the forecast, and the answer said "for all 12".
        n, k = entities["expected"], entities["delivered"]
        detail = f"{min(k, n)} of {n} entities forecast"
        if entities["missing"]:
            detail += "; missing: " + ", ".join(entities["missing"][:12])
        items.append({"item": "every panel entity has a delivered forecast (forecast_report on the whole panel)",
                      "status": "done" if k >= n else "todo", "detail": detail})
    if forecasts:
        verdict = delivered_health(forecasts[-1][1])
        items.append({"item": "forecast health is not flat_unexpected", "status": "todo" if verdict == "flat_unexpected" else "done",
                      "detail": f"health: {verdict or 'not reported'}"})
    if wants_files:
        reports = [r for h in obs if isinstance(r := _ok_result(h, "forecast_report"), dict)]
        items.append({"item": "produce the requested charts, workbook or report (forecast_report)",
                      "status": "done" if reports else "todo"})
    facts = trust_facts(obs)
    if facts and facts.get("verdict") == "baseline_better" and (wants_files or forecasts):
        # The delivery gate: a simple rule that beat TimesFM in the backtest is the centre line the person should see.
        packs = [h for h in obs if _ok_result(h, "forecast_report") is not None]
        # Recentred on the rule the backtest ranked best (or the checked blend). Live WTI rep4: this item named "naive or
        # drift" whatever the backtest said, and the pack went out on naive while mean_last_season was best.
        best = str(facts["best_baseline"])
        recentred = any(str((h.get("args") or {}).get("centre", "timesfm")) in (best, "blend") for h in packs)
        items.append({"item": f"the backtest found {best} more accurate than TimesFM: build the pack with centre '{best}' "
                              "(TimesFM's band around it), or 'blend' for one series, or say plainly why not",
                      "status": "done" if recentred or not packs and not wants_files else "todo"})
    first = next((it for it in items if it["status"] == "todo"), None)
    if first is not None:
        first["next"] = True
    return items


def render_progress(items: list[dict]) -> str:
    """The checklist as plain lines for the planner, the next undone item marked."""
    if not items:
        return ""
    lines = []
    for it in items:
        mark = "[x]" if it["status"] == "done" else "[ ]"
        tail = f" ({it['detail']})" if it.get("detail") else ""
        lines.append(f"{mark} {it['item']}{tail}" + ("   <- next" if it.get("next") else ""))
    return "Mission checklist (derived from the evidence; finish only when every box is ticked, or say plainly why one cannot be):\n" + "\n".join(lines)


RULE_TOLERANCE = 0.02   # a rule centre within 2 percent of the best rule's MASE counts as the best (2026-11-22)


def _baseline_gate(delivered: dict, backtests: list) -> dict:
    """The baseline check for the forecast that stands (audit 2026-11-22, item 4). A backtest counts only if it ran at the
    delivered horizon or longer, so a 14-day check does not vouch for a 60-day forecast. When the centre is TimesFM's own
    line and a simple rule beat it in such a backtest, the gate blocks it; a centre switch (forecast_report centre=rule)
    is the repair."""
    fh = delivered.get("horizon") if isinstance(delivered, dict) else None
    covering = [b for b in backtests if _is_int(b.get("horizon")) and (not _is_int(fh) or b["horizon"] >= fh)]
    if not covering:
        ran = sorted({b.get("horizon") for b in backtests if _is_int(b.get("horizon"))})
        return _check("baseline_compared", False,
                      f"the forecast runs {fh} periods ahead but no backtest covers that horizon (backtests ran at {ran})")
    centre = delivered.get("centre") if isinstance(delivered, dict) else None
    on_timesfm = not (isinstance(centre, dict) and centre.get("rule") not in (None, "timesfm"))
    for b in covering:
        models = b.get("models") if isinstance(b.get("models"), dict) else {}
        best = b.get("best_baseline")
        tmodel = models.get("timesfm")
        bmodel = models.get(best) if isinstance(best, str) else None
        tm = _finite(tmodel.get("mase")) if isinstance(tmodel, dict) else None
        bm = _finite(bmodel.get("mase")) if isinstance(bmodel, dict) else None
        if on_timesfm and tm is not None and bm is not None and tm > bm and not _flagged(b):
            return _check("baseline_compared", False,
                          f"the forecast keeps TimesFM's centre line, but the backtest found {best} more accurate "
                          f"(MASE {bm:.3f} against {tm:.3f}); switch the centre to that rule")
        rule = centre.get("rule") if isinstance(centre, dict) else None
        rmodel = models.get(rule) if isinstance(rule, str) else None
        rm = _finite(rmodel.get("mase")) if isinstance(rmodel, dict) else None
        if (not on_timesfm and rule != "blend" and isinstance(best, str) and rule != best and rm is not None
                and bm is not None and rm > bm * (1 + RULE_TOLERANCE) and not _flagged(b)):
            # A rule centre must be the rule the backtest ranked best, or close to it (live WTI rep4: naive 7.27 delivered
            # while mean_last_season scored 6.66). The blend carries its own backtest, so it is judged there.
            return _check("baseline_compared", False,
                          f"the centre is the {rule} rule, but the backtest found {best} more accurate (MASE {bm:.3f} "
                          f"against {rm:.3f}); switch the centre to {best}")
    return _check("baseline_compared", True, "the forecast was compared with the baselines in a backtest at its horizon")


_SHIFT_ENTRY = re.compile(r"shift|break|changepoint", re.I)
_CAUSAL_DECISION = re.compile(r"correspond|\bregimes?\b|crisis|crash|\bcaused?\b|because|due to|driven by|known (?:event|oil|market)|"
                              r"lockdown|covid|launch|promotion", re.I)
_CITED_DECISION = re.compile(r"\bS\d{1,3}\b|control_lab|backtest|walk-forward|ignore_ranges|\btested\b|unexplained", re.I)
_PROMISED_TEST = re.compile(r"\bwill (?:test|run|check|be tested|be run)\b|\bto be (?:tested|run)\b", re.I)


def check_ledger_claims(history, run_root) -> list[dict]:
    """Two checks on the newest saved ledger (A3-04). A shift decision that attributes the shift to a cause must cite a research
    source id (S1, S2 ...) or a test, or say the shift is unexplained; otherwise it is a warning. A decision that promises a test
    ("will test ... in control_lab") must find that test run in the mission. Live evidence (measured, audit round 3, A3): WTI rep4
    named the regimes of five level shifts from memory and promised an ignore_ranges test no run performed. Empty when no ledger
    was saved or ``run_root`` is unknown."""
    ledgers = _saved_ledgers(history, run_root) if run_root is not None else []
    if not ledgers:
        return []
    entries = [e for e in ledgers[0] if isinstance(e, dict) and not e.get("superseded")]
    out: list[dict] = []
    shifts = [e for e in entries if _SHIFT_ENTRY.search(f"{e.get('stage', '')} {e.get('finding', '')}")]
    if shifts:
        bare = [e for e in shifts if _CAUSAL_DECISION.search(str(e.get("decision", "")))
                and not _CITED_DECISION.search(str(e.get("decision", "")))]
        if bare:
            out.append(_check("causal_shift_cited", False,
                              f"the ledger attributes a shift to a cause with no research source id or test (stage "
                              f"'{bare[0].get('stage')}'); cite the S-id or the control_lab result, or record the shift as "
                              "unexplained, kept as observed", "warn"))
        else:
            out.append(_check("causal_shift_cited", True, "every shift decision cites a source or a test, or says it is unexplained",
                              "warn"))
    promised = [e for e in entries if _PROMISED_TEST.search(str(e.get("decision", "")))]
    if promised:
        ran = any(_ok_result(h, "control_lab") is not None or _ok_result(h, "backtest_series") is not None
                  for h in with_delegated(history if isinstance(history, list) else []))
        out.append(_check("promised_tests_run", ran,
                          "the test the ledger promised ran in this mission" if ran else
                          f"the ledger promises a test ('{str(promised[0].get('decision'))[:120]}') and no control_lab or "
                          "backtest_series ran; run it, or drop the promise", "warn"))
    return out


def _likely_recording_break(ev: dict) -> bool:
    """A break the scan flags as possibly a recording change: on a calendar boundary, or shared by several entities on one date.
    The flags are read from the finding and from each break row when present (the scan slice adds them; absent means unflagged)."""
    rows = ev.get("breaks") if isinstance(ev.get("breaks"), list) else []
    for src in [ev, *[r for r in rows if isinstance(r, dict)]]:
        if src.get("on_calendar_boundary") is True:
            return True
        n = src.get("coincident_entities")
        if _is_int(n) and n > 1:
            return True
    return False


def check_shift_decisions(history, run_root) -> dict | None:
    """Where the scan flags a level shift as a likely recording break, the ledger must name the treatment options (truncate, ignore
    ranges, or flag) and a control_lab or backtest must have run before the shift is kept (A4-09). A warning. None when no flagged
    shift exists or the scan cannot be read."""
    findings = _saved_scan_findings(history, run_root)
    if findings is None:
        return None
    flagged = [f for f in findings if f.get("id") == "changepoints.level_shift" and isinstance(f.get("evidence"), dict)
               and _likely_recording_break(f["evidence"])]
    if not flagged:
        return None
    ledgers = _saved_ledgers(history, run_root)
    text = " ".join(str(e.get("decision", "")) for e in (ledgers[0] if ledgers else []) if isinstance(e, dict)
                    and not e.get("superseded"))
    names_options = bool(re.search(r"truncat|ignore_ranges|\bflag", text, re.I))
    tested = any(_ok_result(h, "control_lab") is not None or _ok_result(h, "backtest_series") is not None
                 for h in with_delegated(history if isinstance(history, list) else []))
    ok = names_options and tested
    if ok:
        return _check("shift_decision_tested", True,
                      "the flagged shift was tested (truncate, ignore or flag) before it was kept", "warn")
    gap = "no control_lab or backtest_series ran" if not tested else "the ledger does not name the options (truncate, ignore_ranges or flag)"
    return _check("shift_decision_tested", False,
                  f"{len(flagged)} level shift(s) are flagged as likely recording breaks (calendar boundary, or shared by several "
                  f"entities), and {gap}; test them before keeping the shift", "warn")


OUTLIER_SHARE_RULE = 0.02     # the outliers skill: above about 2% of steps, treat the flags as noise or a broken feed until shown otherwise
_OUTLIER_TEST = re.compile(r"winsori|robust|replace_outliers|ignore_ranges|control_lab|backtest", re.I)
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s?%")


def check_outlier_share(history, run_root) -> dict | None:
    """Where the saved scan flags more than OUTLIER_SHARE_RULE of the steps as outliers (Hampel share, the scan's own field),
    the ledger must state the share and test a winsorised or robust copy (A4-11). Measured, audit round 3: Beijing kept 6.3% to
    8.8% of rows as real spikes with one template sentence. A warning. None when the scan shows no such series or no folder is
    known."""
    findings = _saved_scan_findings(history, run_root)
    if findings is None:
        return None
    high = []
    for f in findings:
        ev = f.get("evidence") if isinstance(f.get("evidence"), dict) else {}
        shares = ev.get("share_by_method") if isinstance(ev.get("share_by_method"), dict) else {}
        share = _finite(shares.get("hampel"))
        if f.get("id") == "outliers.share" and share is not None and share > OUTLIER_SHARE_RULE:
            high.append((str(f.get("scope")), share))
    if not high:
        return None
    ledgers = _saved_ledgers(history, run_root)
    relevant = [e for e in (ledgers[0] if ledgers else []) if isinstance(e, dict) and not e.get("superseded")
                and re.search(r"outlier|spike", " ".join(str(e.get(k, "")) for k in ("stage", "finding", "decision")), re.I)]
    text = " ".join(f"{e.get('finding', '')} {e.get('decision', '')}" for e in relevant)
    names = ", ".join(f"{s} ({v:.1%})" for s, v in high[:6])
    if _PERCENT.search(text) and _OUTLIER_TEST.search(text):
        return _check("outlier_share_decided", True,
                      f"the ledger states the outlier share and tests a winsorised or robust copy for {names}", "warn")
    return _check("outlier_share_decided", False,
                  f"the scan flags more than {OUTLIER_SHARE_RULE:.0%} of the steps as outliers for {names}; the ledger must state the "
                  "share and test a winsorised or robust copy (replace_outliers, ignore_ranges or control_lab) before keeping them",
                  "warn")


_HYPOTHESIS_STAGE = re.compile(r"hypothes", re.I)
_HYPOTHESIS_OUTCOME = re.compile(r"\b(confirmed|refuted|supported|rejected)\b", re.I)


def check_hypotheses(history, run_root) -> dict | None:
    """The mission should state two or more falsifiable hypotheses and mark each confirmed or refuted (A4-10). An entry counts when it
    carries an ``expected`` statement and an ``outcome`` of confirmed or refuted. Without those fields, an entry staged as a
    hypothesis counts when its text states an outcome. A warning, not a block. None when no ledger was saved. Live evidence (measured,
    audit round 3, A4): none of six ledgers held an expectation, although planner step 4 asks for one."""
    ledgers = _saved_ledgers(history, run_root) if run_root is not None else []
    if not ledgers:
        return None
    entries = [e for e in ledgers[0] if isinstance(e, dict) and not e.get("superseded")]
    tested = 0
    for e in entries:
        expected = str(e.get("expected") or "").strip()
        outcome = str(e.get("outcome") or "").strip().lower()
        if expected and outcome in ("confirmed", "refuted"):
            tested += 1
        elif not e.get("expected") and not e.get("outcome") and _HYPOTHESIS_STAGE.search(str(e.get("stage", ""))) \
                and _HYPOTHESIS_OUTCOME.search(f"{e.get('finding', '')} {e.get('decision', '')}"):
            tested += 1
    if tested >= 2:
        return _check("hypotheses_tested", True,
                      f"{tested} falsifiable hypotheses were stated and marked confirmed or refuted", "warn")
    return _check("hypotheses_tested", False,
                  f"{tested} hypothesis test(s) recorded; the mission should state at least two falsifiable expectations and mark "
                  "each confirmed or refuted", "warn")


def check_frame(history, run_root) -> dict | None:
    """The mission should record a frame before it forecasts: what the forecast is for, its horizon, and the decision it feeds (A3-13).
    Read from the newest saved ledger, as an entry staged "frame" with a non-empty decision. A warning, not a block. None when no
    ledger was saved."""
    ledgers = _saved_ledgers(history, run_root) if run_root is not None else []
    if not ledgers:
        return None
    frames = [e for e in ledgers[0] if isinstance(e, dict) and not e.get("superseded")
              and str(e.get("stage", "")).strip().lower() == "frame" and str(e.get("decision", "")).strip()]
    if frames:
        return _check("frame_recorded", True, "the ledger records the frame: purpose, horizon and the decision it feeds", "warn")
    return _check("frame_recorded", False,
                  "the ledger has no frame entry; record what the forecast is for, its horizon and the decision it feeds before "
                  "forecasting", "warn")


def check_evidence(history, *, run_root=None) -> list[dict]:
    """Checks on a mission's observations, in order. Never raises: a missing or malformed piece skips its check."""
    obs = [h for h in history if isinstance(h, dict)] if isinstance(history, list) else []
    checks: list[dict] = []

    forecasts: list[tuple[int, dict, dict]] = []   # (position, forecast record, the call's arguments)
    for i, h in enumerate(obs):
        rec = _forecast_record(h)
        if rec is not None:
            args = h.get("args") if isinstance(h.get("args"), dict) else {}
            forecasts.append((i, rec, args))
    inspections = [i for i, h in enumerate(obs)
                   if any(_ok_result(h, t) is not None for t in INSPECTION_TOOLS)]
    backtests = [r for h in obs if isinstance(r := _ok_result(h, "backtest_series"), dict)]

    if forecasts:
        delivered = forecasts[-1][0]      # the forecast that stands must follow an inspection, so a re-run after one repairs it
        inspected = any(i < delivered for i in inspections)
        checks.append(_check("data_inspected", inspected,
                             "the data was inspected before the forecast that stands" if inspected
                             else "the forecast was run before any sandbox or data inspection step succeeded"))
        checks.append(_check("forecast_exists", True, "a forecast result exists", "warn"))
    else:
        checks.append(_check("data_inspected", True, "no forecast yet; nothing to check"))
        checks.append(_check("forecast_exists", False, "no forecast has been produced yet", "warn"))

    if not forecasts:
        checks.append(_check("baseline_compared", True, "no forecast yet; nothing to compare"))
    elif not backtests and forecasts[-1][1].get("panel") and any(isinstance(_ok_result(h, "panel_backtest"), dict) for h in obs):
        checks.append(_check("baseline_compared", True,
                             "the panel forecast was compared with each entity's simple rules in panel_backtest"))
    elif not backtests:
        checks.append(_check("baseline_compared", False,
                             "the forecast was never compared with the simple baselines (no successful backtest)"))
    else:
        checks.append(_baseline_gate(forecasts[-1][1], backtests))
    if (panel_centre := check_panel_centre(obs)) is not None:
        checks.append(panel_centre)

    if not forecasts:
        checks.append(_check("health_repaired", True, "no forecast yet; nothing to check"))
    else:
        # The gate follows the pack the person receives (forecast_report). A healthy timesfm_forecast run after a flat pack is a
        # diagnostic; it does not repair the band that ships (A6-07; the pack is rebuilt only by a later forecast_report).
        pack = next((rec for i, rec, _ in reversed(forecasts) if obs[i].get("tool") == "forecast_report"), None)
        if pack is None:
            pack = forecasts[-1][1]          # a run with no pack: its last forecast is what the answer delivers
        verdict = delivered_health(pack)
        if verdict == "flat_unexpected":
            checks.append(_check("health_repaired", False,
                                 "the forecast the answer delivers says the cycle was lost, and no later pack was built on a "
                                 "repaired forecast (rebuild the pack after the repair)"))
            period = _health_period(pack)
            if period is not None:
                tested = any(isinstance(_ok_result(h, "control_lab"), dict) and (h.get("args") or {}).get("period") == period
                             for h in obs)
                checks.append(_check("flagged_period_tested", tested,
                                     f"control_lab ran with period={period} on this flagged band" if tested else
                                     f"the flagged band has health period {period}, and no control_lab ran with period={period}; "
                                     "run control_lab with that period on the same series and horizon", "warn"))
        else:
            checks.append(_check("health_repaired", True, f"the pack's health is {verdict or 'not reported'}"))

        _, rec, args = forecasts[-1]
        point = rec.get("point")
        if _numbers(point) is not None:
            cfg = rec.get("config")
            horizon = cfg.get("horizon") if isinstance(cfg, dict) else None
            # A compact observation shows long forecasts thinned, with a note saying so: its length is not the horizon.
            thinned = isinstance(rec.get("note"), str) and "periods" in rec["note"]
            if thinned or not _is_int(horizon):
                horizon = None
            hist_vals = args.get("values")
            checks.extend(check_forecast(point, _quantile(rec, "0.1", "q10"), _quantile(rec, "0.9", "q90"),
                                         horizon=horizon, history=hist_vals if _numbers(hist_vals) is not None else None))

    if forecasts:
        controls = check_controls(obs, forecasts[-1][2], forecasts[-1][1])
        if controls is not None:
            checks.append(controls)
        scan = check_scan(obs, run_root)
        if scan is not None:
            checks.append(scan)
        ledger = check_ledger(obs, run_root)
        if ledger is not None:
            checks.append(ledger)
        kept = check_ledger_kept(obs, run_root)
        if kept is not None:
            checks.append(kept)
        decisions = check_scan_decisions(obs, run_root)
        if decisions is not None:
            checks.append(decisions)
        outliers = check_outlier_share(obs, run_root)
        if outliers is not None:
            checks.append(outliers)
        checks.extend(check_ledger_claims(obs, run_root))
        shifts = check_shift_decisions(obs, run_root)
        if shifts is not None:
            checks.append(shifts)
        hypotheses = check_hypotheses(obs, run_root)
        if hypotheses is not None:
            checks.append(hypotheses)
        frame = check_frame(obs, run_root)
        if frame is not None:
            checks.append(frame)

    selection = check_selection(obs)
    if selection is not None:
        checks.append(selection)

    if backtests and isinstance(backtests[-1].get("models"), dict):
        checks.extend(check_backtest(backtests[-1]))

    reports = [r for h in obs if isinstance(r := _ok_result(h, "forecast_report"), dict)]
    if reports and isinstance(reports[-1].get("files"), list):
        last = reports[-1]
        run_id = last.get("run_id")
        run_dir = None
        if run_root is not None and isinstance(run_id, str) and run_id.strip() and run_id not in (".", "..") \
                and "/" not in run_id and "\\" not in run_id:
            run_dir = Path(run_root) / run_id
        checks.extend(check_deliverables(last["files"], run_dir))
    return checks


# ---------------------------------------------------------------- reading the checks

def blocking(checks) -> list[dict]:
    """The checks that failed with severity block: these stop a finish from being accepted."""
    return [c for c in checks if isinstance(c, dict) and not c.get("ok") and c.get("severity") == "block"]


def summarise(checks, limit: int = 5) -> str:
    """Plain sentences for the failing checks: blocking ones first, then warnings. Each names its check in brackets."""
    failing = [c for c in checks if isinstance(c, dict) and not c.get("ok")]
    ordered = [c for c in failing if c.get("severity") == "block"] + [c for c in failing if c.get("severity") != "block"]
    if not ordered:
        return "No contract checks failed."
    shown = ordered[:max(int(limit), 0)]
    parts = []
    for c in shown:
        text = str(c.get("detail") or "").strip().rstrip(".") or "A check failed"
        parts.append(f"{text} [{c.get('name')}]")
    rest = len(ordered) - len(shown)
    if rest > 0:
        parts.append(f"and {rest} more")
    return "; ".join(parts)


BASELINE_NAMES = {"naive": "repeating the last value", "seasonal_naive": "repeating the value one season back",
                  "drift": "extending the recent average change", "mean_last_season": "the average of the last season",
                  "tsb_expected": "an intermittent-demand rule (TSB)"}
HEADLINE_MARGIN = 2.0     # percent: TimesFM is the headline only when it beats the best simple rule by more than this


def trust_facts(history) -> dict | None:
    """The accuracy facts a reader needs, taken from the newest successful backtest and calibration, never from model text.
    Returns None when no backtest ran."""
    obs = [h for h in history if isinstance(h, dict)] if isinstance(history, list) else []
    bt = next((r for h in reversed(obs) if isinstance(r := _ok_result(h, "backtest_series"), dict) and r.get("models")), None)
    if bt is None:
        return None
    models = bt.get("models") or {}
    best = bt.get("best_baseline")
    tf, base = (models.get("timesfm") or {}).get("mase"), (models.get(best) or {}).get("mase") if best else None
    facts = {"origins": bt.get("origins") or bt.get("n_origins"), "horizon": bt.get("horizon"), "timesfm_mase": tf,
             "best_baseline": best, "baseline_mase": base, "gain_pct": bt.get("timesfm_vs_best_baseline_pct"),
             "backtest_coverage": bt.get("timesfm_range_coverage_80"), "verdict": bt.get("verdict"),
             "random_walk": (bt.get("random_walk") or {}).get("verdict")}
    te = bt.get("total_error")
    if isinstance(te, dict) and isinstance(te.get("variance_ratio"), (int, float)):
        facts.update(total_variance_ratio=te.get("variance_ratio"), total_origins=te.get("origins"),
                     total_held_measured=te.get("held_measured"), total_held_independent=te.get("held_independent"),
                     total_held_together=te.get("held_together"))
    cal = next((r for h in reversed(obs) if isinstance(r := _ok_result(h, "calibrate_intervals"), dict)
                and "heldout_coverage_raw" in r), None)
    if cal is not None:
        facts.update(heldout_raw=cal.get("heldout_coverage_raw"), heldout_calibrated=cal.get("heldout_coverage_calibrated"),
                     calibration=cal.get("recommend"))
    if facts["verdict"] is None and isinstance(facts["gain_pct"], (int, float)):
        g = facts["gain_pct"]
        facts["verdict"] = "timesfm_better" if g > HEADLINE_MARGIN else ("baseline_better" if g < -HEADLINE_MARGIN else "no_clear_difference")
    return facts


def _pct(v: Any) -> str:
    return f"{100 * float(v):.0f}%" if isinstance(v, (int, float)) else "not measured"


TOTAL_Z90 = 1.2815515655446004      # as in report_tool: the 10-90 band is +-Z90 standard deviations


def _measured_total(history) -> tuple[dict, int] | None:
    """The pack's horizon total with its range built from the measured day-to-day error correlation, and the horizon it is for.

    The newest successful forecast_report gives the per-step band and the total's centre; the newest backtest with a
    total_error of the same horizon gives the variance ratio. The range is centre +- Z90 * sqrt(R * sum of day variances), so
    R = 1 is independent days and R = horizon is days that move together. None when either piece is missing or the horizons
    differ (a truncated table cannot be summed)."""
    obs = [h for h in history if isinstance(h, dict)] if isinstance(history, list) else []
    rep = next((r for h in reversed(obs) if isinstance(r := _ok_result(h, "forecast_report"), dict)), None)
    if rep is None:
        return None
    table, tr = rep.get("table"), rep.get("total_range_80")
    if not isinstance(table, dict) or table.get("rows_omitted") != 0 or not isinstance(tr, dict):
        return None
    rows = table.get("rows") or []
    horizon = len(rows)
    te, measured_h = None, None
    with_te = [r for h in reversed(obs) if isinstance(r := _ok_result(h, "backtest_series"), dict)
               and isinstance(r.get("total_error"), dict) and _is_int(r.get("horizon"))]
    same = next((r for r in with_te if r["horizon"] == horizon), None)
    pick = same or (with_te[0] if with_te else None)
    if pick is None:
        return None
    te, measured_h = pick["total_error"], pick["horizon"]
    try:
        sd2 = sum(((float(row["q90"]) - float(row["q10"])) / (2 * TOTAL_Z90)) ** 2 for row in rows)
        centre, ratio = float(tr["centre"]), float(te["variance_ratio"])
    except (KeyError, TypeError, ValueError):
        return None
    if measured_h != horizon and measured_h > 1 and horizon > 1:
        # Live retail_biz rep3: the backtest ran 20 steps and the pack 43, so no measured range applied and the answer mixed
        # the two bounds into a range no tool printed. The ratio is carried to the pack's horizon at the same average
        # correlation between days, rho = (R - 1) / (h - 1), so R' = 1 + (h' - 1) * rho; the sentence says it was extended.
        rho = min(max((ratio - 1.0) / (measured_h - 1), -1.0 / (horizon - 1)), 1.0)
        ratio = min(max(1.0 + (horizon - 1) * rho, 1.0 / horizon), float(horizon))
    half = TOTAL_Z90 * (ratio * sd2) ** 0.5
    return ({"q10": centre - half, "q90": centre + half, "centre": centre, "variance_ratio": ratio,
             "mean": centre / horizon, "mean_q10": (centre - half) / horizon, "mean_q90": (centre + half) / horizon,
             "origins": te.get("origins"), "held_measured": te.get("held_measured"), "measured_horizon": measured_h}, horizon)


def measured_total_range(history) -> dict | None:
    """The 80 percent range of the pack's horizon total, using the day-to-day error correlation measured on the backtest.
    None when the pieces are missing or the backtest horizon differs from the pack's table length."""
    found = _measured_total(history)
    return found[0] if found else None


CENTRE_WORDS = {"timesfm": "TimesFM", "global_gbm": "the pooled gradient-boosted model",
                "global_blend": "the equal-weight blend of the pooled model and TimesFM", "naive": "the last value",
                "seasonal_naive": "the same step a season earlier", "mean_last_season": "the average of the last season",
                "drift": "the drift rule", "blend": "the blend"}


def _panel_rules(h: dict, res: dict) -> list[str]:
    """The centre rule of each series a panel pack drew, from the full result when stored, else the shown lines."""
    full = h.get("full") if isinstance(h.get("full"), dict) else {}
    lines = ((full.get("forecast") or {}).get("series") if isinstance(full.get("forecast"), dict) else None) \
        or res.get("series") or {}
    return [((ln.get("centre") or {}).get("rule", "timesfm") if isinstance(ln.get("centre"), dict)
             else ln.get("centre") or "timesfm") for ln in lines.values() if isinstance(ln, dict)]


# A level (a price, a rate, a concentration, a daily mean) has no meaningful sum over days or series; a flow (sales, units,
# kWh in a day) does. Read from the measure's own names: the forecast pack's title and the uploaded data's numeric columns.
LEVEL_MEASURE = re.compile(r"price|spot|\brate\b|_rate|pm2|pm10|pm_|concentration|temperature|\btemp|index|\blevel|yield|"
                           r"ratio|percent|pct|%|\u00b5g|ug_?m3|humidity|pressure|_mean\b|\bmean\b|_avg|\bavg\b|average|"
                           r"per_barrel|per barrel|/bbl|usd_per|exchange|fx\b|"
                           # Panel round 2 (2026-11-26): "S&P 500 Daily Close" was given a 20-day total. Closes, stock
                           # indexes, stored quantities and anomalies are levels too.
                           r"\bclose\b|closing|adj_close|\bclose_|s&p|nasdaq|dow jones|\bdjia\b|ftse|nikkei|"
                           r"storage|inventor(?:y|ies)|\bppm\b|co2|anomal|unemployment|inflation|\bcpi\b", re.I)
FLOW_MEASURE = re.compile(r"_sum\b|\bsum\b|total|sales|revenue|units|count|volume|kwh|orders|demand|quantity|qty", re.I)


def is_level_name(text: str) -> bool:
    """True when a measure's name reads as a level (a price, rate, concentration), and not as a flow that sums over time."""
    return bool(LEVEL_MEASURE.search(text or "")) and not FLOW_MEASURE.search(text or "")


def measure_is_level(history) -> bool:
    """True when the forecast measure is a level whose total over the horizon means nothing (live WTI rep9: "80% range for the
    20-day total is $1,644-$1,987" of an oil price; Beijing: a 14-day panel total of PM2.5 concentrations, 12,222 ug/m3). A
    name that reads as a flow (sales, kWh, a daily sum) wins over one that reads as a level."""
    names: list[str] = []
    for h in history:
        if not isinstance(h, dict) or h.get("kind") != "tool":
            continue
        if h.get("tool") == "forecast_report" and isinstance((h.get("args") or {}).get("title"), str):
            names.append(h["args"]["title"])
        if _ok_result(h, "inspect_data") is not None:
            r = h["full"] if isinstance(h.get("full"), dict) else h["result"]
            prof = r.get("profile") if isinstance(r.get("profile"), dict) else r
            for col in prof.get("columns") or []:
                if isinstance(col, dict) and col.get("kind") == "number":
                    names.append(str(col.get("name")))
                elif isinstance(col, str) and ": number" in col:
                    names.append(col.split(":", 1)[0])
    return is_level_name(" ".join(names))


PANEL_VERDICTS_LISTED = 20     # up to this many series get one line each; beyond it, counts and the exceptions


def panel_verdicts(obs: dict, history) -> dict | None:
    """Each series of a panel pack with its verdicts side by side: the pack's health check and the panel backtest's best model
    for that series, with its MASE and the best simple rule's. None unless obs is a panel pack and a panel backtest exists.

    Live Beijing rep8 (measured, 2026-11-23), goal "give each station's forecast with its verdict": the pack held each
    station's health and panel_backtest each station's scores, in two results; the planner tried to join them from files, the
    evaluator rejected three finishes for missing verdicts, and the mission ended partial with a correct forecast."""
    res = obs.get("result") if isinstance(obs, dict) and obs.get("tool") == "forecast_report" and not obs.get("error") else None
    if not isinstance(res, dict) or not res.get("n_series"):
        return None
    bt = next((r for h in reversed(with_delegated(history)) if isinstance(r := _ok_result(h, "panel_backtest"), dict)
               and isinstance(r.get("per_entity"), list)), None)
    if bt is None:
        return None
    rules = ("naive", "seasonal_naive", "mean_last_season", "drift")
    scores = {str(e.get("entity")): e for e in bt["per_entity"] if isinstance(e, dict)}
    rows: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {"health": {}, "backtest_best": {}}
    exceptions: list[str] = []
    full = obs.get("full") if isinstance(obs.get("full"), dict) else {}
    full_series = (full.get("forecast") or {}).get("series") if isinstance(full.get("forecast"), dict) else None
    if isinstance(full_series, dict) and full_series:
        lines = {n: {"health": (fc.get("health") or {}).get("verdict") if isinstance(fc, dict) else None}
                 for n, fc in full_series.items()}
    else:
        lines = res.get("series") or {}
    for name, line in lines.items():
        health = line.get("health") if isinstance(line, dict) else None
        e = scores.get(str(name)) or {}
        models = {k: v for k, v in e.items() if k != "entity" and isinstance(v, (int, float))}
        best = min(models, key=models.get) if models else None
        rule = min((k for k in models if k in rules), key=models.get, default=None)
        text = f"health {health}"
        if best:
            text += f"; backtest best {best} (MASE {models[best]:.3g}" + (f", best simple rule {rule} {models[rule]:.3g}"
                                                                          if rule and rule != best else "") + ")"
        rows[str(name)] = text
        counts["health"][str(health)] = counts["health"].get(str(health), 0) + 1
        if best:
            counts["backtest_best"][best] = counts["backtest_best"].get(best, 0) + 1
        if health != "ok" or (best in rules):
            exceptions.append(f"{name}: {text}")
    drawn = set(_panel_rules(obs, res))
    basis = "health from this pack; scores from the newest panel_backtest" + (
        "; the health check judged TimesFM's line, not the drawn centre" if drawn and drawn != {"timesfm"} else "")
    if len(rows) <= PANEL_VERDICTS_LISTED:
        return {"per_series": rows, "basis": basis}
    return {"counts": counts, "series_scored": f"{len(scores)} of {len(rows)} (the stored panel_backtest view)",
            "exceptions": exceptions[:PANEL_VERDICTS_LISTED],
            "basis": basis + "; exceptions are series whose health is not ok or whose best model was a simple rule"}


PANEL_SCORE_OF = {"global_blend": "blend"}     # the pack's rule name -> panel_backtest's model name


def check_panel_centre(history) -> dict | None:
    """A panel pack that draws one rule for every series draws one the panel backtest did not score worse than its best by more
    than RULE_TOLERANCE. None when there is no panel pack, no panel backtest, or the pack sets rules per series.

    Live electricity rep11 (measured, 2026-11-23): panel_backtest put TimesFM first (overall MASE 1.331) and the pooled model
    fourth (1.539, 16 % worse); the pack drew the pooled model for all 60 clients, and its total was 3.08 million kWh where
    rep10's TimesFM pack gave 4.38 million. Nothing compared the drawn panel centre with the ranking."""
    pack = next(((h, r) for h in reversed(history) if isinstance(r := _ok_result(h, "forecast_report"), dict)
                 and r.get("n_series")), None)
    bt = next((r for h in reversed(with_delegated(history)) if isinstance(r := _ok_result(h, "panel_backtest"), dict)
               and isinstance(r.get("overall"), dict) and r.get("best_overall")), None)
    if pack is None or bt is None:
        return None
    rules = set(_panel_rules(*pack))
    if len(rules) != 1:
        return None
    rule = rules.pop()
    scored, best = PANEL_SCORE_OF.get(rule, rule), str(bt["best_overall"])
    overall = bt["overall"]
    if not isinstance(overall.get(scored), (int, float)) or not isinstance(overall.get(best), (int, float)):
        return None
    if scored == best or overall[scored] <= overall[best] * (1 + RULE_TOLERANCE):
        return _check("panel_centre_matches_backtest", True,
                      f"the panel pack draws {rule}, within {RULE_TOLERANCE:.0%} of the best panel score ({best})")
    pct = 100 * (overall[scored] / overall[best] - 1)
    return _check("panel_centre_matches_backtest", False,
                  f"the panel pack draws {rule} for every series, but panel_backtest scored it {overall[scored]:.3g} overall "
                  f"against {overall[best]:.3g} for {best}, {pct:.0f}% worse; rebuild the pack with centre "
                  f"'{ {'blend': 'global_blend'}.get(best, best) }', or set rules per series from the per-entity scores")


GLOBAL_RULES = ("global_gbm", "global_blend")
# A sentence that says which rule is the centre line, as opposed to ranking the methods ("TimesFM wins on 21 clients, global_gbm on 17").
_CENTRE_CLAIM = re.compile(r"centre|center|\bpack\b|delivered|\bdraws?\b|\bassigns?\b|forecast (?:uses|used|was|is)\b", re.I)


def _centre_named_wrong(text: str, obs: list) -> dict | None:
    """The answer says the centre line is one global rule while the newest panel pack drew the other for every series. Blocks.

    Live electricity 29acb1c0 (2026-11-23): the last forecast_report drew global_blend for all 60 clients, but a ledger written
    earlier said global_gbm, and the answer stated "the panel pack assigns global_gbm as the centre for every client". The
    check reads the pack's own rules; a sentence that names both rules, or negates the claim, is not read as a claim."""
    pack = next(((h, r) for h in reversed(obs) if isinstance(r := _ok_result(h, "forecast_report"), dict) and r.get("n_series")),
                None)
    if pack is None:
        return None
    rules = set(_panel_rules(*pack))
    if len(rules) != 1 or next(iter(rules)) not in GLOBAL_RULES:
        return None
    drawn = next(iter(rules))
    for unit in grounding.sentences(text):
        low = unit.lower()
        named = [r for r in GLOBAL_RULES if r in low]
        if not named or drawn in named or not _CENTRE_CLAIM.search(low):
            continue
        if re.search(r"\bnot\b|\bnor\b|rather than|instead of|requested", low):
            continue
        return _check("answer_centre_named_right", False,
                      f"the answer says the centre is {named[0]} ({unit.strip()[:160]}), but the newest forecast pack draws "
                      f"{drawn} for every series; name {drawn}, and take the other rule out")
    return None


# A table in the answer that gives each station's day-by-day figures must copy them from that station's own day. The figure
# check matches a number against every number in the evidence, so a table cell of 52 is "grounded" by any 52 anywhere.
# Live Beijing 0137871e (2026-11-23): the table gave Aotizhongxin day 1 as 52 (12 to 132); the stored forecast gives 40.7
# (4.5 to 114.2), and the figure check passed it.
_DAY_HEAD = re.compile(r"\bday\s*(\d{1,3})\b", re.I)
_AVG_HEAD = re.compile(r"average|\bmean\b", re.I)
_BAND_HEAD = re.compile(r"\bq\s*\d|\d\s*%|\bband|\brange\b|\blow|\bhigh|\blower\b|\bupper\b|\binterval|\bmin\b|\bmax\b|"
                        r"percentile|quantile", re.I)
_NUMBER_TOKEN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


_SEP_ROW = re.compile(r"\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?")


def _row_cells(line: str) -> list[str]:
    """A table row's cells, with or without its outer pipes ("| a | b |" and "a | b" are both two cells)."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return s.split("|")


def _table_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """(start, stop) of each markdown table in the lines.

    A run of pipe-bounded lines is a table, as before. Audit round 6m: a run with a separator row is a table too, even without
    outer pipes ("Station | Day 1" over "--- | ---"); its header is the line above the separator, so a pipe in the prose above
    does not join it."""
    out, i = [], 0
    while i < len(lines):
        if "|" not in lines[i] or not lines[i].strip():
            i += 1
            continue
        j = i
        while j < len(lines) and "|" in lines[j] and lines[j].strip():
            j += 1
        block = lines[i:j]
        if all(_is_table_line(l) for l in block):
            out.append((i, j))
        else:
            seps = [k for k, l in enumerate(block) if _SEP_ROW.fullmatch(l.strip())]
            if seps:
                out.append((i + max(seps[0] - 1, 0), j))
        i = j
    return out


def _tables(text: str) -> list[list[list[str]]]:
    """The markdown tables in a text, each as its rows of cells, separator rows dropped."""
    lines, tables = text.splitlines(), []
    for start, stop in _table_blocks(lines):
        rows = [[c.strip() for c in _row_cells(l)] for l in lines[start:stop]]
        rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r if c)]
        if rows:
            tables.append(rows)
    return tables


def _printed_matches(token: str, value: float) -> bool:
    """A printed figure matches the value when it is the value rounded to the printed decimals."""
    digits = token.replace(",", "")
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    return abs(float(digits) - value) <= 0.5 * 10 ** -decimals + 1e-9


def _panel_series(obs: list) -> dict | None:
    """The newest panel forecast's series keyed by lower-cased name, read from its full result; None when there is none."""
    pack = next(((h, r) for h in reversed(obs) if isinstance(r := _ok_result(h, "forecast_report"), dict) and r.get("n_series")),
                None)
    if pack is None:
        return None
    h, res = pack
    full = h.get("full") if isinstance(h.get("full"), dict) else res
    fc_part = full.get("forecast") if isinstance(full, dict) else None
    series = fc_part.get("series") if isinstance(fc_part, dict) else None
    if not isinstance(series, dict) or not series:
        return None
    return {str(k).lower(): v for k, v in series.items() if isinstance(v, dict)}


def _day_columns(header: list[str]) -> tuple[dict[int, int], int | None]:
    """A header's day columns (column index -> day number) and its average column. A band header is never a day.

    A bare number header ("1", "14") is a day column too (audit round 6l): a station table often writes its days that way,
    and the day is only compared when its row names a station, so a plain number elsewhere is not checked.
    """
    day_cols, avg_col = {}, None
    for i, cell in enumerate(header):
        days = _DAY_HEAD.findall(cell)
        if _BAND_HEAD.search(cell):
            continue                       # "Day 14 q10" holds a band edge, not the point
        if not days and re.fullmatch(r"\d{1,3}", cell.strip()):
            day_cols[i] = int(cell.strip())
        elif len(days) == 1 and "→" not in cell and "->" not in cell:
            day_cols[i] = int(days[0])
        elif _AVG_HEAD.search(cell):
            avg_col = i
    return day_cols, avg_col


def _station_series(cell: str, by_name: dict) -> dict | None:
    """The series a table's first cell names: the exact name, or the name with a suffix in brackets or after a dash.

    Audit round 6m: "Aotizhongxin (north)" and "Aotizhongxin - north" named a station, but an exact match alone left their wrong
    figures unchecked. A suffix must follow a space and a separator, so "Dingling Annex" is not read as "Dingling"."""
    name = re.sub(r"[*_`]", "", cell).strip().lower()
    if name in by_name:
        return by_name[name]
    m = re.match(r"^(.+?)\s*(?:\([^)]*\)|\s[-–—:,]\s.*)$", name)
    return by_name.get(m.group(1).strip()) if m else None


_TOTAL_HEAD = re.compile(r"^(?:total|sum)\b", re.I)


def _total_column(header: list[str]) -> int | None:
    """The column named Total or Sum: the horizon total of the station's point forecast. A band or day header is never one.

    Audit round 6m: a "Total" column had no day and no average name, so its figures were never checked."""
    for i, cell in enumerate(header):
        c = re.sub(r"[*_`]", "", cell).strip()
        if _TOTAL_HEAD.match(c) and not _BAND_HEAD.search(c) and not _DAY_HEAD.search(c):
            return i
    return None


def _table_matches_panel(text: str, obs: list) -> dict | None:
    """The day-by-day cells of a station table match that station's own forecast on that day. Blocks on the first mismatch.

    Read from the newest panel forecast's full result. A row is a station when its first cell names a series exactly; a
    day column is a header with one "Day n"; a cell is checked when it holds the point alone or the point with its band.
    """
    by_name = _panel_series(obs)
    if by_name is None:
        return None
    checked, wrong = 0, []
    for table in _tables(text):
        day_cols, avg_col = _day_columns(table[0])
        total_col = _total_column(table[0])
        if not day_cols and avg_col is None and total_col is None:
            continue
        for row in table[1:]:
            fc = _station_series(row[0], by_name)
            if fc is None:
                continue
            point = _numbers(fc.get("point")) or []
            quant = fc.get("quantiles") if isinstance(fc.get("quantiles"), dict) else {}
            lo_all, hi_all = _numbers(quant.get("0.1")) or [], _numbers(quant.get("0.9")) or []
            for i, k in day_cols.items():
                if i >= len(row) or not (1 <= k <= min(len(point), len(lo_all), len(hi_all))):
                    continue
                toks = _NUMBER_TOKEN.findall(row[i])
                if not toks:
                    continue
                expected = (point[k - 1], lo_all[k - 1], hi_all[k - 1])
                # Audit round 6l: a point with its band is read whole. Any other shape ("52 (12)", "52 ± 5", "52 (12 to 132,
                # 90%)") still has its leading figure, the point, checked, so the extra figures cannot hide a wrong point.
                pairs = zip(toks, expected) if len(toks) in (1, 3) else [(toks[0], expected[0])]
                ok = all(_printed_matches(t, e) for t, e in pairs)
                checked += 1
                if not ok:
                    wrong.append(f"{row[0].strip(' *')} day {k} is {row[i]}; the forecast gives {expected[0]:.1f} "
                                 f"({expected[1]:.1f} to {expected[2]:.1f})")
            if avg_col is not None and avg_col < len(row) and point:
                toks = _NUMBER_TOKEN.findall(row[avg_col])
                if toks:                   # the leading figure is the average, whatever follows it (audit round 6l)
                    checked += 1
                    mean = sum(point) / len(point)
                    if not _printed_matches(toks[0], mean):
                        wrong.append(f"{row[0].strip(' *')} average is {row[avg_col]}; the forecast averages {mean:.1f}")
            if total_col is not None and total_col < len(row) and point:
                toks = _NUMBER_TOKEN.findall(row[total_col])
                if toks:                   # the leading figure is the horizon total of the point forecast (audit round 6m)
                    checked += 1
                    total = sum(point)
                    if not _printed_matches(toks[0], total):
                        wrong.append(f"{row[0].strip(' *')} total is {row[total_col]}; the forecast sums to {total:.1f}")
    if not checked:
        return None
    if wrong:
        return _check("answer_table_matches_forecast", False,
                      f"{len(wrong)} table cell(s) do not match the station's own forecast: {wrong[0][:200]}. Copy each "
                      "station's figures for each day from the forecast, not from another row")
    return _check("answer_table_matches_forecast", True, f"{checked} table cell(s) match each station's own forecast")


def _is_table_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and len(s) > 1


def _restate(cell: str, values: list[float]) -> str:
    """The cell's numbers replaced in order by ``values``, each printed to the decimals it had; the rest of the cell is kept."""
    it = iter(values)

    def _one(m: re.Match) -> str:
        tok = m.group(0)
        decimals = len(tok.split(".")[1]) if "." in tok else 0
        return f"{next(it):.{decimals}f}"
    return _NUMBER_TOKEN.sub(_one, cell)


def _snap_block(lines: list[str], start: int, stop: int, by_name: dict, fixed: list[tuple[str, str]]) -> None:
    """Corrects one markdown table's station day and average cells in place, recording each (old, new) cell."""
    rows = [_row_cells(lines[k]) for k in range(start, stop)]
    day_cols, avg_col = _day_columns([c.strip() for c in rows[0]])
    total_col = _total_column([c.strip() for c in rows[0]])
    if not day_cols and avg_col is None and total_col is None:
        return
    for k in range(1, len(rows)):
        cells = rows[k]
        if all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in cells if c.strip()):
            continue                                   # the separator row
        fc = _station_series(cells[0], by_name)
        if fc is None:
            continue
        point = _numbers(fc.get("point")) or []
        quant = fc.get("quantiles") if isinstance(fc.get("quantiles"), dict) else {}
        lo_all, hi_all = _numbers(quant.get("0.1")) or [], _numbers(quant.get("0.9")) or []
        targets = [(i, (point[d - 1], lo_all[d - 1], hi_all[d - 1])) for i, d in day_cols.items()
                   if 1 <= d <= min(len(point), len(lo_all), len(hi_all))]
        if avg_col is not None and point:
            targets.append((avg_col, (sum(point) / len(point),)))
        if total_col is not None and point:
            targets.append((total_col, (sum(point),)))
        changed = False
        for i, expected in targets:
            if i >= len(cells) or "," in cells[i]:
                continue
            toks = _NUMBER_TOKEN.findall(cells[i])
            if len(toks) not in (1, len(expected)):
                continue
            if all(_printed_matches(t, e) for t, e in zip(toks, expected)):
                continue
            new = _restate(cells[i], list(expected[:len(toks)]))
            if new != cells[i]:
                fixed.append((cells[i].strip(), new.strip()))
                cells[i] = new
                changed = True
        if changed:
            line = lines[start + k]
            lead, trail = line[: len(line) - len(line.lstrip())], line[len(line.rstrip()):]
            s = line.strip()
            body = "|".join(cells)
            # The row keeps its own form: a row written without outer pipes is rewritten without them (audit round 6m).
            lines[start + k] = lead + ("|" if s.startswith("|") else "") + body + ("|" if s.endswith("|") else "") + trail


def snap_tables(text: str, obs: list) -> tuple[str, list[tuple[str, str]]]:
    """The answer with each station table's day and average cells set to the newest panel forecast, and the (old, new) pairs.

    The same kind of copying slip snap corrects in a figure, found by the same check: the row names the station and the
    column names the day, so the printed value is known. Only a cell holding one number, or a point with its band and no
    thousands comma, is changed, and only when it differs from the forecast at its printed decimals. Live Beijing 80bbc853
    (2026-11-23): eight stations' day cells were not their stored values; the one rewrite copied them wrong again, and the
    answer left partial.
    """
    by_name = _panel_series(obs)
    if by_name is None:
        return text, []
    lines = text.split("\n")
    fixed: list[tuple[str, str]] = []
    for start, stop in _table_blocks(lines):
        _snap_block(lines, start, stop, by_name, fixed)
    return "\n".join(lines), fixed


def panel_sentence(history) -> str:
    """The newest panel pack in plain words: which centre line it drew for how many series, and the panel total with both
    bounds named for what they assume.

    Live electricity rep10 (measured, 2026-11-23): the pack drew TimesFM for all 60 clients and the answer said the blend was
    the centre line; it called the sum of the clients' 10-90% bounds the likely range and said that assumed independent
    errors (summing bounds assumes every client misses the same way at once)."""
    for h in reversed(with_delegated(history)):
        res = _ok_result(h, "forecast_report")
        if not isinstance(res, dict) or not res.get("n_series"):
            continue
        rules = _panel_rules(h, res)
        counts: dict[str, int] = {}
        for r in rules:
            counts[r] = counts.get(r, 0) + 1
        n = res.get("n_series")
        shown = sum(counts.values())
        if counts and shown == n:
            parts = [f"{CENTRE_WORDS.get(r, r)} for {'all ' if k == n else ''}{k} series" for r, k in
                     sorted(counts.items(), key=lambda kv: -kv[1])]
            text = " The panel pack's centre line: " + "; ".join(parts) + "."
        else:
            text = ""
        pt = res.get("panel_total")
        if isinstance(pt, dict) and measure_is_level(history):
            text += (" No panel total is given: the measure is a level (a price, rate or concentration), so a sum over series "
                     "or days means nothing; read each series' own line and range.")
            pt = None
        if isinstance(pt, dict) and isinstance(pt.get("centre"), (int, float)):
            ind, tog = pt.get("if_entities_miss_independently") or {}, pt.get("if_entities_miss_together") or {}
            if all(isinstance(d.get(k), (int, float)) for d in (ind, tog) for k in ("q10", "q90")):
                text += (f" Panel total over {pt.get('steps')} steps: {pt['centre']:,.0f}. Its 80% range lies between two bounds: "
                         f"{ind['q10']:,.0f} to {ind['q90']:,.0f} if the series miss independently, and {tog['q10']:,.0f} to "
                         f"{tog['q90']:,.0f} if they all miss the same way at once (the sum of each series' own bounds, the "
                         "cautious one).")
        return text
    return ""


def trust_statement(history) -> str:
    """One computed paragraph on how far to trust the forecast: the backtest score against the best simple rule, what that
    means for the centre line, and how often the range held. It is appended to the answer as written by code, so a model can
    never misquote it (live: an answer said a range "reached 87.5%" when the stored calibration said otherwise)."""
    f = trust_facts(history)
    extra = panel_sentence(history)
    found = _measured_total(history)
    if found is not None:
        mr, h = found
        # SciFore (2026-11-24): this sentence is appended to the answer, then the withheld-total check reads it. A level,
        # or a pack that withheld its total, must not be called a total here or the check fails on Prophet's own line.
        level = measure_is_level(history)
        withheld = _withheld_totals(history if isinstance(history, list) else [])
        if level or withheld:
            why = next(iter(withheld.values()), "")
            head_range = ((" The pack withheld a summed total (" + why + ")." if why else "")
                          + (" This measure is a level, so it is not summed over the horizon." if level else "")
                          + f" 80% range of its average over the {h} steps: {mr['mean_q10']:,.2f} to "
                          f"{mr['mean_q90']:,.2f} (centre {mr['mean']:,.2f})")
            noun = "value"
        else:
            head_range = (f" 80% range of the {h}-step total: {mr['q10']:,.0f} to {mr['q90']:,.0f} (per step on average: "
                          f"{mr['mean']:,.2f}, range {mr['mean_q10']:,.2f} to {mr['mean_q90']:,.2f})")
            noun = "total"
        extra = (head_range + f", using the day-to-day error correlation "
                 f"measured over {mr['origins']} past windows of {mr['measured_horizon']} steps"
                 + (f", extended to {h} steps at the same average correlation between days"
                    if mr["measured_horizon"] != h else "")
                 + f"; built the same way, the range held the actual {noun} in {mr['held_measured']} of {mr['origins']} windows."
                 + extra)
    if not f or not isinstance(f.get("timesfm_mase"), (int, float)) or not isinstance(f.get("baseline_mase"), (int, float)):
        return ("**Accuracy check (computed from the stored backtest):**" + extra) if extra else ""
    rule = BASELINE_NAMES.get(f["best_baseline"], str(f["best_baseline"]))
    head = (f"Over {f['origins']} past windows of {f['horizon']} steps, TimesFM scored MASE {f['timesfm_mase']:.3g} against "
            f"{f['baseline_mase']:.3g} for the best simple rule ({rule}).")
    v = f.get("verdict")
    if v == "baseline_better":
        call = (" The simple rule was more accurate, so use it for the centre line and treat TimesFM's range as the band; "
                "the TimesFM centre line is shown for reference only.")
    elif v == "timesfm_better":
        call = f" TimesFM was {abs(f['gain_pct']):.1f}% more accurate." if isinstance(f.get("gain_pct"), (int, float)) else ""
    else:
        call = " The two are not distinguishable over this many windows, so the centre line is no better than the simple rule."
    if f.get("random_walk") == "consistent_with_random_walk":
        call += (" A variance-ratio test finds no evidence of predictability at this sample size: the changes look like a "
                 "random walk, so no method is expected to call its direction; read the range, not the line.")
    rng = f" In that backtest the 10-90% range held {_pct(f.get('backtest_coverage'))} of outcomes (target 80%)."
    if f.get("heldout_raw") is not None:
        used = "the calibrated range is used" if f.get("calibration") == "use_calibrated" else "the raw range is kept"
        rng += (f" On later held-out windows it held {_pct(f['heldout_raw'])} raw and {_pct(f.get('heldout_calibrated'))} "
                f"after calibration; {used}.")
    return "**Accuracy check (computed from the stored backtest):** " + head + call + rng + extra


# ---------------------------------------------------------------- the written answer (A6-09)

def _health_period(rec: dict) -> int | None:
    health = rec.get("health") if isinstance(rec, dict) else None
    p = health.get("period") if isinstance(health, dict) else None
    return p if _is_int(p) and p >= 2 else None


def _newest_backtest(obs: list) -> dict | None:
    return next((r for h in reversed(obs) if isinstance(r := _ok_result(h, "backtest_series"), dict)
                 and isinstance(r.get("models"), dict)), None)


def _scan_shift_counts(history: list, run_root) -> list[int] | None:
    """The level-shift counts the saved scan reports, one per entity. A saved scan with no level-shift finding holds zero."""
    findings = _saved_scan_findings(history, run_root)
    if findings is None:
        return None
    counts = [f["evidence"]["count"] for f in findings if f.get("id") == "changepoints.level_shift"
              and isinstance(f.get("evidence"), dict) and _is_int(f["evidence"].get("count"))]
    return counts or [0]


def _withheld_totals(obs) -> dict[str, str]:
    """Why the delivered pack (the newest forecast_report) withheld its headline total: {"": reason} for one series,
    {entity: reason} for a panel. Empty when the total stands or no pack was built."""
    for h in reversed(obs):
        res = _ok_result(h, "forecast_report")
        if not isinstance(res, dict):
            continue
        head = res.get("headline")
        if isinstance(head, dict):
            return {"": str(head["withheld"])} if head.get("withheld") else {}
        series = res.get("series")
        if isinstance(series, dict):
            return {str(n): str(ln["total_withheld"]) for n, ln in series.items()
                    if isinstance(ln, dict) and ln.get("total_withheld")}
        return {}
    return {}


def check_ledger_answer(answer, history, run_root) -> dict | None:
    """The answer and the newest saved ledger must take the same stance on negatives, outliers and gaps (A4-04). Live evidence
    (measured): retail rep3 ledger "keep negative values" against an answer that "clipped to 0"; WTI rep3 "no gaps filled"
    against a ledger of 109 gaps filled by interpolation. An entry marked superseded is left out, so a later correction counts.
    None when no ledger was saved or ``run_root`` is unknown."""
    ledgers = _saved_ledgers(history, run_root) if run_root is not None else []
    if not ledgers:
        return None
    decided = " . ".join(str(e.get("decision", "")) for e in ledgers[0]
                         if isinstance(e, dict) and not e.get("superseded") and not e.get("superseded_by"))
    ledger = grounding.treatment_stances(decided)
    said = grounding.treatment_stances(answer if isinstance(answer, str) else "")
    clashes = [(topic, l, a) for topic in sorted(set(ledger) & set(said)) for l in sorted(ledger[topic]) for a in sorted(said[topic])
               if (l, a) in grounding.OPPOSED_STANCES or (a, l) in grounding.OPPOSED_STANCES]
    if clashes:
        topic, l, a = clashes[0]
        return _check("ledger_answer_consistent", False,
                      f"the ledger decides {topic} as '{l}', but the answer says '{a}'; make the answer follow the ledger, or "
                      "record a superseding decision in the ledger first", "warn")
    if set(ledger) & set(said):
        return _check("ledger_answer_consistent", True,
                      "the answer's treatments of " + ", ".join(sorted(set(ledger) & set(said))) + " agree with the ledger", "warn")
    return None


def _level_summed(text: str, obs: list) -> dict | None:
    """An answer that states the pack's total over days or series for a level measure (a price, rate or concentration). Blocks.

    Live WTI rep9 (2026-11-23) gave "the 80% range for the 20-day total is $1,644-$1,987" of an oil price; Beijing reps 8 and 9
    gave a 14-day panel total of PM2.5 concentrations, 12,222 ug/m3. The figures are the pack's own, so the figure check passed."""
    if not measure_is_level(obs):
        return None
    res = next((r for h in reversed(obs) if isinstance(r := _ok_result(h, "forecast_report"), dict)), None)
    if not isinstance(res, dict):
        return None
    tr = res.get("panel_total") if res.get("n_series") else res.get("total_range_80")
    if not isinstance(tr, dict):
        return None
    figs = [tr.get("centre")] + [b.get(k) for b in tr.values() if isinstance(b, dict) for k in ("q10", "q90")]
    mr = measured_total_range(obs) if not res.get("n_series") else None
    if mr:
        figs += [mr.get(k) for k in ("q10", "q90", "centre")]
    figs = [f for f in figs if isinstance(f, (int, float)) and f]
    for s in grounding.sentences(text):
        if not re.search(r"\btotal\b|\bsum\b|\bsummed\b|\bcumulative\b", s, re.I):
            continue                     # a year or an outside figure near the total is not a claim of one
        vals = grounding._sentence_values(s)
        if any(abs(v - f) <= 0.01 * abs(f) for v in vals for f in figs):
            return _check("answer_level_not_summed", False,
                          f"the measure is a level (a price, rate or concentration), so its sum over days or series means "
                          f"nothing, but the answer states one ({s[:160]}); give the average over the horizon and each "
                          "series' or day's own values instead")
    return None


HEALTH_WORDS = ("flat_expected", "flat_unexpected", "check")
HEALTH_TOKEN = re.compile(r"\b(flat[_ ]expected|flat[_ ]unexpected)\b|\b(check)\b(?= verdict| health)", re.I)


def _series_health_named(text: str, obs: list) -> dict | None:
    """A sentence that gives one panel series a health verdict the pack did not give it. Blocks.

    Live Beijing rep9 (measured, 2026-11-23): "Two stations (Nongzhanguan, Huairou) received a flat_expected verdict" while the
    pack, and the answer's own table, gave Huairou "ok". Only a sentence with one non-ok verdict and no "ok" in it is read, so a
    sentence contrasting two verdicts is left alone."""
    pack = next(((h, r) for h in reversed(obs) if isinstance(r := _ok_result(h, "forecast_report"), dict) and r.get("n_series")),
                None)
    if pack is None:
        return None
    h, res = pack
    full = h.get("full") if isinstance(h.get("full"), dict) else {}
    fs = (full.get("forecast") or {}).get("series") if isinstance(full.get("forecast"), dict) else None
    health = ({str(n): (fc.get("health") or {}).get("verdict") for n, fc in fs.items() if isinstance(fc, dict)}
              if isinstance(fs, dict) and fs else
              {str(n): ln.get("health") for n, ln in (res.get("series") or {}).items() if isinstance(ln, dict)})
    if not health:
        return None
    for s in grounding.sentences(text):
        if s.lstrip().startswith("|"):
            continue                                   # a table row states its own series' cell
        tokens = {m.group(0).lower().replace(" ", "_") for m in HEALTH_TOKEN.finditer(s)}
        if len(tokens) != 1 or re.search(r"\bok\b", s, re.I):
            continue
        said = tokens.pop()
        wrong = [n for n in health if re.search(rf"\b{re.escape(n)}\b", s) and health[n] != said]
        if wrong:
            return _check("answer_series_health_matches", False,
                          f"the answer gives {', '.join(wrong[:4])} the health verdict {said} ({s[:160]}), but the pack gave "
                          + "; ".join(f"{n}: {health[n]}" for n in wrong[:4]) + ". State each series' verdict as the pack gives it")
    return None


MAE_WORD = re.compile(r"\bMAE\b", re.I)


def _metric_named(text: str, obs: list) -> dict | None:
    """Scores called MAE when the backtests scored MASE. Blocks unless an analysis step printed an MAE of its own.

    Live electricity rep12 (measured, 2026-11-23): the answer's table headed "Overall MAE" listed panel_backtest's MASE scores
    (1.47 and so on). MAE is in the data's own units; MASE is scaled by the series' typical step change, so the two are not
    interchangeable and a reader would misjudge the error by orders of magnitude."""
    if not MAE_WORD.search(text):
        return None
    scored = any(isinstance(_ok_result(h, t), dict) for h in with_delegated(obs) for t in ("backtest_series", "panel_backtest"))
    if not scored:
        return None
    own = any(MAE_WORD.search(json.dumps(h.get("result"), default=str)) for h in with_delegated(obs)
              if isinstance(h, dict) and h.get("tool") == "sandbox_python")
    if own:
        return None
    return _check("answer_metric_named", False,
                  "the answer calls the scores MAE, but the backtests scored MASE (mean absolute scaled error: the error divided "
                  "by the series' typical step change, 1 meaning as good as the naive rule); name the metric MASE")


AVERAGE_GOAL = re.compile(r"\baverage\b|\bmean\b", re.I)


def _average_range(text: str, obs: list, goal: str) -> dict | None:
    """A goal that asks for the average over the horizon gets the average's own 80% range in the answer: the per-step average
    of the measured total range. Blocks when the answer gives neither edge of it.

    Live wti_budget rep3 (measured, 2026-11-23), "what will crude cost on average next month": the answer gave the average,
    $90.76, with "80% range: $77-$110 by month-end", one day's band; the measured range of the month's average was 78.86 to
    102.65 and appeared only in the computed paragraph below."""
    if not AVERAGE_GOAL.search(goal or ""):
        return None
    mr = measured_total_range(obs)
    if not mr or not all(isinstance(mr.get(k), (int, float)) for k in ("mean", "mean_q10", "mean_q90")):
        return None
    vals = grounding._sentence_values(text)
    has = lambda t: any(abs(v - t) <= 0.01 * abs(t) for v in vals)
    if has(mr["mean_q10"]) and has(mr["mean_q90"]):
        return _check("answer_average_range", True, "the answer gives the measured 80% range of the average")
    return _check("answer_average_range", False,
                  f"the goal asks for an average over the horizon; the measured 80% range of that average is "
                  f"{mr['mean_q10']:,.2f} to {mr['mean_q90']:,.2f} around {mr['mean']:,.2f}. Give that range for the average; "
                  "one step's band is wider and answers a different question")


def _misnamed_bound(text: str, obs: list) -> dict | None:
    """A bound of the horizon total (or the panel total) stated as the range without saying what it assumes. Blocks.

    Live retail_biz rep7 (measured, 2026-11-23) opened with "an 80% range of £1,257,168 to £2,610,558", the bound if every day
    misses the same way at once, while the measured range appended below it read 1,358,450 to 2,142,805; the run passed as
    verified. Electricity rep10 called the sum of the clients' bounds "the likely range" and said it assumed independence."""
    pack = next((r for h in reversed(obs) if isinstance(r := _ok_result(h, "forecast_report"), dict)), None)
    if pack is None:
        return None
    if pack.get("n_series"):
        tr, keys, what = pack.get("panel_total"), ("if_entities_miss_together", "if_entities_miss_independently"), "series"
    else:
        tr, keys, what = pack.get("total_range_80"), ("if_days_miss_together", "if_days_miss_independently"), "days"
    if not isinstance(tr, dict):
        return None
    measured = measured_total_range(obs) if not pack.get("n_series") else None
    # Both bounds, each with its assumption, in the words the answer must use. Live Beijing 97ab893f (2026-11-23): the rewrite
    # was told only "give both bounds", kept the wide bound under "if stations vary independently", and was blocked twice.
    words = (f"if every one of the {what} misses the same way at once", f"if the {what} miss independently")
    both = "; ".join(f"{w}: {b['q10']:,.0f} to {b['q90']:,.0f}" for w, k in zip(words, keys)
                     if isinstance(b := tr.get(k), dict) and all(isinstance(b.get(x), (int, float)) for x in ("q10", "q90")))
    ranges = {k: (v["q10"], v["q90"]) for k in keys
              if isinstance(v := tr.get(k), dict) and all(isinstance(v.get(x), (int, float)) for x in ("q10", "q90"))}
    if measured and all(isinstance(measured.get(x), (int, float)) for x in ("q10", "q90")):
        ranges["measured"] = (measured["q10"], measured["q90"])
    for key, together in zip(keys, (True, False)):
        b = tr.get(key)
        if not isinstance(b, dict) or not all(isinstance(b.get(k), (int, float)) for k in ("q10", "q90")):
            continue
        others = tuple(r for k, r in ranges.items() if k != key)
        s = grounding.bound_misnamed(text, b["q10"], b["q90"], together, others)
        if s:
            assumes = f"every one of the {what} misses the same way at once" if together else f"the {what} miss independently"
            fix = (f"; the measured 80% range is {measured['q10']:,.0f} to {measured['q90']:,.0f}: give that as the range"
                   if measured else f"; give both bounds, each with its own assumption: {both}")
            return _check("answer_total_range_named", False,
                          f"the answer gives {b['q10']:,.0f} to {b['q90']:,.0f} as the total's range ({s[:160]}), but that is the "
                          f"bound if {assumes}{fix}")
    return None


# The pooled blend is one equal-weight blend of the pooled model and TimesFM, the same for every client. A per-client blend
# (TimesFM with each client's best simple rule) is another method. Live electricity rep13 (2026-11-23) wrote the pooled blend
# as "TimesFM combined with the best simple rule per client", which sends the reader to the wrong method.
PER_CLIENT_BLEND = re.compile(
    r"\bblend\b[^.\n|]{0,90}?\bper[- ](?:client|series|entity|station|sku)s?\b"
    r"|\bper[- ](?:client|series|entity|station|sku)s?\b[^.\n|]{0,30}?\bblend\b", re.I)


def _blend_misdescribed(text: str, obs: list) -> dict | None:
    """A block when the panel's centre is the pooled blend and the answer calls it a per-client blend. None otherwise."""
    found = next(((h, r) for h in reversed(obs) if isinstance(r := _ok_result(h, "forecast_report"), dict) and r.get("n_series")),
                 None)
    if found is None or set(_panel_rules(*found)) != {"global_blend"}:
        return None
    m = PER_CLIENT_BLEND.search(text)
    if m is None:
        return None
    return _check("answer_blend_described_right", False,
                  f"the panel's centre is the equal-weight blend of the pooled model and TimesFM, the same for every client, but "
                  f"the answer describes it as a per-client blend ({m.group()[:160]}); describe it as that blend")


_NAMED_IMAGE = re.compile(r"(?<![\w./:-])((?:[\w-]+/)*[\w.-]+\.(?:png|svg|jpe?g|gif|webp))(?![\w-])", re.I)


def _made_files(history) -> set[str]:
    """Lower-case base names of every file a tool call in this mission (or a sub-agent's) listed as written."""
    names: set[str] = set()
    for h in with_delegated(history):
        res = h.get("result") if isinstance(h, dict) and h.get("kind") == "tool" else None
        if not isinstance(res, dict):
            continue
        for key in ("outputs", "files"):
            for o in res.get(key) or []:
                path = o.split(" (")[0] if isinstance(o, str) else o.get("path") if isinstance(o, dict) else None
                if isinstance(path, str) and path:
                    names.add(Path(path).name.lower())
    return names


def check_answer_files(answer, history) -> dict | None:
    """An image the answer names must be a file a run made, so the person sees it.

    Live NVDA diagram (2026-11-25): three sandbox_python cells failed with a SyntaxError and wrote nothing, and the answer
    said "The diagram has been created and saved as nvda_forecast_pipeline_diagram.png". Nothing was shown. Web addresses are
    not read as file names. None when the answer names no image."""
    text = answer if isinstance(answer, str) else ""
    named = list(dict.fromkeys(Path(m.group(1)).name for m in _NAMED_IMAGE.finditer(text)))
    if not named:
        return None
    made = _made_files(history)
    missing = [n for n in named if n.lower() not in made]
    if not missing:
        return _check("answer_file_exists", True, "every image the answer names is a file a run made")
    return _check("answer_file_exists", False,
                  f"the answer names {', '.join(missing[:4])}, but no run made that file (no tool result lists it). Make "
                  "it in sandbox_python and save it under out/, or take the file name out and say the image was not made")


def check_answer_real_series(answer, goal: str = "") -> dict | None:
    """The answer may not say it used approximate or synthetic data in place of a requested real series.

    Motivation vs Logic
    Motivation (task 2, 2026-11-26): a planner that cannot find the real series can fall back on invented numbers and still write
    "completed". The person asked for NVDA prices; a forecast of simulated prices is not the answer to that question.
    Logic: a sentence that offers invented, simulated or approximate data (questions.invented_data_phrase) blocks the finish. A
    sentence that negates it is not a claim of use. A goal that itself asks for a synthetic or simulated series (a twin, a test)
    is left alone, because then the synthetic data is the test the person asked for. None when the answer makes no such claim.
    """
    if SYNTHETIC_GOAL.search(goal or ""):
        return None
    phrase = invented_data_phrase(answer if isinstance(answer, str) else "")
    if phrase is None:
        return None
    return _check("answer_real_series", False,
                  f"the answer says it used {phrase!r}, which stands in for a real series the person asked for. Do not "
                  "present invented or approximate data as the series: find the real series in public sources, or state that "
                  "it could not be found and finish as partial")


def check_answer(answer, history, *, run_root=None, goal: str = "") -> list[dict]:
    """Checks on what the written answer says against the stored output. A check is returned only for a claim the answer
    makes, so an answer with no such claim adds nothing.

    Motivation (audit round 3, A6-09): the answer is written by a model and the figures in it are matched, but its plain
    claims were not. Live retail rep4 said "weekly (7-day) confirmed" with a health period of 5; live WTI said "naive is the
    appropriate benchmark" with a backtest that ranked mean_last_season ahead. Those two reach the person, so they block.
    Logic: the claims are read by grounding.py. A cycle must equal the measured health period, and a named best rule must be
    the backtest's best or within RULE_TOLERANCE of it. A level-shift count and a cause with no citation are warnings.
    """
    text = answer if isinstance(answer, str) else ""
    obs = [h for h in history if isinstance(h, dict)] if isinstance(history, list) else []
    checks: list[dict] = []

    periods = grounding.claimed_periods(text)
    if periods:
        stored = {p for h in obs if (rec := _forecast_record(h)) is not None and (p := _health_period(rec)) is not None}
        if stored:
            # A long cycle is read within 1 percent: "365 days" against a measured 364 is the same annual cycle (CO2 live).
            wrong = [(n, s) for n, s in periods if not any(abs(n - p) <= p // 100 for p in stored)]
            if wrong:
                n, s = wrong[0]
                # Live (retail business user, 2026-11-23): the rewrite was asked to "state the measured period" and kept the
                # backtest's 3 steps. Name the number to write, and why the other figure is not the cycle.
                measured = " or ".join(f"{p} steps" for p in sorted(stored))
                checks.append(_check("answer_cycle_matches_health", False,
                                     f"the answer states a cycle of {n} steps ({s[:160]}), but the health record measured "
                                     f"{measured}; write {measured} as the cycle, and do not use the backtest's season check "
                                     f"for it"))
            else:
                checks.append(_check("answer_cycle_matches_health", True,
                                     "every cycle the answer states is the period the health record measured"))

    forecast_claims = grounding.claimed_forecast_results(text)
    stored_forecast = any(_forecast_record(h) for h in obs)
    if forecast_claims and not stored_forecast:
        checks.append(_check(
            "answer_forecast_stored", False,
            "the answer states a forecast result, but no forecast tool result is stored "
            f"({forecast_claims[0][:160]}); run the forecast, or take the result out"))
    claims = grounding.claimed_best_baselines(text)
    bt = _newest_backtest(obs)
    models = bt["models"] if bt is not None else {}
    best = bt.get("best_baseline") if bt is not None else None
    bm = _finite(models[best].get("mase")) if isinstance(best, str) and isinstance(models.get(best), dict) else None
    if claims and bm is None:
        # Audit round 6l: with no backtest, or a backtest that names no usable best baseline, nothing verifies the claim,
        # so it must not pass. The check blocks and asks for the backtest's best rule, or for the claim to be dropped.
        name, sentence = claims[0]
        why = "no backtest is stored" if bt is None else "the stored backtest records no usable best baseline"
        checks.append(_check("answer_benchmark_matches", False,
                             f"the answer calls {name} the best benchmark, but {why}, so the claim is not verified "
                             f"({sentence[:160]}); run the backtest and name the rule it ranks best, or drop the claim"))
    elif claims:
        # Audit round 6m: a rule the backtest never ranked (no MASE for it) cannot be verified as the best benchmark, so it blocks.
        # Before, its MASE read as None and the claim passed as "the one the backtest ranked best".
        unranked = [(n, s) for n, s in claims
                    if not isinstance(models.get(n), dict) or _finite(models[n].get("mase")) is None]
        wrong_rules = []
        for name, sentence in claims:
            nm = _finite(models[name].get("mase")) if isinstance(models.get(name), dict) else None
            if name != best and nm is not None and bm is not None and nm > bm * (1 + RULE_TOLERANCE):
                wrong_rules.append((name, nm, sentence))
        if unranked:
            name, sentence = unranked[0]
            checks.append(_check("answer_benchmark_matches", False,
                                 f"the answer calls {name} the best benchmark, but the stored backtest does not rank {name} "
                                 f"(no MASE for it), so the claim is not verified ({sentence[:160]}); name the rule the backtest "
                                 f"ranks best ({best}), or drop the claim"))
        elif wrong_rules:
            name, nm, sentence = wrong_rules[0]
            checks.append(_check("answer_benchmark_matches", False,
                                 f"the answer calls {name} the best benchmark (MASE {nm:.3f}), but the backtest ranked {best} "
                                 f"best (MASE {bm:.3f}) ({sentence[:160]}); name {best}, or say why it was not used"))
        else:
            checks.append(_check("answer_benchmark_matches", True,
                                 "the benchmark the answer names is the one the backtest ranked best, or within tolerance of it"))

    counts = grounding.claimed_shift_counts(text)
    stored_counts = _scan_shift_counts(obs, run_root) if counts else None
    if counts and stored_counts is not None:
        allowed = set(stored_counts) | {sum(stored_counts)}
        wrong_counts = [(n, s) for n, s in counts if n not in allowed]
        if wrong_counts:
            n, s = wrong_counts[0]
            checks.append(_check("answer_shift_count_matches", False,
                                 f"the answer counts {n} level shifts ({s[:160]}), but the saved scan reports "
                                 f"{', '.join(str(c) for c in stored_counts)}", "warn"))
        else:
            checks.append(_check("answer_shift_count_matches", True,
                                 "the level-shift count the answer states is the scan's count", "warn"))

    causes = grounding.uncited_causes(text)
    if causes:
        checks.append(_check("answer_causes_cited", False,
                             f"the answer attributes a shift to a cause with no research source id or test ({causes[0][:160]}); "
                             "cite the S-id of a research passage or the control_lab test, or say the shift is unexplained",
                             "warn"))

    reconciled = check_ledger_answer(text, obs, run_root)
    if reconciled is not None:
        checks.append(reconciled)

    if (bound := _misnamed_bound(text, obs)) is not None:
        checks.append(bound)
    if (avg := _average_range(text, obs, goal)) is not None:
        checks.append(avg)
    if (metric := _metric_named(text, obs)) is not None:
        checks.append(metric)
    if (health := _series_health_named(text, obs)) is not None:
        checks.append(health)
    if (summed := _level_summed(text, obs)) is not None:
        checks.append(summed)

    withheld = _withheld_totals(obs)
    if withheld:
        named = [n for n in withheld if n]               # panel entities; a single series is keyed ""
        totals = [s for s in grounding.claimed_totals(text) if not named or any(n.lower() in s.lower() for n in named)]
        why = "; ".join(f"{n}: {r}" if n else r for n, r in list(withheld.items())[:2])
        if totals:
            checks.append(_check("answer_total_withheld", False,
                                 f"the pack withheld the total ({why[:200]}), but the answer states one ({totals[0][:160]}); "
                                 "give the reason first and quote the daily table or the range, not a total"))
        elif not grounding.explains_withheld(text):
            checks.append(_check("answer_total_withheld", False,
                                 f"the pack withheld the total ({why[:200]}); the answer does not say so. State the reason", "warn"))
        else:
            checks.append(_check("answer_total_withheld", True, "the answer gives no total the pack withheld, and says why"))

    methods = grounding.unsupported_method_claims(text, obs)
    if methods:
        checks.append(_check("answer_methods_run", False,
                             f"the answer says a method was tested or computed, but no tool call runs it ({methods[0][:160]}); "
                             "run the treatment, or take the claim out", "warn"))
    if (blend := _blend_misdescribed(text, obs)) is not None:
        checks.append(blend)
    if (centre := _centre_named_wrong(text, obs)) is not None:
        checks.append(centre)
    if (table := _table_matches_panel(text, obs)) is not None:
        checks.append(table)
    if (short := _series_shorter_than_file(obs)) is not None:
        checks.append(short)
    if (level := _flat_expected_level_sentence(text, obs)) is not None:
        checks.append(level)
    if (named := check_answer_files(text, obs)) is not None:
        checks.append(named)
    if (real := check_answer_real_series(text, goal)) is not None:
        checks.append(real)
    return checks


_LEVEL_LINE = re.compile(r"centre line is a level|center line is a level", re.I)


def _latest_health_verdict(obs: list) -> str | None:
    return delivered_verdict(obs)


TRENDING_CENTRES = ("drift",)


def _delivered_centre(obs: list) -> str | None:
    """The centre named on the newest forecast_report (timesfm when none was named); None when no pack was built."""
    for h in reversed(obs if isinstance(obs, list) else []):
        if isinstance(h, dict) and _ok_result(h, "forecast_report") is not None:
            centre = (h.get("args") or {}).get("centre", "timesfm")
            return centre if isinstance(centre, str) else None
    return None


def _level_line_applies(obs: list) -> bool:
    """flat_expected, and the delivered centre is not a trending rule. Panel round 3 SPX (2026-11-26): the pack's centre was
    drift, the sentence "the centre line is a level" was added, and the evaluator rightly called the answer wrong."""
    return _latest_health_verdict(obs) == "flat_expected" and _delivered_centre(obs) not in TRENDING_CENTRES


def _flat_expected_level_sentence(text: str, obs: list) -> dict | None:
    """The level sentence is required on the written answer only for flat_expected.

    SciRes-20261124-124543 stored check. The evaluator demanded the sentence anyway. A check verdict
    adds no check. A flat_expected verdict blocks until the answer says the centre line is a level.
    """
    if not _level_line_applies(obs):
        return None
    if _LEVEL_LINE.search(text or ""):
        return _check("answer_level_sentence", True, "the answer says the centre line is a level")
    return _check("answer_level_sentence", False,
                  "the health verdict is flat_expected, so the answer must say the centre line is a level because "
                  "the history shows no repeating pattern. A check verdict does not require this sentence.")


LEVEL_SENTENCE = ("The centre line is a level: the history shows no repeating pattern, so the line carries no directional "
                  "view and the range is the forecast.")


def add_level_sentence(text: str, obs: list) -> str:
    """The answer with the level sentence added when the delivered verdict is flat_expected and the answer lacks it.

    Motivation: panel runs NVDA (round 1) and SPX (baseline), 2026-11-26: the composer was told to add the sentence and
    the answer still missed it, so a price forecast that was right to be flat ended partial. The sentence follows from
    the stored verdict alone, like a snapped figure. Logic: only for flat_expected, only when _LEVEL_LINE is absent.
    """
    if not _level_line_applies(obs) or _LEVEL_LINE.search(text or ""):
        return text
    return (text or "").rstrip() + "\n\n" + LEVEL_SENTENCE


def _series_read_from_download(obs: list, paths: list[str]) -> bool:
    """True when the forecast's source file was written by sandbox code that read a downloaded file, directly or through
    files that earlier cells derived from it. Such a series is a filter of the file (one region of a long table), not a paste.

    Panel NEM (2026-11-26): the download held 604 rows for several regions; the run kept the 120 NSW rows in code and was
    blocked as if it had pasted a preview."""
    names = {p.rsplit("/", 1)[-1] for p in paths if p}
    if not names:
        return False
    source_run = None
    for h in reversed(obs):
        if isinstance(h, dict) and h.get("kind") == "tool" and h.get("tool") in ("forecast_report", "timesfm_forecast"):
            src = (h.get("args") or {}).get("source")
            if isinstance(src, dict) and src.get("run_id"):
                source_run = str(src["run_id"])
                break
    if source_run is None:
        return False
    derived = set(names)
    for h in obs:
        res = _ok_result(h, "sandbox_python")
        if not isinstance(res, dict):
            continue
        code = str((h.get("args") or {}).get("code") or "")
        reads = any(n in code for n in derived)
        if reads:
            for out in res.get("outputs") or []:
                derived.add(str(out).split(" (", 1)[0].rsplit("/", 1)[-1])
        if str(res.get("run_id")) == source_run:
            return reads
    return False


def _series_shorter_than_file(obs: list) -> dict | None:
    """The forecast used a small paste of a file the mission already downloaded.

    FinRes-20261124-111851 forecast 228 points ending 2017-10-10. SciRes re-run pasted 13 rows from 1974 out of the
    NOAA file. The file's row count is on the download or on the fetch that landed it."""
    files: list[tuple[int, str]] = []
    for h in obs:
        for tool in ("web_fetch", "download_web_data", "download_open_data"):
            res = _ok_result(h, tool)
            if isinstance(res, dict) and isinstance(res.get("row_count"), int) and res["row_count"] >= 30:
                files.append((res["row_count"], str(res.get("sandbox_file") or "")))
    if not files:
        return None
    used = None
    for h in reversed(obs):
        # forecast_report nests the same config. FinRes called that tool, not timesfm_forecast,
        # so a 228-point sample of a downloaded file never reached this check.
        rec = _forecast_record(h)
        if not isinstance(rec, dict):
            continue
        cfg = rec.get("config") if isinstance(rec.get("config"), dict) else None
        if cfg and isinstance(cfg.get("history_points"), int):
            used = cfg["history_points"]
            break
    # A mission can download two series. The forecast covers the file it actually used when its history
    # is at least half of that file. Block only when the history is a short paste of every downloaded file.
    if used is None or any(used >= rows * 0.5 for rows, _path in files):
        return None
    if _series_read_from_download(obs, [p for _r, p in files if p]):
        return None
    rows, path = max(files)
    where = f" at {path}" if path else ""
    return _check("series_covers_download", False,
                  f"the forecast used {used} points, but the downloaded file{where} has {rows} rows. "
                  "Read that file in sandbox_python and forecast the series it holds, not a pasted preview")
