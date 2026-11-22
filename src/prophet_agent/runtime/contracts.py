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
    if hist and all(math.isfinite(v) for v in hist) and p and min(hist) >= 0:
        neg = sum(1 for v in p if not v >= 0)
        checks.append(_check("non_negative_when_history_is", neg == 0,
                             "the history is never negative and the forecast is never negative" if neg == 0
                             else f"the history is never negative but {neg} forecast steps are below zero"))
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
        # A panel pack, seen as one line per series. It is held only when a series delivered on TimesFM's own line lost its
        # cycle; a series centred on a simple rule does not deliver TimesFM's line.
        lost = [n for n, ln in res["series"].items() if isinstance(ln, dict) and ln.get("health") == "flat_unexpected"
                and ln.get("centre", "timesfm") == "timesfm"]
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


def _saved_ledgers(history, run_root) -> list[list]:
    """The entries of every saved ledger in this mission (newest first). A step that saved the ledger and then failed later in
    the cell still saved it: the file is the evidence. Unreadable or unsafe paths are skipped."""
    out: list[list] = []
    for h in reversed(with_delegated(history)):
        res = h.get("result") if isinstance(h, dict) and h.get("kind") == "tool" and h.get("tool") == "sandbox_python" else None
        if not isinstance(res, dict) or not isinstance(res.get("run_id"), str) or "/" in res["run_id"] or ".." in res["run_id"]:
            continue
        outputs = [_output_path(o) for o in (res.get("outputs") or [])]
        if "out/ledger.json" not in outputs:
            continue
        try:
            entries = json.loads((Path(run_root) / res["run_id"] / "out" / "ledger.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(entries, list):
            out.append(entries)
    return out


def ledger_stages(history, run_root) -> int | None:
    """The most distinct stages (with a finding and a decision each) any saved ledger in this mission records. None when
    ``run_root`` is unknown, so a caller can tell "zero" from "cannot look"."""
    if run_root is None:
        return None
    best = 0
    for entries in _saved_ledgers(history, run_root):
        stages = {str(e.get("stage", "")).strip().lower() for e in entries
                  if isinstance(e, dict) and str(e.get("finding", "")).strip() and str(e.get("decision", "")).strip()}
        best = max(best, len(stages - {""}))
    return best


def _saved_scan_findings(history, run_root) -> list[dict] | None:
    """The findings of every scan the mission saved (out/scan*.json in each run that listed it). None when no scan was saved
    or the files cannot be read."""
    if run_root is None:
        return None
    found: list[dict] = []
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
                found.extend(f for f in doc["findings"] if isinstance(f, dict))
    return found if seen else None


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
                  + "; record each one with ps.Ledger().add(stage, finding, decision) and put the finding id in the text")


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


def used_controls(args: dict) -> list[str]:
    """The TimesFM tuning options a forecast call switched on (a false or empty value is the default)."""
    return [k for k in CONTROL_OPTIONS if args.get(k)]


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
    used = used_controls(args)
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


def delivered_health(rec: dict) -> str | None:
    """The health verdict of the line actually delivered. When the pack's centre line is a simple rule (the backtest found it
    more accurate), TimesFM's line is shown for reference only, so its health does not hold the mission (live retail: 17
    pack builds with a rule as centre, each held because TimesFM's own line had lost the weekly cycle)."""
    centre = rec.get("centre") if isinstance(rec, dict) else None
    if isinstance(centre, dict) and centre.get("rule"):
        return f"centre_rule:{centre['rule']}"
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


def researched(history) -> bool:
    """True once outside context was sought: a web tool was called (even if it failed) or a researcher sub-agent ran. An attempt
    counts, so a mission with no network is told to say so in its answer rather than being held on this item."""
    for h in history if isinstance(history, list) else []:
        if not isinstance(h, dict):
            continue
        if h.get("kind") == "tool" and h.get("tool") in RESEARCH_TOOLS:
            return True
        if h.get("kind") == "subagent" and h.get("agent_type") == "researcher":
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
    backtested = any(isinstance(_ok_result(h, "backtest_series"), dict) for h in obs)
    stages = ledger_stages(obs, run_root)
    items: list[dict] = [{"item": "inspect the data (sandbox_python or inspect_data)", "status": "done" if inspected else "todo"}]
    if run_root is not None:
        items.append({"item": "scan the data and save it (prophet_scan: sc.scan(...) then sc.write_scan(r))",
                      "status": "done" if scan_ran(obs) else "todo"})
    if stages is not None:
        items.append({"item": f"analysis ledger: at least {MIN_LEDGER_STAGES} stages with a finding and a decision "
                              "(time index, coverage, zeros and censoring, outliers and level shifts, seasonality)",
                      "status": "done" if stages >= MIN_LEDGER_STAGES else "todo",
                      "detail": f"{stages} of {MIN_LEDGER_STAGES} recorded"})
    if external:
        items.append({"item": "outside context: the series is driven by things its history cannot show, so delegate a researcher "
                              "(agent_type researcher) for the regime, scheduled events in the horizon and the main risks, or say "
                              "plainly why web research could not be used",
                      "status": "done" if researched(obs) else "todo"})
    items.append({"item": "compare the forecast method with the simple baselines (backtest_series)",
                  "status": "done" if backtested else "todo"})
    items.append({"item": "forecast the series (timesfm_forecast)", "status": "done" if forecasts else "todo"})
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
    elif not backtests:
        checks.append(_check("baseline_compared", False,
                             "the forecast was never compared with the simple baselines (no successful backtest)"))
    else:
        checks.append(_baseline_gate(forecasts[-1][1], backtests))

    if not forecasts:
        checks.append(_check("health_repaired", True, "no forecast yet; nothing to check"))
    else:
        verdict = delivered_health(forecasts[-1][1])
        if verdict == "flat_unexpected":
            checks.append(_check("health_repaired", False,
                                 "the last forecast's health says the cycle was lost, and it was never repaired"))
        else:
            checks.append(_check("health_repaired", True, f"the last forecast's health is {verdict or 'not reported'}"))

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
        decisions = check_scan_decisions(obs, run_root)
        if decisions is not None:
            checks.append(decisions)

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


def trust_statement(history) -> str:
    """One computed paragraph on how far to trust the forecast: the backtest score against the best simple rule, what that
    means for the centre line, and how often the range held. It is appended to the answer as written by code, so a model can
    never misquote it (live: an answer said a range "reached 87.5%" when the stored calibration said otherwise)."""
    f = trust_facts(history)
    if not f or not isinstance(f.get("timesfm_mase"), (int, float)) or not isinstance(f.get("baseline_mase"), (int, float)):
        return ""
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
    return "**Accuracy check (computed from the stored backtest):** " + head + call + rng
