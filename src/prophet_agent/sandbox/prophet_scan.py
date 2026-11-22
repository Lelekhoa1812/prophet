"""Principal-analyst scan of a long table: ``import prophet_scan as sc; r = sc.scan(df, "date", "units", ["sku"], freq="D")``.

Motivation: aggregates hide defects. A total can look smooth while one entity repeats its timestamps, a clock change
repeats or drops a local hour, one entity is recorded in thousands while the rest are in units, or a stockout reads as
zero demand. A scan that only plots the total will miss all of these.

Logic: one call runs a fixed battery of checks and returns typed findings. Every finding has an id, a severity
(blocker, major, minor or info), a scope (one entity name, or "all"), the numbers it was based on, one decision the
analyst must record (or None), and a concrete next call. Blockers stop the entity or the table until the question is
answered. The scan never cleans, drops or fills anything, and it never changes the input frame.

Sampling: structural checks (duplicates, spacing, gaps, clock changes, zeros, missing values) run on every entity.
Statistical checks (outliers, level shifts, censoring, intermittency, stationarity, seasonality, neighbours) run on at
most ``max_entities`` entities, chosen with a fixed seed, and the summary says which were analysed.

Thresholds are named constants. Those marked "proposal" are proposed defaults to calibrate on synthetic replay before
they are trusted on real data. Those marked with a source reuse a published or in-repository rule.

Runs in the sandbox next to ``prophet_stats`` (imported as ``prophet_stats``) and needs only numpy, pandas, scipy and
statsmodels. Every result is plain JSON-safe data: no numpy types and no NaN (a missing number is reported as None).
"""
from __future__ import annotations

import json
import re
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from scipy import stats as _stats
from statsmodels.tsa.stattools import acf as _acf

try:                                       # inside the sandbox the helpers sit side by side; in tests they are a package
    from prophet_pair import ReportDict, accepts_aliases
except ImportError:                        # pragma: no cover
    from prophet_agent.sandbox.prophet_pair import ReportDict, accepts_aliases

try:  # inside the sandbox, prophet_stats is copied next to this file
    import prophet_stats as ps
except ImportError:  # in the repository it is imported as part of the package
    from prophet_agent.sandbox import prophet_stats as ps

_SAMPLE_SEED = 0  # fixed seed, so the same entities are analysed on every run (proposal)
_DAY_NS = 86_400 * 10**9
_HOUR_NS = 3_600 * 10**9
_SEVERITY_RANK = {"blocker": 0, "major": 1, "minor": 2, "info": 3}

# Data quality
_PARSE_FAIL_BLOCK = 0.01  # share of unreadable time or value cells above which the table is blocked (proposal)
_MISSING_VALUE_MAJOR = 0.20  # share of stamps with no value (proposal)
_FREQ_AGREE = 0.5  # share of entities that must agree on pd.infer_freq before it is used (proposal)

# Time grid and gaps
_OFF_GRID_MAJOR = 0.05  # share of stamps off the regular grid (proposal)
_GAP_SHARE_MAJOR = 0.20  # share of grid steps that are missing (proposal)
_WEEKEND_GAP_SHARE = 0.90  # share of all Saturdays and Sundays in range with no record that makes the series a business-day calendar (proposal)
_TOD_MIN_DAYS = 10  # days in a calendar month needed to read the first time of day (proposal)

# Coverage and zeros
_DEAD_RUN_MIN = 14  # consecutive zeros treated as a dead segment, in observed steps (proposal: two weeks of daily data)
_MIN_CYCLES = 2  # minimum active history in seasonal cycles (first-contact-profile, step 5)

# Scale and concentration
_SCALE_SPREAD_MINOR = 1.0  # log10 range of entity medians, that is a 10x spread (proposal)
_SCALE_SPREAD_MAJOR = 3.0  # log10 range, that is a 1000x spread (proposal)
_UNIT_GAP_LOG10 = 2.0  # largest gap in sorted log10 medians that suggests two unit systems, a 100x gap (proposal)
_UNIT_SIDE_MIN_SHARE = 0.10  # each side of a unit gap must hold this share of entities (proposal)
_UNIT_RATIO = 10.0  # median jump across a break that counts as a unit change (first-contact-profile, step 7)
_CONC_TOP_SHARE = 0.20  # the top 20 percent of entities by volume (proposal)
_CONC_MINOR = 0.60  # volume share held by that top group (proposal)
_CONC_MAJOR = 0.80  # volume share held by that top group (proposal)
_TAIL_VOLUME_SHARE = 0.001  # an entity below this share of total volume is tail (proposal)
_TAIL_ENTITY_SHARE = 0.50  # share of entities that must be tail before it is reported (proposal)

# Outliers and breaks
# Calibrated on white noise (n = 500, sd 2, 12 replicates): with a 7-point window and 3 sigmas, Hampel flagged about
# 5.6 percent of pure noise, far above the Gaussian 0.3 percent, because a 7-point median absolute deviation is too noisy.
# A 15-point window at 5 sigmas flags about 0.4 percent of noise. The STL remainder at k = 5 flagged 2 to 6 percent of
# noise, so it is reported as a diagnostic only and does not set severity.
_HAMPEL_WINDOW = 15
_HAMPEL_SIGMAS = 5.0
_STL_K = 5.0
# Severity uses the Hampel share. Thresholds sit at about 2.5 and 7.5 times the measured null rate. Proposal, to calibrate.
_OUTLIER_MINOR = 0.01
_OUTLIER_MAJOR = 0.03
_CP_MIN_SIZE = 20  # minimum segment length, the prophet_stats.changepoints default
_CP_SHIFT_SD = 2.0  # break size in within-segment standard deviations (proposal)
_REPEAT_RUN_MIN = 6  # identical consecutive non-zero values (proposal; integer demand can repeat naturally)

# Intermittency and zeros
_ADI_CUTOFF = 1.32  # average demand interval cutoff (Syntetos and Boylan, 2005)
_CV2_CUTOFF = 0.49  # squared coefficient of variation cutoff (Syntetos and Boylan, 2005)
_ZERO_INFLATED = 0.20  # zero share, the same proposal as prophet_stats.censoring_check

# Heterogeneity
_ETA2_MINOR = 0.06  # Cohen (1988) medium effect benchmark
_ETA2_MAJOR = 0.14  # Cohen (1988) large effect benchmark

# Seasonality and trend
_SEASON_MIN = 0.6  # seasonal strength, FPP3 reading that the seasonal component is clearly present (proposal)
_TREND_STRONG = 0.8  # trend strength, FPP3 reading that the trend dominates (proposal)
_ACF_MIN_PEAK = 0.2  # autocorrelation needed before a peak counts (proposal)
_ACF_MAX_LAG = 200  # upper bound on the lags searched, to bound run time (proposal)

# Neighbours
_NEIGH_WINDOW = 52  # last steps of the pooled timeline used to compare entities (proposal: one year of weekly data)
_NEIGH_MIN_OVERLAP = 26  # overlapping observed steps needed for a correlation (proposal)
_NEIGH_CORR = 0.7  # correlation at which an entity is flagged as borrowable (proposal)
_NEIGH_TOP_K = 3  # neighbours reported per entity (proposal)
_BORROW_CYCLES = 3.0  # an entity with fewer active cycles than this can borrow history (proposal)

# Candidate seasonal periods, in steps, for each calendar grain
_DAILY_CYCLES = (7, 365)  # week and year, for daily data
_WEEKLY_CYCLES = (52,)  # year, for weekly data
_MONTHLY_CYCLES = (12,)  # year, for monthly data
_QUARTERLY_CYCLES = (4,)  # year, for quarterly data
_BUSINESS_CYCLES = (5,)  # week, for business-day data

_CHECKS = (
    "schema", "values.non_finite", "values.missing", "values.negative", "time.duplicate_timestamps",
    "time.modal_frequency", "time.irregular_spacing", "gaps.missing", "time.dst_shift", "time.timezone_shift",
    "coverage", "scale", "concentration", "heterogeneity", "changepoints.level_shift", "outliers.share",
    "censoring", "intermittency", "stationarity.verdict", "seasonality", "trend.strength", "neighbours.borrowable",
)


# ---------------------------------------------------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    """Convert numpy, pandas and tuple values to plain JSON types. NaN and infinity become None."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return str(obj)


class Finding(dict):
    """A finding is a plain dict whose missing-key error names the keys that exist: a model that guesses a key (``message``,
    ``text``) reads the fix in the traceback instead of repeating the call."""

    # Names models write for a finding's fields (live: finding['title'] failed two ledger cells, finding['detail'] one).
    ALIASES = {"title": "message", "detail": "message", "description": "message", "summary": "message", "text": "message",
               "finding": "message", "decision": "decision_needed", "action": "next_step", "recommendation": "next_step"}

    def __missing__(self, key: Any) -> Any:
        target = self.ALIASES.get(key) if isinstance(key, str) else None
        if target is not None and target in self:
            return self[target]
        raise KeyError(f"a finding has no key {key!r}; its keys are {sorted(self)} (use 'message' for a one-line summary)")

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class ScanResult(dict):
    """The scan result: a dict with "findings" and "summary", which also reads as ``result.findings`` and
    ``result.summary``. A wrong key or attribute names the real ones, so one traceback is enough to fix the call."""

    @property
    def findings(self) -> list:
        return self["findings"]

    @property
    def summary(self) -> dict:
        return self["summary"]

    def __missing__(self, key: Any) -> Any:
        raise KeyError(f"a scan result has no key {key!r}; its keys are {sorted(self)}")

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"a scan result has no attribute {name!r}; use result['findings'] or result.findings "
                             f"(a list of findings) and result['summary']")


def _finding(fid: str, severity: str, scope: Any, evidence: Any, decision: str | None, next_step: str | None) -> dict:
    """One typed finding. ``scope`` is an entity name or "all"."""
    ev = _jsonable(evidence)
    return Finding({
        "id": fid,
        "severity": severity,
        "scope": str(scope),
        "message": f"{fid} ({scope}): {decision}" if decision else f"{fid} ({scope})",
        "evidence": ev,
        "decision_needed": decision,
        "next_step": next_step,
    })


def _result(findings: list[dict], summary: dict) -> dict:
    """Sort findings (most severe first, then by id and scope) and attach counts. The order is deterministic."""
    ordered = sorted(findings, key=lambda f: (_SEVERITY_RANK[f["severity"]], f["id"], f["scope"]))
    counts = {name: 0 for name in _SEVERITY_RANK}
    for f in ordered:
        counts[f["severity"]] += 1
    out_summary = dict(summary)
    out_summary["severity_counts"] = counts
    out_summary["blocked"] = counts["blocker"] > 0
    # Coverage totals over the analysed entities: analysts read summary['observed_steps'] and ['periods'] (live EUR/JPY: 7
    # failed cells), which until now lived only in each profile.
    profs = summary.get("profiles")
    profs = list(profs.values()) if isinstance(profs, dict) else (profs if isinstance(profs, list) else [])
    profs = [p for p in profs if isinstance(p, dict) and isinstance(p.get("periods"), int)]
    if profs:
        obs, per = sum(int(p.get("observed_steps") or 0) for p in profs), sum(p["periods"] for p in profs)
        out_summary.update(observed_steps=obs, periods=per, observed_share=round(obs / per, 4) if per else None)
    # ReportDict: a guessed key ('blockers', 'missing_steps') is answered by its alias or named with the real keys.
    return ScanResult({"findings": ordered, "summary": ReportDict(_jsonable(out_summary))})


def _ts(ns: Any) -> str:
    """A nanosecond timestamp as text."""
    return str(pd.Timestamp(int(ns)))


def _median(x: np.ndarray) -> float | None:
    """Median of the observed values, or None when nothing is observed."""
    ok = x[~np.isnan(x)]
    return float(np.median(ok)) if ok.size else None


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of each run of True in a boolean array."""
    if mask.size == 0 or not mask.any():
        return []
    padded = np.concatenate([[0], mask.astype(np.int8), [0]])
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]


def _fixed_step(off: Any) -> int | None:
    """Length of a fixed-length grain in nanoseconds (hours, minutes, days), or None for calendar grains.

    pandas 3 does not make ``Day`` a Tick, so both classes are checked explicitly.
    """
    if isinstance(off, pd.offsets.Tick):
        return int(off.nanos)
    if isinstance(off, pd.offsets.Day):
        return int(off.n) * _DAY_NS
    return None


def _cycle_periods(off: Any) -> list[int]:
    """Candidate seasonal periods, in steps, implied by the grain. Empty when the grain implies none."""
    if off is None:
        return []
    step = _fixed_step(off)
    if step is not None:
        if step <= 0 or _DAY_NS % step:
            return []
        per_day = _DAY_NS // step
        if per_day >= 2:  # sub-daily: the day and the week
            return [per_day, 7 * per_day]
        return list(_DAILY_CYCLES) if step == _DAY_NS else []
    code = str(getattr(off, "rule_code", "") or "").upper()
    if isinstance(off, pd.offsets.Week) or code.startswith("W"):
        return list(_WEEKLY_CYCLES)
    if code.startswith("M") or code.startswith("BM"):
        return list(_MONTHLY_CYCLES)
    if code.startswith("Q"):
        return list(_QUARTERLY_CYCLES)
    if code.startswith("B"):
        return list(_BUSINESS_CYCLES)
    return []


def _is_cycle(lag: int, period: int) -> bool:
    """True when a lag sits on a multiple of the period, allowing one step either side."""
    remainder = lag % period
    return remainder <= 1 or period - remainder <= 1


def _demand_class(adi: float, cv2: float | None) -> str:
    """Syntetos and Boylan (2005) classes from the average demand interval and the squared CV of non-zero sizes."""
    if cv2 is None:
        return "unknown"
    intermittent = adi >= _ADI_CUTOFF
    erratic = cv2 >= _CV2_CUTOFF
    if intermittent and erratic:
        return "lumpy"
    if intermittent:
        return "intermittent"
    if erratic:
        return "erratic"
    return "smooth"


def _acf_peak(arr: np.ndarray) -> tuple[int, float] | None:
    """The strongest local autocorrelation peak at lag 2 or more, on the span between first and last observed value."""
    observed = np.flatnonzero(~np.isnan(arr))
    if observed.size < 6:
        return None
    sub = pd.Series(arr[observed[0]: observed[-1] + 1]).interpolate(limit_area="inside").to_numpy(dtype=float)
    if np.isnan(sub).any() or np.nanstd(sub) == 0:
        return None
    nlags = min(sub.size // 3, _ACF_MAX_LAG)  # a third of the span is the usual upper bound for lags
    if nlags < 3:
        return None
    r = _acf(sub, nlags=nlags, fft=True)
    peaks = [
        lag for lag in range(2, nlags)
        if r[lag] >= r[lag - 1] and r[lag] >= r[lag + 1] and r[lag] >= _ACF_MIN_PEAK
    ]
    if not peaks:
        return None
    best = max(peaks, key=lambda lag: (r[lag], -lag))
    return int(best), float(r[best])


def _remove_profile(arr: np.ndarray, period: int) -> np.ndarray:
    """Subtract the mean value at each phase of the period, so a break search does not confuse the season with a shift."""
    phase = np.arange(arr.size) % period
    ok = ~np.isnan(arr)
    sums = np.bincount(phase, weights=np.where(ok, arr, 0.0), minlength=period)
    counts = np.bincount(phase, weights=ok.astype(float), minlength=period)
    profile = sums / np.maximum(counts, 1.0)
    return arr - profile[phase]


def _grid(ut: np.ndarray, off: Any) -> dict:
    """Map each unique stamp (int64 nanoseconds, sorted) to a step on the regular grid that starts at the first stamp.

    ``pos`` is the grid step, ``on`` says whether the stamp sits exactly on the grid, ``n`` is the number of grid steps
    and ``label`` turns a step number into a timestamp. Fixed-length grains use arithmetic; calendar grains (months,
    business days) use ``pd.date_range`` anchored at the first stamp.
    """
    step = _fixed_step(off)
    if step is not None:
        # The grid phase is the most common remainder of the stamps (for a daily grain, the usual time of day), so one
        # off-grid first stamp does not make every other stamp look off the grid. Ties go to the smallest remainder.
        remainders, counts = np.unique(ut % step, return_counts=True)
        phase = int(remainders[int(np.argmax(counts))])
        shifted = ut - phase
        cell = shifted // step
        return {
            "pos": cell - cell[0],
            "on": shifted % step == 0,
            "n": int(cell[-1] - cell[0]) + 1,
            "label": lambda k: pd.Timestamp(phase + (int(cell[0]) + int(k)) * step),
        }
    start = pd.Timestamp(int(ut[0]))
    if not off.is_on_offset(start):
        start = off.rollforward(start)
    end = pd.Timestamp(int(ut[-1]))
    if start > end:
        grid = np.empty(0, dtype=np.int64)
    else:
        grid = pd.date_range(start, end, freq=off).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    n = int(grid.size)
    if n == 0:
        return {"pos": np.zeros(ut.size, dtype=np.int64), "on": np.zeros(ut.size, dtype=bool), "n": 0,
                "label": lambda k: pd.NaT}
    idx = np.searchsorted(grid, ut)
    idx_c = np.minimum(idx, n - 1)
    on = (idx < n) & (grid[idx_c] == ut)
    return {"pos": idx_c.astype(np.int64), "on": on, "n": n, "label": lambda k: pd.Timestamp(int(grid[int(k)]))}


def _gap_stats(pos_on: np.ndarray, n_grid: int) -> dict:
    """Missing grid steps: their count, the runs they form, the longest run and its position."""
    present = np.zeros(n_grid, dtype=bool)
    present[pos_on] = True
    runs = _true_runs(~present)
    lengths = [length for _, length in runs]
    missing = int((~present).sum())
    longest = max(runs, key=lambda r: r[1]) if runs else None
    return {
        "missing": missing,
        "runs": len(runs),
        "longest": int(longest[1]) if longest else 0,
        "longest_start": int(longest[0]) if longest else None,
        "share": (missing / n_grid) if n_grid else 0.0,
        "runs_total": int(sum(lengths)),
    }


def _stamps(t: np.ndarray, v: np.ndarray) -> dict:
    """Collapse one entity's rows (sorted by time) to one row per stamp, without touching the input arrays.

    Returns the unique stamps, the number of rows per stamp, the mean value per stamp (NaN ignored), and whether the
    rows sharing a stamp disagree (different values, or some missing and some not).
    """
    ut, first, inv, counts = np.unique(t, return_index=True, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    is_nan = np.isnan(v)
    size = ut.size
    sums = np.bincount(inv, weights=np.where(is_nan, 0.0, v), minlength=size)
    ok_n = np.bincount(inv, weights=(~is_nan).astype(float), minlength=size)
    nan_n = counts.astype(float) - ok_n
    vs = np.full(size, np.nan)
    has = ok_n > 0
    vs[has] = sums[has] / ok_n[has]
    mx = np.fmax.reduceat(v, first)
    mn = np.fmin.reduceat(v, first)
    differ = (mx != mn) & ~(np.isnan(mx) & np.isnan(mn))
    partial = (nan_n > 0) & (nan_n < counts)
    conflict = (counts > 1) & (differ | partial)
    return {"ut": ut, "rows": counts, "vs": vs, "conflict": conflict}


def _infer_offset(structs: list[dict], modal_ns: int | None) -> tuple[Any, str | None]:
    """Grain from pd.infer_freq when most entities agree, else the modal step between stamps."""
    names: Counter = Counter()
    eligible = 0
    for s in structs:
        ut = s["ut"]
        if ut.size < 3:
            continue
        eligible += 1
        try:
            with warnings.catch_warnings():  # pandas announces a return-type change; the code reads both types
                warnings.filterwarnings("ignore", message="A future version of pandas will return a BaseOffset")
                inferred = pd.infer_freq(pd.DatetimeIndex(pd.to_datetime(ut, unit="ns")))
            # pandas may return a string now and an offset object later; both are read as the grain's frequency string
            name = None if inferred is None else getattr(inferred, "freqstr", inferred)
        except (ValueError, TypeError):
            name = None
        if name:
            names[name] += 1
    if names and eligible:
        top, top_n = names.most_common(1)[0]
        if top_n >= _FREQ_AGREE * eligible:
            return to_offset(top), "pd.infer_freq (most entities agree)"
    if modal_ns:
        return to_offset(pd.Timedelta(modal_ns, unit="ns")), "modal step between stamps"
    return None, None


# ---------------------------------------------------------------------------------------------------------------------
# Per-entity battery
# ---------------------------------------------------------------------------------------------------------------------

class _Battery:
    """Collects findings and the cross-entity state. Each check is a method; a failure in one check is reported, not raised."""

    def __init__(self, off: Any, key_hint: str, time_name: str, value_name: str, entity_hint: str):
        self.off = off
        self.step = _fixed_step(off)
        self.periods = _cycle_periods(off)
        self.key_hint = key_hint
        self.time_name = time_name
        self.value_name = value_name
        self.entity_hint = entity_hint
        self.findings: list[dict] = []
        self.profiles: dict[str, dict] = {}
        self.neighbours: dict[str, list] = {}
        self.volume: dict[str, float] = {}
        self.medians: dict[str, float] = {}
        self.series: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def add(self, fid: str, severity: str, scope: Any, evidence: Any, decision: str | None, next_step: str | None) -> None:
        self.findings.append(_finding(fid, severity, scope, evidence, decision, next_step))

    def _try(self, label: str, name: str, fn: Any, prof: dict) -> Any:
        """Run one check. A ValueError means the entity is too short or too sparse for it: recorded in the profile.
        Any other error is reported as a minor finding, so the rest of the scan still runs."""
        try:
            return fn()
        except ValueError as exc:
            prof.setdefault("skipped", {})[name] = str(exc)
        except Exception as exc:  # the scan reports an unexpected failure rather than raising it
            self.add("scan.check_skipped", "minor", label, {"check": name, "error": f"{type(exc).__name__}: {exc}"},
                     "Is this entity's data outside what the check expects? Decide whether the other checks for it can be trusted.",
                     "Run the failing check on this entity's series alone to see the full traceback.")
        return None

    # -- entity entry point -------------------------------------------------------------------------------------------

    def entity(self, label: str, s: dict, raw_t: np.ndarray, analysed: bool) -> None:
        ut, vs = s["ut"], s["vs"]
        prof: dict[str, Any] = {"rows": int(raw_t.size), "stamps": int(ut.size), "first": _ts(ut[0]), "last": _ts(ut[-1])}
        self._duplicates(label, s, raw_t)
        grid = _grid(ut, self.off) if self.off is not None else None
        if grid is not None:
            prof["grid_steps"] = grid["n"]
            self._spacing(label, ut, grid, prof)
        if self.step is not None and self.step <= _HOUR_NS and _DAY_NS % self.step == 0:
            self._clock_changes(label, raw_t)
            self._time_of_day(label, ut)
        self._values(label, ut, vs, prof)
        ok = ~np.isnan(vs)
        if ok.any():
            self.volume[label] = float(np.clip(vs[ok], 0.0, None).sum())
            self.medians[label] = float(np.median(vs[ok]))
        if analysed:
            if grid is not None:
                self._statistics(label, grid, vs, prof)
            else:
                prof.setdefault("skipped", {})["grid"] = "no grain could be determined"
            self.profiles[label] = prof
            self.series[label] = (ut, vs)

    # -- structure (every entity) -------------------------------------------------------------------------------------

    def _duplicates(self, label: str, s: dict, raw_t: np.ndarray) -> None:
        extra = int(raw_t.size - s["ut"].size)
        if not extra:
            return
        conflicting = int(s["conflict"].sum())
        repeated = s["ut"][s["rows"] > 1]
        self.add(
            "time.duplicate_timestamps", "blocker" if conflicting else "major", label,
            {"extra_rows": extra, "repeated_stamps": int((s["rows"] > 1).sum()),
             "conflicting_stamps": conflicting, "examples": [_ts(x) for x in repeated[:3]]},
            "For each repeated stamp, which row is right: keep the first, keep the last, or sum them? Conflicting values "
            "need the source system before any row is dropped.",
            f"dup = df[df.duplicated(subset={self.key_hint}, keep=False)]; print(dup.sort_values({self.key_hint}).head(20))",
        )

    def _spacing(self, label: str, ut: np.ndarray, grid: dict, prof: dict) -> None:
        on = grid["on"]
        off_n = int((~on).sum())
        prof["off_grid_stamps"] = off_n
        if off_n:
            share = off_n / ut.size
            self.add(
                "time.irregular_spacing", "major" if share > _OFF_GRID_MAJOR else "minor", label,
                {"off_grid_stamps": off_n, "share": share, "grain": self.off.freqstr,
                 "examples": [_ts(x) for x in ut[~on][:5]]},
                "Are the off-grid stamps a time-of-day or timezone offset, a late feed, or a different grain? Decide "
                "before any lag or calendar feature is built on them.",
                f"print(pd.to_datetime(df[{self.time_name!r}]).dt.time.value_counts().head(10))",
            )
        gaps = _gap_stats(grid["pos"][on], grid["n"])
        prof.update({"gap_runs": gaps["runs"], "missing_steps": gaps["missing"], "gap_share": gaps["share"]})
        if gaps["missing"] >= 10 and self.step == _DAY_NS:
            present = np.zeros(grid["n"], dtype=bool)
            present[grid["pos"][on]] = True
            absent = np.flatnonzero(~present)
            days = np.array([grid["label"](k).day_of_week for k in absent])
            # A weekday-only calendar is one where (nearly) every Saturday and Sunday in range has no record. Judged on the
            # weekend days themselves, so a long outage on weekdays does not hide it.
            all_days = np.array([grid["label"](k).day_of_week for k in range(grid["n"])])
            weekend_total = int((all_days >= 5).sum())
            weekend = float((days >= 5).sum() / weekend_total) if weekend_total else 0.0
            if weekend >= _WEEKEND_GAP_SHARE:
                other = int((days < 5).sum())
                prof["business_day_calendar"] = True
                self.add(
                    "calendar.business_days", "info", label,
                    {"missing_steps": gaps["missing"], "weekend_days_without_record": weekend, "other_missing_weekdays": other,
                     "grid_steps": grid["n"]},
                    "The series is only recorded on weekdays (a trading or working calendar), not missing weekends. "
                    "Forecast in business days: count the horizon in trading days, do not fill weekends, and treat the "
                    "weekday gaps (holidays) as closed days rather than data loss.",
                    "df = df.set_index(time_col).asfreq('B')  # business-day grid; holidays stay as gaps",
                )
                gaps = {**gaps, "missing": other, "share": other / grid["n"], "runs": other}
        if gaps["missing"]:
            start = grid["label"](gaps["longest_start"]) if gaps["longest_start"] is not None else None
            self.add(
                "gaps.missing", "major" if gaps["share"] > _GAP_SHARE_MAJOR else "minor", label,
                {"runs": gaps["runs"], "missing_steps": gaps["missing"], "grid_steps": grid["n"],
                 "share": gaps["share"], "longest_steps": gaps["longest"], "longest_start": start},
                "Are the gaps outages (flag them and treat as censored), real zero-activity periods (fill with zero), or "
                "missing records (ask the data owner)?",
                f"print(pd.date_range(first, last, freq={self.off.freqstr!r}).difference(pd.to_datetime(stamps)))  # per entity",
            )

    def _clock_changes(self, label: str, raw_t: np.ndarray) -> None:
        """Days whose stamp count differs from the typical day by exactly one hour of steps: a clock change candidate."""
        sph = _HOUR_NS // self.step  # steps per hour
        day = raw_t // _DAY_NS
        days, counts = np.unique(day, return_counts=True)
        if days.size < 5:
            return
        d_in, c_in = days[1:-1], counts[1:-1]  # the first and last day are partial, so they are excluded
        vals, nvals = np.unique(c_in, return_counts=True)
        expected = int(vals[np.argmax(nvals)])
        lost = d_in[c_in == expected - sph]
        gained = d_in[c_in == expected + sph]
        if not lost.size and not gained.size:
            return
        flagged = sorted([(int(d), "missing_hour") for d in lost] + [(int(d), "repeated_hour") for d in gained])
        examples = []
        for d, kind in flagged[:5]:
            per_hour = np.bincount((raw_t[day == d] % _DAY_NS) // _HOUR_NS, minlength=24)
            present = np.flatnonzero(per_hour)
            absent = [h for h in range(int(present[0]), int(present[-1]) + 1) if per_hour[h] == 0] if present.size else []
            examples.append({
                "date": str(pd.Timestamp(d * _DAY_NS).date()), "kind": kind, "stamps": int(per_hour.sum()),
                "typical_day_stamps": expected,
                "hours_repeated": [int(h) for h in np.flatnonzero(per_hour > sph)],
                "hours_absent": [int(h) for h in absent],
            })
        dates = [e["date"] for e in examples]
        self.add(
            "time.dst_shift", "major", label,
            {"candidate_days": len(flagged), "missing_hour_days": int(lost.size), "repeated_hour_days": int(gained.size),
             "examples": examples,
             "rule": "stamp count differs from the typical day by exactly one hour of steps"},
            "Are these real clock changes? If yes, convert the stamps to one timezone without daylight saving (or to UTC) "
            "before modelling. A repeated hour is two different demands and cannot share one stamp.",
            f"print(df[pd.to_datetime(df[{self.time_name!r}]).dt.strftime('%Y-%m-%d').isin({dates})].head(30))",
        )

    def _time_of_day(self, label: str, ut: np.ndarray) -> None:
        """A shift of the first stamp of the day, by one hour or more between consecutive months: a timezone change."""
        days, first = np.unique(ut // _DAY_NS, return_index=True)
        if days.size < 2 * _TOD_MIN_DAYS:
            return
        tod = (ut % _DAY_NS)[first]
        months = pd.DatetimeIndex(pd.to_datetime(days, unit="D")).to_period("M")
        modes: dict = {}
        for month in pd.unique(months):
            sel = months == month
            if int(sel.sum()) < _TOD_MIN_DAYS:
                continue
            vals, cnt = np.unique(tod[sel], return_counts=True)
            modes[month] = int(vals[np.argmax(cnt)])
        ordered = sorted(modes)
        shifts = []
        for a, b in zip(ordered[:-1], ordered[1:]):
            if b.ordinal - a.ordinal != 1:
                continue
            delta = modes[b] - modes[a]
            if abs(delta) >= _HOUR_NS:
                shifts.append({"from_month": str(a), "to_month": str(b), "shift_hours": delta / _HOUR_NS,
                               "whole_hours": delta % _HOUR_NS == 0})
        if shifts:
            self.add(
                "time.timezone_shift", "major", label,
                {"shifts": shifts[:5], "count": len(shifts),
                 "rule": "modal time of the first stamp per day, compared across consecutive months"},
                "Did the source change its timezone or clock setting in this month? Convert all stamps to one timezone "
                "before modelling, or split the series at the change.",
                "print(pd.to_datetime(df[{!r}]).dt.floor('D').value_counts().sort_index().head())".format(self.time_name),
            )

    def _values(self, label: str, ut: np.ndarray, vs: np.ndarray, prof: dict) -> None:
        nan_n = int(np.isnan(vs).sum())
        if nan_n:
            share = nan_n / ut.size
            self.add(
                "values.missing", "major" if share > _MISSING_VALUE_MAJOR else "minor", label,
                {"missing_values": nan_n, "stamps": int(ut.size), "share": share},
                "Are the missing values unrecorded (keep them as gaps) or unknown quantities (ask the owner)? Do not fill "
                "them silently.",
                f"print(df.loc[df[{self.value_name!r}].isna()].head(20))",
            )
        negative = int(np.sum(vs < 0))
        if negative:
            self.add(
                "values.negative", "minor", label, {"negative_values": negative, "min": float(np.nanmin(vs))},
                "Are the negative values returns, corrections or entry errors? Decide whether they belong in demand.",
                f"print(df.loc[df[{self.value_name!r}] < 0].head(20))",
            )
        ok = ~np.isnan(vs)
        self._zeros(label, ut[ok], vs[ok], prof)

    def _zeros(self, label: str, ut_ok: np.ndarray, vv: np.ndarray, prof: dict) -> None:
        if vv.size == 0:
            return
        zero = vv == 0
        prof["zero_share_stamps"] = float(zero.mean())
        if zero.all():
            self.add("coverage.all_zero", "major", label, {"stamps": int(vv.size)},
                     "The entity has no demand at all. Is it inactive, unlisted, or a feed that never reported?",
                     "Exclude the entity only after the owner confirms it is inactive, and log the reason.")
            return
        nz = np.flatnonzero(~zero)
        lead, trail = int(nz[0]), int(vv.size - 1 - nz[-1])
        if lead:
            self.add(
                "coverage.leading_zeros", "major" if lead >= _DEAD_RUN_MIN else "minor", label,
                {"zeros": lead, "stamps": int(vv.size), "until": _ts(ut_ok[lead - 1])},
                "Are these zeros before launch (trim them) or a feed that had no records yet (ask the owner)?",
                "Trim the leading run only after the owner confirms it predates launch, and log the trim.",
            )
        if trail:
            self.add(
                "coverage.trailing_zeros", "major" if trail >= _DEAD_RUN_MIN else "minor", label,
                {"zeros": trail, "stamps": int(vv.size), "from": _ts(ut_ok[vv.size - trail])},
                "Did the product end here (drop the tail and log it) or did the feed stop (treat as censored)?",
                "Confirm the end date with the owner before trimming the trailing run.",
            )
        interior = [(s, length) for s, length in _true_runs(zero)
                    if s > 0 and s + length < vv.size and length >= _DEAD_RUN_MIN]
        if interior:
            longest = max(length for _, length in interior)
            self.add(
                "coverage.dead_segment", "major", label,
                {"segments": len(interior), "longest_zeros": longest,
                 "examples": [{"from": _ts(ut_ok[s]), "zeros": int(length)} for s, length in interior[:5]]},
                "Are the dead segments outages (censored, not zero demand) or real zero demand? Outages must not be learned as zeros.",
                "Cross-check the dead segments against outage or stock records before any model sees them.",
            )

    # -- statistics (analysed entities only) --------------------------------------------------------------------------

    def _statistics(self, label: str, grid: dict, vs: np.ndarray, prof: dict) -> None:
        pos, on, n_grid = grid["pos"], grid["on"], grid["n"]
        arr = np.full(n_grid, np.nan)
        arr[pos[on]] = vs[on]  # off-grid stamps are reported in the structure checks and left out of the series here
        obs = ~np.isnan(arr)
        n_obs = int(obs.sum())
        prof["observed_steps"] = n_obs
        prof["periods"] = int(n_grid)
        prof["observed_share"] = round(n_obs / n_grid, 4) if n_grid else None
        if n_obs < 2:
            prof.setdefault("skipped", {})["statistics"] = "fewer than two observed steps on the grid"
            return
        p0 = self.periods[0] if self.periods else None
        label_at = grid["label"]
        resid = _remove_profile(arr, p0) if p0 and n_obs >= 2 * p0 else arr

        self._outliers(label, arr, resid, n_obs, p0, label_at, prof)
        self._breaks(label, arr, resid, n_obs, p0, label_at, prof)
        self._censoring(label, arr, obs, label_at, prof)
        self._intermittency(label, arr, obs, n_obs, prof)
        verdict = self._try(label, "stationarity", lambda: ps.stationarity(arr), prof)
        if verdict:
            prof["stationarity"] = verdict["verdict"]
            if verdict["verdict"] == "unit_root":
                self.add("stationarity.verdict", "major", label,
                         {"verdict": "unit_root", "adf_p": verdict["adf_p"], "kpss_level_p": verdict["kpss_level_p"],
                          "kpss_trend_p": verdict["kpss_trend_p"], "n": verdict["n"]},
                         "Model the differences (d = 1) or the levels? A unit root makes long forecasts wander, so the "
                         "horizon decides it.",
                         "Plot the series with its differences, then choose d with the backtest, not by eye.")
            elif verdict["verdict"] == "inconclusive":
                self.add("stationarity.verdict", "minor", label,
                         {"verdict": "inconclusive", "adf_p": verdict["adf_p"], "kpss_level_p": verdict["kpss_level_p"],
                          "kpss_trend_p": verdict["kpss_trend_p"], "n": verdict["n"]},
                         "The ADF and KPSS tests disagree. Treat the series as non-stationary, or ask for a longer history?",
                         "Compare the seasonal and trend strengths in the profile before choosing a model class.")
        self._seasonality(label, arr, n_obs, prof)
        peak = _acf_peak(arr)
        prof["acf_peak"] = list(peak) if peak else None
        if peak and not any(_is_cycle(peak[0], p) for p in self.periods):
            self.add("seasonality.acf_peak", "minor", label,
                     {"lag": peak[0], "acf": peak[1], "candidate_periods": self.periods},
                     f"Is lag {peak[0]} a real cycle the forecast must cover? If yes, set it as the period; if not, record "
                     "why it is ignored.",
                     "Plot the series around that lag before choosing a period.")
        cyc = p0 or (peak[0] if peak else None)
        active = np.flatnonzero(obs & (arr != 0))
        if active.size and cyc:
            span = int(active[-1] - active[0] + 1)
            prof["active_cycles"] = round(span / cyc, 2)
            if span < _MIN_CYCLES * cyc:
                self.add("coverage.short_history", "major", label,
                         {"active_steps": span, "cycle_steps": cyc, "active_cycles": round(span / cyc, 2),
                          "minimum_cycles": _MIN_CYCLES},
                         "Forecast at this grain, or aggregate to a coarser one? Two cycles of active history is the "
                         "minimum for a seasonal model (first-contact-profile, step 5).",
                         "Check the active span with the coverage fields in the profile before modelling.")

    def _outliers(self, label: str, arr: np.ndarray, resid: np.ndarray, n_obs: int, p0: int | None, label_at: Any,
                  prof: dict) -> None:
        shares: dict[str, float] = {}
        examples: list[str] = []
        # Hampel runs on the seasonal residual: a local median over a weekly cycle would flag every weekly peak.
        res = self._try(label, "hampel", lambda: ps.hampel(resid, window=_HAMPEL_WINDOW, n_sigmas=_HAMPEL_SIGMAS,
                                                          causal=False), prof)
        if res is not None:
            flags = np.asarray(res[0], dtype=bool)
            shares["hampel"] = float(flags.sum()) / n_obs
            examples = [str(label_at(k)) for k in np.flatnonzero(flags)[:5]]
        if p0:
            res = self._try(label, "stl_outliers", lambda: ps.stl_outliers(arr, period=p0, k=_STL_K), prof)
            if res is not None:
                shares["stl_remainder"] = float(np.sum(np.asarray(res[0], dtype=bool))) / n_obs
        if not shares:
            return
        prof["outlier_share"] = shares
        if "hampel" not in shares:
            return
        worst = shares["hampel"]
        if worst > _OUTLIER_MINOR:
            self.add(
                "outliers.share", "major" if worst > _OUTLIER_MAJOR else "minor", label,
                {"share_by_method": shares, "severity_uses": "hampel", "observed_steps": n_obs,
                 "hampel_window": _HAMPEL_WINDOW, "hampel_sigmas": _HAMPEL_SIGMAS, "stl_period": p0,
                 "hampel_examples": examples},
                "Are the flagged points entry errors (correct or drop, with a log) or real events (keep, and mark them as "
                "known)? A spike can be a promotion, which a model must see.",
                "Show the flagged steps with their neighbours before any treatment is chosen.",
            )

    def _breaks(self, label: str, arr: np.ndarray, resid: np.ndarray, n_obs: int, p0: int | None, label_at: Any,
                prof: dict) -> None:
        res = self._try(label, "changepoints", lambda: ps.changepoints(resid, min_size=_CP_MIN_SIZE), prof)
        if res is None:
            return
        changes = res[0]
        prof["changepoints"] = len(changes)
        if not changes:
            return
        rows, worst = [], "minor"
        rank = {"minor": 0, "major": 1, "blocker": 2}
        for c in changes:
            idx = int(c["index"])
            before, after = _median(arr[:idx]), _median(arr[idx:])
            ratio = None
            if before and after and before > 0 and after > 0:
                ratio = max(after / before, before / after)
            sd = c["shift_in_sd"]
            if ratio is not None and ratio >= _UNIT_RATIO:
                sev = "blocker"
            elif sd is not None and abs(sd) >= _CP_SHIFT_SD:
                sev = "major"
            else:
                sev = "minor"
            if rank[sev] > rank[worst]:
                worst = sev
            rows.append({"at": str(label_at(idx)), "shift_in_sd": sd, "median_before": before,
                         "median_after": after, "median_ratio": ratio, "severity": sev})
        self.add(
            "changepoints.level_shift", worst, label,
            {"breaks": rows[:5], "count": len(rows), "search": "binary segmentation on the seasonal residual"
             if p0 and n_obs >= 2 * p0 else "binary segmentation on the raw series"},
            "For each break: is there a known event (launch, price, supply, unit or system change) on that date? A median "
            "jump of ten times or more with no event is a unit change and must be resolved first.",
            "Split the history at the break only after the event is recorded; never fit the break on the full series in a backtest.",
        )

    def _censoring(self, label: str, arr: np.ndarray, obs: np.ndarray, label_at: Any, prof: dict) -> None:
        cen = self._try(label, "censoring_check", lambda: ps.censoring_check(arr), prof)
        if cen is None:
            return
        stock = cen["probable_stockouts"]
        if stock:
            longest = max(stock, key=lambda r: r["length"])
            self.add(
                "censoring.stockout_runs", "major", label,
                {"runs": len(stock), "longest": longest["length"],
                 "examples": [{"from": str(label_at(r["start"])), "length": r["length"]} for r in stock[:5]]},
                "Are these zero runs stockouts or outages (censored: the lost demand is missing) or real zero periods?",
                "Check stock, outage or opening records for the runs before treating them as demand.",
            )
        if cen["floor_plateau"] or cen["ceiling_plateau"]:
            self.add(
                "censoring.plateau", "major", label,
                {"floor": cen["floor_plateau"], "ceiling": cen["ceiling_plateau"]},
                "Is the repeated floor or cap a system limit (censored, so true demand is higher) or a real value?",
                "Compare the plateau with the capacity or price limits in the source system.",
            )
        idx_nz = np.flatnonzero(obs & (arr != 0))
        if idx_nz.size > 1:
            x = arr[idx_nz]
            runs = _true_runs(np.diff(x) == 0)
            if runs:
                start, length = max(runs, key=lambda r: r[1])
                if length + 1 >= _REPEAT_RUN_MIN:
                    self.add(
                        "censoring.repeated_values", "minor", label,
                        {"run_length": length + 1, "value": float(x[start]), "from": str(label_at(idx_nz[start]))},
                        "Is the repeated value a cap, a default, or a copied figure? Integer demand can repeat naturally, so "
                        "check before treating it as censored.",
                        "Print the run with its neighbours to see whether the value is a default.",
                    )

    def _intermittency(self, label: str, arr: np.ndarray, obs: np.ndarray, n_obs: int, prof: dict) -> None:
        vals = arr[obs]
        nz = vals[vals != 0]
        zero_share = 1.0 - nz.size / n_obs
        prof["zero_share"] = zero_share
        if nz.size == 0:
            prof["demand_class"] = "all_zero"
            return
        adi = n_obs / nz.size
        mean = float(nz.mean())
        cv2 = float(np.var(nz) / mean**2) if mean != 0 else None
        cls = _demand_class(adi, cv2)
        prof.update({"adi": adi, "cv2": cv2, "demand_class": cls})
        if cls in ("intermittent", "lumpy"):
            self.add(
                "intermittency.class", "major", label,
                {"class": cls, "adi": adi, "cv2": cv2, "zero_share": zero_share,
                 "cutoffs": {"adi": _ADI_CUTOFF, "cv2": _CV2_CUTOFF}},
                "Model with an intermittent method (Croston or TSB), or aggregate to a coarser grain? A normal model "
                "fitted to these zeros forecasts a level that never occurs.",
                "Compare the intermittent methods on the backtest, using the same origins for each.",
            )
        elif cls == "erratic":
            self.add(
                "intermittency.class", "minor", label,
                {"class": cls, "adi": adi, "cv2": cv2, "zero_share": zero_share},
                "Is the size variation real, or caused by outliers or a unit mix? Decide before choosing a noise model.",
                "Compare the outlier share in the profile with the size distribution.",
            )
        if zero_share > _ZERO_INFLATED:
            self.add(
                "intermittency.zero_inflation", "minor", label,
                {"zero_share": zero_share, "threshold": _ZERO_INFLATED},
                "Are the zeros sales (intermittent demand) or missing records (censored)? The answer changes the model.",
                "Check the zeros against stock and opening records.",
            )

    def _seasonality(self, label: str, arr: np.ndarray, n_obs: int, prof: dict) -> None:
        strengths: dict[str, dict] = {}
        for p in self.periods:
            if n_obs < 2 * p:
                prof.setdefault("skipped", {})[f"seasonal_{p}"] = f"needs {2 * p} observed steps, has {n_obs}"
                continue
            ss = self._try(label, f"seasonal_strength_{p}", lambda p=p: ps.seasonal_strength(arr, p), prof)
            if ss is None:
                continue
            strengths[str(p)] = {"seasonal": ss["seasonal_strength"], "trend": ss["trend_strength"]}
        if strengths:
            prof["seasonal_strength"] = strengths

    # -- cross-entity (after every entity) ----------------------------------------------------------------------------

    def cross_entity(self) -> None:
        self._scale()
        self._concentration()
        self._heterogeneity()
        self._neighbours()
        self._strength_summary()

    def _strength_summary(self) -> None:
        """One info line per candidate period: how many analysed entities carry a strong season or trend.

        Per-entity values are in the profiles. A strong season is a modelling input, not a defect, so it is not listed
        entity by entity.
        """
        for p in self.periods:
            key = str(p)
            pairs = [prof["seasonal_strength"][key] for prof in self.profiles.values()
                     if key in prof.get("seasonal_strength", {})]
            if not pairs:
                continue
            seas = [x["seasonal"] for x in pairs]
            trend = [x["trend"] for x in pairs]
            self.add(
                "seasonality.strength", "info", "all",
                {"period_steps": p, "entities_seasonal_at_least": _SEASON_MIN,
                 "entities_seasonal": int(sum(x >= _SEASON_MIN for x in seas)), "analysed": len(seas),
                 "median_seasonal_strength": float(np.median(seas)), "median_trend_strength": float(np.median(trend)),
                 "entities_trend_at_least": _TREND_STRONG, "entities_trend": int(sum(x >= _TREND_STRONG for x in trend))},
                None,
                f"Confirm period {p} with a seasonal naive baseline on the backtest; per-entity values are in profiles.",
            )

    def _scale(self) -> None:
        meds = sorted((m, lab) for lab, m in self.medians.items() if m > 0)
        if len(meds) < 2:
            return
        logs = np.log10([m for m, _ in meds])
        spread = float(logs[-1] - logs[0])
        if spread >= _SCALE_SPREAD_MINOR:
            self.add(
                "scale.spread", "major" if spread >= _SCALE_SPREAD_MAJOR else "minor", "all",
                {"range_log10": spread, "min_median": meds[0][0], "max_median": meds[-1][0],
                 "entities": len(meds), "lowest": meds[0][1], "highest": meds[-1][1]},
                "Do entities of very different size share one model? Decide whether to pool them, scale them, or model "
                "them apart.",
                "Compare the per-entity medians in the profiles before pooling.",
            )
        gaps = np.diff(logs)
        k = int(np.argmax(gaps))
        gap = float(gaps[k])
        low, high = k + 1, logs.size - (k + 1)
        if gap >= _UNIT_GAP_LOG10 and min(low, high) >= max(2, _UNIT_SIDE_MIN_SHARE * logs.size):
            self.add(
                "scale.unit_mix", "major", "all",
                {"gap_log10": gap, "low_group": low, "high_group": high,
                 "low_examples": [lab for _, lab in meds[:low][:5]], "high_examples": [lab for _, lab in meds[low:][:5]]},
                "Are the two groups in different units (for example units and thousands, cases and eaches)? Convert to one "
                "unit, or model the groups apart. Pooling across units is a silent error.",
                f"print(df.groupby({self.entity_hint})[{self.value_name!r}].median().sort_values())  # confirm the unit per group",
            )

    def _concentration(self) -> None:
        labs = list(self.volume)
        vols = np.array([self.volume[lab] for lab in labs], dtype=float)
        n = vols.size
        total = float(vols.sum())
        if n < 2 or total <= 0:
            return
        order = np.argsort(-vols, kind="stable")
        k = max(1, math.ceil(_CONC_TOP_SHARE * n))
        top_share = float(vols[order[:k]].sum() / total)
        if top_share >= _CONC_MINOR:
            self.add(
                "concentration.top_share", "major" if top_share >= _CONC_MAJOR else "minor", "all",
                {"top_entities": k, "share_of_entities": _CONC_TOP_SHARE, "top_volume_share": top_share,
                 "largest": labs[int(order[0])], "largest_volume_share": float(vols[order[0]] / total)},
                "Do a few entities carry the volume? If yes, give the top entities their own models and pool the tail, "
                "and say so in the brief.",
                f"print(df.groupby({self.entity_hint})[{self.value_name!r}].sum().sort_values(ascending=False).head(20))",
            )
        tail_share = float(np.mean(vols / total < _TAIL_VOLUME_SHARE))
        if tail_share >= _TAIL_ENTITY_SHARE:
            self.add(
                "concentration.long_tail", "minor", "all",
                {"tail_entities": int(round(tail_share * n)), "share_of_entities": tail_share,
                 "volume_threshold": _TAIL_VOLUME_SHARE},
                "Model the long tail as one pooled group, or drop it with a logged reason? Most entities carry little volume.",
                "Check whether the tail entities are active enough to forecast at all (see coverage findings).",
            )

    def _heterogeneity(self) -> None:
        groups = []
        for ut, vs in self.series.values():
            ok = vs[~np.isnan(vs)]
            med = float(np.median(ok)) if ok.size else 0.0
            if ok.size >= 3 and med > 0:
                # Absolute deviation from each entity's own median removes scale and keeps spread, skew and zero-heavy
                # shape. Ranking the scaled values alone would miss these: after median scaling, groups share a centre.
                groups.append(np.abs(ok / med - 1.0))
        if len(groups) < 2:
            return
        try:
            res = _stats.kruskal(*groups)
        except ValueError:  # every value identical: nothing to compare
            return
        h, p = float(res.statistic), float(res.pvalue)
        n_total = int(sum(g.size for g in groups))
        k = len(groups)
        eta2 = max(0.0, (h - k + 1) / (n_total - k)) if n_total > k else 0.0
        if eta2 >= _ETA2_MINOR:
            self.add(
                "heterogeneity.between_entities", "major" if eta2 >= _ETA2_MAJOR else "minor", "all",
                {"kruskal_h": h, "p_value": p, "eta_squared": eta2, "groups": k, "values": n_total,
                 "measure": "absolute deviation from each entity's median", "analysed_entities_only": True},
                "Do the entities share one pattern once scale is removed? If not, pool them with a group effect or model "
                "them separately, and say which in the brief.",
                "Plot the scaled series of the most different entities side by side before pooling.",
            )

    def _neighbours(self) -> None:
        if len(self.series) < 3:
            return
        stamps = np.unique(np.concatenate([ut for ut, _ in self.series.values()]))
        window = stamps[-_NEIGH_WINDOW:]
        frames = []
        for lab, (ut, vs) in self.series.items():
            ok = ~np.isnan(vs) & np.isin(ut, window)
            if int(ok.sum()) >= _NEIGH_MIN_OVERLAP:
                frames.append(pd.DataFrame({"t": ut[ok], "e": lab, "v": vs[ok]}))
        if len(frames) < 3:
            return
        wide = pd.concat(frames).pivot_table(index="t", columns="e", values="v", aggfunc="mean")
        wide = wide.reindex(window)
        corr = wide.corr(min_periods=_NEIGH_MIN_OVERLAP)
        for lab in corr.columns:
            row = corr[lab].drop(labels=[lab], errors="ignore").dropna().sort_values(ascending=False)
            if row.empty:
                continue
            nbrs = [{"entity": str(k), "corr": float(v)} for k, v in row.head(_NEIGH_TOP_K).items()]
            self.neighbours[str(lab)] = nbrs
            own = self.profiles.get(str(lab), {})
            short = own.get("active_cycles") is not None and own["active_cycles"] < _BORROW_CYCLES
            gappy = (own.get("gap_share") or 0) > 0
            if nbrs[0]["corr"] >= _NEIGH_CORR and (short or gappy):
                self.add(
                    "neighbours.borrowable", "minor", str(lab),
                    {"neighbours": nbrs, "window_steps": int(len(window)), "min_overlap": _NEIGH_MIN_OVERLAP,
                     "rule": "Pearson correlation on the last window; equal to KNN on z-scored series"},
                    "Borrow history from the nearest neighbour for a cold start or a sparse entity, or model this entity alone?",
                    "Compare the two series on their overlap (level, units, timing) before pooling them.",
                )


# ---------------------------------------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------------------------------------

def _check_arguments(df: Any, time_col: Any, value_col: Any, entity_cols: Any, max_entities: Any) -> tuple[list, list]:
    """Return (entity column names, blocker findings). Never raises."""
    if not isinstance(df, pd.DataFrame):
        return [], [_finding("schema.not_a_table", "blocker", "all", {"got": type(df).__name__},
                             "Pass the data as a pandas DataFrame.",
                             "Load the file with pandas and call sc.scan(df, time_col, value_col, ...) again.")]
    if entity_cols is None:
        ents: list = []
    elif isinstance(entity_cols, str):
        ents = [entity_cols]
    elif isinstance(entity_cols, (list, tuple)):
        ents = list(entity_cols)
    else:
        return [], [_finding("schema.bad_argument", "blocker", "all",
                             {"entity_cols": type(entity_cols).__name__, "expected": "a column name or a list"},
                             "Pass the entity columns as a list of column names, or None.",
                             "Call sc.scan(df, time_col, value_col, entity_cols=['sku']).")]
    wanted = [time_col, value_col] + ents
    problems = []
    available = [str(c) for c in df.columns][:40]
    for col in wanted:
        try:
            present = col in df.columns
        except TypeError:
            present = False
        if not present:
            problems.append(_finding("schema.missing_column", "blocker", str(col), {"available_columns": available},
                                     f"Column {col!r} is not in the table. Confirm the name before any analysis.",
                                     "print(df.columns.tolist())"))
    if not problems and len(set(map(str, wanted))) != len(wanted):
        problems.append(_finding("schema.bad_argument", "blocker", "all", {"wanted": [str(c) for c in wanted]},
                                 "The time, value and entity roles must name different columns.",
                                 "Pass each role a different column."))
    if not problems and df.columns.duplicated().any():
        dupes = [str(c) for c in df.columns[df.columns.duplicated()]]
        if any(str(c) in dupes for c in wanted):
            problems.append(_finding("schema.duplicate_column", "blocker", "all", {"duplicated_names": dupes},
                                     "A column name appears twice, so its values cannot be read unambiguously.",
                                     "Rename the duplicated columns, then rerun the scan."))
    if not (isinstance(max_entities, (int, np.integer)) and not isinstance(max_entities, bool) and max_entities >= 1):
        problems.append(_finding("schema.bad_argument", "blocker", "all", {"max_entities": str(max_entities)},
                                 "max_entities must be a whole number of at least 1.", "Pass max_entities=200 or similar."))
    return ents, problems


def _run(df: Any, time_col: Any, value_col: Any, entity_cols: Any, freq: Any, max_entities: Any) -> dict:
    ents, problems = _check_arguments(df, time_col, value_col, entity_cols, max_entities)
    rows_in = len(df) if isinstance(df, pd.DataFrame) else None
    base = {"rows_in": rows_in, "time_col": str(time_col), "value_col": str(value_col), "entity_cols": ents,
            "freq_given": None if freq is None else str(freq)}
    if problems:
        return _result(problems, base)

    off_given = None
    if freq is not None:
        try:
            off_given = to_offset(freq)
        except (ValueError, TypeError) as exc:
            return _result([_finding(
                "schema.bad_frequency", "blocker", "all", {"freq": str(freq), "reason": str(exc)[:300]},
                "Give a valid pandas frequency alias (for example D, h, W-SUN or MS), or leave freq empty to infer it.",
                "Call sc.scan(df, ..., freq=None) to infer the grain, then confirm it.")], base)

    findings: list[dict] = []
    t_raw = df[time_col]
    order_finding = None
    try:
        if pd.api.types.is_datetime64_any_dtype(t_raw):
            t_parsed = t_raw
        elif pd.api.types.is_numeric_dtype(t_raw):
            # Epoch numbers: seconds for the usual range (1e8 to 1e11), otherwise pandas' own default. Audit: 1704067200 read as ns.
            med = float(pd.Series(t_raw).dropna().median()) if t_raw.notna().any() else 0.0
            unit = "s" if 1e8 <= med <= 1e11 else ("ms" if 1e11 < med <= 1e14 else "ns")
            t_parsed = pd.to_datetime(t_raw, errors="coerce", unit=unit)
        else:
            dayfirst, order_finding = _day_order(t_raw, time_col)
            t_parsed = pd.to_datetime(t_raw, errors="coerce", format="mixed", dayfirst=dayfirst)
    except (ValueError, TypeError, OverflowError) as exc:
        findings.append(_finding(
            "schema.unparseable_time", "blocker", "all", {"reason": str(exc)[:300]},
            "The time column cannot be read. Confirm its format and timezone with the data owner.",
            f"print(df[{time_col!r}].head(20))"))
        return _result(findings, base)

    tz_name = None
    if isinstance(t_parsed.dtype, pd.DatetimeTZDtype):
        tz_name = str(t_parsed.dtype.tz)
        t_parsed = t_parsed.dt.tz_localize(None)
        findings.append(_finding(
            "time.timezone_aware", "info", "all", {"timezone": tz_name},
            None, "The scan reads local wall-clock times. Confirm the timezone is the one the brief uses."))
    if order_finding is not None:
        findings.append(order_finding)
    t_nat = t_parsed.isna().to_numpy()
    t_ns = t_parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    n_rows = len(df)
    unreadable_t = t_nat & t_raw.notna().to_numpy()
    if unreadable_t.any():
        share = float(unreadable_t.sum()) / n_rows
        findings.append(_finding(
            "schema.unparseable_time", "blocker" if share > _PARSE_FAIL_BLOCK else "major", "all",
            {"rows": int(unreadable_t.sum()), "share": share,
             "examples": [str(x) for x in t_raw[unreadable_t].head(5)]},
            "Which reading of these time values is right, or should they be dropped? Ask the owner before dropping rows.",
            f"print(df.loc[pd.to_datetime(df[{time_col!r}], errors='coerce').isna(), {time_col!r}].head(20))"))

    v_raw = df[value_col]
    try:
        v_num = v_raw if pd.api.types.is_numeric_dtype(v_raw) else pd.to_numeric(v_raw, errors="coerce")
    except (ValueError, TypeError) as exc:
        return _result(findings + [_finding(
            "schema.non_numeric_value", "blocker", "all", {"reason": str(exc)[:300]},
            "The value column holds text that cannot be read as numbers. Fix the parse before anything else.",
            f"print(df[{value_col!r}].map(type).value_counts())")], base)
    v_all = v_num.to_numpy(dtype=float, na_value=np.nan)
    unreadable_v = np.isnan(v_all) & v_raw.notna().to_numpy()
    if unreadable_v.any():
        share = float(unreadable_v.sum()) / n_rows
        findings.append(_finding(
            "schema.non_numeric_value", "blocker" if share > _PARSE_FAIL_BLOCK else "major", "all",
            {"rows": int(unreadable_v.sum()), "share": share,
             "examples": [str(x) for x in v_raw[unreadable_v].head(5)]},
            "Are these cells text (a note, a thousands separator, a unit label) or errors? Fix the parse before modelling.",
            f"print(df.loc[pd.to_numeric(df[{value_col!r}], errors='coerce').isna() & df[{value_col!r}].notna(), {value_col!r}].head(20))"))
    infinite = np.isinf(v_all)
    if infinite.any():
        findings.append(_finding(
            "values.non_finite", "major", "all", {"rows": int(infinite.sum())},
            "Are the infinite values division errors or sentinels? They are treated as missing here; replace or drop them before modelling.",
            f"print(df.loc[np.isinf(pd.to_numeric(df[{value_col!r}], errors='coerce')), {value_col!r}].head(20))"))
    v_all = np.where(infinite, np.nan, v_all)

    if ents:
        missing_entity = df[ents].isna().any(axis=1).to_numpy()
        label_parts = []
        for c in ents:
            col = df[c]
            label_parts.append(col.astype(object).where(col.notna(), "(missing)").map(str).to_numpy(dtype=object))
        labels = label_parts[0]
        for part in label_parts[1:]:
            labels = np.array([f"{a}|{b}" for a, b in zip(labels, part)], dtype=object)
    else:
        missing_entity = np.zeros(n_rows, dtype=bool)
        labels = np.array(["all"] * n_rows, dtype=object)
    if missing_entity.any():
        findings.append(_finding(
            "schema.missing_entity", "major", "all", {"rows": int(missing_entity.sum())},
            "Rows with no entity label are grouped as (missing). Which entity do they belong to?",
            f"print(df.loc[df[{ents!r}].isna().any(axis=1)].head(20))"))

    keep = ~t_nat
    n_used = int(keep.sum())
    if n_used == 0:
        findings.append(_finding(
            "schema.no_valid_rows", "blocker", "all", {"rows_in": n_rows},
            "No row has a readable time. Nothing can be analysed until the time column is fixed.",
            f"print(df[{time_col!r}].head(20))"))
        return _result(findings, base)

    t_k = t_ns[keep]
    v_k = v_all[keep]
    names, codes = np.unique(labels[keep].astype(str), return_inverse=True)
    codes = codes.ravel()
    order = np.lexsort((t_k, codes))
    t_s, v_s, codes_s = t_k[order], v_k[order], codes[order]
    n_ent = int(names.size)
    starts = np.searchsorted(codes_s, np.arange(n_ent), side="left")
    ends = np.searchsorted(codes_s, np.arange(n_ent), side="right")

    structs = [_stamps(t_s[a:b], v_s[a:b]) for a, b in zip(starts, ends)]

    modal_ns = None
    modal_share = None
    deltas = [np.diff(s["ut"]) for s in structs if s["ut"].size > 1]
    if deltas:
        dd = np.concatenate(deltas)
        vals, cnt = np.unique(dd, return_counts=True)
        modal_ns = int(vals[np.argmax(cnt)])
        modal_share = float(cnt.max()) / float(dd.size)
    off, freq_source = off_given, ("given" if off_given is not None else None)
    if off is None and modal_ns is not None:
        off, freq_source = _infer_offset(structs, modal_ns)
    if off is not None:
        findings.append(_finding(
            "time.modal_frequency", "info", "all",
            {"grain": off.freqstr, "source": freq_source,
             "modal_step": str(pd.Timedelta(modal_ns, unit="ns")) if modal_ns is not None else None,
             "share_of_steps_at_modal": modal_share},
            None, None))
    else:
        findings.append(_finding(
            "time.no_spacing", "major", "all", {"entities": n_ent},
            "No entity has two distinct stamps, so no grain can be read. Is this a snapshot rather than a series?",
            f"print(df.groupby({ents!r})[{time_col!r}].nunique().describe())" if ents else
            f"print(df[{time_col!r}].nunique())"))

    sampled = n_ent > max_entities
    if sampled:
        rng = np.random.default_rng(_SAMPLE_SEED)
        chosen = set(int(i) for i in np.sort(rng.choice(n_ent, size=int(max_entities), replace=False)))
        findings.append(_finding(
            "scan.entities_sampled", "info", "all",
            {"entities_total": n_ent, "entities_analysed": len(chosen), "seed": _SAMPLE_SEED},
            None, None))
    else:
        chosen = set(range(n_ent))

    key_hint = repr([*ents, time_col])
    entity_hint = repr(ents)
    battery = _Battery(off, key_hint, str(time_col), str(value_col), entity_hint)
    for e in range(n_ent):
        a, b = int(starts[e]), int(ends[e])
        battery.entity(str(names[e]), structs[e], t_s[a:b], e in chosen)
    battery.cross_entity()

    sample_note = (
        f"{len(chosen)} of {n_ent} entities analysed for statistics (seed {_SAMPLE_SEED}); structural checks ran on all "
        f"{n_ent}. Cross-entity checks use the analysed entities, except volume and scale, which use all."
        if sampled else f"All {n_ent} entities analysed."
    )
    summary = {
        **base,
        "rows_used": n_used,
        "rows_dropped": int(n_rows - n_used),
        "entities_total": n_ent,
        "entities_analysed": len(chosen),
        "sampled": bool(sampled),
        "sample_seed": _SAMPLE_SEED if sampled else None,
        "sample_note": sample_note,
        "grain_used": off.freqstr if off is not None else None,
        "grain_source": freq_source,
        "modal_step": str(pd.Timedelta(modal_ns, unit="ns")) if modal_ns is not None else None,
        "candidate_periods": list(battery.periods),
        "time_range": [_ts(t_s.min()), _ts(t_s.max())],
        "timezone": tz_name,
        "checks_run": list(_CHECKS),
        "neighbours": battery.neighbours,
        "profiles": battery.profiles,
    }
    return _result(findings + battery.findings, summary)


_ENTITY_ALIASES = {name: "entity_cols" for name in
                   ("entity_col", "entity", "entities", "group_col", "group_cols", "id_col", "id_cols", "key_cols")}


_DMY = re.compile(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{2,4}")


def _day_order(t_raw: pd.Series, time_col: str) -> tuple[bool, dict | None]:
    """Day-first or month-first for a text time column. A field above 12 settles it; when every such date has both fields at
    or under 12, the order is ambiguous: month-first is used and a major finding asks the person to confirm it (before this,
    01/02/2024 to 28/02/2024 was read as 2 January to 2 December, with no finding)."""
    fields = t_raw.dropna().astype(str).str.extract(_DMY)
    fields = fields.dropna()
    if fields.empty:
        return False, None
    first = fields[0].astype(int)
    second = fields[1].astype(int)
    if (first > 12).any() and not (second > 12).any():
        return True, None
    if (second > 12).any() and not (first > 12).any():
        return False, None
    if (first > 12).any() and (second > 12).any():
        return False, _finding(
            "time.mixed_day_month", "blocker", "all", {"rows": int(len(fields))},
            "Some dates have the day first and some the month first. Confirm the order before the scan can read the grain.",
            f"print(df[{time_col!r}].head(20))")
    return False, _finding(
        "time.ambiguous_day_month", "major", "all", {"rows": int(len(fields))},
        "Every date has both fields at most 12, so the order cannot be told apart. Month-first is assumed; confirm it with the "
        "data owner, or re-read with the order stated.",
        f"print(df[{time_col!r}].head(20))")


@accepts_aliases({**_ENTITY_ALIASES, "time": "time_col", "value": "value_col"})
def scan(df: Any, time_col: str, value_col: str, entity_cols: Any = None, freq: str | None = None,
         max_entities: int = 200) -> dict:
    """Run the principal-analyst battery on a long table and return typed findings. Never mutates ``df``.

    ``entity_cols`` is a column name, a list of names, or None for one series. Aliases: ``entity_col``, ``entity`` and
    similar names are ``entity_cols``; ``time`` is ``time_col``; ``value`` is ``value_col``. An unknown keyword raises a
    TypeError that shows this signature. Data problems never raise: they come back as blocker findings.
    ``freq`` is a pandas alias such as "D", "h" or "MS"; when it is None the grain is inferred. Entities beyond
    ``max_entities`` are sampled with a fixed seed for the statistical checks, and the summary says so.

    Returns {"findings": [...], "summary": {...}}. Each finding has id, severity, scope, evidence, decision_needed and
    next_step. Record a ledger decision for every blocker and major finding before any forecast, and put each finding id
    (for example time.duplicate_timestamps) in that ledger entry's finding or decision text: the forecast is checked for it.
    """
    try:
        return _run(df, time_col, value_col, entity_cols, freq, max_entities)
    except Exception as exc:  # the caller gets a blocker with the reason, never a traceback
        rows = len(df) if isinstance(df, pd.DataFrame) else None
        return _result([_finding(
            "scan.failed", "blocker", "all", {"error": f"{type(exc).__name__}: {exc}"[:400], "rows_in": rows},
            "The scan could not finish. Fix the input named in the error before any forecast.",
            "Check the columns with df.dtypes and df.head(), then call sc.scan again.")], {"rows_in": rows})


def write_scan(result: dict, path: str = "out/scan.json") -> str:
    """Write a scan result as strict JSON, creating the folder if needed. Returns the path written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_jsonable(result), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return str(target)
