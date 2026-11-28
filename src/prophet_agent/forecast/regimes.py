"""Risk regimes: where a policy or geopolitical risk index stood at each forecast origin, and how the target moved after spikes.

Motivation vs Logic
Motivation (owner review 2026-11-28, Mission 10 batch 2): the forecasts did not account for political, geopolitical and policy
risk. The research behind Mission 10 (reports/Fundamentals models for six panel targets.md) found no measured out-of-sample
gain from putting a geopolitical-risk (GPR) or policy-uncertainty (EPU) index into a point forecast, and recommended using the
indices to condition the band and to trigger scenarios. Either use has to be measured on the target's own history, or it is
decoration.
Logic:
- regime_at: at an origin's cutoff date, the index's latest value is ranked against the index's own last WINDOW observations
  up to that date (at least MIN_HISTORY, a value no older than MAX_LAG_DAYS): low below the 1/3 point, high from the 2/3
  point, mid between. A trailing window, not the whole history: newspaper-count indices drift with the media they count
  (Australian EPU ranked against its whole history put 113 of 214 cash-rate windows in "high" and 20 in "low"), and a
  regime is a comparison with the recent past. Nothing after the cutoff is read. ensemble.run uses these labels to report the combination's
  backtest coverage by regime and, when the windows that share today's regime held fewer than 75 per cent of their outcomes,
  to widen the band with side scales fitted on those windows only (class-conditional, or Mondrian, split conformal).
- event_study: a spike onset is an origin at which the index stands at or above its trailing 90th percentile after at least
  ``gap`` origins below it. The target's change over h steps after each onset is compared with its change over h steps from
  every origin. The interval of the difference in medians comes from resampling onsets and moving blocks of origins (windows
  that overlap are serially correlated, so resampling single origins would understate the spread). The shift is reported as
  measured only when there are at least MIN_EVENTS onsets and the 80 per cent interval excludes zero.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np

MIN_HISTORY = 36          # index observations before an origin can be ranked
WINDOW = 120              # an index value is ranked against its last 120 observations (ten years of a monthly index)
MAX_LAG_DAYS = 70         # a monthly index read later than this at the cutoff is stale: no label
LOW, HIGH = 1 / 3, 2 / 3
SPIKE = 0.9               # an onset is the index at or above its expanding 90th percentile
MIN_EVENTS = 5            # fewer onsets than this never give a measured shift
BOOT = 2000


def percentile_of_last(hist: np.ndarray) -> float:
    """Mid-rank percentile of the last value within the whole array (0 to 1)."""
    now = hist[-1]
    return float(np.mean(hist < now) + 0.5 * np.mean(hist == now))


def regime_at(days: np.ndarray, vals: np.ndarray, cutoff: dt.date, *, min_history: int = MIN_HISTORY,
              max_lag_days: int = MAX_LAG_DAYS, window: int = WINDOW) -> tuple[str | None, float | None]:
    """(label, percentile) of the index as known at ``cutoff``; (None, None) with too little or too stale history."""
    k = int(np.searchsorted(days, cutoff.toordinal(), side="right"))
    if k < min_history or cutoff.toordinal() - int(days[k - 1]) > max_lag_days:
        return None, None
    pct = percentile_of_last(vals[max(0, k - window):k])
    return ("low" if pct < LOW else "high" if pct >= HIGH else "mid"), pct


def labels_for(cutoffs: list[dt.date], days: np.ndarray, vals: np.ndarray, **kw: Any) -> list[str | None]:
    """The regime label known with t values of the target, for t = 0..n (n + 1 labels; label t uses cutoff t - 1)."""
    out: list[str | None] = [None]
    for c in cutoffs:
        out.append(regime_at(days, vals, c, **kw)[0])
    return out


def _change(y: np.ndarray, t: int, h: int, relative: bool) -> float:
    a, b = y[t - 1], y[t - 1 + h]
    return float(np.log(b / a)) if relative else float(b - a)


def event_study(y: np.ndarray, cutoffs: list[dt.date], days: np.ndarray, vals: np.ndarray, h: int, *, relative: bool,
                gap: int = 6, threshold: float = SPIKE, min_history: int = MIN_HISTORY, window: int = WINDOW,
                seed: int = 0) -> dict[str, Any]:
    """How the target moved over ``h`` steps after past spikes of the index, against its moves from every origin.

    ``cutoffs[t - 1]`` is the date by which the first t values were known. Changes are log ratios when ``relative`` (prices),
    else differences in the target's units."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    origins, onsets, last_spike = [], [], -10 ** 9
    for t in range(1, n - h + 1):
        k = int(np.searchsorted(days, cutoffs[t - 1].toordinal(), side="right"))
        if k < min_history or cutoffs[t - 1].toordinal() - int(days[k - 1]) > MAX_LAG_DAYS:
            continue
        if relative and (y[t - 1] <= 0 or y[t - 1 + h] <= 0):
            continue
        origins.append(t)
        if percentile_of_last(vals[max(0, k - window):k]) >= threshold:
            if t - last_spike > gap:
                onsets.append(t)
            last_spike = t
    unit = "log change" if relative else "change in the target's units"
    base = {"h": h, "threshold": threshold, "gap": gap, "unit": unit, "origins": len(origins), "n": len(onsets),
            "onsets": [cutoffs[t - 1].isoformat() for t in onsets]}
    if len(origins) < 24 or not onsets:
        return {**base, "measured": False, "reason": "too few origins or no spike in the target's history"}
    all_ch = np.array([_change(y, t, h, relative) for t in origins])
    ev_ch = np.array([_change(y, t, h, relative) for t in onsets])
    shift = float(np.median(ev_ch) - np.median(all_ch))
    rng = np.random.default_rng(seed)
    block = max(1, h)
    m = len(all_ch)
    starts = np.arange(max(1, m - block + 1))
    diffs = np.empty(BOOT)
    for b in range(BOOT):
        ev = rng.choice(ev_ch, size=len(ev_ch), replace=True)
        idx = np.concatenate([np.arange(s, min(s + block, m)) for s in rng.choice(starts, size=-(-m // block))])[:m]
        diffs[b] = np.median(ev) - np.median(all_ch[idx])
    lo, hi = (float(x) for x in np.quantile(diffs, [0.1, 0.9]))
    measured = len(onsets) >= MIN_EVENTS and (lo > 0 or hi < 0)
    out = {**base, "median_change_after": float(np.median(ev_ch)), "median_change_all": float(np.median(all_ch)),
           "shift": shift, "ci80": [lo, hi], "measured": measured,
           "after_each": [{"onset": cutoffs[t - 1].isoformat(), "change": round(c, 6)} for t, c in zip(onsets, ev_ch)],
           "beyond_band": _beyond(ev_ch, all_ch)}
    if not measured:
        out["reason"] = (f"only {len(onsets)} onsets" if len(onsets) < MIN_EVENTS
                         else "the 80 per cent interval of the shift includes zero")
    return out


def _beyond(ev: np.ndarray, base: np.ndarray) -> float:
    """Share of onsets after which the target's move fell outside the 10-90 range of its moves from every origin."""
    lo, hi = np.quantile(base, [0.1, 0.9])
    return round(float(np.mean((ev < lo) | (ev > hi))), 3)
