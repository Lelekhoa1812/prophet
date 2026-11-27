"""A direction for a flat centre line.

Motivation vs Logic
Motivation (owner review of round 14 charts, 2026-11-27): the NVDA, AUUNEMP, SPX and NEM packs drew a dead-flat centre line.
The line was the last value (or a mean of it), chosen because a random walk is hard to beat. A reader who asks for a
forecast wants the best-supported direction drawn, with the range around it. A flat line is the answer only when the
history gives no direction at all.
Logic: when the centre is flat, fit a damped additive trend (Holt) to the history (to its logarithm when the series is
positive, so the path cannot cross zero), and score it on rolling origins of the same history against the flat line (the last
value repeated). Deliver the damped path as the centre when it is no more than TOLERANCE times the flat line's error and it
moves visibly against the band; otherwise keep the flat line and say what was tried. The result is a tilt chosen on these windows:
it is labelled as such and never presented as a validated gain.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

TOLERANCE = 1.05        # the tilt may score at most this many times the flat line's mean absolute error
MIN_MOVE = 0.04         # the path must move at least this share of the band's final half-width, or it is not visible
FLAT_SHARE = 0.08       # a centre whose range is under this share of the band's width is flat
AGREE_FLOOR = 0.02      # the model's own median moves at least this share of the final half-range to count as a direction
FIT_POINTS = 400        # the trailing history the trend is fitted to
MIN_POINTS = 24         # shorter histories give no trend
N_ORIGINS = 8
MIN_WINDOWS = 3


def is_flat(point: Any, q10: Any, q90: Any) -> bool:
    """True when the centre line's range is a small share of the 10-90 band's mean width (a level line)."""
    p = np.asarray(point, dtype=float)
    width = float(np.mean(np.asarray(q90, dtype=float) - np.asarray(q10, dtype=float)))
    if p.size < 2:
        return False
    span = float(p.max() - p.min())
    return span <= FLAT_SHARE * width if width > 0 else span <= 1e-9


def damped_path(x: Any, h: int) -> np.ndarray | None:
    """A damped additive trend over h steps, from the last FIT_POINTS values (their logarithm when all are positive).
    None when too short or the fit fails."""
    y = np.asarray(x, dtype=float)[-FIT_POINTS:]
    if len(y) < MIN_POINTS or not np.isfinite(y).all():
        return None
    positive = bool(y.min() > 0)
    z = np.log(y) if positive else y
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(z, trend="add", damped_trend=True, initialization_method="estimated").fit()
            path = np.asarray(fit.forecast(h), dtype=float)
    except Exception:       # noqa: BLE001 - any fitting failure means no tilt, never a crash
        return None
    if not np.isfinite(path).all():
        return None
    return np.exp(path) if positive else path


DRIFT_POINTS = 250      # the trailing window the recent drift averages its change over


def recent_drift_path(x: Any, h: int) -> np.ndarray | None:
    """The average change over the last DRIFT_POINTS steps, extended (in logarithms when all values are positive)."""
    y = np.asarray(x, dtype=float)[-DRIFT_POINTS:]
    if len(y) < MIN_POINTS or not np.isfinite(y).all():
        return None
    positive = bool(y.min() > 0)
    z = np.log(y) if positive else y
    slope = (z[-1] - z[0]) / (len(z) - 1)
    path = z[-1] + slope * np.arange(1, h + 1)
    return np.exp(path) if positive else path


MOMENTUM_HORIZONS = 4   # the momentum window is this many horizons long (at least 12 points): the trend at the scale being forecast


def momentum_path(x: Any, h: int) -> np.ndarray | None:
    """The average change over the last MOMENTUM_HORIZONS horizons (at least 12 points, at most DRIFT_POINTS), extended. It
    reads what the series did lately at the scale of the forecast; a rate that has climbed for a year shows it where a
    twenty-year drift and a smoothed trend average it away (round 15, AUUNEMP)."""
    w = int(min(max(MOMENTUM_HORIZONS * h, 12), DRIFT_POINTS))
    y = np.asarray(x, dtype=float)[-w:]
    if len(y) < MIN_POINTS // 2 or not np.isfinite(y).all():
        return None
    positive = bool(y.min() > 0)
    z = np.log(y) if positive else y
    slope = (z[-1] - z[0]) / (len(z) - 1)
    path = z[-1] + slope * np.arange(1, h + 1)
    return np.exp(path) if positive else path


CANDIDATES = {"damped_trend": damped_path, "recent_drift": recent_drift_path, "momentum": momentum_path}
TILT_RULES = tuple(CANDIDATES)


def _origins(n: int, h: int) -> list[int]:
    lo = max(MIN_POINTS, n // 3)
    hi = n - h
    if hi < lo:
        return []
    k = min(N_ORIGINS, hi - lo + 1)
    return sorted({int(round(v)) for v in np.linspace(lo, hi, k)})


def tilt_evidence(values: Any, h: int, rule: str = "damped_trend") -> dict[str, Any]:
    """Score one trend rule against the flat line (the last value repeated) on rolling origins of the history."""
    x = np.asarray(values, dtype=float)
    orig = _origins(len(x), h)
    errs_d, errs_n = [], []
    for o in orig:
        path = CANDIDATES[rule](x[:o], h)
        if path is None:
            continue
        actual = x[o:o + h]
        errs_d.append(float(np.mean(np.abs(actual - path))))
        errs_n.append(float(np.mean(np.abs(actual - x[o - 1]))))
    if len(errs_d) < MIN_WINDOWS or not np.mean(errs_n) > 0:
        return {"scored": False, "reason": f"only {len(errs_d)} usable windows of {h} steps; too few to score a trend"}
    ratios = [d / n for d, n in zip(errs_d, errs_n) if n > 0]
    return {"scored": True, "windows": len(errs_d), "horizon": h,
            "error_ratio": round(float(np.sum(errs_d) / np.sum(errs_n)), 3),
            "windows_won": int(sum(1 for r in ratios if r < 1.0))}


def _shifted(q: dict, old: np.ndarray, new: np.ndarray) -> dict:
    """The band moved with the centre: multiplicatively when everything is positive (the band stays above zero), else by the
    same amount at each step."""
    positive = bool(old.min() > 0 and new.min() > 0 and all(min(v) > 0 for v in q.values()))
    if positive:
        ratio = new / old
        return {k: [round(float(v) * float(r), 4) for v, r in zip(vals, ratio)] for k, vals in q.items()}
    shift = new - old
    return {k: [round(float(v) + float(d), 4) for v, d in zip(vals, shift)] for k, vals in q.items()}


def tilt(fc: dict, values: Any) -> dict:
    """The forecast with a trend centre when its centre was flat and a trend rule passes; the forecast unchanged otherwise.
    Each rule in CANDIDATES is tried; the one with the lowest error ratio among those that move visibly and stay within
    TOLERANCE is delivered. ``tilt_check`` is recorded whenever the centre was flat, so the answer can say what was tried."""
    q = fc["quantiles"]
    old = np.asarray(fc["point"], dtype=float)
    h = len(old)
    if not is_flat(old, q["0.1"], q["0.9"]):
        return fc
    x = np.asarray(values, dtype=float)
    final_half = float(q["0.9"][-1] - q["0.1"][-1]) / 2 or float(np.mean(np.asarray(q["0.9"]) - np.asarray(q["0.1"]))) / 2 or 1.0
    own = np.asarray(fc.get("timesfm_point", fc["point"]), dtype=float)     # the model's own median, before any rule replaced it
    own_move = float(own[-1] - x[-1])
    own_dir = 0 if abs(own_move) < AGREE_FLOOR * final_half else (1 if own_move > 0 else -1)
    tried: dict[str, Any] = {}
    passing: list[tuple[float, str, np.ndarray, dict]] = []
    for rule, fn in CANDIDATES.items():
        path = fn(x, h)
        if path is None:
            tried[rule] = {"result": "the history is too short or the trend did not fit"}
            continue
        move = float(path[-1] - x[-1])
        share = abs(move) / final_half
        info: dict[str, Any] = {"move": round(move, 4), "move_share_of_final_half_range": round(share, 3)}
        if share < MIN_MOVE:
            tried[rule] = {**info, "result": f"moves under {MIN_MOVE:.0%} of the range: no visible direction"}
            continue
        if own_dir and (move > 0) != (own_dir > 0):
            tried[rule] = {**info, "result": f"points {'up' if move > 0 else 'down'} while the model's own median moves "
                                             f"{'up' if own_dir > 0 else 'down'}: no agreement, so no tilt"}
            continue
        ev = tilt_evidence(x, h, rule)
        if not ev["scored"]:
            tried[rule] = {**info, "result": ev["reason"]}
            continue
        info["evidence"] = ev
        if ev["error_ratio"] > TOLERANCE:
            tried[rule] = {**info, "result": f"error {ev['error_ratio']} times the flat line's, over the {TOLERANCE} allowed"}
            continue
        tried[rule] = {**info, "result": "passed"}
        passing.append((ev["error_ratio"], rule, np.asarray(path, dtype=float), {**info, "move": move}))
    if not passing:
        return {**fc, "tilt_check": {"flat_centre": True, "adopt": False, "tried": tried,
                                     "reason": "no trend rule moved visibly and stayed within the flat line's error on these "
                                               "windows: the history gives no direction, and the range is the forecast"}}
    passing.sort(key=lambda t: t[0])
    ratio, rule, new, info = passing[0]
    ev = info["evidence"]
    reason = (f"the centre was flat; {rule.replace('_', ' ')} moves {info['move']:+.4g} over the horizon and scored {ratio} times "
              f"the flat line's error on {ev['windows']} windows (it won {ev['windows_won']}); chosen on those same windows, so "
              "it is a tilt, not a validated gain")
    centre = {"rule": rule, "reason": reason, "evidence": ev}
    health = fc.get("health")
    if isinstance(health, dict):
        # The verdict judged the model's own median, which was flat. The line drawn is the tilt, and the answer must say so
        # (round 17 SPX: "a flat median is the honest forecast" was copied beside a chart that rises).
        note = (f"the model's own median was flat; the line drawn is {rule.replace('_', ' ')}, a tilt chosen on the backtest "
                "windows, not a validated gain")
        causes = [c.replace("so a flat median is the honest forecast", "so the model's own median is flat")
                  for c in health.get("causes", [])]
        health = {**health, "causes": causes, "drawn_line": note}
    return {**fc, **({"health": health} if isinstance(health, dict) else {}), "point": [round(float(v), 4) for v in new], "quantiles": _shifted(q, old, new), "timesfm_point": fc["point"],
            "centre": centre, "tilt_check": {"flat_centre": True, "adopt": True, "rule": rule, "tried": tried, "reason": reason}}
