"""One season choice for every forecast tool: the cycle length that the backtest, health check and lab all share.

Motivation vs Logic
Motivation: the same series gave three different seasons in one mission (252 from the planner, 340 from the health check,
and a lag-1 scale elsewhere), so the MASE figures could not be compared. A business-day price series with a 252 season
looked like a seasonal naive win that did not exist. The season must come from one place, and it must default to "no
season" unless a cycle is really there.
Logic: pick the calendar candidates from the frequency (business days: a week of 5 and a month of 21; daily: 7 and 365;
hourly: 24 and 168; weekly 52; monthly 12; quarterly 4). Measure each candidate on the one-step changes of the series
rather than on its levels: a random walk has large level autocorrelation at every lag, but its changes have none, so
only a repeating pattern survives. The candidate with the strongest autocorrelation wins if it clears the threshold;
otherwise the season is 1, meaning none. Without a frequency hint or dates, the candidates are the general lags used by
the existing find_season, measured the same way. The reason names what was measured so the caller can quote it.

A second measure covers the case the changes miss: a cycle that is small against day-to-day noise. Daily temperature has an
annual cycle of about 10 degrees under weather noise of about 3 degrees; the one-step changes carry that cycle at about a
thousandth of the noise, so their autocorrelation at lag 365 is near zero (measured on the Berlin series: -0.02). The levels
show it plainly. So a candidate is also scored on the seasonal strength of the detrended levels: the share of variance the
phase means explain, 1 - Var(residual) / Var(series). A random walk has no fixed phase, so its strength is small; its null
distribution is simulated at the same length and the bar is the larger of THRESHOLD and the (1 - alpha) quantile of that null.
"""
from __future__ import annotations

from functools import lru_cache
from statistics import NormalDist
from typing import Any

import numpy as np

from prophet_agent.sandbox.prophet_lib import find_season

THRESHOLD = 0.3             # the same bar the existing season picker used: above about 0.3 suggests a repeating pattern
ALPHA = 0.05                # family-wise false-season rate the bar is set for (A2-11)
MIN_POINTS = 24             # below this a cycle cannot be told from noise
MIN_CYCLES = 3              # a level test needs at least this many whole cycles to compare phases
LEVEL_REPS = 300            # random walks simulated for the level null at each length
LEVEL_SEED = 0              # fixed, so the same series always gets the same answer
CANDIDATES: dict[str, tuple[int, ...]] = {
    "business": (5, 21),    # a trading week and a trading month
    "daily": (7, 365),
    "hourly": (24, 168),
    "weekly": (52,),
    "monthly": (12,),
    "quarterly": (4,),
}
_HINTS = {"B": "business", "BD": "business", "BUS": "business", "BUSINESS": "business", "D": "daily", "DAY": "daily",
          "DAILY": "daily", "H": "hourly", "HOURLY": "hourly", "W": "weekly", "WEEKLY": "weekly", "M": "monthly",
          "MS": "monthly", "ME": "monthly", "MONTHLY": "monthly", "Q": "quarterly", "QS": "quarterly", "QE": "quarterly",
          "QUARTERLY": "quarterly"}


def _kind_from_dates(dates: Any) -> str | None:
    """Business days when every date is a weekday; otherwise the kind of the typical gap between dates."""
    try:
        d = np.asarray(dates).astype("datetime64[s]")
    except (TypeError, ValueError):
        return None
    d = d[~np.isnat(d)]                     # a missing date says nothing about the calendar
    if d.size < 3:
        return None
    days = d.astype("datetime64[D]").astype(np.int64)
    if np.all((days + 3) % 7 < 5):          # 1970-01-01 was a Thursday; Monday is 0
        return "business"
    # Gaps between distinct timestamps only: a repeated stamp is a zero gap, and a median over those read a daily series with
    # each date repeated as hourly (probe: 400 daily dates, each twice, gave "hourly" before this).
    gaps = np.diff(np.unique(d)).astype(np.int64) / 86400.0
    if gaps.size == 0:
        return None
    gap = float(np.median(gaps))
    if gap < 1:
        return "hourly"
    if abs(gap - 1) < 0.01:
        return "daily"
    if abs(gap - 7) < 0.01:
        return "weekly"
    if 28 <= gap <= 31:
        return "monthly"
    if 89 <= gap <= 92:
        return "quarterly"
    return None


def _kind(freq_hint: str | None, dates: Any) -> str | None:
    if freq_hint:
        return _HINTS.get(str(freq_hint).strip().upper())
    return _kind_from_dates(dates) if dates is not None else None


def _acf(z: np.ndarray, lag: int) -> float:
    """Autocorrelation over the pairs where both points are measured; NaN when no pair is measured."""
    a, b = z[lag:], z[:-lag]
    both = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(a[both] * b[both])) if both.any() else float("nan")


def _bar(n_changes: int, n_lags: int, alpha: float) -> float:
    """The autocorrelation a candidate must clear: the Bonferroni bound over the candidate lags, floored at THRESHOLD.

    Motivation (A2-11, measured): the strongest of many lags against a fixed 0.3 gave white-noise changes a false season
    8 percent of the time at n = 60. Under no cycle the lag autocorrelation of n standardised changes is about N(0, 1/n),
    so z(1 - alpha / (2K)) / sqrt(n) bounds the largest of K lags with chance alpha. Long series keep the 0.3 floor."""
    z = NormalDist().inv_cdf(1 - alpha / (2 * max(1, n_lags)))
    return max(THRESHOLD, z / np.sqrt(max(1, n_changes)))


def _detrend(x: np.ndarray) -> np.ndarray:
    """Residual from a linear trend fitted on the measured points. A missing point stays missing (NaN) at its position."""
    t = np.arange(len(x), dtype=float)
    obs = np.isfinite(x)
    return x - np.polyval(np.polyfit(t[obs], x[obs], 1), t)


def _level_strength(residual: np.ndarray, lag: int) -> float | None:
    """Share of the detrended level's variance that the phase means at this lag explain. Missing points are left out of
    each phase mean and of the variances, and keep their position. None when there are fewer than MIN_CYCLES whole
    cycles or the series is flat."""
    k = len(residual) // lag
    if k < MIN_CYCLES:
        return None
    m = residual[: k * lag].reshape(k, lag)
    ok = np.isfinite(m)
    total = float(m[ok].var()) if ok.any() else 0.0
    if total <= 1e-12:
        return None
    counts = ok.sum(axis=0)
    phase_mean = np.where(ok, m, 0.0).sum(axis=0) / np.maximum(counts, 1)
    return float(1 - (m - phase_mean)[ok].var() / total)


def _level_strengths(x: np.ndarray, lags: tuple[int, ...]) -> dict[int, float]:
    residual = _detrend(x)
    out = {}
    for lag in lags:
        s = _level_strength(residual, lag)
        if s is not None:
            out[int(lag)] = round(s, 3)
    return out


SHORTER_TOLERANCE = 0.1     # the tolerance find_season uses for the same tie-break (SEASON_TOLERANCE in sandbox prophet_lib)


def _shortest_repeat(scores: dict[int, float], bar: float) -> int:
    """The lag to report: the shortest lag that clears the bar and scores within SHORTER_TOLERANCE of the best, else the best.

    Motivation vs Logic
    Motivation (measured, synthetic): a clean 24-step sine gave 48, 96 or 168 from the largest of its multiples, which score
    within a few thousandths of each other, so the reported cycle was a multiple of the real one.
    Logic: the same tie-break find_season applies. Whether a season is found is still decided by the best lag against the bar,
    so this changes only which of equal lags is reported.
    """
    best = max(scores, key=scores.get)
    for lag in sorted(scores):
        if lag < best and scores[lag] >= scores[best] - SHORTER_TOLERANCE and scores[lag] > bar:
            return int(lag)
    return int(best)


@lru_cache(maxsize=512)
def _level_bar(n: int, lags: tuple[int, ...], alpha: float) -> float:
    """The bar a level strength must clear: the (1 - alpha) quantile of the strongest lag on simulated random walks of the
    same length, floored at THRESHOLD. Random walks are the null because they are what the changes test guards against."""
    rng = np.random.default_rng(LEVEL_SEED)
    tops = []
    for _ in range(LEVEL_REPS):
        residual = _detrend(np.cumsum(rng.normal(size=n)))
        vals = [s for s in (_level_strength(residual, lag) for lag in lags) if s is not None]
        if vals:
            tops.append(max(vals))
    if not tops:
        return float("inf")
    return max(THRESHOLD, round(float(np.quantile(tops, 1 - alpha)), 3))


def choose_season(values: Any, freq_hint: str | None = None, dates: Any = None, alpha: float = ALPHA) -> dict[str, Any]:
    """Return {"season": int, "reason": str, "threshold": float, ...}. Season 1 means no season: use one-step changes for
    the MASE scale. The threshold is the bar the strongest candidate had to clear (see ``_bar``)."""
    x = np.asarray(values, dtype=float).reshape(-1)
    n_all = len(x)
    if dates is not None and len(dates) != n_all:
        return {"season": 1, "reason": f"the dates ({len(dates)}) do not match the values ({n_all}), so the calendar cannot be "
                                        "read; no season measured"}
    # Missing values keep their position: the changes and the level phases are measured on positions, so a gap neither shifts
    # a cycle nor joins two steps that were not one step apart. Only the dates of measured values are used for the calendar.
    obs = np.isfinite(x)
    n = int(obs.sum())
    if n < MIN_POINTS:
        return {"season": 1, "reason": f"only {n} finite points: too few to measure a cycle, so no season is assumed"}
    kind = _kind(freq_hint, np.asarray(dates, dtype=object)[obs] if dates is not None else None)
    d = np.diff(x)
    d[~np.isfinite(d)] = np.nan                  # a change across a gap, or one that overflows, is not measured
    dobs = np.isfinite(d)
    n_ch = int(dobs.sum())
    if n_ch < 2:
        return {"season": 1, "reason": "the one-step changes are too large to measure or too few values are consecutive, so no "
                                        "season is assumed"}
    sd = float(np.std(d[dobs]))
    if not np.isfinite(sd):
        return {"season": 1, "reason": "the values are too large to measure their changes, so no season is assumed"}
    if sd <= 1e-12:
        return {"season": 1, "reason": "the series does not change between steps, so no cycle can be measured"}
    z = (d - float(np.mean(d[dobs]))) / sd
    if kind is not None:
        cands = tuple(l for l in CANDIDATES[kind] if l < len(z) // 2)
        if not cands:
            return {"season": 1, "reason": f"{kind} data with too few points for its candidate cycles; no season assumed"}
        raw = {l: _acf(z, l) for l in cands}
        strengths = {l: round(v, 3) for l, v in raw.items() if np.isfinite(v)}
        label = f"{kind} data"
    else:
        found = find_season(d)
        strengths = {int(k): float(v) for k, v in found.get("autocorrelation_by_lag", {}).items()}
        label = "data with no frequency hint"
        if not strengths:
            return {"season": 1, "reason": f"{label}: no candidate cycle fits the history; no season assumed"}
    if not strengths:
        return {"season": 1, "reason": f"{label}: no candidate cycle has enough measured pairs; no season assumed"}
    bar = round(_bar(n_ch, len(strengths), alpha), 3)
    best = _shortest_repeat(strengths, bar)
    if strengths[best] > bar:
        return {"season": int(best), "reason": f"{label}: the one-step changes repeat every {best} periods "
                                                f"(autocorrelation {strengths[best]:.2f}, above {bar})",
                "strengths": strengths, "kind": kind, "threshold": bar}
    # The changes test found nothing. A cycle can still be in the levels when it is small against the day-to-day noise.
    level = _level_strengths(x, tuple(sorted(strengths)))
    if level:
        level_bar = _level_bar(n_all, tuple(sorted(level)), alpha)
        top = _shortest_repeat(level, level_bar)
        if level[top] > level_bar:
            return {"season": int(top), "reason": f"{label}: the levels repeat every {top} periods (seasonal strength "
                                                    f"{level[top]:.2f}, above {level_bar}); the one-step changes do not show it, "
                                                    f"because the cycle is small against the day-to-day noise",
                    "strengths": strengths, "level_strengths": level, "kind": kind, "threshold": bar,
                    "level_threshold": level_bar, "method": "levels"}
    if kind == "business":
        why = "no weekly or monthly cycle is measured in the one-step changes, so the price-like series has no season"
    else:
        why = f"no candidate cycle is above {bar} in the one-step changes ({len(strengths)} lags tested)"
    return {"season": 1, "reason": f"{label}: {why}", "strengths": strengths, "kind": kind, "threshold": bar}


def mase_scale_text(season: int) -> str:
    """The definition of the MASE denominator that matches ``season``: the same text every tool prints."""
    if season <= 1:
        return "mean absolute one-step change"
    return f"mean absolute seasonal change at lag {season}"
