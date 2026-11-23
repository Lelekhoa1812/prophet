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
"""
from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np

from prophet_agent.sandbox.prophet_lib import find_season

THRESHOLD = 0.3             # the same bar the existing season picker used: above about 0.3 suggests a repeating pattern
ALPHA = 0.05                # family-wise false-season rate the bar is set for (A2-11)
MIN_POINTS = 24             # below this a cycle cannot be told from noise
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
    return float(np.mean(z[lag:] * z[:-lag]))


def _bar(n_changes: int, n_lags: int, alpha: float) -> float:
    """The autocorrelation a candidate must clear: the Bonferroni bound over the candidate lags, floored at THRESHOLD.

    Motivation (A2-11, measured): the strongest of many lags against a fixed 0.3 gave white-noise changes a false season
    8 percent of the time at n = 60. Under no cycle the lag autocorrelation of n standardised changes is about N(0, 1/n),
    so z(1 - alpha / (2K)) / sqrt(n) bounds the largest of K lags with chance alpha. Long series keep the 0.3 floor."""
    z = NormalDist().inv_cdf(1 - alpha / (2 * max(1, n_lags)))
    return max(THRESHOLD, z / np.sqrt(max(1, n_changes)))


def choose_season(values: Any, freq_hint: str | None = None, dates: Any = None, alpha: float = ALPHA) -> dict[str, Any]:
    """Return {"season": int, "reason": str, "threshold": float, ...}. Season 1 means no season: use one-step changes for
    the MASE scale. The threshold is the bar the strongest candidate had to clear (see ``_bar``)."""
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < MIN_POINTS:
        return {"season": 1, "reason": f"only {n} points: too few to measure a cycle, so no season is assumed"}
    kind = _kind(freq_hint, dates)
    d = np.diff(x)
    sd = float(d.std())
    if sd <= 1e-12:
        return {"season": 1, "reason": "the series does not change between steps, so no cycle can be measured"}
    z = (d - d.mean()) / sd
    if kind is not None:
        cands = tuple(l for l in CANDIDATES[kind] if l < len(z) // 2)
        if not cands:
            return {"season": 1, "reason": f"{kind} data with too few points for its candidate cycles; no season assumed"}
        strengths = {l: round(_acf(z, l), 3) for l in cands}
        label = f"{kind} data"
    else:
        found = find_season(d)
        strengths = {int(k): float(v) for k, v in found.get("autocorrelation_by_lag", {}).items()}
        label = "data with no frequency hint"
        if not strengths:
            return {"season": 1, "reason": f"{label}: no candidate cycle fits the history; no season assumed"}
    best = max(strengths, key=strengths.get)
    bar = round(_bar(len(z), len(strengths), alpha), 3)
    if strengths[best] > bar:
        return {"season": int(best), "reason": f"{label}: the one-step changes repeat every {best} periods "
                                                f"(autocorrelation {strengths[best]:.2f}, above {bar})",
                "strengths": strengths, "kind": kind, "threshold": bar}
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
