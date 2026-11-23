"""Honest baselines with empirical bands from their own past errors. Ported from src/lab/engine/baselines.py.

The `run` name dispatcher from the legacy file is not ported. Callers use the three functions directly.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .tsb import is_intermittent

Z = np.array([-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816])
MEAN_LAST_FLOOR = 4         # the shortest window mean_last_season averages; equal to rolling.MEAN_LAST_FLOOR (a test checks it)
TSB_SIZE_WINDOW = 52        # the non-zero order sizes TSB averages; tsb.expected_volume uses the last 52
# Rules that only apply to one demand class. Everything else applies to every series.
DEMAND_CLASS = {"tsb_expected": "intermittent"}


def candidate_rules(rules: Sequence[str], series: np.ndarray) -> tuple[str, ...]:
    """The rules that may be backtested as a baseline for this series' demand class, in the order given.

    Motivation vs Logic
    Motivation: TSB is a model for intermittent demand. A rule backtested on a continuous series can win on noise and be
    named the best baseline (audit A4-12: a random walk of 160 points picked tsb_expected, MASE 1.08 against 1.35 for naive,
    measured, synthetic).
    Logic: a rule tagged with a class is kept only when the series is in that class. intermittent uses tsb.is_intermittent,
    which is ADI above 1.32 and is false for any series that goes below zero. The caller drops tsb_expected when the
    delivered point already is TSB's expected volume (a lumpy series), because the two would be the same forecast.
    """
    lumpy = None
    kept = []
    for r in rules:
        cls = DEMAND_CLASS.get(r)
        if cls == "intermittent":
            if lumpy is None:
                lumpy = is_intermittent(series)
            if not lumpy:
                continue
        kept.append(r)
    return tuple(kept)


def rule_windows(season: int, n_points: int) -> dict[str, int]:
    """How many trailing points each simple rule uses, so a backtest can print it beside the rule's score.

    Motivation vs Logic
    Motivation: the rule named mean_last_season averaged a 4-point window with no season (audit A4-12, WTI). A reader
    could not see the window from the result.
    Logic: each rule's window is a function of the season and the length of the history, and is returned rather than
    left as an internal parameter. drift uses every point; tsb_expected uses the last 52 non-zero sizes.
    """
    m = max(1, int(season))
    return {"naive": 1, "seasonal_naive": m, "drift": int(n_points),
            "mean_last_season": max(m, MEAN_LAST_FLOOR), "tsb_expected": TSB_SIZE_WINDOW}


def _nonneg(series: np.ndarray) -> bool:
    """A history that never goes below zero (stock, sales, counts) keeps its bands above zero. One that does (a
    balance, a temperature, a margin) is left unclipped: flooring it would bend a legitimate band."""
    return bool(len(series) == 0 or (series >= 0).all())


def _band(point: np.ndarray, resid_sd: float, floor: bool = True) -> np.ndarray:
    """point (h,) -> quantiles (h, 9), widening with sqrt of lead; floored at zero only when ``floor``."""
    h = np.arange(1, len(point) + 1)[:, None]
    q = point[:, None] + Z[None, :] * resid_sd * np.sqrt(h)
    return np.clip(q, 0, None) if floor else q


def moving_average(series: np.ndarray, h: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(series, dtype=float)
    w = min(window, len(s))
    level = s[-w:].mean() if w else 0.0
    # residual sd from rolling one-step errors of the same rule
    if len(s) > w + 4:
        roll = np.convolve(s, np.ones(w) / w, mode="valid")[:-1]
        sd = float(np.std(s[w:] - roll))
    else:
        sd = float(np.std(s)) if len(s) > 1 else 0.0
    p = np.full(h, level)
    return p, _band(p, sd, floor=_nonneg(s))


def _phase_sd(s: np.ndarray, season: int) -> np.ndarray:
    """Residual sd of the lag-season difference for each phase of the season (length ``season``).

    A phase with fewer than three differences takes the pooled sd of all of them, so a short history never gives a band
    of zero width by accident.
    """
    resid = s[season:] - s[:-season]
    pooled = float(np.std(resid, ddof=1) / np.sqrt(2)) if len(resid) > 2 else float(np.std(s))
    sd = np.full(season, pooled)
    phase = (np.arange(len(resid)) + season) % season
    for ph in range(season):
        r = resid[phase == ph]
        if len(r) >= 3:
            sd[ph] = float(np.std(r, ddof=1) / np.sqrt(2))
    return sd


def seasonal_naive(series: np.ndarray, h: int, season: int, k: int = 4,
                   band_mode: str = "phase") -> tuple[np.ndarray, np.ndarray]:
    """Mean of the last k same-phase values, e.g. the last 4 same weekdays for daily data with season=7.

    Motivation vs Logic
    Motivation: the h-step error of a seasonal naive forecast repeats one season and does not accumulate, so a band that
    widens with sqrt(h) is too wide at 2 and 3 seasons out (audit A2-13: 4.54 at horizon 1 against 16.37 at horizon 13 on a
    period-4 series with noise 2, measured, synthetic).
    Logic: band_mode "phase" (the default) uses the residual sd of the lag-season difference for the forecast's own phase,
    with no growth in h, and widened by sqrt(1 + 1/k) for the mean of k values (R6l, audit item 1). band_mode "sqrt" keeps
    the earlier random-walk band for callers that need it.
    A history shorter than one season has no same-phase value to repeat, so it returns moving_average(series, h, season)
    instead (the mean of the whole history when it is shorter than the season). That is the documented fallback; check the
    length first if the seasonal rule is wanted.
    """
    if band_mode not in ("phase", "sqrt"):
        raise ValueError("band_mode must be 'phase' or 'sqrt'")
    s = np.asarray(series, dtype=float)
    if len(s) < season:
        return moving_average(s, h, season)
    p = np.zeros(h)
    n_avg = np.zeros(h)
    for i in range(h):
        # The k most recent observed values with this step's phase. Step i's same-phase values sit at len(s) + i - season * j,
        # and the smallest j that lies in the history is i // season + 1, so every step averages k values however far ahead.
        j0 = i // season + 1
        idx = len(s) + i - season * np.arange(j0, j0 + k)
        idx = idx[idx >= 0]
        p[i] = s[idx].mean()
        n_avg[i] = len(idx)
    if band_mode == "sqrt":
        resid = s[season:] - s[:-season]
        sd = float(np.std(resid) / np.sqrt(2)) if len(resid) > 2 else float(np.std(s))
        return p, _band(p, sd, floor=_nonneg(s))
    # The forecast is a mean of n_avg same-phase values, so its error is sigma * sqrt(1 + 1/n_avg), not sigma alone: the
    # lag-season difference measures a single value. Without this factor the 10 to 90 band covered about 70 percent.
    phase_sd = _phase_sd(s, season)
    sd_h = phase_sd[(len(s) + np.arange(h)) % season] * np.sqrt(1.0 + 1.0 / n_avg)
    q = p[:, None] + Z[None, :] * sd_h[:, None]
    return p, (np.clip(q, 0, None) if _nonneg(s) else q)


def seasonal_year(series: np.ndarray, h: int, season: int = 52) -> tuple[np.ndarray, np.ndarray]:
    """Same period last season, scaled by the recent level ratio. Needs more than one season of history."""
    s = np.asarray(series, dtype=float)
    if len(s) < season + h + 8:
        return moving_average(s, h, 8)
    recent = s[-8:].mean()
    ly = s[-season - 8:-season].mean()
    ratio = recent / ly if ly > 0 else 1.0
    ratio = float(np.clip(ratio, 0.5, 2.0))
    # Step i is the same period one season earlier; beyond one season ahead it repeats the last observed season (the index
    # len(s)+i-season ran past the data when h > season: IndexError, audit maths 5).
    p = np.array([s[len(s) - season + (i % season)] for i in range(h)]) * ratio
    sd = float(np.std(s[-26:] - s[-26 - season:-season] * ratio)) if len(s) > season + 26 else float(np.std(s))
    return p, _band(p, sd, floor=_nonneg(s))
