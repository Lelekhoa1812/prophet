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


def _band_sd(point: np.ndarray, sd_h: np.ndarray, floor: bool = True) -> np.ndarray:
    """point (h,) with a residual sd for each lead (h,) -> quantiles (h, 9); floored at zero only when ``floor``."""
    q = point[:, None] + Z[None, :] * np.asarray(sd_h, dtype=float)[:, None]
    return np.clip(q, 0, None) if floor else q


MIN_PAIRS = 6               # fewest h-step errors a lead needs before its spread is measured from the history


def _measured_lead_sd(s: np.ndarray, w: int, h: int) -> np.ndarray | None:
    """Residual sd of the moving-average rule's own j-step errors for j = 1..h, or None when no lead can be measured.

    Motivation vs Logic
    Motivation: the band was sd * Z * sqrt(h), a random-walk growth. A moving-average level does not accumulate error on a
    stationary or seasonal series, so the band over-covered: about 96 percent on stationary noise and 98 percent on a
    period-4 series against a label of 80 (measured, synthetic, 400 replications). The growth belongs to a series whose
    level drifts, and that is measurable from the history rather than assumed.
    Logic: the forecast made at origin t is mean(s[t-w:t]), and its j-step error is s[t+j-1] minus that. Each lead's error sd
    is taken from the history, with the variance fitted as a line in j (least squares, slope not below zero). A flat fit
    gives a flat band; a drifting level gives a widening one. The fit extends past the last measurable lead, so the
    band beyond the history is a straight-line extrapolation of the measured variance, not a hold.
    """
    n = len(s)
    roll = np.convolve(s, np.ones(w) / w, mode="valid")        # roll[u] = mean(s[u:u+w]): the forecast at origin u + w
    sds: list[float] = []
    for j in range(1, h + 1):
        cnt = n - w - j + 1                                    # origins whose j-step target lies inside the history
        if cnt < MIN_PAIRS:
            break
        sds.append(float(np.std(s[w + j - 1:] - roll[:cnt], ddof=1)))
    if not sds:
        return None
    v = np.square(sds)
    if len(v) == 1:
        a, b = float(v[0]), 0.0
    else:
        X = np.column_stack([np.ones(len(v)), np.arange(len(v), dtype=float)])
        (a, b), *_ = np.linalg.lstsq(X, v, rcond=None)
        if b < 0:
            a, b = float(v.mean()), 0.0
    return np.sqrt(np.maximum(a + b * np.arange(h, dtype=float), 0.0))


def moving_average(series: np.ndarray, h: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(series, dtype=float)
    # An empty history or a window below one has no level. Returning zero would be a forecast with no basis (R6m probe).
    if len(s) == 0 or window < 1:
        raise ValueError("moving_average needs at least one point and a window of at least 1")
    w = min(window, len(s))
    level = s[-w:].mean() if w else 0.0
    sd_h = _measured_lead_sd(s, w, h) if w >= 1 else None
    if sd_h is None:
        # too short to measure a lead's error: the spread of the whole history, the same at every lead
        sd_h = np.full(h, float(np.std(s, ddof=1)) if len(s) > 1 else 0.0)
    p = np.full(h, level)
    return p, _band_sd(p, sd_h, floor=_nonneg(s))


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
    # The error is one stored value against one value a season earlier, so it is the same at every lead: no sqrt(h) growth
    # (R6m, audit item 1: the growth covered 77 to 100 percent across horizons on a synthetic period-4 series).
    sd = float(np.std(s[-26:] - s[-26 - season:-season] * ratio)) if len(s) > season + 26 else float(np.std(s))
    return p, _band_sd(p, np.full(h, sd), floor=_nonneg(s))
