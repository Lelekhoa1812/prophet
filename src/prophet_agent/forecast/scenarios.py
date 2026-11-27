"""Scenario-weighted forecasts: political, geopolitical and policy scenarios applied to a delivered forecast distribution.

Motivation vs Logic
Motivation (owner review, 2026-11-27): the forecasts left out the political, geopolitical and economic events that drive the
outcome. A statistical model of a series' history cannot know that a central bank has signalled a hike, that a tariff starts
in March, or that an election may bring a stimulus. An analyst states such events as scenarios with a probability and an
impact and mixes them into the forecast, so the delivered range carries them.
Logic: the base distribution at each step is the delivered pack's quantiles (10, 30, 50, 70, 90; tails extended as a normal
from the outer quantiles). Each simulated path draws one uniform rank used at every step (so a path stays high or low through
the horizon, as the pack's own paths do), and one scenario by its probability (the rest of the mass is the base case). A
scenario adds a shift from its start step, reached over ``ramp`` steps and held (or faded out after ``end``); the shift is
in the series' units, or a percentage of the base median, and is itself uncertain (``shift_sd`` as a share of the shift).
``band_scale`` widens or narrows the spread around the median inside that scenario. The output is the mixture's quantiles,
each scenario's own median path, and the probability-weighted shift at the last step.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import ndtri

KNOTS = (0.1, 0.3, 0.5, 0.7, 0.9)
OUT_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
Z90 = float(ndtri(0.9))


def _base_draws(q: dict[float, np.ndarray], u: np.ndarray) -> np.ndarray:
    """Values at rank u (n,) for each step from the quantile knots q {level: (h,)} -> (n, h). Linear between knots; beyond
    the outer knots, a normal tail with the outer half-width read as 1.28 standard deviations."""
    levels = np.array(sorted(q))
    vals = np.vstack([q[k] for k in levels])                     # (k, h)
    n, h = u.size, vals.shape[1]
    out = np.empty((n, h))
    lo_sd = np.maximum((vals[levels.searchsorted(0.5)] - vals[0]) / Z90, 1e-12)
    hi_sd = np.maximum((vals[-1] - vals[levels.searchsorted(0.5)]) / Z90, 1e-12)
    inner = (u >= levels[0]) & (u <= levels[-1])
    for j in range(h):
        out[inner, j] = np.interp(u[inner], levels, vals[:, j])
    z = ndtri(np.clip(u, 1e-6, 1 - 1e-6))
    lo, hi = u < levels[0], u > levels[-1]
    out[lo] = vals[0][None, :] + (z[lo, None] - ndtri(levels[0])) * lo_sd[None, :]
    out[hi] = vals[-1][None, :] + (z[hi, None] - ndtri(levels[-1])) * hi_sd[None, :]
    return out


def _profile(h: int, start: int, ramp: int, end: int | None) -> np.ndarray:
    """Share of the full shift at each of h steps (1-based start): 0 before start, rising over ramp steps, held, and faded
    over ramp steps after end."""
    k = np.arange(1, h + 1, dtype=float)
    up = np.clip((k - start + 1) / max(ramp, 1), 0.0, 1.0)
    if end is not None:
        down = np.clip(1.0 - (k - end) / max(ramp, 1), 0.0, 1.0)
        up = np.minimum(up, down)
    return up


def validate(scenarios: Any, h: int, dates: list[str] | None) -> list[dict]:
    """Checked scenarios: name, probability (each above 0, together at most 1), shift or shift_pct, start (step 1..h or a
    date in the forecast window), ramp, end, shift_sd, band_scale. Raises ValueError with the fix."""
    if not isinstance(scenarios, list) or not 1 <= len(scenarios) <= 6:
        raise ValueError("scenarios must be a list of 1 to 6 objects {name, probability, shift or shift_pct, start, ...}")
    out = []
    for i, s in enumerate(scenarios):
        if not isinstance(s, dict):
            raise ValueError(f"scenario {i + 1} must be an object")
        name = str(s.get("name") or "").strip()[:60]
        if not name:
            raise ValueError(f"scenario {i + 1} needs a name (for example 'RBA hikes in February')")
        p = s.get("probability")
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not 0 < p < 1:
            raise ValueError(f"scenario {name!r}: probability must be a number between 0 and 1 (0.25 for one in four)")
        has_abs, has_pct = s.get("shift") is not None, s.get("shift_pct") is not None
        if has_abs == has_pct:
            raise ValueError(f"scenario {name!r}: give exactly one of shift (in the series' units, for example 0.25 for "
                             "a quarter-point rate rise) or shift_pct (per cent of the base median, for example -8)")
        val = s.get("shift") if has_abs else s.get("shift_pct")
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not np.isfinite(val):
            raise ValueError(f"scenario {name!r}: the shift must be a number")
        start = s.get("start", 1)
        if isinstance(start, str):
            if not dates:
                raise ValueError(f"scenario {name!r}: start is a date but the forecast has no dates; give a step number")
            hit = [j for j, d in enumerate(dates, start=1) if d[:len(start)] >= start]
            if not hit:
                raise ValueError(f"scenario {name!r}: start {start} is after the forecast window ({dates[0]} to {dates[-1]})")
            start = hit[0]
        if not isinstance(start, int) or isinstance(start, bool) or not 1 <= start <= h:
            raise ValueError(f"scenario {name!r}: start must be a step from 1 to {h} or a date in the window")
        ramp = s.get("ramp", 1)
        end = s.get("end")
        if not isinstance(ramp, int) or isinstance(ramp, bool) or not 1 <= ramp <= h:
            raise ValueError(f"scenario {name!r}: ramp must be a whole number of steps from 1 to {h}")
        if end is not None and (not isinstance(end, int) or isinstance(end, bool) or not start <= end <= h):
            raise ValueError(f"scenario {name!r}: end must be a step from start to {h}, or left out to hold the shift")
        sd = s.get("shift_sd", 0.3)
        scale = s.get("band_scale", 1.0)
        if not isinstance(sd, (int, float)) or isinstance(sd, bool) or not 0 <= sd <= 2:
            raise ValueError(f"scenario {name!r}: shift_sd is the impact's uncertainty as a share of the shift, 0 to 2")
        if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not 0.5 <= scale <= 3:
            raise ValueError(f"scenario {name!r}: band_scale must be from 0.5 to 3")
        out.append({"name": name, "probability": float(p), "kind": "level" if has_abs else "pct", "shift": float(val),
                    "start": int(start), "ramp": int(ramp), "end": end, "shift_sd": float(sd), "band_scale": float(scale),
                    "why": str(s.get("why") or "")[:300], "evidence": str(s.get("evidence") or "")[:300]})
    total = sum(s["probability"] for s in out)
    if total > 1 + 1e-9:
        raise ValueError(f"the scenario probabilities add to {total:.2f}; together they must be at most 1 (the rest is the "
                         "base case)")
    names = [s["name"].lower() for s in out]
    if len(set(names)) != len(names):
        raise ValueError("each scenario needs its own name")
    return out


def mix(quantiles: dict[float, list[float]], scenarios: list[dict], *, n: int = 20000, seed: int = 0,
        floor: float | None = None) -> dict[str, Any]:
    """The scenario-weighted distribution. ``quantiles`` holds the base pack's knots {0.1: [...], 0.5: [...], ...} (0.5 is
    required); ``scenarios`` come from ``validate``. ``floor`` clips draws (0 for a rate or a price that cannot go below).
    Returns point (median), the OUT_LEVELS quantiles, each scenario's median path and the weighted last-step shift."""
    q = {float(k): np.asarray(v, float) for k, v in quantiles.items()}
    if 0.5 not in q:
        raise ValueError("the base forecast needs its median")
    h = q[0.5].size
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0005, 0.9995, n)
    base = _base_draws(q, u)
    median = q[0.5]
    probs = np.array([s["probability"] for s in scenarios] + [max(0.0, 1 - sum(s["probability"] for s in scenarios))])
    pick = rng.choice(len(probs), size=n, p=probs / probs.sum())
    out = base.copy()
    per: list[dict] = []
    for j, s in enumerate(scenarios):
        rows = pick == j
        prof = _profile(h, s["start"], s["ramp"], s["end"])
        full = s["shift"] if s["kind"] == "level" else s["shift"] / 100.0 * median
        noise = 1.0 + s["shift_sd"] * rng.standard_normal(int(rows.sum()))
        shift = np.outer(noise, prof * full)
        spread = (base[rows] - median[None, :]) * s["band_scale"]
        out[rows] = median[None, :] + spread + shift
        cond = median + prof * full
        per.append({"name": s["name"], "probability": s["probability"], "median": cond.tolist(),
                    "last_shift": float((prof * full)[-1]), "start": s["start"], "why": s["why"]})
    if floor is not None:
        out = np.maximum(out, floor)
    qs = np.quantile(out, OUT_LEVELS, axis=0)
    levels = {f"{lv:.1f}": qs[i].tolist() for i, lv in enumerate(OUT_LEVELS)}
    weighted = float(sum(p["probability"] * p["last_shift"] for p in per))
    return {"point": levels["0.5"], "quantiles": levels, "scenarios": per, "weighted_last_shift": weighted,
            "base_case_probability": float(probs[-1]), "draws": n}
