"""Is this forecast believable for this history? A deterministic check that names the cause when it is not.

Motivation vs Logic
Motivation: TimesFM 3 returns a nearly flat median in two very different situations. When the history has no repeatable
pattern (a price, a rate, rare counts) a flat median is the honest answer and the information is in the interval. When
the history has a strong cycle and the forecast is flat anyway, something upstream is wrong: the model was given too
little history to see the cycle, the history ends in a filled stretch, or the horizon is too long for the context.
Measured on the real example series (see docs/prophet/GOAL.md): the daily CO2 record forecasts with a range of 3.1 ppm
from 3,600 points but 0.25 ppm from 400 points, because 400 days hold about one annual cycle. A reader cannot tell
those two cases apart from the chart, and neither could the planner.
Logic: find the history's dominant period from the autocorrelation of the detrended series (no calendar assumed: the
period is whatever repeats), build the path that history would imply for the horizon from its average cycle, and compare
the forecast's shape with it. ``assess`` returns a verdict (ok, flat_expected, flat_unexpected, check), the evidence and
the likely causes with what to try, so the planner can repair a bad forecast and the answer can say plainly when a flat
line is the honest one.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from prophet_agent.sandbox.prophet_lib import _detrend, dominant_period   # one implementation, also used inside the sandbox

MIN_POINTS = 24
MIN_CYCLES = 3                # cycles of history the model needs to see to pick a cycle up
RECENT_CYCLES = 8             # cycles the expected cycle is averaged over
SHAPE_FLAT = 0.5              # forecast shape below this share of the expected shape is flat
LEVEL_FLAT = 0.30             # without a cycle: forecast moves less than this share of the recent history's own movement
SHORT_WINDOW = 0.25           # a horizon under this share of a cycle shows too little of it to judge the forecast's shape
CONST_RUN = 7                 # identical consecutive values at the end of a history that suggest a filled gap


def _expected_path(history: np.ndarray, period: int, horizon: int) -> np.ndarray | None:
    """The cycle the recent history implies for the next ``horizon`` points. Classical decomposition: a moving average over
    one period is the level, what is left is the cycle, averaged by position in the period over the latest cycles so the
    recent amplitude is used (a weekly ripple is far smaller in a quiet year than in a wave). None with too few cycles."""
    x = np.asarray(history, dtype=float)
    if len(x) < (MIN_CYCLES + 1) * period:
        return None
    x = x[-(RECENT_CYCLES + 1) * period:]
    m = len(x)
    level = np.convolve(x, np.ones(period) / period, mode="valid")        # level[j] is centred near position j + offset
    offset = (period - 1) // 2
    pos = np.arange(len(level)) + offset
    cycle = x[pos] - level
    phase = pos % period
    profile = np.array([cycle[phase == k].mean() if (phase == k).any() else 0.0 for k in range(period)])
    return np.array([profile[(m + k) % period] for k in range(horizon)])


def _shape(a: np.ndarray) -> float:
    return float(np.std(_detrend(np.asarray(a, dtype=float)))) if len(a) > 2 else 0.0


def _constant_tail(history: np.ndarray) -> int:
    """Length of the identical-value run at the end, 0 when that value is the series' floor (a count that is legitimately
    zero for weeks is not a filled gap)."""
    x = np.asarray(history, dtype=float)
    if len(x) < 2 or x[-1] <= x.min():
        return 0
    run = 1
    for i in range(len(x) - 1, 0, -1):
        if x[i] == x[i - 1]:
            run += 1
        else:
            break
    return run


def assess(history: Any, point: Any, q10: Any = None, q90: Any = None, *, context_used: int | None = None,
           context_points: int | None = None) -> dict[str, Any]:
    """Judge a forecast against its history. Returns verdict, period, strengths, causes and what to try."""
    x = np.asarray(history, dtype=float).reshape(-1)
    p = np.asarray(point, dtype=float).reshape(-1)
    horizon = len(p)
    n = len(x)
    ctx = int(context_used or n)
    if n < MIN_POINTS or horizon < 2:
        return {"verdict": "check", "reason": f"only {n} history points; too few to judge the forecast's shape"}

    period, strength = dominant_period(x)
    fc_shape = _shape(p)
    recent = x[-max(horizon, 30):]
    recent_shape = _shape(recent)
    out: dict[str, Any] = {"period": period, "cycle_strength": round(strength, 2), "history_points": n}
    expected = _expected_path(x, period, horizon) if period else None
    causes: list[str] = []
    fixes: list[str] = []

    # Shape check, against the cycle when there is one, against the recent movement when there is not.
    if period and expected is not None and _shape(expected) > 1e-9:
        ratio = fc_shape / _shape(expected)
        out["shape_ratio"] = round(ratio, 2)
        flat = ratio < SHAPE_FLAT
        structured = True
    else:
        ratio = fc_shape / recent_shape if recent_shape > 1e-12 else 1.0
        out["shape_ratio"] = round(ratio, 2)
        flat = ratio < LEVEL_FLAT
        structured = bool(period)        # a cycle exists but too few cycles are in the context to build the expected path

    # Naive comparison: how far the median path sits from the last observed value, in typical one-step moves of the history.
    step = float(np.mean(np.abs(np.diff(x)))) if n > 1 else 0.0
    if step > 1e-12:
        out["vs_last_value"] = round(float(np.mean(np.abs(p - x[-1]))) / step, 2)
        if flat:
            if out["vs_last_value"] <= 1.0:
                out["vs_last_value_note"] = (f"the median is within {out['vs_last_value']:.1f} typical one-step moves of the last "
                                             "value, so it carries no directional view")
            else:
                out["vs_last_value_note"] = (f"the median sits {out['vs_last_value']:.1f} typical one-step moves from the last "
                                             "value, so it does carry a level view even though its shape is flat")

    if q10 is not None and q90 is not None:
        band = float(np.mean(np.asarray(q90, dtype=float) - np.asarray(q10, dtype=float)))
        spread = float(np.std(x[-min(n, 365):]))
        out["interval_vs_spread"] = round(band / spread, 2) if spread > 1e-12 else None

    if period:
        cycles_in_ctx = ctx / period
        out["cycles_in_context"] = round(cycles_in_ctx, 1)
        if cycles_in_ctx < MIN_CYCLES:
            if context_points and context_points < n:
                causes.append(f"context_points={context_points} holds {cycles_in_ctx:.1f} cycles of the {period}-step pattern; "
                              f"the model needs about {MIN_CYCLES} or more to see it")
                fixes.append("drop context_points, or set it to at least " + str(MIN_CYCLES * period))
            elif n < MIN_CYCLES * period:
                causes.append(f"the history holds {n / period:.1f} cycles of a {period}-step pattern; too few to learn it")
                fixes.append("say the cycle cannot be learned from this much history; report the level and the interval, "
                             "or use a seasonal baseline from backtest_series as the point forecast")
    tail = _constant_tail(x)
    if tail >= CONST_RUN:
        causes.append(f"the history ends with {tail} identical values, which looks like a gap that was filled forward")
        fixes.append("cut the filled stretch off the end (or use ignore_ranges for it) and forecast from the last real value")
    if horizon > max(ctx // 2, 1) and ctx >= MIN_POINTS:
        causes.append(f"the horizon ({horizon}) is longer than half the context ({ctx}); long horizons drift toward the mean")
        fixes.append("forecast a shorter horizon, or say how far ahead the forecast can be trusted")

    # Measured on the daily CO2 record (13 holdouts, 60-day horizon, full history): the shape ratio read below 0.5 in 7 of 13, yet
    # TimesFM's error was below seasonal naive's in all 13. A window of a sixth of a cycle is nearly a straight line once detrended,
    # so its curvature is noise. Over so short a window the shape check cannot call a forecast flat; the backtest against
    # seasonal naive is the test.
    short_window = bool(period) and horizon < SHORT_WINDOW * period
    if short_window:
        out["shape_reliable"] = False
    if flat and structured and short_window:
        verdict = "check"
        causes.insert(0, f"the forecast has {out['shape_ratio']:.0%} of the history's {period}-step cycle shape, but {horizon} steps "
                         f"is only {horizon / period:.0%} of one cycle, too short a window to judge shape")
        fixes.append("do not call this forecast flat on shape alone: run backtest_series with season set to the period and trust "
                     "its error against seasonal naive")
    elif flat and structured:
        verdict = "flat_unexpected"
        causes.insert(0, f"the history repeats every {period} steps (strength {strength:.2f}) but the forecast has "
                         f"{out['shape_ratio']:.0%} of that shape")
        # A6-07: the instruction names the detected period, so the planner runs the lab on it instead of guessing one.
        had_fixes = bool(fixes)
        fixes.insert(0, f"run control_lab with period={period} and backtest_series with season={period} on the same series; "
                        "keep a treatment only if the lab adopts it")
        if not had_fixes:
            fixes.append("check how the series was prepared (aggregation, gap filling, rounding) and run backtest_series to see "
                         "whether a seasonal baseline beats it")
    elif flat:
        verdict = "flat_expected"
        causes = ["no repeating pattern stands out in the history, so a flat median is the honest forecast"] + causes
        fixes = ["present the level and the 10 to 90 percent interval, not a precise path; say the day-to-day movement is not predictable"]
    elif causes and structured:
        verdict = "check"
    else:
        verdict = "ok"
    out.update(verdict=verdict)
    if causes:
        out["causes"] = causes
    if fixes:
        out["try"] = fixes
    return out
