"""Control lab: a bounded, leak-free walk-forward experiment on which TimesFM 3 input treatments help one series.

Motivation vs Logic
Motivation: the agent can set the history window, ignore points, add drivers, z-normalise and average symmetrically, but
nothing shows which of these helps a given series, and hand-built trials are slow and easy to read by eye.
Logic: a fixed grid of treatments runs over the same origins as the default model and the reference baselines. Each
treatment is built from a FoldContext that holds only what was known at its origin, so nothing after the origin can be
read, and every treatment is causal: trailing windows only, no centred smoothing, no full-sample decomposition. Errors
are paired fold by fold against the default (MASE, RMSSE, pinball loss, sMAPE). A seeded bootstrap of the mean paired
MASE difference gives the interval, a paired Wilcoxon test gives a p-value for reporting, and named rules turn the
evidence into adopt, harmful or no evidence. The result is a measured verdict on this series in these windows, not a
claim that a treatment works in general.
Two guards sit on top. A simple baseline that beats the default by more than 5 percent on these folds is reported as
beating it, whatever the treatments do. And an adoption must survive a Holm step-down correction across every treatment
tested on the same folds, since trying many treatments on one set of folds invites a lucky winner.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

from . import metrics
from .baselines import moving_average, seasonal_naive
from .control import Control, build_control
from .paired import bootstrap_gain
from .tsb import INTERMITTENT_ADI, TSB, is_intermittent  # noqa: F401  one definition of intermittent (A6-01)
from .types import Forecast

# ----------------------------------------------------------------------------------------- decision thresholds --
MIN_GAIN = 0.02             # mean relative MASE gain a treatment must show; a proposal, to calibrate on real runs
MIN_WIN_RATE = 0.6          # share of folds where the treatment beats the default; stops one fold carrying the mean
CONFIDENCE = 0.90           # level of the bootstrap interval of the mean paired MASE difference
BOOTSTRAP_RESAMPLES = 2000  # resamples of the folds; enough for stable 5th and 95th percentiles on a few folds
PINBALL_TOLERANCE = 0.0     # adopt needs mean pinball loss no worse than the default's by more than this relative amount
WILCOXON_MIN_FOLDS = 6      # below six folds the exact signed-rank test cannot reach conventional p-values

# ------------------------------------------------------------------------------------------- experiment shape --
DEFAULT_FOLDS = 8           # origins scored for every treatment
MIN_FOLDS = 6               # fewer folds makes the win rate jump by more than 15 points and the interval unreadable
MAX_FOLDS = 30
FOLD_GAP_HORIZONS = 1       # horizons left unscored between one test window and the next origin
MIN_HISTORY = 16            # points before the first origin; shorter histories say nothing about the model or a season
MAX_CALLS = 240             # engine calls per run; the full grid over 8 folds needs about 70, the rest is headroom
MAX_HORIZON = 512           # the same limit as timesfm_forecast
HOLM_ALPHA = 0.05           # family-wise level for adopting a treatment when several are tested on the same folds
BASELINE_GATE_PCT = 5.0     # a simple baseline with this much lower MASE than the default is reported as beating it
BASELINE_NOTE = "a simple baseline beats default TimesFM on these folds; consider it or a combination"
MAX_PERIOD = 1000           # the same limit as backtest_series' season

# ------------------------------------------------------------------------------------------- treatment settings --
MIN_CONTEXT = 8             # the shortest history window the control validator accepts
SEASON_MIN_CYCLES = 2       # a seasonal window holds at least two whole cycles
CONTEXT_MULTIPLES = (2, 4, 8)  # history windows as multiples of the horizon
OUTLIER_K = 3.5             # Iglewicz and Hoaglin cut-off on the robust z score
HAMPEL_WINDOW = 7           # a point's level is the median of the 6 points before it: causal, and a spike cannot drag it
MAX_FLAG_SHARE = 0.10       # above this share flagged the series is volatile rather than broken, so nothing is removed
MIN_USABLE = 8              # the usable-point floor build_control enforces
N_QUANTILES = 9

# ------------------------------------------------------------------------------------------------ baselines --
TSB_MIN_SHARE = 0.5         # the TSB baseline is run when at least this share of fold histories are intermittent
SIMPLE_BASELINES = ("naive", "mean_last_8", "seasonal_naive")
CANDIDATE_NAMES = ("default", "ctx_2h", "ctx_4h", "ctx_8h", "ctx_season", "ignore_outliers", "replace_outliers",
                   "znorm", "symmetric_averaging", "with_covariates", "covariates_edge_pad")


@dataclass(frozen=True)
class FoldContext:
    """Everything a treatment may read at one origin. Built by slicing the full arrays, so the history stops at the
    origin, past covariates stop there too, and future covariates stop at origin + horizon (they are known ahead)."""

    hist: np.ndarray
    origin: int
    horizon: int
    period: int | None
    past: dict[str, np.ndarray]
    future: dict[str, np.ndarray]


@dataclass(frozen=True)
class Prepared:
    """What the engine is given for one fold: the history it reads, the control, and a note for the report."""

    history: np.ndarray
    control: Control
    note: str = ""


@dataclass(frozen=True)
class Candidate:
    name: str
    summary: str
    prepare: Callable[[FoldContext], Prepared]


@dataclass
class _Run:
    fcs: list[Forecast]
    spent: int


# ------------------------------------------------------------------------------------------------ validation --

def _series(series: Any) -> np.ndarray:
    try:
        x = np.asarray(series, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("series must be a flat list of numbers, oldest first")
    if x.ndim != 1 or x.size == 0:
        raise ValueError("series must be a non-empty flat list of numbers, oldest first")
    if not np.isfinite(x).all():
        raise ValueError("series has missing or non-finite values; fill or drop them first so the choice is explicit")
    return x


def _whole(name: str, value: Any, lo: int, hi: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not lo <= value <= hi:
        raise ValueError(f"{name} must be a whole number from {lo} to {hi}")
    return value


def _period(period: Any) -> int | None:
    """None and 1 both mean no cycle; anything else is a number of periods per cycle."""
    if period is None:
        return None
    p = _whole("period", period, 1, MAX_PERIOD)
    return None if p == 1 else p


def _horizon(horizon: Any) -> int:
    return _whole("horizon", horizon, 1, MAX_HORIZON)


# ------------------------------------------------------------------------------------------------- folds --

def _min_history(period: int | None) -> int:
    """Two full cycles are needed for a seasonal baseline or window to mean anything."""
    return max(MIN_HISTORY, 2 * period if period and period > 1 else 0)


def plan_folds(n: int, horizon: int, folds: int = DEFAULT_FOLDS, period: int | None = None) -> tuple[list[int], int, int]:
    """Return (origins oldest first, minimum history, folds that fit).

    The last origin is n - horizon, so the final test window ends at the end of the series. Origins are spaced one
    test window plus FOLD_GAP_HORIZONS windows apart, so consecutive test windows never touch."""
    min_hist = _min_history(period)
    spacing = horizon * (1 + FOLD_GAP_HORIZONS)
    last = n - horizon
    if last < min_hist:
        return [], min_hist, 0
    fit = (last - min_hist) // spacing + 1
    used = min(folds, fit)
    return [last - spacing * k for k in range(used - 1, -1, -1)], min_hist, fit


def _fold_context(x: np.ndarray, past: Mapping[str, np.ndarray], future: Mapping[str, np.ndarray], origin: int,
                  horizon: int, period: int | None) -> FoldContext:
    return FoldContext(hist=np.array(x[:origin], dtype=float), origin=origin, horizon=horizon, period=period,
                       past={k: np.asarray(v, dtype=float)[:origin] for k, v in past.items()},
                       future={k: np.asarray(v, dtype=float)[:origin + horizon] for k, v in future.items()})


def _scales(hist: np.ndarray, lag: int) -> tuple[float, float]:
    """(MAE scale, MSE scale) of the in-sample change at the given lag over the history. Zero or undefined means the
    fold cannot be scored by MASE or RMSSE."""
    if len(hist) <= lag:
        return math.nan, math.nan
    d = hist[lag:] - hist[:-lag]
    mae, mse = float(np.abs(d).mean()), float((d * d).mean())
    return (mae if mae > 0 else math.nan), (mse if mse > 0 else math.nan)


# ------------------------------------------------------------------------------------------- outlier detection --

def robust_flags(hist: Sequence[float] | np.ndarray, period: int | None = None) -> np.ndarray:
    """Points of the history that a robust z test flags as errors. Causal: uses the history only, and each point is
    judged against a trailing level (its median with the newest HAMPEL_WINDOW - 1 points before it), never a centred
    one. The scale is the median absolute deviation of the residuals over the history, which is before the origin.

    With a period, the phase median of the residual is removed first, so a regular seasonal peak is not flagged. A
    sustained level shift is flagged until the trailing level catches up; MAX_FLAG_SHARE stops that from removing
    much of a series."""
    x = np.asarray(hist, dtype=float)
    n = len(x)
    flags = np.zeros(n, dtype=bool)
    if n < HAMPEL_WINDOW:
        return flags
    # The level of point t is the median of the HAMPEL_WINDOW - 1 points before it, so a spike cannot drag its own level.
    k = HAMPEL_WINDOW - 1
    padded = np.concatenate([np.full(k, np.nan), x])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # the first k points have fewer than k values before them
        level = np.nanmedian(sliding_window_view(padded, k)[:n], axis=1)
    r = x - level
    if period and period > 1 and n >= 2 * period:
        phase = np.arange(n) % period
        profile = np.array([np.nanmedian(r[phase == p]) for p in range(period)])
        r = r - profile[phase]
    dev = np.abs(r - np.nanmedian(r))
    scale = 1.4826 * float(np.nanmedian(dev))
    if not scale > 0:
        return flags
    return dev > OUTLIER_K * scale


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((int(i), int(j)))
            i = j
        else:
            i += 1
    return out


def _flagged(ctx: FoldContext) -> np.ndarray | None:
    """Flags for this origin, or None when there is nothing to treat (no flags, or too many to be errors)."""
    flags = robust_flags(ctx.hist, ctx.period)
    count = int(flags.sum())
    if count == 0:
        return None
    if count / len(flags) > MAX_FLAG_SHARE or len(flags) - count < MIN_USABLE:
        return None
    return flags


def _carry_forward(hist: np.ndarray, flags: np.ndarray) -> tuple[np.ndarray, int]:
    """Each flagged point takes the last unflagged value before it. Only earlier points are read, so this is causal.
    A flagged point with no unflagged value before it is left alone and not counted."""
    out = hist.copy()
    last: int | None = None
    replaced = 0
    for t in range(len(hist)):
        if not flags[t]:
            last = t
        elif last is not None:
            out[t] = hist[last]
            replaced += 1
    return out, replaced


# -------------------------------------------------------------------------------------------- the grid --

def _default(ctx: FoldContext) -> Prepared:
    return Prepared(ctx.hist, Control(), "no treatment")


def _window(k: int) -> Callable[[FoldContext], Prepared]:
    def prepare(ctx: FoldContext) -> Prepared:
        points = max(MIN_CONTEXT, k * ctx.horizon)
        return Prepared(ctx.hist, Control(context_points=points if points < len(ctx.hist) else None))
    return prepare


def _season_window(period: int) -> Callable[[FoldContext], Prepared]:
    def prepare(ctx: FoldContext) -> Prepared:
        cycles = max(SEASON_MIN_CYCLES, math.ceil(2 * ctx.horizon / period))
        points = max(MIN_CONTEXT, period * cycles)
        return Prepared(ctx.hist, Control(context_points=points if points < len(ctx.hist) else None),
                        f"{cycles} whole cycles of {period}")
    return prepare


def _ignore(ctx: FoldContext) -> Prepared:
    flags = _flagged(ctx)
    if flags is None:
        return Prepared(ctx.hist, Control(), "no treatment at this origin")
    return Prepared(ctx.hist, Control(ignore=tuple(_runs(flags))), f"ignored {int(flags.sum())} flagged points")


def _replace(ctx: FoldContext) -> Prepared:
    flags = _flagged(ctx)
    if flags is None:
        return Prepared(ctx.hist, Control(), "no treatment at this origin")
    hist, replaced = _carry_forward(ctx.hist, flags)
    return Prepared(hist, Control(), f"replaced {replaced} flagged points by the last unflagged value before them")


def _znorm(ctx: FoldContext) -> Prepared:
    return Prepared(ctx.hist, Control(znorm=True))


def _symmetric(ctx: FoldContext) -> Prepared:
    return Prepared(ctx.hist, Control(symmetric=True))


def _covariates(ctx: FoldContext) -> Prepared:
    return Prepared(ctx.hist, Control(past=dict(ctx.past), future=dict(ctx.future)))


def _covariates_edge(ctx: FoldContext) -> Prepared:
    # A6-10: the same drivers with the package's edge padding past the horizon, measured as its own treatment.
    return Prepared(ctx.hist, Control(past=dict(ctx.past), future=dict(ctx.future), padding_mode="edge"),
                    "covariates with edge padding past the horizon")


def _grid(period: int | None, has_covariates: bool) -> list[Candidate]:
    """The treatments in priority order. The default comes first: it is the reference every other treatment is paired
    with, and when the budget runs out the treatments at the end are the ones skipped."""
    grid = [Candidate("default", "the model as it is, no treatment", _default)]
    grid += [Candidate(f"ctx_{k}h", f"history window of the last {k} horizons", _window(k)) for k in CONTEXT_MULTIPLES]
    if period:
        grid.append(Candidate("ctx_season", f"history window of whole cycles of {period} covering two horizons",
                              _season_window(period)))
    grid += [
        Candidate("ignore_outliers", "points flagged as errors are masked, so the model does not read them", _ignore),
        Candidate("replace_outliers", "flagged points replaced by the last unflagged value before them", _replace),
        Candidate("znorm", "series and covariates standardised before the model reads them", _znorm),
        Candidate("symmetric_averaging", "a forecast of the negated series is averaged in, reducing bias", _symmetric),
    ]
    if has_covariates:
        grid.append(Candidate("with_covariates", "the past and future covariates as supplied", _covariates))
        grid.append(Candidate("covariates_edge_pad", "the covariates as supplied, last future value repeated past the horizon "
                                                     "as the package's edge padding does", _covariates_edge))
    return grid


_ABSENT_REASON = {
    "ctx_season": "it needs period (periods per cycle, above 1)",
    "with_covariates": "it needs past_covariates or future_covariates",
    "covariates_edge_pad": "it needs past_covariates or future_covariates",
}


def _choose(grid: list[Candidate], requested: Sequence[str] | None) -> list[Candidate]:
    if requested is None:
        return grid
    if isinstance(requested, str) or not isinstance(requested, (list, tuple)):
        raise ValueError(f"candidates must be a list of names from {list(CANDIDATE_NAMES)}")
    if not requested:
        raise ValueError("candidates is empty; leave it out to run the full grid")
    unknown = [r for r in requested if r not in CANDIDATE_NAMES]
    if unknown:
        raise ValueError(f"unknown candidates {unknown}; choose from {list(CANDIDATE_NAMES)}")
    available = {c.name for c in grid}
    absent = [r for r in requested if r not in available]
    if absent:
        why = "; ".join(f"{r}: {_ABSENT_REASON.get(r, 'does not apply to this call')}" for r in absent)
        raise ValueError(f"candidates do not apply to this call: {why}")
    wanted = set(requested) | {"default"}
    return [c for c in grid if c.name in wanted]


# ------------------------------------------------------------------------------------------- baselines --

def _baseline(name: str, hist: np.ndarray, horizon: int, period: int | None) -> Forecast:
    """Reference forecasts. All read only the history they are given. The zero forecast is what a MASE-only reading
    rewards on intermittent demand, so it is always reported beside the others."""
    if name == "zero":
        return Forecast(np.zeros(horizon), np.zeros((horizon, N_QUANTILES)))
    if name == "tsb":
        fc = TSB().predict([hist], horizon)
        return Forecast(np.asarray(fc.point[0], dtype=float), np.asarray(fc.q[0], dtype=float))
    if name == "naive":
        p, q = moving_average(hist, horizon, 1)
    elif name == "mean_last_8":
        p, q = moving_average(hist, horizon, 8)
    elif period and len(hist) >= period:
        p, q = seasonal_naive(hist, horizon, period)
    else:
        p, q = moving_average(hist, horizon, 1)   # no cycle given: seasonal naive is the same as repeating the last value
    return Forecast(np.asarray(p, dtype=float), np.asarray(q, dtype=float))


# ------------------------------------------------------------------------------------------- scoring --

def _sane(fc: Any, horizon: int) -> bool:
    point, q = np.asarray(fc.point, dtype=float), np.asarray(fc.q, dtype=float)
    return point.shape == (horizon,) and q.shape == (horizon, N_QUANTILES) and bool(
        np.isfinite(point).all() and np.isfinite(q).all())


def _smape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Symmetric MAPE in percent. Points where both actual and forecast are zero count as zero error, not as a division."""
    denom = np.abs(actual) + np.abs(pred)
    term = np.where(denom > 0, np.abs(actual - pred) / np.where(denom > 0, denom, 1.0), 0.0)
    return float(200.0 * term.mean())


def _nanmean(v: Any) -> float:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else math.nan


def _scores(actual: np.ndarray, fcs: Sequence[Forecast], mae: np.ndarray, mse: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mase": np.array([np.abs(actual[k] - fc.point).mean() / mae[k] for k, fc in enumerate(fcs)]),
        "rmsse": np.array([math.sqrt(float(((actual[k] - fc.point) ** 2).mean()) / mse[k]) for k, fc in enumerate(fcs)]),
        "smape": np.array([_smape(actual[k], fc.point) for k, fc in enumerate(fcs)]),
        "pinball": np.array([metrics.pinball(actual[k], fc.q) for k, fc in enumerate(fcs)]),
    }


def _paired(base: np.ndarray, cand: np.ndarray, seed: int) -> dict[str, Any]:
    """Paired fold differences of MASE (positive means the candidate has the lower error): the relative mean gain, the
    number of folds won, a seeded bootstrap interval of the mean gain relative to the default's mean error, and a
    two-sided paired Wilcoxon p-value (None when there are too few folds or no non-zero difference)."""
    diffs = base - cand
    wins = int((diffs > 0).sum())
    p_value = _wilcoxon(diffs)
    g = bootstrap_gain(base, cand, seed, CONFIDENCE, BOOTSTRAP_RESAMPLES)
    return {"gain": g["gain"], "wins": wins, "lo": g["lo"], "hi": g["hi"], "p": p_value}


def _holm(pvals: Mapping[str, float | None]) -> dict[str, float]:
    """Holm step-down adjusted p-values for the entries that have one. Entries with None are left out of the family."""
    items = sorted(((p, nm) for nm, p in pvals.items() if p is not None))
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for i, (p, nm) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[nm] = running
    return out


def _wilcoxon(diffs: np.ndarray) -> float | None:
    if len(diffs) < WILCOXON_MIN_FOLDS or not np.any(diffs != 0):
        return None
    p = float(stats.wilcoxon(diffs).pvalue)
    return p if math.isfinite(p) else None


def _pinball_gain(base: np.ndarray, cand: np.ndarray) -> float | None:
    b, c = _nanmean(base), _nanmean(cand)
    if not (math.isfinite(b) and math.isfinite(c)) or b <= 0:
        return None
    return (b - c) / b


def _decide(pair: dict[str, Any], pinball_gain: float | None, n_folds: int) -> tuple[str, str]:
    """adopt: the MASE gain clears MIN_GAIN, enough folds are won, the interval is above zero, and the pinball loss is no
    worse than the default's. harmful: the MASE interval lies entirely below zero. Anything else is no evidence, with
    the unmet rules named. The Wilcoxon p-value is reported but does not decide."""
    gain, lo, hi, wins = pair["gain"], pair["lo"], pair["hi"], pair["wins"]
    if not all(math.isfinite(v) for v in (gain, lo, hi)):
        return "no evidence", "the error could not be scored on these folds"
    win_rate = wins / n_folds
    pinball_ok = pinball_gain is not None and pinball_gain >= -PINBALL_TOLERANCE
    pinball_txt = "pinball n/a" if pinball_gain is None else f"pinball {pinball_gain:+.1%}"
    stats_txt = (f"MASE gain {gain:+.1%}, wins {wins}/{n_folds}, {CONFIDENCE:.0%} interval [{lo:+.1%}, {hi:+.1%}], "
                 f"{pinball_txt}")
    if gain >= MIN_GAIN and win_rate >= MIN_WIN_RATE and lo > 0 and pinball_ok:
        return "adopt", stats_txt + "; meets the gain, win-rate, interval and pinball rules"
    if hi < 0:
        return "harmful", stats_txt + "; the MASE interval lies entirely below zero"
    unmet = []
    if gain < MIN_GAIN:
        unmet.append(f"gain below {MIN_GAIN:.0%}")
    if win_rate < MIN_WIN_RATE:
        unmet.append(f"wins below {MIN_WIN_RATE:.0%} of folds")
    if lo <= 0:
        unmet.append("interval includes zero")
    if not pinball_ok:
        unmet.append("pinball loss not as good as the default")
    return "no evidence", stats_txt + "; " + ", ".join(unmet)


def is_scale_effect(zero_mase: float, default_mase: float, zero_pinball: float, default_pinball: float) -> bool:
    """True when the zero forecast beats the default on MASE but loses on pinball loss. MASE then rewards small forecasts
    on this series, so MASE alone cannot justify adopting anything."""
    return bool(zero_mase < default_mase and zero_pinball > default_pinball)


def _round(v: Any, nd: int = 4) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v), nd)


def _row(name: str, kind: str, sc: dict[str, np.ndarray], d_sc: dict[str, np.ndarray], pair: dict | None,
         decision: str) -> dict[str, Any]:
    pin = _pinball_gain(d_sc["pinball"], sc["pinball"])
    return {
        "name": name,
        "kind": kind,
        "mean_mase": _round(_nanmean(sc["mase"])),
        "mean_rmsse": _round(_nanmean(sc["rmsse"])),
        "mean_pinball": _round(_nanmean(sc["pinball"])),
        "gain_pct": 0.0 if pair is None else _round(pair["gain"] * 100, 2),
        "pinball_gain_pct": None if pin is None else _round(pin * 100, 2),
        "wins": None if pair is None else pair["wins"],
        "interval_pct": None if pair is None else [_round(pair["lo"] * 100, 2), _round(pair["hi"] * 100, 2)],
        "wilcoxon_p": None if pair is None else _round(pair["p"], 4),
        "smape_pct": _round(_nanmean(sc["smape"]), 2),
        "decision": decision,
    }


# ----------------------------------------------------------------------------------------------- the run --

def run_lab(series: Any, horizon: int, engine: Any, candidates: Sequence[str] | None = None, *,
            folds: int = DEFAULT_FOLDS, seed: int = 0, period: int | None = None,
            past_covariates: Mapping[str, Any] | None = None, future_covariates: Mapping[str, Any] | None = None,
            max_calls: int = MAX_CALLS) -> dict[str, Any]:
    """Run the control lab on one series and return a compact decision table.

    ``engine`` needs ``forecast_controlled(series, horizon, control)`` returning a Forecast, as TimesFMEngine has.
    ``candidates`` names treatments from CANDIDATE_NAMES; the default is always run as the reference. Past covariates
    have the length of the series, future covariates the length of the series plus the horizon.
    """
    x = _series(series)
    n = len(x)
    h = _horizon(horizon)
    m = _period(period)
    n_folds = _whole("folds", folds, MIN_FOLDS, MAX_FOLDS)
    budget = _whole("max_calls", max_calls, 1, 10_000_000)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be a whole number")
    cov = build_control({"past_covariates": dict(past_covariates or {}),
                         "future_covariates": dict(future_covariates or {})}, n, h)
    chosen = _choose(_grid(m, bool(cov.past or cov.future)), candidates)
    by_name = {c.name: c for c in chosen}

    origins, min_hist, fit = plan_folds(n, h, n_folds, m)
    spacing = h * (1 + FOLD_GAP_HORIZONS)
    if fit < MIN_FOLDS:
        need = min_hist + (MIN_FOLDS - 1) * spacing + h
        cycle = f", and two full cycles of {m} periods" if m else ""
        raise ValueError(
            f"the series has {n} points; the lab needs at least {need} for {MIN_FOLDS} walk-forward folds at horizon {h}. "
            f"Each fold needs {min_hist} points of history{cycle}, and folds are {spacing} periods apart "
            f"(one horizon tested, one horizon left unscored). Use a longer series or a shorter horizon.")

    lag = m or 1
    scales = {o: _scales(x[:o], lag) for o in origins}
    kept = [o for o in origins if math.isfinite(scales[o][0]) and math.isfinite(scales[o][1])]
    dropped = len(origins) - len(kept)
    if len(kept) < MIN_FOLDS:
        raise ValueError(f"only {len(kept)} of the {len(origins)} folds have history that varies (a flat stretch makes "
                         f"the naive error zero, so MASE is undefined); the lab needs {MIN_FOLDS}")
    if budget < len(kept):
        raise ValueError(f"max_calls {budget} cannot cover the {len(kept)} folds the reference needs")
    F = len(kept)
    mae = np.array([scales[o][0] for o in kept])
    mse = np.array([scales[o][1] for o in kept])
    actual = np.stack([x[o:o + h] for o in kept])
    ctxs = [_fold_context(x, cov.past, cov.future, o, h, m) for o in kept]

    intermittent_share = float(np.mean([is_intermittent(c.hist) for c in ctxs]))
    baselines = list(SIMPLE_BASELINES) + (["tsb"] if intermittent_share >= TSB_MIN_SHARE else []) + ["zero"]
    base_sc = {b: _scores(actual, [_baseline(b, c.hist, h, m) for c in ctxs], mae, mse) for b in baselines}

    left = budget
    skipped: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []
    runs: dict[str, _Run] = {}
    default_fcs: list[Forecast] = []
    for cand in chosen:
        preps = [cand.prepare(c) for c in ctxs]
        if cand.name == "default":
            reuse = [False] * F
        else:
            # A treatment that gives the engine exactly the default's input at a fold reuses the default's forecast.
            reuse = [bool(default_fcs) and p.control.is_default and np.array_equal(p.history, c.hist)
                     for p, c in zip(preps, ctxs)]
        need = F - sum(reuse)
        if cand.name != "default" and need == 0:
            skipped.append({"candidate": cand.name, "reason": "identical to the default on every fold; nothing to test"})
            continue
        if need > left:
            skipped.append({"candidate": cand.name,
                            "reason": f"call budget: needs {need} engine calls, {left} left of {budget}"})
            continue
        fcs: list[Forecast] = []
        spent, error = 0, None
        for k, (p, reused) in enumerate(zip(preps, reuse)):
            if reused:
                fcs.append(default_fcs[k])
                continue
            spent += 1
            try:
                fc = engine.forecast_controlled(p.history, h, p.control)
            except Exception as exc:  # one failing fold excludes this treatment; the lab itself carries on
                error = f"fold {kept[k]}: {type(exc).__name__}: {exc}"[:300]
                break
            if not _sane(fc, h):
                error = f"fold {kept[k]}: forecast has the wrong shape or non-finite values"
                break
            fcs.append(fc)
        left -= spent
        if error is not None:
            if cand.name == "default":
                raise RuntimeError(f"the default forecast failed, so there is no reference to compare with: {error}")
            failed.append({"candidate": cand.name, "error": error})
            continue
        if cand.name == "default":
            default_fcs = fcs
        runs[cand.name] = _Run(fcs, spent)

    d_sc = _scores(actual, default_fcs, mae, mse)
    rows = [_row("default", "reference", d_sc, d_sc, None, "reference")]
    decisions: dict[str, dict[str, Any]] = {}
    gain_of: dict[str, float] = {}
    pair_of: dict[str, dict[str, Any]] = {}
    for name, run in runs.items():
        if name == "default":
            continue
        sc = _scores(actual, run.fcs, mae, mse)
        pair = _paired(d_sc["mase"], sc["mase"], seed)
        decision, reason = _decide(pair, _pinball_gain(d_sc["pinball"], sc["pinball"]), F)
        decisions[name] = {"decision": decision, "reason": reason, "calls": run.spent}
        gain_of[name] = pair["gain"]
        pair_of[name] = pair
        rows.append(_row(name, "treatment", sc, d_sc, pair, decision))
    base_pair: dict[str, dict[str, Any]] = {}
    for b in baselines:
        base_pair[b] = _paired(d_sc["mase"], base_sc[b]["mase"], seed)
        rows.append(_row(b, "baseline", base_sc[b], d_sc, base_pair[b], "reference"))
    rows.sort(key=lambda r: (r["gain_pct"] is None, -(r["gain_pct"] or 0.0)))

    # Scale effect: the zero forecast beats the default on MASE but loses on pinball. MASE then rewards small forecasts on
    # this series, so no treatment may be adopted on MASE alone.
    default_mase, zero_mase = _nanmean(d_sc["mase"]), _nanmean(base_sc["zero"]["mase"])
    default_pin, zero_pin = _nanmean(d_sc["pinball"]), _nanmean(base_sc["zero"]["pinball"])
    scale_effect = is_scale_effect(zero_mase, default_mase, zero_pin, default_pin)
    if scale_effect:
        for name, d in decisions.items():
            if d["decision"] == "adopt":
                d["decision"] = "no evidence"
                d["reason"] = ("scale effect: the zero forecast beats the default on MASE but loses on pinball loss, so "
                               "MASE rewards small forecasts here; not adopted. " + d["reason"])

    simple = [b for b in baselines if b != "zero"]
    best = min(simple, key=lambda b: _nanmean(base_sc[b]["mase"]))
    best_m = _nanmean(base_sc[best]["mase"])
    # Multiplicity: every treatment is tested on the same folds, so an adoption must survive a Holm step-down correction
    # across all treatments that have a p-value.
    holm = _holm({nm: pair_of[nm]["p"] for nm in decisions})
    for nm, d in decisions.items():
        d["holm_p"] = None if holm.get(nm) is None else round(holm[nm], 4)
        if d["decision"] == "adopt" and (holm.get(nm) is None or holm[nm] > HOLM_ALPHA):
            adj = "n/a" if holm.get(nm) is None else f"{holm[nm]:.3f}"
            d["decision"] = "no evidence"
            d["reason"] = (f"does not survive Holm correction at {HOLM_ALPHA:.0%} across {len(holm)} treatments "
                           f"(adjusted p {adj}). " + d["reason"])
    # Baseline gate: the default TimesFM is only the reference, so a simple rule that beats it on these folds must be said.
    gain_base = (default_mase - best_m) / default_mase * 100 if default_mase > 0 and math.isfinite(best_m) else math.nan
    baseline_beats = ({"name": best, "gain": round(gain_base, 2)} if math.isfinite(gain_base) and gain_base > BASELINE_GATE_PCT
                      else None)

    adopted = [nm for nm, d in decisions.items() if d["decision"] == "adopt"]
    if adopted:
        top = max(adopted, key=lambda nm: gain_of[nm])
        final = by_name[top].prepare(_fold_context(x, cov.past, cov.future, n, h, m))
        recommended: Any = {"candidate": top, "treatment": by_name[top].summary,
                            "control": final.control.describe(), "note": final.note}
        if baseline_beats is not None:
            recommended["baseline_note"] = BASELINE_NOTE
    elif baseline_beats is not None:
        recommended = BASELINE_NOTE
    else:
        recommended = "keep the default"

    caveats = [
        f"{F} folds of horizon {h}: the verdict describes this series in these windows, not a general rule.",
        f"MASE and RMSSE scale each fold by the in-sample change at lag {lag} "
        f"({'seasonal' if lag > 1 else 'naive'}).",
        f"{len(decisions)} treatments were tested on the same folds; with that many, one can look useful by chance. "
        "Re-check an adopted treatment on newer data before relying on it.",
        "Figures are measured on the series supplied; label them synthetic replay, real staging or measured when reporting.",
        "Every treatment is causal: it reads only the history before its origin. Outlier flags use trailing levels, so a "
        "sustained level shift is flagged for a few periods before the level catches up.",
    ]
    if scale_effect:
        caveats.append("Scale effect: the zero forecast beats the default on MASE but loses on pinball loss; no treatment "
                       "was adopted on MASE alone.")
    if any(p["p"] is None for p in base_pair.values()) or any(p["p"] is None for p in pair_of.values()):
        caveats.append(f"Wilcoxon p-values are blank where there are fewer than {WILCOXON_MIN_FOLDS} folds or every "
                       "paired difference is zero.")
    if intermittent_share >= TSB_MIN_SHARE:
        caveats.append("Demand is intermittent on most folds, so the TSB baseline is reported; the zero forecast is "
                       "reported beside it because MASE alone rewards it.")
    if m is None:
        caveats.append("No period was given: seasonal naive repeats the last value and the seasonal window is not in the grid.")
    if dropped:
        caveats.append(f"{dropped} origin(s) dropped because their history is flat, so MASE is undefined there.")
    if skipped:
        caveats.append(f"{len(skipped)} candidate(s) were not run (budget or identical to the default); see skipped.")
    if cov.past or cov.future:
        caveats.append("Covariates are used as classified: past means known only up to each origin, future means known "
                       "ahead. The lab does not check that classification.")
    if "symmetric_averaging" in runs:
        caveats.append("Symmetric averaging runs the model twice inside each call, so its real cost is about double the "
                       "call count shown.")
    if baseline_beats is not None:
        caveats.append(f"{best} has {baseline_beats['gain']:.1f}% lower MASE than the default on these folds "
                       "(gain measured against the default's MASE).")
    if adopted:
        caveats.append(f"Adopted treatments pass a Holm correction at {HOLM_ALPHA:.0%} across the {len(holm)} treatments with a p-value.")
    if len(adopted) > 1:
        caveats.append("More than one treatment was adopted. Each was measured alone, so their combined effect is not known.")

    calls_spent = budget - left
    return {
        "horizon": h,
        "period": m,
        "points": n,
        "folds": kept,
        "fold_count": F,
        "folds_requested": n_folds,
        "fold_gap_horizons": FOLD_GAP_HORIZONS,
        "scale_lag": lag,
        "calls_spent": calls_spent,
        "max_calls": budget,
        "seed": seed,
        "thresholds": {"min_gain_pct": round(MIN_GAIN * 100, 2), "min_win_rate": MIN_WIN_RATE,
                       "pinball_tolerance": PINBALL_TOLERANCE, "confidence": CONFIDENCE,
                       "bootstrap_resamples": BOOTSTRAP_RESAMPLES},
        "reference": {"default_mase": _round(default_mase), "best_simple_baseline": best,
                      "best_simple_baseline_mase": _round(best_m),
                      "default_vs_best_simple_pct": _round((best_m - default_mase) / best_m * 100, 2) if best_m > 0 else None,
                      "zero_forecast_mase": _round(zero_mase), "scale_effect": scale_effect},
        "table": rows,
        "decisions": decisions,
        "recommended": recommended,
        "baseline_beats_default": baseline_beats,
        "skipped": skipped,
        "failed": failed,
        "caveats": caveats,
    }
