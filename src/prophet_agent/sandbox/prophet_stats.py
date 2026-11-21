"""Statistics toolkit for the analysis sandbox: ``import prophet_stats as ps``.

Motivation vs Logic
Motivation: the agent cleans and tests a series before it forecasts it. Each cleaning step can quietly help or hurt. A
centred filter reads the future, an imputer fitted on the whole table leaks the test period, a spike filter can erase
real demand, and a zero can be a stockout rather than a sale. Mistakes like these produce confident forecasts that look
better than they are.
Logic: every function returns its result together with a report of what it changed, what it could not change and why.
Causal versions use only data up to each point. The backtest in ``backtest_treatment`` gives each cleaning step only the
history before each origin and scores it against raw actuals. Nothing is filled or removed without a report entry.
Thresholds are proposed defaults and must be calibrated on synthetic replay before they are trusted on real data.

Runs inside the sandbox only, where numpy, pandas, scipy and statsmodels are available. scikit-learn is imported lazily
by the two functions that need it (``impute`` with ``method="knn"`` and ``isolation_outliers``), so the rest of the
module loads and works without it. Every report is plain JSON-safe data: no numpy types, and no NaN (a missing number
is reported as None).
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import interpolate as _interp
from scipy import signal as _signal
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.stattools import adfuller, kpss

_MAD_SCALE = 1.4826  # makes the median absolute deviation comparable to a standard deviation for normal data
_IQR_TO_SD = 1.349  # for normal data, the interquartile range is about 1.349 standard deviations
_IMPUTE_METHODS = ("auto", "linear", "spline", "seasonal", "knn", "kalman")
_TWO_SIDED_IMPUTERS = ("spline", "knn", "kalman")
_SMOOTH_METHODS = ("ewm", "median", "mean", "sg", "loess")
_BASELINES = ("naive", "seasonal_naive", "drift")


# ---------------------------------------------------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------------------------------------------------

def _values(x: Any) -> np.ndarray:
    """Return the input as a 1-D float array, with NaN for missing values. Infinite values are refused, not guessed."""
    if isinstance(x, pd.DataFrame):
        raise ValueError("expected one series, got a table; pass a single column")
    if isinstance(x, pd.Series):
        arr = x.to_numpy(dtype=float, na_value=np.nan)
    else:
        arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D series; got an array with shape {arr.shape}")
    if np.isinf(arr).any():
        raise ValueError("input has infinite values; replace or drop them before calling this function")
    return arr


def _index_of(x: Any, n: int) -> pd.Index:
    """The index to hand back with results: the Series index when there is one, otherwise 0..n-1."""
    return x.index if isinstance(x, pd.Series) else pd.RangeIndex(n)


def _py(v: Any) -> Any:
    """Convert numpy and pandas values to plain Python. Non-finite floats become None so the report is valid JSON."""
    if isinstance(v, dict):
        return {str(k): _py(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_py(i) for i in v]
    if isinstance(v, np.ndarray):
        return [_py(i) for i in v.tolist()]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    if v is None or isinstance(v, str):
        return v
    return str(v)


def _core(arr: np.ndarray) -> tuple[int, int] | None:
    """Span from the first to the last observed value as (start, stop), stop exclusive. None if nothing is observed."""
    idx = np.flatnonzero(np.isfinite(arr))
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1]) + 1


def _decomposable(arr: np.ndarray, who: str) -> tuple[tuple[int, int], np.ndarray, int]:
    """Prepare the span between the first and last observed value for STL.

    Leading and trailing missing values are left out (STL needs a solid run). Interior gaps are filled by linear
    interpolation for the decomposition only. The returned count says how many were filled, so callers can report it.
    """
    span = _core(arr)
    if span is None:
        raise ValueError(f"{who} needs at least one observed value; the series is empty or entirely missing")
    a, b = span
    core = arr[a:b]
    n_interp = int(np.isnan(core).sum())
    sub = pd.Series(core).interpolate(method="linear", limit_area="inside").to_numpy(dtype=float)
    return span, sub, n_interp


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Runs of True in a boolean array, as (start, stop) pairs with stop exclusive."""
    if not mask.any():
        return []
    padded = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def _shifted(arr: np.ndarray, offset: int) -> np.ndarray:
    """out[i] = arr[i + offset], with NaN where i + offset falls outside the series."""
    n = arr.size
    out = np.full(n, np.nan)
    if abs(offset) >= n:
        return out
    if offset >= 0:
        out[: n - offset] = arr[offset:]
    else:
        out[-offset:] = arr[: n + offset]
    return out


def _ffill(arr: np.ndarray) -> np.ndarray:
    return pd.Series(arr).ffill().to_numpy(dtype=float)


# ---------------------------------------------------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------------------------------------------------

def hampel(x: Any, window: int = 7, n_sigmas: float = 3.0, causal: bool = True) -> tuple[np.ndarray, dict]:
    """Flag spikes against a rolling median and a rolling MAD (median absolute deviation, scaled by 1.4826).

    Use it to find single-point spikes that are far from their neighbours. A point is flagged when it lies more than
    ``n_sigmas`` robust standard deviations from the local median. With ``causal=True`` the window is the points up to
    and including the current one, so no future value can change an earlier flag. That is the only setting to use for a
    model input or a backtest. ``causal=False`` centres the window, which is fine for description only.

    Failure modes: a level shift is flagged as a run of outliers at its edges, so check changepoints first. A constant
    window has a MAD of zero, so any change from it is flagged. Points before the window has half filled are not tested.
    Missing values are never flagged. Input shorter than ``window`` raises ValueError.

    Returns (flags, report). ``flags`` is a boolean array the length of the input.
    """
    arr = _values(x)
    if window < 3:
        raise ValueError("hampel needs a window of at least 3 points")
    if n_sigmas <= 0:
        raise ValueError("n_sigmas must be positive")
    if arr.size < window:
        raise ValueError(f"hampel needs at least {window} points for a window of {window}; got {arr.size}")

    s = pd.Series(arr)
    center = not causal
    min_periods = window // 2 + 1  # do not judge a point against a window that is mostly empty
    med = s.rolling(window, center=center, min_periods=min_periods).median()
    dev = (s - med).abs()
    mad = _MAD_SCALE * dev.rolling(window, center=center, min_periods=min_periods).median()
    flags = ((dev > n_sigmas * mad) & s.notna()).fillna(False).to_numpy(dtype=bool)

    count = int(flags.sum())
    observed = int(np.isfinite(arr).sum())
    report = {
        "count": count,
        "share": count / observed if observed else 0.0,
        "window": window,
        "n_sigmas": n_sigmas,
        "causal": causal,
    }
    return flags, _py(report)


def stl_outliers(x: Any, period: int, k: float = 3.0) -> tuple[np.ndarray, dict]:
    """Flag points whose robust STL remainder is unusually large (two-sided, for description and diagnosis).

    The series is decomposed with robust STL. The remainder is flagged where it lies more than ``k`` IQR-based standard
    deviations (IQR / 1.349) from the median remainder. Use it to see which points the seasonal pattern cannot explain.
    It reads the whole series, so it is not causal: do not feed its flags into a model that is backtested as causal.

    Failure modes: needs at least two full periods between the first and last observed value, else ValueError.
    Leading and trailing missing values are not scored. Interior gaps are interpolated for the decomposition only, and
    the count is reported. A constant or near-constant remainder flags nothing.

    Returns (flags, report). The report has causal=False.
    """
    arr = _values(x)
    period = int(period)
    if period < 2:
        raise ValueError("stl_outliers needs a period of at least 2")
    if k <= 0:
        raise ValueError("k must be positive")
    span, sub, n_interp = _decomposable(arr, "stl_outliers")
    if sub.size < 2 * period:
        raise ValueError(
            f"stl_outliers needs at least {2 * period} points between the first and last observed value for period "
            f"{period}; got {sub.size}"
        )

    res = STL(sub, period=period, robust=True).fit()
    resid = np.asarray(res.resid, dtype=float)
    q25, q75 = np.percentile(resid, [25, 75])
    sigma = float((q75 - q25) / _IQR_TO_SD)
    median = float(np.median(resid))
    sub_flags = np.abs(resid - median) > k * sigma if sigma > 0 else np.zeros(sub.size, dtype=bool)

    a, b = span
    flags = np.zeros(arr.size, dtype=bool)
    flags[a:b] = sub_flags & np.isfinite(arr[a:b])
    count = int(flags.sum())
    observed = int(np.isfinite(arr).sum())
    report = {
        "count": count,
        "share": count / observed if observed else 0.0,
        "period": period,
        "k": k,
        "sigma": sigma,
        "interpolated_for_fit": n_interp,
        "missing_outside_span": int(arr.size - (b - a)),
        "causal": False,
    }
    return flags, _py(report)


# ---------------------------------------------------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------------------------------------------------

def _seasonal_candidate(arr: np.ndarray, period: int, causal: bool) -> np.ndarray:
    """Mean of the same-phase values in other cycles. Causal use only earlier cycles. NaN where none is observed."""
    n = arr.size
    fin = np.isfinite(arr)
    vals = np.where(fin, arr, 0.0)
    total = np.zeros(n)
    cnt = np.zeros(n)
    for m in range(1, n // period + 1):
        off = m * period
        if off >= n:
            break
        total[off:] += vals[:-off]
        cnt[off:] += fin[:-off]
        if not causal:
            total[:-off] += vals[off:]
            cnt[:-off] += fin[off:]
    return np.where(cnt > 0, total / np.maximum(cnt, 1.0), np.nan)


def _spline_candidate(arr: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    """Quadratic spline through the observed points (two-sided)."""
    obs = np.flatnonzero(np.isfinite(arr))
    if obs.size < 3:
        return None, "spline needs at least 3 observed points"
    spl = _interp.make_interp_spline(obs, arr[obs], k=2)
    out = np.full(arr.size, np.nan)
    lo, hi = int(obs[0]), int(obs[-1])
    out[lo : hi + 1] = spl(np.arange(lo, hi + 1))
    return out, None


def _knn_candidate(arr: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    """KNN imputation on a matrix of lag and lead values, so each row's neighbours shape its fill (two-sided)."""
    from sklearn.impute import KNNImputer  # lazy: scikit-learn is optional for the rest of the module

    n_obs = int(np.isfinite(arr).sum())
    if n_obs < 3:
        return None, "knn needs at least 3 observed points"
    offsets = [-3, -2, -1, 0, 1, 2, 3]
    cols = [_shifted(arr, o) for o in offsets]
    keep = [i for i, c in enumerate(cols) if np.isfinite(c).any()]
    matrix = np.column_stack([cols[i] for i in keep])
    center = keep.index(offsets.index(0))
    neighbours = np.delete(matrix, center, axis=1)
    no_neighbours = ~np.isfinite(neighbours).any(axis=1)
    imputer = KNNImputer(n_neighbors=min(5, n_obs - 1))
    filled = imputer.fit_transform(matrix)[:, center].astype(float)
    out = filled.copy()
    out[no_neighbours & ~np.isfinite(arr)] = np.nan  # the imputer would fall back to a global mean; refuse that
    return out, None


def _kalman_candidate(arr: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    """Smoothed local level from a Kalman smoother (two-sided). Missing values are handled inside the state space."""
    if int(np.isfinite(arr).sum()) < 10:
        return None, "kalman needs at least 10 observed points"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = UnobservedComponents(arr, level="llevel").fit(disp=False)
        except Exception as exc:  # statsmodels raises several exception types for degenerate data
            raise ValueError(f"kalman fit failed: {exc}") from exc
    return np.asarray(res.smoothed_state[0], dtype=float).ravel(), None


def _candidate(name: str, arr: np.ndarray, period: int | None, causal: bool) -> tuple[np.ndarray | None, str | None]:
    """Fill values for every position from one method. Gaps are only filled from these where the method applies."""
    if name == "linear":
        s = pd.Series(arr)
        vals = s.ffill() if causal else s.interpolate(method="linear", limit_area="inside")
        return vals.to_numpy(dtype=float), None
    if name == "seasonal":
        if period is None:
            raise ValueError("seasonal imputation needs a period")
        return _seasonal_candidate(arr, period, causal), None
    if name == "spline":
        return _spline_candidate(arr)
    if name == "knn":
        return _knn_candidate(arr)
    if name == "kalman":
        return _kalman_candidate(arr)
    raise ValueError(f"unknown imputation method {name!r}")


def impute(
    x: Any,
    method: str = "auto",
    period: int | None = None,
    max_gap: int | None = None,
    causal: bool = False,
) -> tuple[pd.Series, dict]:
    """Fill interior gaps, reporting every fill. Leading and trailing missing values are never filled.

    Methods:
      "linear"   time-ordered linear interpolation. With causal=True it becomes a forward fill (the last observed value).
      "spline"   quadratic spline through the observed points (two-sided).
      "seasonal" mean of the same phase in other cycles (needs period). With causal=True only earlier cycles are used.
      "knn"      sklearn KNNImputer on lag and lead values (two-sided, lazy import).
      "kalman"   smoothed local level from a statsmodels Kalman smoother (two-sided).
      "auto"     a gap of up to 3 points: linear. A gap longer than 3 and no longer than ``period``: seasonal.
                 Any longer gap is left unfilled, because imputed values in a long gap are mostly the imputer's own
                 assumptions.

    ``max_gap`` leaves any gap longer than it unfilled, whatever the method. ``causal=True`` forbids the two-sided
    methods (spline, knn, kalman), so no future value reaches a past position.

    Failure modes: gaps a method cannot fill (for example, no same-phase value for seasonal) are left unfilled and listed
    with a reason. Fewer than two points raises ValueError. The report lists each filled gap with its method, each
    unfilled gap with its reason, and the counts of leading and trailing missing values that were left alone.

    Returns (series, report). The series has the input's index and name.
    """
    arr = _values(x)
    n = arr.size
    index = _index_of(x, n)
    name = getattr(x, "name", None)
    if method not in _IMPUTE_METHODS:
        raise ValueError(f"method must be one of {_IMPUTE_METHODS}; got {method!r}")
    if causal and method in _TWO_SIDED_IMPUTERS:
        raise ValueError(
            f"{method} is two-sided and reads values after each gap. Pass causal=False, or use method='linear' or "
            "'seasonal' with causal=True"
        )
    if method == "seasonal" and period is None:
        raise ValueError("seasonal imputation needs a period, for example period=7 for weekly cycles")
    if period is not None and int(period) < 2:
        raise ValueError("period must be at least 2")
    if period is not None:
        period = int(period)
    if max_gap is not None and max_gap < 1:
        raise ValueError("max_gap must be at least 1")
    if n < 2:
        raise ValueError("impute needs at least 2 points")

    missing = ~np.isfinite(arr)
    span = _core(arr)
    if span is None:
        leading, trailing = n, 0  # nothing observed: every position is outside the span
    else:
        leading, trailing = span[0], n - span[1]

    out = arr.copy()
    filled_positions: list[int] = []
    method_by_gap: list[dict] = []
    unfilled: list[dict] = []
    cache: dict[str, tuple[np.ndarray | None, str | None]] = {}

    def gap_label(a: int, b: int) -> dict:
        return {"start": a, "end": b - 1, "length": b - a, "start_label": str(index[a]), "end_label": str(index[b - 1])}

    if span is not None:
        for a, b in _runs(missing):
            if a < span[0] or b > span[1]:
                continue  # leading or trailing, never filled
            length = b - a
            if max_gap is not None and length > max_gap:
                unfilled.append({**gap_label(a, b), "reason": f"longer than max_gap={max_gap}"})
                continue
            if method == "auto":
                if length <= 3:
                    choice = "linear"
                elif period is not None and length <= period:
                    choice = "seasonal"
                else:
                    unfilled.append({
                        **gap_label(a, b),
                        "reason": "longer than 3 points and longer than one period; auto leaves it unfilled",
                    })
                    continue
            else:
                choice = method
            if choice not in cache:
                cache[choice] = _candidate(choice, arr, period, causal)
            vals, why = cache[choice]
            if vals is None or np.isnan(vals[a:b]).any():
                reason = why or "the method has no value for every missing position in this gap"
                unfilled.append({**gap_label(a, b), "reason": reason})
                continue
            out[a:b] = vals[a:b]
            filled_positions.extend(range(a, b))
            method_by_gap.append({**gap_label(a, b), "method": choice})

    series = pd.Series(out, index=index, name=name)
    report = {
        "method": method,
        "causal": causal,
        "period": period,
        "max_gap": max_gap,
        "missing_before": int(missing.sum()),
        "filled_count": len(filled_positions),
        "filled_positions": filled_positions,
        "method_by_gap": method_by_gap,
        "unfilled_gaps": unfilled,
        "leading_missing": leading,
        "trailing_missing": trailing,
    }
    return series, _py(report)


# ---------------------------------------------------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------------------------------------------------

def smooth(x: Any, method: str = "ewm", window: int = 7, causal: bool = True, **kw: Any) -> pd.Series:
    """Smooth a series. Missing values stay missing in the output.

    Causal methods (use these for model inputs): "ewm" (exponentially weighted mean; pass alpha, span, halflife or com
    through **kw, otherwise span=window), "median" and "mean" (trailing windows of ``window`` points).
    Two-sided methods, which read values after each point and so need causal=False: "sg" (Savitzky-Golay, odd window,
    polyorder via **kw, default 2) and "loess" (statsmodels lowess; frac via **kw, default about window / observed count).
    Passing causal=True with "sg" or "loess" raises ValueError.

    With causal=False, "ewm" becomes the average of a forward and a backward pass, and "median" and "mean" use centred
    windows.

    Failure modes: "sg" needs at least ``window`` points between the first and last observed value. "loess" needs at
    least 3 observed points. Unused keyword arguments for median and mean raise rather than being ignored.
    Interior gaps are not filled by any method here.

    Returns a pd.Series with the input's index and name.
    """
    arr = _values(x)
    index = _index_of(x, arr.size)
    name = getattr(x, "name", None)
    if method not in _SMOOTH_METHODS:
        raise ValueError(f"method must be one of {_SMOOTH_METHODS}; got {method!r}")
    if window < 1:
        raise ValueError("window must be at least 1")
    if method in ("sg", "loess") and causal:
        raise ValueError(
            f"{method} is two-sided: it reads values after each point. Pass causal=False to use it, and do not feed "
            "its output to a model that is backtested as causal"
        )

    missing = ~np.isfinite(arr)
    s = pd.Series(arr)
    if method == "ewm":
        args = dict(kw)
        if not any(key in args for key in ("alpha", "span", "halflife", "com")):
            args["span"] = window
        forward = s.ewm(**args).mean().to_numpy(dtype=float)
        if causal:
            out = forward
        else:
            backward = pd.Series(arr[::-1]).ewm(**args).mean().to_numpy(dtype=float)[::-1]
            out = (forward + backward) / 2.0
    elif method in ("median", "mean"):
        if kw:
            raise ValueError(f"unused keyword arguments for {method}: {sorted(kw)}")
        roll = s.rolling(window, min_periods=1, center=not causal)
        out = (roll.median() if method == "median" else roll.mean()).to_numpy(dtype=float)
    elif method == "sg":
        polyorder = int(kw.pop("polyorder", 2))
        if kw:
            raise ValueError(f"unused keyword arguments for sg: {sorted(kw)}")
        if window % 2 == 0 or window <= polyorder:
            raise ValueError("sg needs an odd window larger than polyorder")
        out = np.full(arr.size, np.nan)
        span = _core(arr)
        if span is not None:
            a, b = span
            sub = pd.Series(arr[a:b]).interpolate(limit_area="inside").to_numpy(dtype=float)
            if sub.size < window:
                raise ValueError(
                    f"sg needs at least {window} points between the first and last observed value; got {sub.size}"
                )
            out[a:b] = _signal.savgol_filter(sub, window_length=window, polyorder=polyorder, mode="interp")
    else:  # loess
        frac = kw.pop("frac", None)
        obs = np.flatnonzero(~missing)
        if obs.size < 3:
            raise ValueError("loess needs at least 3 observed points")
        if frac is None:
            frac = min(1.0, max(window, 3) / obs.size)
        fitted = lowess(arr[obs], obs.astype(float), frac=float(frac), return_sorted=False, **kw)
        out = np.full(arr.size, np.nan)
        out[obs] = fitted

    out = np.where(missing, np.nan, out)
    return pd.Series(out, index=index, name=name)


# ---------------------------------------------------------------------------------------------------------------------
# Changepoints
# ---------------------------------------------------------------------------------------------------------------------

def changepoints(
    x: Any,
    min_size: int = 20,
    penalty: float | None = None,
    max_changes: int = 5,
) -> tuple[list[dict], dict]:
    """Find shifts in the mean by binary segmentation with a BIC-style penalty (numpy only).

    The best split of each current segment is found by least squares. The split with the largest drop in residual sum
    of squares is accepted while that drop exceeds ``penalty``, up to ``max_changes`` splits. The default penalty is
    3 * log(n) * variance, in the same units as the residual sum of squares. Use it on history only: a changepoint fitted
    on the full series moves the segment means that a backtest sees in the past.

    Each change reports the index (position in the input), the mean before and after it, the shift, and the shift in
    within-segment standard deviations (None when that deviation is zero). Missing values are dropped for the search and
    counted in the report.

    Failure modes: needs at least 2 * min_size observed points, else ValueError. A constant series returns no changes.
    A change that lasts less than about a seasonal cycle is an outlier or a transient, not a regime; decide that on the
    segment lengths, not on this function alone.

    Returns (changes, report). ``changes`` is sorted by index.
    """
    arr = _values(x)
    index = _index_of(x, arr.size)
    if min_size < 2:
        raise ValueError("min_size must be at least 2")
    if max_changes < 0:
        raise ValueError("max_changes must be zero or more")
    pos = np.flatnonzero(np.isfinite(arr))
    y = arr[pos]
    n = y.size
    dropped = int(arr.size - n)
    if n < 2 * min_size:
        raise ValueError(f"changepoints needs at least {2 * min_size} observed points for min_size={min_size}; got {n}")

    var = float(np.var(y))
    if penalty is None:
        penalty = 3.0 * math.log(n) * var
    if penalty < 0:
        raise ValueError("penalty must be zero or more")
    report = {
        "n_changes": 0,
        "min_size": min_size,
        "penalty": float(penalty),
        "max_changes": max_changes,
        "observed_points": n,
        "dropped_missing": dropped,
    }
    if var == 0 or max_changes == 0:
        return [], _py(report)

    cs1 = np.concatenate([[0.0], np.cumsum(y)])
    cs2 = np.concatenate([[0.0], np.cumsum(y * y)])

    def best_split(a: int, b: int) -> tuple[float, int | None]:
        ks = np.arange(a + min_size, b - min_size + 1)
        if ks.size == 0:
            return -math.inf, None
        left = (cs2[ks] - cs2[a]) - (cs1[ks] - cs1[a]) ** 2 / (ks - a)
        right = (cs2[b] - cs2[ks]) - (cs1[b] - cs1[ks]) ** 2 / (b - ks)
        total = left + right
        j = int(np.argmin(total))
        base = (cs2[b] - cs2[a]) - (cs1[b] - cs1[a]) ** 2 / (b - a)
        return float(base - total[j]), int(ks[j])

    segments: list[tuple[int, int]] = [(0, n)]
    splits: list[int] = []
    while len(splits) < max_changes:
        candidates = []
        for a, b in segments:
            if b - a >= 2 * min_size:
                gain, k = best_split(a, b)
                if k is not None:
                    candidates.append((gain, a, b, k))
        if not candidates:
            break
        gain, a, b, k = max(candidates)
        if gain <= penalty:
            break
        splits.append(k)
        segments.remove((a, b))
        segments.extend([(a, k), (k, b)])

    bounds = [0] + sorted(splits) + [n]
    residual_ss = 0.0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        seg = y[lo:hi]
        residual_ss += float(np.sum((seg - seg.mean()) ** 2))
    sd = math.sqrt(residual_ss / n)

    changes: list[dict] = []
    for j, k in enumerate(sorted(splits), start=1):
        before = float(y[bounds[j - 1] : k].mean())
        after = float(y[k : bounds[j + 1]].mean())
        shift = after - before
        changes.append({
            "index": int(pos[k]),
            "label": str(index[int(pos[k])]),
            "before_mean": before,
            "after_mean": after,
            "shift": shift,
            "shift_in_sd": (shift / sd) if sd > 0 else None,
        })
    report["n_changes"] = len(changes)
    return _py(changes), _py(report)


# ---------------------------------------------------------------------------------------------------------------------
# Derivative features
# ---------------------------------------------------------------------------------------------------------------------

def derivative_features(x: Any, windows: tuple[int, ...] = (3, 7, 28)) -> pd.DataFrame:
    """Causal features about the rate of change and how unusual each value is.

    Columns: diff1 (change from the previous point), diff2 (change in that change), and for each window w:
    slope_w (trailing least-squares slope per step over the last w points, including the current point), std_w
    (trailing standard deviation including the current point), and z_w (the current value minus the mean of the previous
    w points, divided by their standard deviation). z_w excludes the current point from its window so a value is not
    compared with itself. Zero spread gives NaN, not infinity.

    Every column uses only the current and earlier points, so no column changes when a future value changes.

    Failure modes: each window needs at least 2 points, and the input needs at least as many points as the longest
    window, else ValueError. Missing values give NaN in every column that uses them. The output has the input's index.
    """
    arr = _values(x)
    index = _index_of(x, arr.size)
    windows = tuple(int(w) for w in windows)
    if not windows or min(windows) < 2:
        raise ValueError("each window needs at least 2 points (a slope needs two)")
    if arr.size < max(windows):
        raise ValueError(f"derivative_features needs at least {max(windows)} points for the longest window; got {arr.size}")

    s = pd.Series(arr, index=index)
    t = pd.Series(np.arange(arr.size, dtype=float), index=index)
    d1 = s.diff()
    cols: dict[str, pd.Series] = {"diff1": d1, "diff2": d1.diff()}
    prev = s.shift(1)
    for w in windows:
        cols[f"slope_{w}"] = t.rolling(w, min_periods=w).cov(s) / t.rolling(w, min_periods=w).var()
        cols[f"std_{w}"] = s.rolling(w, min_periods=w).std()
        prev_std = prev.rolling(w, min_periods=w).std().replace(0.0, np.nan)
        cols[f"z_{w}"] = (s - prev.rolling(w, min_periods=w).mean()) / prev_std
    return pd.DataFrame(cols, index=index)


# ---------------------------------------------------------------------------------------------------------------------
# Censoring
# ---------------------------------------------------------------------------------------------------------------------

def _plateau(arr: np.ndarray, fin: np.ndarray, value: float, n_obs: int) -> dict | None:
    """A value at the series extreme that repeats in a run of three or more points, else None."""
    at = fin & (arr == value)
    runs = _runs(at)
    longest = max((b - a for a, b in runs), default=0)
    if longest < 3:
        return None
    return {"value": float(value), "share": float(at.sum()) / n_obs, "longest_run": int(longest)}


def censoring_check(x: Any) -> dict:
    """Look for signs that zeros or extreme values are censored (stockouts, outages, floors or caps), not real values.

    Reports the zero share, the longest zero run, a floor plateau (the minimum repeating in a run of three or more) and a
    ceiling plateau (the same at the maximum). ``probable_stockouts`` lists zero runs of three or more points that sit
    between non-zero values; a zero run at either end of the series is not listed, because it may just be the start or
    the end of the record. A recommendation sentence summarises the findings.

    Failure modes: this is a screen, not a test. It cannot tell a stockout from a real zero without a second signal such
    as inventory, opening hours or an outage log. The 20 percent zero-share threshold in the recommendation is a proposed
    default to calibrate. All-NaN or constant input returns a valid report.
    """
    arr = _values(x)
    fin = np.isfinite(arr)
    obs = arr[fin]
    n_obs = int(obs.size)
    report: dict[str, Any] = {
        "observed": n_obs,
        "zero_share": 0.0,
        "longest_zero_run": 0,
        "floor_plateau": None,
        "ceiling_plateau": None,
        "probable_stockouts": [],
        "recommendation": "",
    }
    if n_obs == 0:
        report["recommendation"] = "No observed values, so there is nothing to check. Check the feed."
        return _py(report)

    zero = fin & (arr == 0)
    zero_runs = _runs(zero)
    report["zero_share"] = float(zero.sum()) / n_obs
    report["longest_zero_run"] = int(max((b - a for a, b in zero_runs), default=0))
    lo, hi = float(obs.min()), float(obs.max())
    report["floor_plateau"] = _plateau(arr, fin, lo, n_obs)
    report["ceiling_plateau"] = _plateau(arr, fin, hi, n_obs)

    nonzero_pos = np.flatnonzero(fin & (arr != 0))
    stockouts: list[dict] = []
    if nonzero_pos.size:
        first, last = int(nonzero_pos[0]), int(nonzero_pos[-1])
        for a, b in zero_runs:
            if b - a >= 3 and a > first and b - 1 < last:
                stockouts.append({"start": int(a), "end": int(b - 1), "length": int(b - a)})
    report["probable_stockouts"] = stockouts

    if lo == hi:
        rec = "The series is constant, so the censoring checks say little. Check that the feed is live."
    elif stockouts:
        rec = (
            f"{len(stockouts)} run(s) of three or more zeros sit between non-zero values. Check stock, outage or "
            "opening records before treating them as demand; they may be censored."
        )
    elif report["zero_share"] > 0.2:  # proposed default, to calibrate on synthetic replay
        rec = (
            f"Zeros make up {report['zero_share']:.0%} of observed points, with no zero run between non-zero values. "
            "Check whether this is intermittent demand before modelling it as continuous."
        )
    elif report["floor_plateau"] or report["ceiling_plateau"]:
        rec = (
            "Values repeat at the series minimum or maximum in runs of three or more. Check for a floor or cap "
            "(censoring) before modelling."
        )
    else:
        rec = "No censoring pattern found. Zeros look like ordinary values."
    report["recommendation"] = rec
    return _py(report)


# ---------------------------------------------------------------------------------------------------------------------
# Seasonality and stationarity
# ---------------------------------------------------------------------------------------------------------------------

def _strength(var_r: float, var_xr: float) -> float:
    if var_xr <= 0:
        return 0.0
    return max(0.0, 1.0 - var_r / var_xr)


def seasonal_strength(x: Any, period: int) -> dict:
    """Strength of trend and seasonality from an STL decomposition (Hyndman, FPP3).

    trend_strength = max(0, 1 - var(R) / var(T + R)) and seasonal_strength = max(0, 1 - var(R) / var(S + R)), where T, S
    and R are the trend, seasonal and remainder parts. Values near 1 mean the component dominates; near 0 means it is
    weak. Use it to decide whether seasonal adjustment or a seasonal model is worth the cost.

    Failure modes: needs at least 2 * period points between the first and last observed value, else ValueError.
    Interior gaps are interpolated for the decomposition only, and the count is reported. Leading and trailing missing
    values are excluded, and ``n`` is the length actually used. A constant series scores 0 on both.
    """
    arr = _values(x)
    period = int(period)
    if period < 2:
        raise ValueError("seasonal_strength needs a period of at least 2")
    _, sub, n_interp = _decomposable(arr, "seasonal_strength")
    if sub.size < 2 * period:
        raise ValueError(
            f"seasonal_strength needs at least {2 * period} points between the first and last observed value for "
            f"period {period}; got {sub.size}"
        )
    res = STL(sub, period=period, robust=True).fit()
    trend = np.asarray(res.trend, dtype=float)
    seas = np.asarray(res.seasonal, dtype=float)
    resid = np.asarray(res.resid, dtype=float)
    var_r = float(np.var(resid))
    report = {
        "trend_strength": _strength(var_r, float(np.var(trend + resid))),
        "seasonal_strength": _strength(var_r, float(np.var(seas + resid))),
        "period": period,
        "n": int(sub.size),
        "interpolated": n_interp,
    }
    return _py(report)


def stationarity(x: Any) -> dict:
    """Run ADF and KPSS tests together and give a verdict on stationarity.

    ADF (null: unit root) gives ``adf_p``. KPSS (null: stationary) is run twice: around a level (``kpss_level_p``) and
    around a trend (``kpss_trend_p``). The verdict rule, applied in order, with alpha = 0.05:
      1. stationary:       adf_p < 0.05 and kpss_level_p >= 0.05 (both tests agree the series is stationary)
      2. trend_stationary: kpss_level_p < 0.05 and kpss_trend_p >= 0.05 (a trend explains the level rejection)
      3. unit_root:        adf_p >= 0.05 and kpss_level_p < 0.05 and kpss_trend_p < 0.05 (both point to a unit root)
      4. inconclusive:     anything else, including the tests disagreeing
    statsmodels clips KPSS p-values to the range 0.01 to 0.10, so they are coarse. Treat them as evidence, not as exact.

    Failure modes: needs at least 20 observed points, else ValueError. A constant series returns "inconclusive" with the
    p-values set to None. A verdict is not a guarantee; check it against the plot and the seasonal strength.
    """
    arr = _values(x)
    y = arr[np.isfinite(arr)]
    if y.size < 20:
        raise ValueError(f"stationarity needs at least 20 observed points; got {y.size}")
    if np.ptp(y) == 0:
        return {
            "adf_p": None,
            "kpss_level_p": None,
            "kpss_trend_p": None,
            "verdict": "inconclusive",
            "note": "constant series; the tests do not apply",
            "n": int(y.size),
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS reports interpolated p-values with a warning we do not need
        adf_p = float(adfuller(y, autolag="AIC")[1])
        kpss_level_p = float(kpss(y, regression="c", nlags="auto")[1])
        kpss_trend_p = float(kpss(y, regression="ct", nlags="auto")[1])

    alpha = 0.05
    if adf_p < alpha and kpss_level_p >= alpha:
        verdict = "stationary"
    elif kpss_level_p < alpha and kpss_trend_p >= alpha:
        verdict = "trend_stationary"
    elif adf_p >= alpha and kpss_level_p < alpha and kpss_trend_p < alpha:
        verdict = "unit_root"
    else:
        verdict = "inconclusive"
    report = {
        "adf_p": adf_p,
        "kpss_level_p": kpss_level_p,
        "kpss_trend_p": kpss_trend_p,
        "verdict": verdict,
        "n": int(y.size),
    }
    return _py(report)


# ---------------------------------------------------------------------------------------------------------------------
# Leakage-safe backtest of a cleaning step
# ---------------------------------------------------------------------------------------------------------------------

def _forecast(train: np.ndarray, horizon: int, baseline: str, period: int | None) -> np.ndarray:
    """Point forecasts from a forward-filled training array. NaN where the baseline has no value."""
    out = np.full(horizon, np.nan)
    if train.size == 0 or not np.isfinite(train[-1]):
        return out
    last = float(train[-1])
    if baseline == "naive":
        return np.full(horizon, last)
    if baseline == "drift":
        first_idx = int(np.flatnonzero(np.isfinite(train))[0])
        span = train.size - 1 - first_idx
        slope = (last - float(train[first_idx])) / span if span > 0 else 0.0
        return last + slope * np.arange(1, horizon + 1)
    # seasonal_naive: repeat the last full cycle
    if period is None or train.size < period:
        return out
    idx = train.size - period + (np.arange(horizon) % period)
    return train[idx].astype(float)


def _scale(raw_train: np.ndarray, lag: int) -> float:
    """Mean absolute in-sample difference at the given lag (the MASE scale), from observed raw training values."""
    if raw_train.size <= lag:
        return float("nan")
    d = np.abs(raw_train[lag:] - raw_train[:-lag])
    d = d[np.isfinite(d)]
    return float(d.mean()) if d.size else float("nan")


def backtest_treatment(
    x: Any,
    treat: Callable[[pd.Series], pd.Series],
    horizon: int,
    origins: int = 5,
    period: int | None = None,
    baseline: str = "seasonal_naive",
) -> dict:
    """Leakage-safe test of whether a cleaning step (``treat``) helps a simple forecast.

    For each of ``origins`` evenly spaced origins, ``treat`` is called with only the data before that origin. A baseline
    ("naive", "seasonal_naive" with period, or "drift") then forecasts the next ``horizon`` points from both the treated
    and the raw training data. Both forecasts are scored against the RAW actuals with MASE, scaled by the mean absolute
    in-sample naive difference (seasonal difference for seasonal_naive) of the raw training data. A NaN in the treated
    series is filled forward before forecasting. A leading NaN that remains falls back to the raw value, and the count is
    reported.

    The decision rule: ``helped`` is True only when the treated forecast wins at more than half the origins AND the mean
    per-origin improvement is above 2.0 percent. 2.0 is a proposed default, to calibrate on synthetic replay. Origins
    without usable actuals or with a zero scale are skipped and counted.

    Failure modes: ``treat`` must return a series of the same length as its input, else ValueError. Too short a series for
    the horizon raises ValueError. If no origin can be scored, ValueError is raised rather than returning a false
    verdict. The function never gives ``treat`` any data after an origin.

    Returns a dict with origins (scored), mase_treated, mase_untreated (means over origins), wins, mean_improvement_pct,
    helped, plus the settings and the skip counts.
    """
    arr = _values(x)
    n = arr.size
    horizon = int(horizon)
    origins = int(origins)
    if horizon < 1 or origins < 1:
        raise ValueError("horizon and origins must both be at least 1")
    if baseline not in _BASELINES:
        raise ValueError(f"baseline must be one of {_BASELINES}; got {baseline!r}")
    if baseline == "seasonal_naive":
        if period is None or int(period) < 1:
            raise ValueError("seasonal_naive needs a period")
        period = int(period)
    lag = period if baseline == "seasonal_naive" else 1
    min_train = max(3, lag + 1)
    last_origin = n - horizon
    if last_origin < min_train:
        raise ValueError(
            f"backtest_treatment needs at least {min_train + horizon} points for horizon {horizon}; got {n}"
        )
    cut_points = np.unique(np.round(np.linspace(min_train, last_origin, origins)).astype(int))

    rows: list[tuple[float, float]] = []
    skipped = 0
    treated_nan = 0
    raw_fallback = 0
    for o in cut_points:
        o = int(o)
        raw_train = arr[:o].copy()
        actual = arr[o : o + horizon]
        ok = np.isfinite(actual)
        if not ok.any():
            skipped += 1
            continue

        treated_out = treat(pd.Series(raw_train))
        t_arr = _values(treated_out)
        if t_arr.size != o:
            raise ValueError("treat must return a series with the same length as its input")
        treated_nan += int(np.isnan(t_arr).sum())
        t_ff = _ffill(t_arr)
        leading = np.isnan(t_ff)
        raw_fallback += int(leading.sum())
        t_ff = np.where(leading, raw_train, t_ff)

        f_raw = _forecast(_ffill(raw_train), horizon, baseline, period)
        f_tr = _forecast(t_ff, horizon, baseline, period)
        scale = _scale(raw_train, lag)
        mask = ok & np.isfinite(f_raw) & np.isfinite(f_tr)
        if not mask.any() or not np.isfinite(scale) or scale <= 0:
            skipped += 1
            continue
        mase_raw = float(np.mean(np.abs(actual[mask] - f_raw[mask]))) / scale
        mase_tr = float(np.mean(np.abs(actual[mask] - f_tr[mask]))) / scale
        rows.append((mase_tr, mase_raw))

    if not rows:
        raise ValueError("no origin had usable data to score; check for missing values or a constant series")

    mt = np.array([r[0] for r in rows])
    mu = np.array([r[1] for r in rows])
    wins = int(np.sum(mt < mu))
    pct = [(m_u - m_t) / m_u * 100.0 for m_t, m_u in rows if m_u > 0]
    mean_improvement = float(np.mean(pct)) if pct else 0.0
    # Proposed rule: wins at more than half of origins AND mean improvement above 2 percent (to calibrate).
    helped = bool(wins > len(rows) / 2 and mean_improvement > 2.0)
    report = {
        "origins": len(rows),
        "mase_treated": float(mt.mean()),
        "mase_untreated": float(mu.mean()),
        "wins": wins,
        "mean_improvement_pct": mean_improvement,
        "helped": helped,
        "baseline": baseline,
        "horizon": horizon,
        "period": period,
        "origins_requested": int(cut_points.size),
        "skipped_origins": skipped,
        "treated_nan_returned": treated_nan,
        "leading_raw_fallback": raw_fallback,
    }
    return _py(report)


# ---------------------------------------------------------------------------------------------------------------------
# Multivariate outliers
# ---------------------------------------------------------------------------------------------------------------------

def isolation_outliers(
    X: Any,
    contamination: float = 0.01,
    random_state: int = 0,
) -> tuple[np.ndarray, dict]:
    """Flag unusual rows with an isolation forest (scikit-learn, imported here only).

    Use it as a second opinion on spikes, with the local-window features that the Hampel and STL flags use. Rows with any
    missing value are not scored and are not flagged; they are counted in the report. ``contamination`` is the expected
    share of outlying rows (0 to 0.5). The fixed ``random_state`` makes the result repeatable.

    Failure modes: fewer than 10 complete rows raises ValueError. The forest flags the most isolated rows whether or not
    they are errors; a flag is a prompt to look, not a reason to delete. Requires scikit-learn, and raises ImportError
    without it.

    Returns (flags, report) with flags the length of the input's rows.
    """
    from sklearn.ensemble import IsolationForest  # lazy: scikit-learn is optional for the rest of the module

    matrix = np.asarray(X, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError("isolation_outliers needs a 1-D or 2-D array")
    if not (0 < contamination <= 0.5):
        raise ValueError("contamination must be above 0 and at most 0.5")
    complete = np.isfinite(matrix).all(axis=1)
    n_ok = int(complete.sum())
    if n_ok < 10:
        raise ValueError(f"isolation_outliers needs at least 10 complete rows; got {n_ok}")

    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(matrix[complete])
    flags = np.zeros(matrix.shape[0], dtype=bool)
    flags[np.flatnonzero(complete)] = labels == -1
    count = int(flags.sum())
    report = {
        "count": count,
        "share": count / n_ok,
        "contamination": contamination,
        "random_state": random_state,
        "rows": int(matrix.shape[0]),
        "rows_scored": n_ok,
        "rows_skipped_missing": int(matrix.shape[0] - n_ok),
        "features": int(matrix.shape[1]),
    }
    return flags, _py(report)


# ---------------------------------------------------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------------------------------------------------

class Ledger:
    """The data-quality and decision ledger: what was found, what was decided, and the evidence for it.

    Motivation: a careful analyst records why every series was cleaned, trimmed or left alone, so a reviewer can trace a
    forecast back to its decisions. Logic: entries are appended in order, each with a running index starting at 1. Every
    entry needs a stage, a finding and a decision. Evidence is optional and is stored as plain JSON-safe data.
    """

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def add(self, stage: str, finding: str, decision: str, evidence: Any = None) -> None:
        """Record one finding and the decision taken about it. stage, finding and decision must be non-empty text."""
        for field_name, value in (("stage", stage), ("finding", finding), ("decision", decision)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"every ledger entry needs a non-empty {field_name}")
        self._entries.append({
            "stage": stage,
            "finding": finding,
            "decision": decision,
            "evidence": _py(evidence),
        })

    def to_dict(self) -> list[dict[str, Any]]:
        """All entries in order, each with its running index (1, 2, 3 and so on)."""
        return [{"index": i, **entry} for i, entry in enumerate(self._entries, start=1)]

    def save(self, path: str = "out/ledger.json") -> str:
        """Write the entries as JSON, creating the parent folder if needed. Returns the path written."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return str(target)
