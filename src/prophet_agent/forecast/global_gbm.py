"""One global gradient-boosted model for a panel of related series, compared with local and foundation models.

Motivation vs Logic
Motivation: in a panel (many products or sites on one calendar) the M5 competition winners fitted one model across all
series rather than one per series. Pooling lets a short series borrow the shape of long ones. The question for the person
is whether that pooling beats a naive rule and TimesFM on the same past windows, and whether the 10 to 90 band it reports
covers what it claims to cover.
Logic: each origin is divided by its own level (the mean absolute value of the history before that origin), so big and
small entities share one scale. Features for an origin are lags, rolling mean and spread, the step position in the season,
the log level and an entity code. Every feature uses only values strictly before the origin. Training rows are stacked for
horizons 1 to H (a horizon feature tells the model which step it is predicting), so one model covers all steps. Three
quantile models give the lower, middle and upper band.
The raw quantile band under-covers (A2-02: 0.68 against 0.80 on a heavy-tailed synthetic panel, and 0.37 on the Beijing
run, measured). The quantile models are scored on rows they were trained on, so the band is too narrow. The band is
therefore calibrated by split conformal prediction (Romano, Patterson and Candes 2019, CQR form). A separate model is
trained on each entity's history before its last k windows, and its out-of-time scores on those windows set one
correction per horizon third. Only the history passed in is used, so windows that a backtest scores never reach the
calibration. Calibration never moves the point forecast, which stays the middle quantile of the raw model.
``panel_backtest`` refits at each of the last few non-overlapping origins using only history before that origin, then scores
every model by MASE on the same windows. Its winner is named only after a paired sign-flip test on per-entity MASE (A2-07):
the best model against the runner-up and against TimesFM. Otherwise the verdict is "no clear winner".
"""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

TRAIN_ROW_TARGET = 20_000       # origins are thinned so training stays near this many rows; the newest origin is always kept
MAX_TRAIN_ROWS = 200_000        # a hard cap: the most recent rows are kept when the panel is still larger
MAX_ENTITIES = 254              # the entity code is a categorical feature; HistGradientBoosting allows 255 categories
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
MIN_HISTORY = 10                # shortest history the model will train on
CALIBRATE_ORIGINS = 6           # calibration windows per entity: its last k windows of length H, all inside its history
CONFORMAL_LEVEL = 0.8           # the band is meant to cover this share (the 10 to 90 percent band)
MIN_CAL_WINDOWS = 2             # fewer calibration windows in the whole panel and the band stays raw, with a reason
MIN_CAL_SCORES = 10             # fewer scores in one horizon third and that third keeps the raw band
THIRDS = 3                      # the correction is set per horizon third: near, middle and far steps
EXACT_MAX_ENTITIES = 16         # sign-flip test enumerates every sign pattern up to this many entities (2^16 rows)
MC_DRAWS = 10_000               # sign-flip draws beyond that, seeded so a repeat gives the same p-value
ALPHA = 0.05


def _lag_count(season: int | None, min_len: int) -> int:
    """Lags 1..L with L = max(2 * season, 14), never more than the shortest history allows."""
    base = max(2 * season, 14) if season else 14
    return max(1, min(base, min_len - 1))


def _windows(season: int | None) -> list[int]:
    """Rolling windows: one week and one season (just one week when there is no season)."""
    return sorted({7, season}) if season else [7]


def _feature_matrix(x: np.ndarray, lags: int, windows: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Features for every possible origin o = 0..n. Row o uses only x[:o], so changing x from index o onward
    leaves row o unchanged. Returns (features normalised by the origin's scale, the scale for each origin)."""
    n = len(x)
    o = np.arange(n + 1)
    # the scale of an origin is the mean absolute value of its history; origin 0 has no history, so it is 1
    cum_abs = np.concatenate([[0.0], np.cumsum(np.abs(x))])
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(o >= 1, cum_abs[o] / np.maximum(o, 1), 1.0)
    scale = np.where(scale > 0, scale, 1.0)           # an all-zero history is scaled by 1, not by 0

    # lag k at origin o is x[o - k]; pad with NaN so early origins have missing lags (the model handles NaN)
    xp = np.concatenate([np.full(lags, np.nan), x])
    cols = [xp[lags + o - k] for k in range(1, lags + 1)]

    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])
    for w in windows:
        start = np.clip(o - w, 0, n)
        mean = (cs[o] - cs[start]) / w
        var = np.maximum((cs2[o] - cs2[start]) / w - mean * mean, 0.0)
        full = o >= w                                  # a window needs w values before the origin
        cols.append(np.where(full, mean, np.nan))
        cols.append(np.where(full, np.sqrt(var), np.nan))

    feats = np.column_stack(cols) / scale[:, None]     # NaN stays NaN after dividing
    return feats, scale


def _entity_block(x: np.ndarray, horizon: int, lags: int, windows: list[int], season: int | None, code: int, stride: int):
    """Training rows for one entity: one row per (origin, horizon step) whose target is inside the history. Origins are
    taken every ``stride`` steps counting back from the latest one, so the newest origin is always used."""
    n = len(x)
    feats, scale = _feature_matrix(x, lags, windows)
    blocks, ys, recency = [], [], []
    for h in range(1, horizon + 1):
        origins = np.arange(n - h, 0, -stride)[::-1]    # target index origin + h - 1 must be <= n - 1
        if origins.size == 0:
            continue
        target = origins + h - 1
        cols = [feats[origins], np.full((origins.size, 1), float(h))]
        if season:
            cols.append((target % season)[:, None].astype(float))
        cols.append(np.log(scale[origins])[:, None])
        cols.append(np.full((origins.size, 1), float(code)))
        blocks.append(np.hstack(cols))
        ys.append(x[target] / scale[origins])
        recency.append(n - 1 - target)                  # 0 for the latest target; used to keep recent rows
    if not blocks:
        return np.empty((0, 0)), np.empty(0), np.empty(0)
    return np.vstack(blocks), np.concatenate(ys), np.concatenate(recency)


def _origin_rows(feats: np.ndarray, scale: np.ndarray, origin: int, horizon: int, season: int | None, code: int) -> np.ndarray:
    """Feature rows for the horizon steps after ``origin``: one row per step, built from x[:origin] only."""
    steps = np.arange(1, horizon + 1, dtype=float)
    parts = [np.tile(feats[origin], (horizon, 1)), steps[:, None]]
    if season:
        parts.append(((origin + steps - 1) % season)[:, None])
    parts.append(np.full((horizon, 1), float(np.log(scale[origin]))))
    parts.append(np.full((horizon, 1), float(code)))
    return np.hstack(parts)


def _check_panel(panel: dict[str, Any], horizon: int, season: int | None, quantiles: tuple, calibrate_origins: int) -> None:
    if not isinstance(panel, dict) or not panel:
        raise ValueError("panel must be a non-empty dict of named series")
    if len(panel) > MAX_ENTITIES:
        raise ValueError(f"{len(panel)} entities; the limit is {MAX_ENTITIES} per model. Split the panel or pick entities")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if season is not None and (not isinstance(season, int) or isinstance(season, bool) or season < 2):
        raise ValueError("season must be None or an integer of 2 or more")
    if len(quantiles) != 3 or not all(0 < q < 1 for q in quantiles) or not quantiles[0] < quantiles[1] < quantiles[2]:
        raise ValueError("quantiles must be three increasing levels between 0 and 1, lower, middle, upper")
    if not isinstance(calibrate_origins, int) or isinstance(calibrate_origins, bool) or not 0 <= calibrate_origins <= 50:
        raise ValueError("calibrate_origins must be an integer from 0 to 50 (0 keeps the raw band)")
    for name, values in panel.items():
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1 or arr.size < max(MIN_HISTORY, horizon + 3):
            raise ValueError(f"series {name!r} needs at least {max(MIN_HISTORY, horizon + 3)} points for horizon {horizon}")
        if not np.isfinite(arr).all():
            raise ValueError(f"series {name!r} has missing or non-finite values; fill or drop them first")


def _fit(X: np.ndarray, y: np.ndarray, quantile: float, seed: int, cat_index: int) -> HistGradientBoostingRegressor:
    # Two threads: the default (all cores) was several times slower on this small data because of thread overhead, and
    # a busy machine makes it worse. Two keeps most of the speed of more threads on large panels.
    model = HistGradientBoostingRegressor(loss="quantile", quantile=quantile, max_iter=200, early_stopping=False,
                                          random_state=seed, categorical_features=[cat_index])
    with threadpool_limits(limits=2):
        return model.fit(X, y)


def _train_rows(items: list[tuple[int, np.ndarray]], horizon: int, lags: int, windows: list[int], season: int | None):
    """Stacked training rows for (entity code, history) pairs, thinned to about TRAIN_ROW_TARGET rows."""
    rows_est = sum(sum(max(0, len(x) - h) for h in range(1, horizon + 1)) for _, x in items)
    stride = max(1, -(-rows_est // TRAIN_ROW_TARGET))
    blocks, ys, recs = [], [], []
    for code, x in items:
        X, y, rec = _entity_block(x, horizon, lags, windows, season, code, stride)
        if X.size:
            blocks.append(X)
            ys.append(y)
            recs.append(rec)
    X = np.vstack(blocks)
    y = np.concatenate(ys)
    rec = np.concatenate(recs)
    if len(y) > MAX_TRAIN_ROWS:                        # keep the most recent rows, so the newest pattern is learned
        keep = np.argsort(rec, kind="stable")[:MAX_TRAIN_ROWS]
        X, y = X[keep], y[keep]
    return X, y


def _fit_quantiles(X: np.ndarray, y: np.ndarray, quantiles: tuple, seed: int) -> list[HistGradientBoostingRegressor]:
    cat_index = X.shape[1] - 1                         # the entity code is the last column
    return [_fit(X, y, q, seed, cat_index) for q in quantiles]


def _band(models: list, Xp: np.ndarray) -> np.ndarray:
    """Quantile predictions for the rows, sorted so the band cannot cross: shape (3, horizon), in normalised units."""
    return np.sort(np.vstack([m.predict(Xp) for m in models]), axis=0)


def _thirds(horizon: int) -> np.ndarray:
    """Horizon third (0, 1 or 2) of each step index 0..H-1."""
    return np.minimum(THIRDS - 1, np.arange(horizon) * THIRDS // horizon)


def _calibrate(series: dict[str, np.ndarray], codes: dict[str, int], horizon: int, lags: int, windows: list[int],
               season: int | None, quantiles: tuple, seed: int, k_req: int) -> tuple[np.ndarray | None, dict]:
    """Split-conformal correction per horizon step, from out-of-time scores on each entity's last windows.

    The calibration model is fitted on each entity's history before its first calibration window, so every score is a
    window the model never saw. Scores are normalised by the scale at the window's origin, the same units as the band,
    and pooled across entities and windows within a horizon third. Returns (correction for each step, or None, record).
    """
    min_train = max(MIN_HISTORY, 2 * (season or 7), horizon + 3)
    k_e = {name: min(k_req, (len(x) - min_train) // horizon) for name, x in series.items()}
    used = {name: k for name, k in k_e.items() if k >= 1}
    n_windows = sum(used.values())
    info: dict[str, Any] = {"method": "split conformal on earlier windows of each entity, one correction per horizon third",
                            "conformal_level": CONFORMAL_LEVEL, "windows": n_windows, "entities_calibrated": len(used),
                            "fallback": None, "q_hat": None, "scores_per_third": None}
    if n_windows < MIN_CAL_WINDOWS:
        info["fallback"] = (f"calibration skipped: {n_windows} calibration window(s) fit before the forecast origin; "
                            f"at least {MIN_CAL_WINDOWS} are needed, so the raw band is used")
        return None, info

    cut = {name: len(series[name]) - k * horizon for name, k in used.items()}
    X, y = _train_rows([(codes[name], series[name][:cut[name]]) for name in used], horizon, lags, windows, season)
    # The score reads only the lower and upper quantiles, so the middle model is not fitted here (a third less work).
    models = _fit_quantiles(X, y, (quantiles[0], quantiles[2]), seed)

    scores = []                                        # one row per window: the score at each horizon step
    for name, k in used.items():
        x = series[name]
        feats, scale = _feature_matrix(x, lags, windows)
        for j in range(k):
            origin = cut[name] + j * horizon
            band = _band(models, _origin_rows(feats, scale, origin, horizon, season, codes[name]))
            actual = x[origin:origin + horizon] / scale[origin]
            scores.append(np.maximum(band[0] - actual, actual - band[-1]))
    S = np.vstack(scores)                              # (windows, horizon)

    third = _thirds(horizon)
    q_third = np.zeros(THIRDS)
    counts, notes = [], []
    for g in range(THIRDS):
        if not (third == g).any():
            counts.append(0)
            continue
        vals = S[:, third == g].ravel()
        counts.append(int(vals.size))
        if vals.size < MIN_CAL_SCORES:
            notes.append(f"horizon third {g + 1} has {vals.size} score(s), below {MIN_CAL_SCORES}, so it keeps the raw band")
            continue
        level = min(1.0, np.ceil((vals.size + 1) * CONFORMAL_LEVEL) / vals.size)
        q_third[g] = float(np.quantile(vals, level, method="higher"))
    info["scores_per_third"] = counts
    info["q_hat"] = [round(float(q), 6) for q in q_third]
    if notes:
        info["fallback"] = "; ".join(notes)
    return q_third[third], info


def _forecast(panel: dict[str, Any], horizon: int, season: int | None, quantiles: tuple, seed: int,
              calibrate_origins: int) -> tuple[dict[str, dict], dict[str, dict], dict]:
    """The calibrated band, the raw band and the calibration record for one panel, at its last point."""
    names = list(panel)
    series = {k: np.asarray(v, dtype=float) for k, v in panel.items()}
    codes = {name: code for code, name in enumerate(names)}
    min_len = min(len(v) for v in series.values())
    lags = _lag_count(season, min_len)
    windows = _windows(season)

    X, y = _train_rows([(codes[n], series[n]) for n in names], horizon, lags, windows, season)
    models = _fit_quantiles(X, y, quantiles, seed)

    if calibrate_origins > 0:
        q_steps, info = _calibrate(series, codes, horizon, lags, windows, season, quantiles, seed, calibrate_origins)
    else:
        q_steps, info = None, {"method": "none", "conformal_level": CONFORMAL_LEVEL, "windows": 0, "entities_calibrated": 0,
                               "fallback": "calibration is off (calibrate_origins 0), so the raw band is used",
                               "q_hat": None, "scores_per_third": None}
    if q_steps is not None:
        q_steps = q_steps[:horizon]

    calibrated: dict[str, dict] = {}
    raw: dict[str, dict] = {}
    for name in names:
        x = series[name]
        feats, scale = _feature_matrix(x, lags, windows)
        band = _band(models, _origin_rows(feats, scale, len(x), horizon, season, codes[name]))
        scale_now = float(np.mean(np.abs(x))) or 1.0
        nonneg = bool((x >= 0).all())
        lo, mid, hi = band[0], band[1], band[2]
        clo, chi = (lo, hi) if q_steps is None else (lo - q_steps, hi + q_steps)
        clo = np.minimum(clo, mid)                     # the band keeps its order: lower <= middle <= upper
        chi = np.maximum(chi, mid)
        out_raw = np.vstack([lo, mid, hi]) * scale_now
        out_cal = np.vstack([clo, mid, chi]) * scale_now
        if nonneg:
            out_raw = np.clip(out_raw, 0.0, None)
            out_cal = np.clip(out_cal, 0.0, None)
        calibrated[name] = _band_dict(out_cal)
        raw[name] = _band_dict(out_raw)
    return calibrated, raw, info


def _band_dict(band: np.ndarray) -> dict[str, list]:
    return {"point": [round(float(v), 6) for v in band[1]],
            "q10": [round(float(v), 6) for v in band[0]],
            "q90": [round(float(v), 6) for v in band[2]]}


def _run_checked(panel, horizon, season, quantiles, seed, calibrate_origins):
    quantiles = tuple(quantiles)
    _check_panel(panel, horizon, season, quantiles, calibrate_origins)
    return _forecast(panel, horizon, season, quantiles, seed, calibrate_origins)


def global_forecast_info(panel: dict[str, Any], horizon: int, season: int | None = None,
                         quantiles: tuple = DEFAULT_QUANTILES, seed: int = 0,
                         calibrate_origins: int = CALIBRATE_ORIGINS) -> tuple[dict[str, dict], dict]:
    """Train one model on every entity; return the calibrated forecasts and the calibration record.

    The forecasts are {name: {"point": [...], "q10": [...], "q90": [...]}} with the middle quantile as the point and the
    band calibrated (see the module note). The record says whether calibration ran, the correction per horizon third and
    any fallback with its reason.
    """
    calibrated, _raw, info = _run_checked(panel, horizon, season, quantiles, seed, calibrate_origins)
    return calibrated, info


def global_forecast(panel: dict[str, Any], horizon: int, season: int | None = None,
                    quantiles: tuple = DEFAULT_QUANTILES, seed: int = 0,
                    calibrate_origins: int = CALIBRATE_ORIGINS, report_into: dict | None = None) -> dict[str, dict]:
    """Train one model on every entity and forecast each one ``horizon`` steps ahead (see ``global_forecast_info``).

    Returns {name: {"point": [...], "q10": [...], "q90": [...]}} with the middle quantile as the point. Values are never
    negative when that entity's history is never negative. Quantiles are sorted so the band cannot cross. When a dict is
    passed as ``report_into`` it receives "raw" (the uncalibrated bands, same shape) and "calibration" (the record), so a
    caller can report both coverages without a second fit.
    """
    calibrated, raw, info = _run_checked(panel, horizon, season, quantiles, seed, calibrate_origins)
    if report_into is not None:
        report_into["raw"] = raw
        report_into["calibration"] = info
    return calibrated


# ---------------------------------------------------------------------------------------------------------------------
# Paired test for the winner (A2-07)

def sign_flip_test(diffs: Any, seed: int = 0, draws: int = MC_DRAWS) -> dict[str, Any]:
    """Two-sided paired sign-flip test of the mean of per-entity differences against zero.

    Under the null that each entity's difference is symmetric about zero, every sign pattern is equally likely. With up to
    EXACT_MAX_ENTITIES entities every pattern is enumerated (exact p). Beyond that, ``draws`` random patterns are drawn
    with a fixed seed, and the p-value counts the observed statistic too (Monte Carlo p, never zero).
    """
    d = np.asarray(diffs, dtype=float).ravel()
    n = d.size
    if n < 1 or not np.isfinite(d).all():
        raise ValueError("a sign-flip test needs at least one finite difference")
    observed = abs(float(d.mean()))
    if n <= EXACT_MAX_ENTITIES:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
        stats = np.abs(signs @ d) / n
        p = float(np.mean(stats >= observed - 1e-12))
        method, used = "exact", 2 ** n
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(draws, n))
        stats = np.abs(signs @ d) / n
        p = float((1 + np.count_nonzero(stats >= observed - 1e-12)) / (draws + 1))
        method, used = "monte_carlo", draws
    return {"mean_diff": round(float(d.mean()), 6), "p_value": round(p, 6), "method": method, "draws": used,
            "entities": n}


def winner_test(per_entity: dict[str, dict[str, float]], models: list[str], alpha: float = ALPHA) -> dict[str, Any]:
    """Name the best model only when a paired test supports it.

    ``per_entity`` maps each entity to its mean MASE per model (unrounded). The best model has the lowest mean over entities
    (ties go to the earlier model). It is compared, entity by entity, with the runner-up and, when it is not TimesFM, with
    TimesFM. Each comparison gives the mean difference (best minus other, never above zero), the sign-flip p-value and a
    verdict: "<best>_better" when p is below alpha, else "no_clear_difference". The verdict is "clear winner: <best>" only
    when every comparison is significant, and "no clear winner" otherwise.
    """
    if len(models) < 2:
        raise ValueError("a winner test needs at least two models")
    entities = list(per_entity)
    if not entities:
        raise ValueError("a winner test needs at least one entity")
    values = {m: np.array([per_entity[e][m] for e in entities], dtype=float) for m in models}
    overall = {m: float(values[m].mean()) for m in models}
    order = sorted(models, key=lambda m: (overall[m], models.index(m)))
    best, runner = order[0], order[1]
    targets = [runner]
    if "timesfm" in models and best != "timesfm" and runner != "timesfm":
        targets.append("timesfm")
    against: dict[str, dict[str, Any]] = {}
    for t in targets:
        stats = sign_flip_test(values[best] - values[t])
        significant = stats["p_value"] < alpha and stats["mean_diff"] < 0
        against[t] = {**stats, "verdict": f"{best}_better" if significant else "no_clear_difference"}
    clear = all(v["verdict"] != "no_clear_difference" for v in against.values())
    return {"best": best, "runner_up": runner, "against": against, "alpha": alpha,
            "verdict": f"clear winner: {best}" if clear else "no clear winner"}


# ---------------------------------------------------------------------------------------------------------------------
# Rolling-origin comparison

MEDIAN = 4          # index of the 0.5 quantile in TimesFM's nine levels
LOW, HIGH = 0, 8    # indices of the 0.1 and 0.9 quantiles
MODELS_ORDER = ["naive", "seasonal_naive", "global_gbm", "timesfm", "blend"]


def _engine_forecast(engine: Any, histories: list[np.ndarray], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Point (n, horizon) and quantiles (n, horizon, levels). One batched call when the engine has one."""
    if hasattr(engine, "forecast_batch"):
        fc = engine.forecast_batch(histories, horizon)
        point, q = np.asarray(fc.point), np.asarray(fc.q)
    else:
        fcs = [engine.forecast(h, horizon) for h in histories]
        point = np.stack([np.asarray(f.point) for f in fcs])
        q = np.stack([np.asarray(f.q) for f in fcs])
    return point.reshape(len(histories), horizon), q.reshape(len(histories), horizon, -1)


def _mase_scale(hist: np.ndarray, season: int | None) -> tuple[float, bool]:
    """Mean absolute one-step change (or seasonal change) of the history. Returns (scale, was_zero)."""
    lag = season or 1
    scale = float(np.mean(np.abs(hist[lag:] - hist[:-lag])))
    return (scale, False) if scale > 0 else (1e-9, True)


def _seasonal_naive(hist: np.ndarray, horizon: int, season: int) -> np.ndarray:
    """Step j (from 0) repeats the value one season earlier, in the same position of the cycle."""
    return hist[len(hist) - season + (np.arange(horizon) % season)]


def _in_band(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> int:
    return int(((actual >= lo) & (actual <= hi)).sum())


def panel_backtest(panel: dict[str, Any], horizon: int, n_origins: int = 6, season: int | None = None,
                   engine: Any = None, extended: bool = False) -> dict[str, Any]:
    """Compare the global model with naive rules and, when ``engine`` is given, with TimesFM and their blend.

    The last ``n_origins`` non-overlapping windows of ``horizon`` steps are the test windows. At each origin the global
    model is refitted on history before that origin only. Scores are MASE per entity, averaged over the origins.

    The default return holds the fields the panel view has always read, and ``coverage_80`` is the calibrated band.
    ``extended=True`` adds ``coverage_80_raw`` (the uncalibrated band), ``calibration`` (the record of each origin's
    correction) and ``best_overall_test`` (the paired winner test, A2-02 and A2-07).
    """
    if not isinstance(n_origins, int) or isinstance(n_origins, bool) or not 1 <= n_origins <= 50:
        raise ValueError("n_origins must be an integer from 1 to 50")
    if not isinstance(panel, dict) or len(panel) < 2:
        raise ValueError("a panel backtest needs at least 2 named series")
    notes: list[str] = []
    raw = {k: np.asarray(v, dtype=float) for k, v in panel.items()}
    for k, v in raw.items():
        if v.ndim != 1 or not np.isfinite(v).all():
            raise ValueError(f"series {k!r} must be one-dimensional with finite values")
    lengths = {len(v) for v in raw.values()}
    N = min(lengths)
    if len(lengths) > 1:
        notes.append(f"series have different lengths; all are aligned on their last {N} points")
    series = {k: v[-N:] for k, v in raw.items()}

    min_history = max(MIN_HISTORY, 2 * (season or 7))
    if N - n_origins * horizon < min_history:
        raise ValueError(f"{n_origins} origins of {horizon} steps need {n_origins * horizon + min_history} points; "
                         f"the series have {N}. Use fewer origins or a shorter horizon")
    cuts = [N - (n_origins - k) * horizon for k in range(n_origins)]
    names = list(series)
    use_engine = engine is not None
    models = [m for m in MODELS_ORDER if (m != "seasonal_naive" or season) and (m not in ("timesfm", "blend") or use_engine)]

    errors = {m: {k: [] for k in names} for m in models}
    inside = {"global_gbm": 0, "timesfm": 0}
    inside_raw = {"global_gbm": 0}
    counted = {"global_gbm": 0, "timesfm": 0}
    zero_scale = 0
    calibration_records: list[dict] = []

    for cut in cuts:
        hist = {k: v[:cut] for k, v in series.items()}          # the only data any model sees at this origin
        actual = {k: v[cut:cut + horizon] for k, v in series.items()}
        report: dict[str, Any] = {}
        gbm = global_forecast(hist, horizon, season=season, report_into=report)
        uncal = report["raw"]
        calibration_records.append(report["calibration"])
        preds: dict[str, dict[str, np.ndarray]] = {
            "naive": {k: np.full(horizon, hist[k][-1]) for k in names},
            "global_gbm": {k: np.asarray(gbm[k]["point"]) for k in names},
        }
        if season:
            preds["seasonal_naive"] = {k: _seasonal_naive(hist[k], horizon, season) for k in names}
        for k in names:
            counted["global_gbm"] += horizon
            inside["global_gbm"] += _in_band(actual[k], np.asarray(gbm[k]["q10"]), np.asarray(gbm[k]["q90"]))
            inside_raw["global_gbm"] += _in_band(actual[k], np.asarray(uncal[k]["q10"]), np.asarray(uncal[k]["q90"]))
        if use_engine:
            point, q = _engine_forecast(engine, [hist[k] for k in names], horizon)
            preds["timesfm"] = {k: point[i] for i, k in enumerate(names)}
            preds["blend"] = {k: 0.5 * (preds["global_gbm"][k] + preds["timesfm"][k]) for k in names}
            for i, k in enumerate(names):
                counted["timesfm"] += horizon
                inside["timesfm"] += _in_band(actual[k], q[i, :, LOW], q[i, :, HIGH])
        for k in names:
            scale, was_zero = _mase_scale(hist[k], season)
            zero_scale += int(was_zero)
            for m in models:
                errors[m][k].append(float(np.mean(np.abs(actual[k] - preds[m][k]))) / scale)

    exact = {k: {m: float(np.mean(errors[m][k])) for m in models} for k in names}
    per_entity = {k: {m: round(exact[k][m], 4) for m in models} for k in names}
    overall = {m: round(float(np.mean([per_entity[k][m] for k in names])), 4) for m in models}
    wins = {m: 0 for m in models}
    for k in names:
        wins[min(models, key=lambda m: (per_entity[k][m], models.index(m)))] += 1   # ties go to the earlier model
    best_overall = min(models, key=lambda m: (overall[m], models.index(m)))
    coverage = {m: round(inside[m] / counted[m], 4) for m in ("global_gbm", "timesfm") if counted[m]}

    if not use_engine:
        notes.append("TimesFM not run: no engine was given, so its scores and the blend are left out")
    if zero_scale:
        notes.append(f"{zero_scale} entity-origin(s) had a flat history; their MASE uses a tiny scale and is not meaningful")
    notes.append("MASE divides the mean absolute error by the mean absolute one-step"
                 + (f" seasonal change (season {season})" if season else " change") + " of the history; lower is better")
    notes.append(f"origins are the last {n_origins} non-overlapping windows of {horizon} steps; the global model is "
                 "refitted at each one on history before it only")
    notes.append("coverage is the share of actual values inside the 10 to 90 percent band, pooled over origins and entities; "
                 "the band is calibrated on earlier windows of each entity (split conformal)")
    result: dict[str, Any] = {"per_entity": per_entity, "overall": overall, "wins": wins, "best_overall": best_overall,
                              "coverage_80": coverage, "origins": n_origins, "horizon": horizon, "season": season,
                              "notes": notes}
    if extended:
        result["coverage_80_raw"] = {"global_gbm": round(inside_raw["global_gbm"] / counted["global_gbm"], 4)}
        fallbacks = sorted({r["fallback"] for r in calibration_records if r.get("fallback")})
        result["calibration"] = {"conformal_level": CONFORMAL_LEVEL, "origins": n_origins,
                                 "origins_calibrated": sum(1 for r in calibration_records if not r.get("fallback")),
                                 "fallbacks": fallbacks}
        result["best_overall_test"] = winner_test(exact, models)
    return result
