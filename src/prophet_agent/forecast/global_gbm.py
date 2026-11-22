"""One global gradient-boosted model for a panel of related series, compared with local and foundation models.

Motivation vs Logic
Motivation: in a panel (many products or sites on one calendar) the M5 competition winners fitted one model across all
series rather than one per series. Pooling lets a short series borrow the shape of long ones. The question for the person
is whether that pooling beats a naive rule and TimesFM on the same past windows.
Logic: each origin is divided by its own level (the mean absolute value of the history before that origin), so big and
small entities share one scale. Features for an origin are lags, rolling mean and spread, the step position in the season,
the log level and an entity code. Every feature uses only values strictly before the origin. Training rows are stacked for
horizons 1 to H (a horizon feature tells the model which step it is predicting), so one model covers all steps. Three
quantile models give the lower, middle and upper band. ``panel_backtest`` refits at each of the last few non-overlapping
origins using only history before that origin, then scores every model by MASE on the same windows.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

TRAIN_ROW_TARGET = 20_000       # origins are thinned so training stays near this many rows; the newest origin is always kept
MAX_TRAIN_ROWS = 200_000        # a hard cap: the most recent rows are kept when the panel is still larger
MAX_ENTITIES = 254              # the entity code is a categorical feature; HistGradientBoosting allows 255 categories
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
MIN_HISTORY = 10                # shortest history the model will train on


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


def _predict_rows(x: np.ndarray, horizon: int, lags: int, windows: list[int], season: int | None, code: int) -> np.ndarray:
    """Feature rows for the future: origin n (the whole history), one row per horizon step."""
    n = len(x)
    feats, scale = _feature_matrix(x, lags, windows)
    rows = []
    for h in range(1, horizon + 1):
        cols = [feats[n], [float(h)]]
        if season:
            cols.append([float((n + h - 1) % season)])
        cols.append([float(np.log(scale[n]))])
        cols.append([float(code)])
        rows.append(np.concatenate([np.asarray(c, dtype=float) for c in cols]))
    return np.vstack(rows)


def _check_panel(panel: dict[str, Any], horizon: int, season: int | None, quantiles: tuple) -> None:
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


def global_forecast(panel: dict[str, Any], horizon: int, season: int | None = None,
                    quantiles: tuple = DEFAULT_QUANTILES, seed: int = 0) -> dict[str, dict]:
    """Train one model on every entity and forecast each one ``horizon`` steps ahead.

    Returns {name: {"point": [...], "q10": [...], "q90": [...]}} with the middle quantile as the point. Values are
    never negative when that entity's history is never negative. Quantiles are sorted so the band cannot cross.
    """
    _check_panel(panel, horizon, season, tuple(quantiles))
    names = list(panel)
    series = {k: np.asarray(v, dtype=float) for k, v in panel.items()}
    min_len = min(len(v) for v in series.values())
    lags = _lag_count(season, min_len)
    windows = _windows(season)
    cat_index = None

    # rows before thinning: each origin gives one row per horizon step whose target is inside the history
    rows_est = sum(sum(max(0, len(v) - h) for h in range(1, horizon + 1)) for v in series.values())
    stride = max(1, -(-rows_est // TRAIN_ROW_TARGET))
    blocks, ys, recs = [], [], []
    for code, name in enumerate(names):
        X, y, rec = _entity_block(series[name], horizon, lags, windows, season, code, stride)
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
    cat_index = X.shape[1] - 1                         # the entity code is the last column

    models = [_fit(X, y, q, seed, cat_index) for q in quantiles]
    out: dict[str, dict] = {}
    for code, name in enumerate(names):
        x = series[name]
        Xp = _predict_rows(x, horizon, lags, windows, season, code)
        scale_now = float(np.mean(np.abs(x))) or 1.0
        band = np.sort(np.vstack([m.predict(Xp) for m in models]), axis=0) * scale_now
        if (x >= 0).all():
            band = np.clip(band, 0.0, None)
        out[name] = {"point": [round(float(v), 6) for v in band[1]],
                     "q10": [round(float(v), 6) for v in band[0]],
                     "q90": [round(float(v), 6) for v in band[2]]}
    return out


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
                   engine: Any = None) -> dict[str, Any]:
    """Compare the global model with naive rules and, when ``engine`` is given, with TimesFM and their blend.

    The last ``n_origins`` non-overlapping windows of ``horizon`` steps are the test windows. At each origin the global
    model is refitted on history before that origin only. Scores are MASE per entity, averaged over the origins.
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
    counted = {"global_gbm": 0, "timesfm": 0}
    zero_scale = 0

    for cut in cuts:
        hist = {k: v[:cut] for k, v in series.items()}          # the only data any model sees at this origin
        actual = {k: v[cut:cut + horizon] for k, v in series.items()}
        gbm = global_forecast(hist, horizon, season=season)
        preds: dict[str, dict[str, np.ndarray]] = {
            "naive": {k: np.full(horizon, hist[k][-1]) for k in names},
            "global_gbm": {k: np.asarray(gbm[k]["point"]) for k in names},
        }
        if season:
            preds["seasonal_naive"] = {k: _seasonal_naive(hist[k], horizon, season) for k in names}
        for k in names:
            counted["global_gbm"] += horizon
            inside["global_gbm"] += _in_band(actual[k], np.asarray(gbm[k]["q10"]), np.asarray(gbm[k]["q90"]))
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

    per_entity = {k: {m: round(float(np.mean(errors[m][k])), 4) for m in models} for k in names}
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
    notes.append("coverage is the share of actual values inside the 10 to 90 percent band, pooled over origins and entities")
    return {"per_entity": per_entity, "overall": overall, "wins": wins, "best_overall": best_overall,
            "coverage_80": coverage, "origins": n_origins, "horizon": horizon, "season": season, "notes": notes}
