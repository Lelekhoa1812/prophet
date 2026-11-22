"""Rolling-origin backtest for one series: does TimesFM beat simple rules on this data, and how sure can we be?

Motivation vs Logic
Motivation: a forecast is only worth trusting if it beats a rule anyone could apply. The agent needs that answer for the
series in front of it, not for a benchmark, and needs it before it spends effort on charts and write-ups. A win on two
windows is not evidence, so the result also says how many windows TimesFM won and how wide the uncertainty is.
Logic: cut the series at several past origins (non-overlapping windows of the forecast horizon, newest last), forecast
each window using only data before it, and score against what happened. Baselines: repeat the last value, repeat the
value one season back, extend the average slope, and the mean of the last season. Scores are MASE (error scaled by how
hard the series is to predict naively), WAPE, and for TimesFM the coverage of its 10 to 90 percent range. The MASE scale
is named in the result, and the season comes from the shared choose_season. Against the best baseline the result gives
the wins per origin and a seeded paired bootstrap interval (90 percent) of the relative MASE gain, and a verdict:
timesfm_better, baseline_better, or no_clear_difference. With combine on, a blend of TimesFM and the baseline that was
best on earlier origins is scored too, with weights fitted on earlier origins only. The result applies no pass or fail
bar beyond that verdict.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from prophet_agent.forecast.combine import combine
from prophet_agent.forecast.paired import bootstrap_gain
from prophet_agent.forecast.season import choose_season, mase_scale_text
from prophet_agent.forecast.tsb import expected_volume

MIN_HISTORY = 8
DEFAULT_ORIGINS = 12        # windows tested when the caller does not say; the minimum-length rule still applies
VOLUME_BIAS_NOTE = 15.0     # percent: a total this far from actual is reported as a volume finding, not left to the reader
CONFIDENCE = 0.90           # level of the paired bootstrap interval of the relative MASE gain over the best baseline
RESAMPLES = 2000
SEED = 20260                # fixed so the same series and settings replay to the same interval
WIN_SHARE = 0.6             # TimesFM must win at least this share of origins for a "timesfm_better" verdict
SIMPLE = ("naive", "seasonal_naive", "drift", "mean_last_season", "tsb_expected")


def _snaive(x: np.ndarray, h: int, m: int) -> np.ndarray:
    if m < 2 or len(x) < m:
        return np.full(h, x[-1])
    return np.array([x[len(x) - m + (i % m)] for i in range(h)])


def _drift(x: np.ndarray, h: int) -> np.ndarray:
    if len(x) < 2:
        return np.full(h, x[-1])
    slope = (x[-1] - x[0]) / (len(x) - 1)
    return x[-1] + slope * np.arange(1, h + 1)


def _mean_last(x: np.ndarray, h: int, m: int) -> np.ndarray:
    return np.full(h, x[-max(m, 4):].mean())


def pick_season(x: np.ndarray, given: int | None) -> int:
    """The season the backtest uses: the caller's, or the one choose_season measures from the data."""
    if given is not None:
        return max(1, int(given))
    return int(choose_season(x)["season"])


def _verdict(base_err: np.ndarray, tsb_err: np.ndarray) -> tuple[dict[str, Any], int, int]:
    """Paired comparison of TimesFM with the best baseline over origins where both errors are finite."""
    keep = np.isfinite(base_err) & np.isfinite(tsb_err)
    n_valid = int(keep.sum())
    wins = int((tsb_err[keep] < base_err[keep]).sum())
    g = bootstrap_gain(base_err, tsb_err, SEED, CONFIDENCE, RESAMPLES)
    if not np.isfinite(g["lo"]):
        return {"verdict": None, "gain_pct": None, "interval_pct": None, "confidence": CONFIDENCE}, wins, n_valid
    if g["lo"] > 0 and n_valid and wins >= WIN_SHARE * n_valid:
        verdict = "timesfm_better"
    elif g["hi"] < 0:
        verdict = "baseline_better"
    else:
        verdict = "no_clear_difference"
    return {"verdict": verdict, "gain_pct": round(g["gain"] * 100, 2),
            "interval_pct": [round(g["lo"] * 100, 2), round(g["hi"] * 100, 2)], "confidence": CONFIDENCE}, wins, n_valid


def _mase_scale(hist: np.ndarray, m: int) -> float:
    """The backtest's MASE denominator for one origin: mean absolute change at lag m (one step when m is 1). NaN if flat."""
    scale = np.mean(np.abs(hist[m:] - hist[:-m])) if len(hist) > m else np.mean(np.abs(np.diff(hist)))
    return float(scale) if scale > 1e-12 else np.nan


def _combined_forecasts(x: np.ndarray, origins: list[int], m: int, preds: list[dict[str, np.ndarray]],
                        actuals: list[np.ndarray]) -> tuple[list[np.ndarray], list[str], list[list[float]]]:
    """Blend TimesFM with the simple baseline that had the lowest mean MASE on earlier origins only. Origin 0 has no
    earlier origin, so it uses the naive baseline and equal weights."""
    combined, chosen, weights = [], [], []
    scales = [_mase_scale(x[:o], m) for o in origins]
    for j in range(len(origins)):
        if j == 0:
            name = "naive"
        else:
            mase_so_far = {nm: float(np.mean([np.abs(actuals[i] - preds[i][nm]).mean() / scales[i] for i in range(j)
                                              if np.isfinite(scales[i])])) for nm in SIMPLE}
            name = min(mase_so_far, key=lambda nm: (np.nan_to_num(mase_so_far[nm], nan=np.inf), SIMPLE.index(nm)))
        err_a = np.stack([actuals[i] - preds[i]["timesfm"] for i in range(j)]) if j else None
        err_b = np.stack([actuals[i] - preds[i][name] for i in range(j)]) if j else None
        blend = combine(preds[j]["timesfm"], preds[j][name], err_a, err_b)
        combined.append(blend["point"])
        chosen.append(name)
        weights.append(blend["weights_a"])
    return combined, chosen, weights


def rolling_backtest(values: list[float] | np.ndarray, horizon: int, *, n_origins: int = DEFAULT_ORIGINS,
                     season: int | None = None, engine: Any = None, control: Any = None, combine_forecasts: bool = False,
                     freq_hint: str | None = None, dates: Any = None) -> dict:
    x = np.asarray(values, dtype=float)
    n = len(x)
    if not np.isfinite(x).all():
        raise ValueError("series contains NaN or infinite values; fill or drop them first")
    m = pick_season(x, season)
    season_reason = "given by the caller" if season is not None else choose_season(x, freq_hint, dates)["reason"]
    min_hist = max(MIN_HISTORY, 2 * m if m > 1 else MIN_HISTORY)
    k = min(int(n_origins), (n - min_hist) // horizon)
    if k < 2:
        hint = ""
        if m > 1:
            fits = max(1, (n - 2 * horizon) // 2)
            hint = (f" The season of {m} needs two full cycles of history before the first test window, which is what sets the "
                    f"minimum. With {n} points a season up to {fits} fits; omit season (or pass 1) if the cycle is not that long.")
        raise ValueError(f"series of {n} points is too short to backtest a horizon of {horizon} "
                         f"(needs at least {min_hist + 2 * horizon} with these settings; n_origins does not change that)." + hint)
    origins = [n - horizon * (k - j) for j in range(k)]           # oldest origin first, last window ends at n
    if engine is None:
        from prophet_agent.forecast.timesfm import TimesFMEngine
        engine = TimesFMEngine.get()
    fc = engine.forecast_batch([x[:o] for o in origins], horizon)
    configured = control is not None and not control.is_default
    if configured:
        # Each origin sees only what was knowable then: arrays are cut at the origin, so nothing leaks from the future.
        runs = [engine.forecast_controlled(x[:o], horizon, control.at_origin(o, horizon)) for o in origins]
        cfc = type(fc)(np.stack([r.point for r in runs]), np.stack([r.q for r in runs]))
    else:
        cfc = fc

    # Forecasts and actuals per origin; the blend (if asked for) is built from these, so it can only see earlier origins.
    preds: list[dict[str, np.ndarray]] = []
    actuals: list[np.ndarray] = []
    for j, o in enumerate(origins):
        hist = x[:o]
        p = {"timesfm": cfc.point[j], "naive": np.full(horizon, hist[-1]), "seasonal_naive": _snaive(hist, horizon, m),
             "drift": _drift(hist, horizon), "mean_last_season": _mean_last(hist, horizon, m),
             "tsb_expected": expected_volume(hist, horizon)}
        if configured:
            p["timesfm_default"] = fc.point[j]
        preds.append(p)
        actuals.append(x[o:o + horizon])
    combination: dict[str, Any] | None = None
    if combine_forecasts:
        blends, chosen, weights = _combined_forecasts(x, origins, m, preds, actuals)
        for p, b in zip(preds, blends):
            p["timesfm_combined"] = b
        combination = {"method": "inverse-MSE weights per horizon third, from earlier origins only; the baseline is the one "
                                 "with the lowest mean MASE on earlier origins only",
                       "baseline_by_origin": chosen, "weight_timesfm_by_origin": weights,
                       "note": "origin 1 has no earlier origin, so it uses naive and equal weights"}

    names = ["timesfm", "naive", "seasonal_naive", "drift", "mean_last_season", "tsb_expected"] + (
        ["timesfm_default"] if configured else []) + (["timesfm_combined"] if combine_forecasts else [])
    errs: dict[str, list[float]] = {nm: [] for nm in names}
    act_sum, abs_err = 0.0, {nm: 0.0 for nm in names}
    signed = {nm: 0.0 for nm in names}          # sum of (forecast - actual): the volume bias, not the error
    covered, pin = [], []
    qs = np.arange(1, 10) / 10
    for j, o in enumerate(origins):
        hist, actual = x[:o], actuals[j]
        scale = _mase_scale(hist, m)
        for nm in names:
            p = preds[j][nm]
            e = np.abs(actual - p)
            errs[nm].append(float(e.mean() / scale) if scale == scale else float("nan"))
            abs_err[nm] += float(e.sum())
            signed[nm] += float((p - actual).sum())
        act_sum += float(np.abs(actual).sum())
        q = cfc.q[j]
        covered.append(float(((actual >= q[:, 0]) & (actual <= q[:, 8])).mean()))
        d = actual[:, None] - q
        pin.append(float(np.mean(np.maximum(qs * d, (qs - 1) * d)) / scale) if scale == scale else float("nan"))

    base = {nm: float(np.nanmean(errs[nm])) for nm in SIMPLE if nm in errs and not np.all(np.isnan(errs[nm]))}
    best = min(base, key=base.get) if base else None
    models: dict[str, Any] = {}
    for nm, v in errs.items():
        entry: dict[str, Any] = {"mase": round(float(np.nanmean(v)), 4) if len(v) and not np.all(np.isnan(v)) else None,
                                 "wape": round(abs_err[nm] / act_sum, 4) if act_sum > 0 else None,
                                 "volume_bias_pct": round(signed[nm] / act_sum * 100, 1) if act_sum > 0 else None}
        if best is not None and nm != best:
            entry["wins_vs_best_baseline"] = int(sum(1 for a, b in zip(v, errs[best])
                                                     if np.isfinite(a) and np.isfinite(b) and a < b))
        models[nm] = entry
    out: dict[str, Any] = {"points": n, "horizon": horizon, "origins": k, "n_origins": k, "season_used": m,
                           "season_reason": season_reason, "mase_scale": mase_scale_text(m),
                           "models": models,
                           "timesfm_range_coverage_80": round(float(np.mean(covered)), 3),
                           "timesfm_pinball_scaled": round(float(np.nanmean(pin)), 4) if not np.all(np.isnan(pin)) else None}
    if k < n_origins:
        out["origins_note"] = (f"only {k} of the {n_origins} requested windows of {horizon} fit in {n} points; "
                               "the evidence rests on these few windows")
    tb = models["timesfm"]["volume_bias_pct"]
    if tb is not None and abs(tb) >= VOLUME_BIAS_NOTE:
        out["timesfm_volume_note"] = (
            f"Over these windows TimesFM's point forecast totals {tb:+.1f}% against what happened. A point forecast is a "
            "median, and for lumpy or skewed demand a median understates volume. Compare with tsb_expected (the mean under TSB) "
            "before presenting the point as expected demand, and say which measure the answer uses.")
    if configured:
        out["control"] = control.describe()
        d, c = models["timesfm_default"]["mase"], models["timesfm"]["mase"]
        if d and c is not None:
            out["configured_vs_default_pct"] = round((d - c) / d * 100, 1)
    if combination is not None:
        out["combination"] = combination
        out["timesfm_combined_mase_note"] = "timesfm_combined is TimesFM blended with the earlier-best baseline; compare its mase with timesfm and best_baseline"
    if best is not None and models["timesfm"]["mase"] is not None:
        out["best_baseline"] = best
        gain = (base[best] - models["timesfm"]["mase"]) / base[best] * 100 if base[best] else None
        out["timesfm_vs_best_baseline_pct"] = round(gain, 1) if gain is not None else None
        ev, wins, n_valid = _verdict(np.asarray(errs[best]), np.asarray(errs["timesfm"]))
        out["wins_vs_best_baseline"] = wins
        out["wins_out_of"] = n_valid
        out["timesfm_vs_best_baseline_interval"] = ev
        out["verdict"] = ev["verdict"]
        out["reading"] = (
            "Lower MASE is better; 1.0 means as accurate as repeating the value one season (or step) back. "
            f"Range coverage should sit near 0.8. Measured over {k} windows of {horizon} periods. The verdict uses a paired "
            f"{CONFIDENCE:.0%} bootstrap of the per-window MASE gain over {best} and the number of windows TimesFM won; "
            "timesfm_better needs the whole interval above zero and at least "
            f"{WIN_SHARE:.0%} of windows won, baseline_better needs the whole interval below zero, anything else is "
            "no_clear_difference."
            + (" 'timesfm' is the configured run (see control); 'timesfm_default' is the same model with no control. "
               "configured_vs_default_pct above 0 means the configuration helped on these windows." if configured else ""))
    return out
