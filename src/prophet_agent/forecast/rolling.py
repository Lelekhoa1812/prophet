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
the wins per origin and a seeded paired sign-flip test (exact up to 16 windows) with a bootstrap interval (90 percent) of the relative MASE gain, and a verdict:
timesfm_better, baseline_better, or no_clear_difference. With combine on, a blend of TimesFM and the baseline that was
best on earlier origins is scored too, with weights fitted on earlier origins only. The result applies no pass or fail
bar beyond that verdict.
With blend_centre on, an equal-weight blend of TimesFM and the simple centre rule with the lowest mean MASE over all the
windows is scored as well. centre_choice says whether the blend beats both TimesFM's centre and that rule on these windows.
The rule and that choice are both selected on these windows, so the result labels them as selected, not unbiased.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from prophet_agent.forecast.combine import adopt_equal_blend, combine, equal_blend
from prophet_agent.forecast.health import season_check
from prophet_agent.forecast.metrics import paired_signflip, wis
from prophet_agent.forecast.paired import bootstrap_gain
from prophet_agent.forecast.season import choose_season, mase_scale_text
from prophet_agent.forecast.tsb import expected_volume, is_intermittent, tsb_band

MIN_HISTORY = 8
DEFAULT_ORIGINS = 12        # windows tested when the caller does not say; the minimum-length rule still applies
VOLUME_BIAS_NOTE = 15.0     # percent: a total this far from actual is reported as a volume finding, not left to the reader
CONFIDENCE = 0.90           # level of the paired bootstrap interval of the relative MASE gain over the best baseline
RESAMPLES = 2000
SEED = 20260                # fixed so the same series and settings replay to the same interval
WIN_SHARE = 0.6             # TimesFM must win at least this share of origins for a "timesfm_better" verdict
SIMPLE = ("naive", "seasonal_naive", "drift", "mean_last_season", "tsb_expected")
EQUAL_BLEND_RULES = ("naive", "drift", "seasonal_naive", "mean_last_season")   # the simple centre rules the report can deliver
BLEND_CAVEAT = ("The rule and the adoption were both chosen on these same windows, so the blend's scores here are optimistic. "
                "This is a selection on the backtest, not an unbiased estimate of accuracy; confirm on later windows before relying on it.")


def _snaive(x: np.ndarray, h: int, m: int) -> np.ndarray:
    if m < 2 or len(x) < m:
        return np.full(h, x[-1])
    return np.array([x[len(x) - m + (i % m)] for i in range(h)])


def _drift(x: np.ndarray, h: int) -> np.ndarray:
    if len(x) < 2:
        return np.full(h, x[-1])
    slope = (x[-1] - x[0]) / (len(x) - 1)
    return x[-1] + slope * np.arange(1, h + 1)


MEAN_LAST_FLOOR = 4         # the shortest window the mean_last_season rule averages (stability, audit A6-06)


def _mean_last_window(m: int) -> int:
    """How many trailing points mean_last_season averages: the season, and at least MEAN_LAST_FLOOR.

    Motivation vs Logic
    Motivation: with no season (m = 1) the rule averaged 4 points while its name and the reports said "the last season"
    (audit A6-06, WTI rep4: best baseline MASE 6.66 was a 4-point mean).
    Logic: the window is explicit. The backtest reports its length, and the note says when it is a floor rather than a season.
    """
    return max(int(m), MEAN_LAST_FLOOR)


def _mean_last(x: np.ndarray, h: int, m: int) -> np.ndarray:
    return np.full(h, x[-_mean_last_window(m):].mean())


def pick_season(x: np.ndarray, given: int | None) -> int:
    """The season the delivered forecast uses: the caller's, or the one choose_season measures from the whole history.
    Backtests and calibration use backtest_season instead, because the whole history includes their test windows."""
    if given is not None:
        return max(1, int(given))
    return int(choose_season(x)["season"])


def season_cut(n: int, horizon: int, n_origins: int) -> int:
    """The number of leading points that no backtest window can touch: every window starts at or after n - horizon *
    n_origins, because the backtest tests at most n_origins windows that end at the last point.

    Motivation vs Logic
    Motivation: a season measured on the full series reads the windows it is scored on (audit A2-01: a random walk with a
    weekly pattern in its last 56 points gave season 1 on its history and 7 on the whole series).
    Logic: the prefix before this cut is shared by every origin the backtest can use, so a season measured on it is known
    before any test window is forecast. Calibration uses the same cut.
    """
    return max(0, int(n) - int(horizon) * int(n_origins))


def backtest_season(x: np.ndarray, horizon: int, n_origins: int, given: int | None, freq_hint: str | None = None,
                    dates: Any = None) -> tuple[int, str]:
    """The season a backtest or calibration uses, and the reason, measured on history before the first test window.

    Motivation vs Logic
    Motivation: the MASE denominator and the seasonal baseline both depend on the season, so a season read from the test
    windows makes the verdict depend on the answer it is scored against (audit A2-01).
    Logic: a caller's season is used as given. Otherwise choose_season reads only x[:season_cut(...)] (and the matching dates),
    and the reason says that it was measured on that prefix. Changing any value in a test window leaves the result unchanged.
    """
    if given is not None:
        return max(1, int(given)), "given by the caller"
    cut = season_cut(len(x), horizon, n_origins)
    dates_h = None if dates is None else list(dates)[:cut]
    res = choose_season(np.asarray(x, dtype=float)[:cut], freq_hint, dates_h)
    return int(res["season"]), f"{res['reason']} (measured on the first {cut} points, before any test window)"


def _without_noop_context(control: Any, n: int, horizon: int) -> tuple[Any, dict[str, Any] | None]:
    """Drop a context_points that covers every origin, and say so. Returns the control to use and the note (None if nothing
    was dropped).

    Motivation vs Logic
    Motivation: a context of at least the last origin (n - horizon) gives every window its whole history, so it changes no
    forecast, yet it was reported as a tested configuration (audit A6-08, retail rep4: configured_vs_default_pct 0.0).
    Logic: the longest origin in rolling_backtest is n - horizon, so a context at or above it truncates nothing. Only that
    option is removed; a real option in the same control (znorm, covariates) is kept and still tested.
    """
    cp = getattr(control, "context_points", None)
    if control is None or not cp or cp < n - horizon:
        return control, None
    note = {"context_points": int(cp),
            "reason": (f"context_points={int(cp)} covers every backtest origin (the longest origin is {n - horizon} points), "
                       "so the default already reads all of it; it was dropped from the configuration and is not a treatment")}
    return dataclasses.replace(control, context_points=None), note


def _verdict(base_err: np.ndarray, tsb_err: np.ndarray) -> tuple[dict[str, Any], int, int]:
    """Paired comparison of TimesFM with the best baseline over origins where both errors are finite.

    The verdict uses a one-sided sign-flip test at the level implied by CONFIDENCE (5 percent each way for 90 percent): the
    percentile bootstrap undercovered at these origin counts (audit A2-06). The bootstrap interval is still reported as a
    description of the gain.
    """
    keep = np.isfinite(base_err) & np.isfinite(tsb_err)
    n_valid = int(keep.sum())
    wins = int((tsb_err[keep] < base_err[keep]).sum())
    g = bootstrap_gain(base_err, tsb_err, SEED, CONFIDENCE, RESAMPLES)
    if not np.isfinite(g["lo"]):
        return {"verdict": None, "gain_pct": None, "interval_pct": None, "confidence": CONFIDENCE}, wins, n_valid
    st = paired_signflip(base_err, tsb_err, seed=SEED)
    alpha = (1 - CONFIDENCE) / 2
    if st["p_cand_better"] <= alpha and n_valid and wins >= WIN_SHARE * n_valid:
        verdict = "timesfm_better"
    elif st["p_base_better"] <= alpha:
        verdict = "baseline_better"
    else:
        verdict = "no_clear_difference"
    return {"verdict": verdict, "gain_pct": round(g["gain"] * 100, 2),
            "interval_pct": [round(g["lo"] * 100, 2), round(g["hi"] * 100, 2)], "confidence": CONFIDENCE,
            "test": st["method"], "p_timesfm_better": round(st["p_cand_better"], 4),
            "p_baseline_better": round(st["p_base_better"], 4)}, wins, n_valid


def _mase_scale(hist: np.ndarray, m: int) -> float:
    """The backtest's MASE denominator for one origin: mean absolute change at lag m (one step when m is 1). NaN if flat."""
    scale = np.mean(np.abs(hist[m:] - hist[:-m])) if len(hist) > m else np.mean(np.abs(np.diff(hist)))
    return float(scale) if scale > 1e-12 else np.nan


def _combined_forecasts(x: np.ndarray, origins: list[int], m: int, preds: list[dict[str, np.ndarray]],
                        actuals: list[np.ndarray], simple: tuple = SIMPLE) -> tuple[list[np.ndarray], list[str], list[list[float]]]:
    """Blend TimesFM with the simple baseline that had the lowest mean MASE on earlier origins only. Origin 0 has no
    earlier origin, so it uses the naive baseline and equal weights."""
    combined, chosen, weights = [], [], []
    scales = [_mase_scale(x[:o], m) for o in origins]
    for j in range(len(origins)):
        if j == 0:
            name = "naive"
        else:
            mase_so_far = {nm: float(np.mean([np.abs(actuals[i] - preds[i][nm]).mean() / scales[i] for i in range(j)
                                              if np.isfinite(scales[i])])) for nm in simple}
            name = min(mase_so_far, key=lambda nm: (np.nan_to_num(mase_so_far[nm], nan=np.inf), simple.index(nm)))
        err_a = np.stack([actuals[i] - preds[i]["timesfm"] for i in range(j)]) if j else None
        err_b = np.stack([actuals[i] - preds[i][name] for i in range(j)]) if j else None
        blend = combine(preds[j]["timesfm"], preds[j][name], err_a, err_b)
        combined.append(blend["point"])
        chosen.append(name)
        weights.append(blend["weights_a"])
    return combined, chosen, weights


def _equal_blend_rule(x: np.ndarray, origins: list[int], m: int, preds: list[dict[str, np.ndarray]],
                      actuals: list[np.ndarray]) -> str | None:
    """The simple centre rule the equal blend uses: the one with the lowest mean MASE over all the windows, among
    EQUAL_BLEND_RULES. None when no rule can be scored (every scale is flat).

    Motivation vs Logic
    Motivation: an equal blend needs one rule, and the rule that did best overall is the natural one. But picking it on the
    same windows the blend is then scored on is a selection, so the caller labels it as one.
    Logic: score each rule's MASE per origin with the backtest's own scale, average over the finite origins, and take the
    lowest, ties going to the rule listed first. Nothing about the TimesFM forecast enters this choice.
    """
    scales = [_mase_scale(x[:o], m) for o in origins]
    mase: dict[str, float] = {}
    for nm in EQUAL_BLEND_RULES:
        vals = [float(np.abs(actuals[i] - preds[i][nm]).mean() / scales[i]) for i in range(len(origins)) if np.isfinite(scales[i])]
        if vals:
            mase[nm] = float(np.mean(vals))
    return min(mase, key=lambda nm: (mase[nm], EQUAL_BLEND_RULES.index(nm))) if mase else None


def _blend_choice(errs: dict[str, list[float]], rule: str | None, k: int, horizon: int, models: dict[str, Any]) -> dict[str, Any]:
    """Whether the equal blend is adopted as the centre, with the evidence and the selection caveat.

    Motivation vs Logic
    Motivation: the report must not deliver a blend because the planner liked it. The adoption is decided here, on the same
    windows as the verdict, and the result says exactly what was compared.
    Logic: adopt_equal_blend compares the blend with TimesFM's centre and with the chosen rule on the same origins. The
    result keeps the counts, the paired intervals and the MASE of each, and always carries the selection caveat, because
    the windows that chose the rule and the adoption are the windows that score it.
    """
    base = {"windows": k, "horizon": horizon, "selected_on_these_windows": True, "caveat": BLEND_CAVEAT}
    if rule is None:
        return {**base, "adopt": False, "reason": "no simple centre rule could be scored on these windows"}
    ev = adopt_equal_blend(errs["timesfm"], errs["timesfm_equal_blend"], errs[rule], seed=SEED, confidence=CONFIDENCE,
                           resamples=RESAMPLES, win_share=WIN_SHARE)
    if ev["adopt"]:
        reason = f"the blend beat TimesFM's centre and {rule} on these {k} windows of {horizon}"
    else:
        lost = [name for name, ok in (("TimesFM's centre", ev["vs_timesfm"]["beats"]), (rule, ev["vs_rule"]["beats"])) if not ok]
        reason = f"the blend did not beat {' or '.join(lost)} on these windows, so the centre stays TimesFM's"
    return {**base, "rule": rule, **ev, "reason": reason,
            "mase": {"timesfm": models["timesfm"]["mase"], "blend": models["timesfm_equal_blend"]["mase"],
                     rule: models[rule]["mase"]}}


def rolling_backtest(values: list[float] | np.ndarray, horizon: int, *, n_origins: int = DEFAULT_ORIGINS,
                     season: int | None = None, engine: Any = None, control: Any = None, combine_forecasts: bool = False,
                     blend_centre: bool = False, freq_hint: str | None = None, dates: Any = None) -> dict:
    x = np.asarray(values, dtype=float)
    n = len(x)
    if not np.isfinite(x).all():
        raise ValueError("series contains NaN or infinite values; fill or drop them first")
    m, season_reason = backtest_season(x, horizon, n_origins, season, freq_hint, dates)
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
    control, noop_note = _without_noop_context(control, n, horizon)      # A6-08: a no-op context is not a treatment
    fc = engine.forecast_batch([x[:o] for o in origins], horizon)
    configured = control is not None and not control.is_default
    if configured:
        # Each origin sees only what was knowable then: arrays are cut at the origin, so nothing leaks from the future.
        runs = [engine.forecast_controlled(x[:o], horizon, control.at_origin(o, horizon)) for o in origins]
        cfc = type(fc)(np.stack([r.point for r in runs]), np.stack([r.q for r in runs]))
    else:
        cfc = fc

    # Forecasts and actuals per origin; the blend (if asked for) is built from these, so it can only see earlier origins.
    # Intermittent demand (Syntetos-Boylan): the delivered point is TSB's expected volume and the delivered band is TSB's
    # quantiles (tools/timesfm_tool.py). The backtest scores that object, and TimesFM's median is kept as its own model
    # (audit A6-01). The classification uses the whole history, as the delivery tool does.
    lumpy = is_intermittent(x)
    simple = tuple(nm for nm in SIMPLE if not (lumpy and nm == "tsb_expected"))   # equal to the delivered point when lumpy
    preds: list[dict[str, np.ndarray]] = []
    actuals: list[np.ndarray] = []
    bands: list[np.ndarray] = []
    for j, o in enumerate(origins):
        hist = x[:o]
        p = {"timesfm": cfc.point[j], "naive": np.full(horizon, hist[-1]), "seasonal_naive": _snaive(hist, horizon, m),
             "drift": _drift(hist, horizon), "mean_last_season": _mean_last(hist, horizon, m),
             "tsb_expected": expected_volume(hist, horizon)}
        if lumpy:
            p["timesfm_median"] = cfc.point[j]
            p["timesfm"] = expected_volume(hist, horizon)
            bands.append(tsb_band(hist, horizon))
        else:
            bands.append(cfc.q[j])
        if configured:
            p["timesfm_default"] = fc.point[j]
        preds.append(p)
        actuals.append(x[o:o + horizon])
    combination: dict[str, Any] | None = None
    if combine_forecasts:
        blends, chosen, weights = _combined_forecasts(x, origins, m, preds, actuals, simple)
        for p, b in zip(preds, blends):
            p["timesfm_combined"] = b
        combination = {"method": "inverse-MSE weights per horizon third, from earlier origins only; the baseline is the one "
                                 "with the lowest mean MASE on earlier origins only",
                       "baseline_by_origin": chosen, "weight_timesfm_by_origin": weights,
                       "note": "origin 1 has no earlier origin, so it uses naive and equal weights"}
    blend_rule: str | None = None
    if blend_centre:
        blend_rule = _equal_blend_rule(x, origins, m, preds, actuals)
        if blend_rule is not None:
            for p in preds:
                p["timesfm_equal_blend"] = equal_blend(p["timesfm"], p[blend_rule])     # plain mean: nothing is fitted

    names = ["timesfm"] + (["timesfm_median"] if lumpy else []) + ["naive", "seasonal_naive", "drift", "mean_last_season",
                                                                  "tsb_expected"] + (
        ["timesfm_default"] if configured else []) + (["timesfm_combined"] if combine_forecasts else []) + (
        ["timesfm_equal_blend"] if blend_rule is not None else [])
    sq_scaled: dict[str, list[float]] = {nm: [] for nm in names}    # MSE in MASE-scale units, reported beside MASE (A2-08)
    errs: dict[str, list[float]] = {nm: [] for nm in names}
    act_sum, abs_err = 0.0, {nm: 0.0 for nm in names}
    signed = {nm: 0.0 for nm in names}          # sum of (forecast - actual): the volume bias, not the error
    covered, pin, wis_scaled = [], [], []
    qs = np.arange(1, 10) / 10
    for j, o in enumerate(origins):
        hist, actual = x[:o], actuals[j]
        scale = _mase_scale(hist, m)
        for nm in names:
            p = preds[j][nm]
            e = np.abs(actual - p)
            errs[nm].append(float(e.mean() / scale) if scale == scale else float("nan"))
            sq_scaled[nm].append(float(np.mean(e ** 2)) / scale ** 2 if scale == scale else float("nan"))
            abs_err[nm] += float(e.sum())
            signed[nm] += float((p - actual).sum())
        act_sum += float(np.abs(actual).sum())
        q = bands[j]
        covered.append(float(((actual >= q[:, 0]) & (actual <= q[:, 8])).mean()))
        d = actual[:, None] - q
        pin.append(float(np.mean(np.maximum(qs * d, (qs - 1) * d)) / scale) if scale == scale else float("nan"))
        wis_scaled.append(float(np.mean(wis(actual, q))) / scale if scale == scale else float("nan"))

    base = {nm: float(np.nanmean(errs[nm])) for nm in simple if nm in errs and not np.all(np.isnan(errs[nm]))}
    best = min(base, key=base.get) if base else None
    models: dict[str, Any] = {}
    for nm, v in errs.items():
        entry: dict[str, Any] = {"mase": round(float(np.nanmean(v)), 4) if len(v) and not np.all(np.isnan(v)) else None,
                                 "wape": round(abs_err[nm] / act_sum, 4) if act_sum > 0 else None,
                                 "volume_bias_pct": round(signed[nm] / act_sum * 100, 1) if act_sum > 0 else None}
        entry["mse_scaled"] = round(float(np.nanmean(sq_scaled[nm])), 4) if not np.all(np.isnan(sq_scaled[nm])) else None
        if best is not None and nm != best:
            entry["wins_vs_best_baseline"] = int(sum(1 for a, b in zip(v, errs[best])
                                                     if np.isfinite(a) and np.isfinite(b) and a < b))
        if nm == "mean_last_season":
            entry["window_points"] = _mean_last_window(m)
        models[nm] = entry
    # Weighted interval score of TimesFM's stored quantiles, in MASE units so it reads beside the MASE (A2-09). Selection
    # still uses MASE; WIS is reported next to it.
    wis_mean = round(float(np.nanmean(wis_scaled)), 4) if not np.all(np.isnan(wis_scaled)) else None
    models["timesfm"]["wis_scaled"] = wis_mean
    out: dict[str, Any] = {"points": n, "horizon": horizon, "origins": k, "n_origins": k, "season_used": m,
                           "season_reason": season_reason, "mase_scale": mase_scale_text(m),
                           "models": models,
                           "timesfm_range_coverage_80": round(float(np.mean(covered)), 3),
                           "timesfm_pinball_scaled": round(float(np.nanmean(pin)), 4) if not np.all(np.isnan(pin)) else None,
                           "timesfm_wis_scaled": wis_mean,
                           "point_basis": ("expected demand (mean under TSB); band: TSB quantiles" if lumpy
                                           else "median; band: TimesFM quantiles"),
                           "intermittent": bool(lumpy)}
    win = _mean_last_window(m)
    out["mean_last_season_window"] = {
        "points": win, "season": m,
        "note": (f"mean of the last {win} points; no season was found, so this is a {win}-point mean, not a season mean"
                 if m <= 1 else (f"mean of the last {win} points, the season" if win == m else
                                 f"mean of the last {win} points: the season is {m}, but the rule keeps at least {MEAN_LAST_FLOOR}")),
    }
    if season is not None:
        # A6-04: a given season is checked against the cycle measured on the prefix no test window can touch, so the check
        # reads nothing the verdict is scored on.
        cut = season_cut(n, horizon, n_origins)
        chk = season_check(season, x[:cut], None if dates is None else list(dates)[:cut])
        if chk is not None:
            out["season_check"] = chk
    if k < n_origins:
        out["origins_note"] = (f"only {k} of the {n_origins} requested windows of {horizon} fit in {n} points; "
                               "the evidence rests on these few windows")
    # For lumpy demand the note is about TimesFM's median, which is the point that understates volume; the delivered point
    # is the expected volume, whose bias is reported on its own entry (audit A6-01).
    tb_key = "timesfm_median" if lumpy else "timesfm"
    tb_name = "TimesFM's median" if lumpy else "TimesFM's point forecast"
    tb = models[tb_key]["volume_bias_pct"]
    if tb is not None and abs(tb) >= VOLUME_BIAS_NOTE:
        out["timesfm_volume_note"] = (
            f"Over these windows {tb_name} totals {tb:+.1f}% against what happened. A point forecast is a "
            "median, and for lumpy or skewed demand a median understates volume. Compare with tsb_expected (the mean under TSB) "
            "before presenting the point as expected demand, and say which measure the answer uses.")
    if noop_note is not None:
        out["control_noop"] = noop_note
    if configured:
        out["control"] = control.describe()
        # A6-08: the configuration was chosen before this call, but its windows are the ones scored here, so the verdict on
        # it is optimistic. Multiplicity across configurations is controlled only by control_lab (Holm).
        out["selection"] = {"selected_on_these_windows": True, "optimistic": True,
                            "note": "the configuration and these windows were both chosen by the caller; this backtest does not "
                                    "correct for the number of configurations tried. Use control_lab for a corrected choice, "
                                    "and confirm on later windows."}
        d, c = models["timesfm_default"]["mase"], models["timesfm"]["mase"]
        if d and c is not None:
            out["configured_vs_default_pct"] = round((d - c) / d * 100, 1)
    if combination is not None:
        out["combination"] = combination
        out["timesfm_combined_mase_note"] = "timesfm_combined is TimesFM blended with the earlier-best baseline; compare its mase with timesfm and best_baseline"
    if blend_centre:
        out["centre_choice"] = _blend_choice(errs, blend_rule, k, horizon, models)
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
            "Lower MASE is better. The error is scaled by the mean absolute change at the series' lag (the one-step change "
            "when there is no season, otherwise the seasonal change at lag m), so 1.0 means the average error equals the "
            "average change at that lag. "
            f"Range coverage should sit near 0.8. Measured over {k} windows of {horizon} periods. The verdict uses a paired "
            f"one-sided sign-flip test ({ev.get('test')}) of the per-window MASE gain over {best}, at "
            f"{(1 - CONFIDENCE) / 2:.0%} each way, and the number of windows TimesFM won; timesfm_better needs the test to "
            f"reject equal error in TimesFM's favour and at least {WIN_SHARE:.0%} of windows won, baseline_better needs it to "
            "reject in the baseline's favour, anything else is no_clear_difference. interval_pct is the bootstrap interval, "
            "a description only."
            + (" 'timesfm' is the configured run (see control); 'timesfm_default' is the same model with no control. "
               "configured_vs_default_pct above 0 means the configuration helped on these windows." if configured else ""))
    from prophet_agent.forecast.randomwalk import random_walk_test
    rw = random_walk_test(x)
    if rw is not None:
        out["random_walk"] = rw      # whether the series' changes are predictable at all, judged on the whole history
    return out
