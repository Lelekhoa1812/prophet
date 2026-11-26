"""Keep an event feature only when a rolling backtest shows it helps (M9-4).

Motivation vs Logic
Motivation: an event table can always be fitted to the past (a spike sits on an earnings day). Whether a known-ahead event
feature improves the next-period forecast is a measured question. A feature kept because it looks right is a choice, not an
improvement, and the figure should be reported whichever way it falls.
Logic: each candidate feature is scored by run_backtest, the existing walk-forward backtest, against two simple rules: an AR(1)
without events and, where the season is known, a seasonal naive. The challenger is the same AR(1) with the one feature as a
known-ahead input. Each origin uses only history before it; the feature's future values are the ones the table already holds
(scheduled dates are known ahead, a source claim). The feature is kept when the challenger's head WAPE beats the best rule.
The verdict is the measured relative improvement and is reported as such.
"""
from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np

from prophet_agent.forecast.backtest import BacktestData, Origin, run_backtest
from prophet_agent.forecast.baselines import seasonal_naive
from prophet_agent.forecast.metrics import paired_signflip
from prophet_agent.forecast.types import Forecast

LEVELS = np.arange(1, 10) / 10.0
Z = np.array([NormalDist().inv_cdf(float(p)) for p in LEVELS])
AR1 = "ar1"
SEASONAL = "seasonal_naive"
N_ORIGINS = 26
P_MAX = 0.05   # one-sided: the feature's origins must beat the rule's by more than chance (paired sign-flip test)


def guess_season(dates: Any) -> int:
    """The cycle the rules use: 5 for a business-day series, 7 for a daily one, else 1 (no seasonal rule). Measured from the
    median step between the saved dates, and whether any date falls on a weekend."""
    import pandas as pd
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    if len(idx) < 3:
        return 1
    step = pd.Series(idx).diff().dropna().median() / pd.Timedelta(days=1)
    if step > 1.5:
        return 1
    return 7 if (idx.day_of_week >= 5).any() else 5


def _band(point: np.ndarray, sd: float) -> np.ndarray:
    return point[:, None] + sd * Z[None, :]


def ar1_predictor(with_feature: bool):
    """AR(1) on the series, optionally with the feature as a same-period input. Fitted by least squares on the history;
    forecasts run forward on the model's own points, with the feature's known values at each step.

    Motivation vs Logic
    Motivation: the comparison has to isolate the feature. The rule and the challenger share everything except the feature.
    Logic: y_t = a + b*y_{t-1} (+ c*feature_t), fitted on the history. The band is the residual sd, normal quantiles. The
    backtest passes the active series as a batch, so each forecast has a leading axis (here one series).
    """
    def predict(origin: Origin) -> Forecast:
        x = np.asarray(origin.contexts[0], dtype=float)
        h = origin.horizon
        feat = None
        if with_feature:
            if not origin.covariates:
                raise ValueError("the feature model needs its covariate at each origin")
            feat = np.atleast_2d(np.asarray(origin.covariates[0], dtype=float))
        y = x[1:]
        cols = [np.ones(len(y)), x[:-1]]
        if feat is not None:
            cols += list(feat[:, 1:len(x)])
        design = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        sd = max(float(resid.std(ddof=design.shape[1])), 1e-9)
        prev, point = float(x[-1]), []
        for step in range(h):
            t = len(x) + step
            row = [1.0, prev] + ([float(feat[j, t]) for j in range(feat.shape[0])] if feat is not None else [])
            prev = float(np.dot(row, beta))
            point.append(prev)
        p = np.asarray(point)
        return Forecast(p[None, :], _band(p, sd)[None, :, :])      # one series in the backtest: a leading axis of 1
    return predict


def seasonal_predictor(season: int):
    def predict(origin: Origin) -> Forecast:
        p, q = seasonal_naive(np.asarray(origin.contexts[0], dtype=float), origin.horizon, season)
        return Forecast(p[None, :], q[None, :, :])
    return predict


def check_feature(values: np.ndarray, name: str, feature: np.ndarray, horizon: int, *, season: int = 1,
                  n_origins: int = N_ORIGINS, p_max: float = P_MAX) -> dict:
    """The measured verdict for one feature: run_backtest over the last ``n_origins`` origins, with the rules and the challenger.
    ``feature`` is as long as the history plus the horizon. Returns the challenger's head WAPE, the best rule's, the relative
    improvement (positive means the feature helps) and kept = improvement > 0."""
    values = np.asarray(values, dtype=float).reshape(-1)
    feature = np.asarray(feature, dtype=float).reshape(-1)
    if feature.size != values.size + horizon:
        raise ValueError(f"feature {name!r} must span the history plus the horizon ({values.size + horizon}), got {feature.size}")
    challenger = f"{AR1}+{name}"
    models = {AR1: ar1_predictor(False), challenger: ar1_predictor(True)}
    baselines = [AR1]
    if season >= 2:
        models[SEASONAL] = seasonal_predictor(season)
        baselines.append(SEASONAL)

    def covariates(origin_index: int, active: np.ndarray, h: int) -> list[np.ndarray]:
        return [feature[None, : origin_index + h] for _ in active]

    res = run_backtest(lambda: BacktestData(demand=values[None, :]), models, horizon=horizon, n_origins=n_origins,
                       covariates=covariates, challenger=challenger, baselines=baselines, season=max(season, 1))
    v = res.get("verdict")
    if v is None:
        raise ValueError("the backtest produced no verdict")
    # Per-origin head WAPE of the rule and the challenger, paired by origin: the sign-flip test asks whether the gap is
    # larger than chance across origins. Origins overlap when the horizon is longer than one step, so p is a guide.
    rule_err = [o["wape"] for o in res["models"][v["best_baseline"]]["by_origin"]]
    feat_err = [o["wape"] for o in res["models"][challenger]["by_origin"]]
    test = paired_signflip(np.asarray(rule_err), np.asarray(feat_err))
    gain = v["improvement_vs_best_baseline"]
    kept = bool(gain is not None and gain > 0 and test["p_cand_better"] is not None
                and test["p_cand_better"] <= p_max)
    return {"feature": name, "challenger_head_wape": round(v["challenger_head_wape"], 4),
            "best_rule": v["best_baseline"], "best_rule_head_wape": round(v["best_baseline_head_wape"], 4),
            "improvement_vs_best_rule": None if gain is None else round(gain, 4),
            "p_feature_better": None if test["p_cand_better"] is None or test["p_cand_better"] != test["p_cand_better"]
            else round(test["p_cand_better"], 4),
            "origins": len(res["origins"]), "horizon": horizon, "kept": kept}


def check_features(values: np.ndarray, features: dict[str, np.ndarray], horizon: int, *, season: int = 1,
                   n_origins: int = N_ORIGINS) -> dict:
    """Check each feature on its own (measured) and keep the ones that help. Returns {kept: [names], verdicts: {name: ...},
    measure: the rule stated in words}. Features are judged one at a time, so one useful feature does not carry a useless one."""
    # Bonferroni over the candidates: each one must reach P_MAX divided by how many were tried, so a table of many weak
    # features does not pass one by chance (measured on a synthetic null feature, see the report).
    per = P_MAX / max(1, len(features))
    verdicts = {name: check_feature(values, name, arr, horizon, season=season, n_origins=n_origins, p_max=per)
                for name, arr in features.items()}
    return {"kept": [n for n, v in verdicts.items() if v["kept"]], "verdicts": verdicts,
            "measure": "measured on this series by a rolling backtest; a feature is kept when its head WAPE is lower than the "
                       f"best of the rules (AR(1) and, with a season, seasonal naive) and its paired sign-flip p is at most {P_MAX} "
                       "divided by the number of candidates (Bonferroni)"}
