"""Walk-forward backtest: forecast each past origin using only data before it, then score what happened.

Ported from src/lab/engine/backtest.py. The data loader, covariates and model set are injected. Nothing is persisted,
and no pass/fail bar is applied: the verdict returns the measured improvement, and the caller compares it with its own
bar. Composite models (such as an ensemble) are built by the caller as predictors that combine other predictors.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from . import metrics
from .types import Forecast


@dataclass(frozen=True)
class BacktestData:
    """Demand matrix plus optional per-series arrays. `weights` defines the head set (the series that make up
    `head_share` of total weight). `segments` labels series for the by-segment WAPE."""

    demand: np.ndarray                  # (series, periods) actual values
    weights: np.ndarray | None = None   # (series,) e.g. revenue
    segments: np.ndarray | None = None  # (series,) labels


@dataclass(frozen=True)
class Origin:
    """What a predictor may see at one origin. The history is demand[:, :index], and nothing later."""

    index: int                              # origin position o
    active: np.ndarray                      # row ids of series with recent demand
    contexts: list[np.ndarray]              # history per active series, all before index
    covariates: list[np.ndarray] | None     # known-future covariates per active series, from the injected provider
    horizon: int
    season: int


Predictor = Callable[[Origin], Forecast]
CovariateProvider = Callable[[int, np.ndarray, int], list[np.ndarray] | None]


def head_mask(weights: np.ndarray | None, n_series: int, share: float = 0.8) -> np.ndarray:
    """Boolean mask of the smallest set of series whose weight reaches `share` of the total. No weights: all series."""
    if weights is None:
        return np.ones(n_series, dtype=bool)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(-w)
    cum = np.cumsum(w[order]) / max(w.sum(), 1e-9)
    mask = np.zeros(n_series, dtype=bool)
    mask[order[: int(np.searchsorted(cum, share)) + 1]] = True
    return mask


def _score(actual, fc: Forecast, scale) -> dict:
    if actual.size == 0:  # e.g. the tail set when every series is in the head
        return dict(wape=np.nan, bias=np.nan, pinball=np.nan, coverage=np.nan, mase=np.nan)
    return dict(wape=metrics.wape(actual, fc.point), bias=metrics.bias(actual, fc.point),
                pinball=metrics.pinball(actual, fc.q), coverage=metrics.coverage(actual, fc.q),
                mase=metrics.mase(actual, fc.point, scale))


def _ratio(num: float, den: float) -> float | None:
    """(num) / (den) as a float, or None when the denominator is zero or missing."""
    if den is None or not np.isfinite(den) or den == 0:
        return None
    return float(num / den)


def run_backtest(load: Callable[[], BacktestData], models: Mapping[str, Predictor], *, horizon: int = 4,
                 n_origins: int = 26, min_history: int = 8, activity_window: int = 26, season: int = 52,
                 scale_window: int = 52, covariates: CovariateProvider | None = None, challenger: str = "timesfm",
                 baselines: Sequence[str] | None = None, head_share: float = 0.8, min_segment_rows: int = 20,
                 progress: Callable[[int, int], None] | None = None) -> dict:
    """Score each predictor at the last `n_origins` origins, using only history before each origin.

    `load` returns the BacktestData. `covariates(origin_index, active_ids, horizon)` may return known-future covariates,
    which reach predictors as `Origin.covariates`. `verdict` is present when `challenger` is among the models: it holds
    the measured relative WAPE improvement over the best baseline (head set, positive means the challenger is better)
    and over the per-series oracle baseline. `baselines` defaults to every other model.
    """
    t0 = time.perf_counter()
    data = load()
    demand = np.asarray(data.demand, dtype=float)
    n_series, n_periods = demand.shape
    names = list(models)
    if not names:
        raise ValueError("no models supplied")
    last_origin = n_periods - horizon
    origins = list(range(max(min_history, last_origin - n_origins + 1), last_origin + 1))
    if not origins:
        raise ValueError(f"not enough history: {n_periods} periods, need > {min_history + horizon}")

    head = head_mask(data.weights, n_series, head_share)
    acc = {m: {k: [] for k in ("act", "pt", "q", "sid", "orig", "scale")} for m in names}
    for k, o in enumerate(origins):
        hist = demand[:, :o]
        active = np.where(hist[:, -min(activity_window, o):].sum(axis=1) > 0)[0]
        if len(active) == 0:
            continue
        ctxs = [hist[i] for i in active]
        actual = demand[active, o:o + horizon]
        cov = covariates(o, active, horizon) if covariates is not None else None
        scale = np.array([np.abs(np.diff(c[-scale_window:])).mean() if len(c) > 2 else np.nan for c in ctxs])
        origin = Origin(index=o, active=active, contexts=ctxs, covariates=cov, horizon=horizon, season=season)
        for m in names:
            fc = models[m](origin)
            a = acc[m]
            a["act"].append(actual); a["pt"].append(fc.point); a["q"].append(fc.q)
            a["sid"].append(active); a["orig"].append(np.full(len(active), o)); a["scale"].append(scale)
        if progress:
            progress(k + 1, len(origins))

    if not acc[names[0]]["act"]:
        raise ValueError("no series had recent demand at any origin")

    out: dict = {"horizon": horizon, "origins": origins, "n_series": n_series,
                 "head_share_series": int(head.sum()), "models": {}}
    flat: dict[str, tuple] = {}
    for m in names:
        a = acc[m]
        act = np.concatenate(a["act"]); pt = np.concatenate(a["pt"]); q = np.concatenate(a["q"])
        sid = np.concatenate(a["sid"]); orig = np.concatenate(a["orig"]); sc = np.concatenate(a["scale"])
        hm = head[sid]
        res = {"all": _score(act, Forecast(pt, q), sc),
               "head": _score(act[hm], Forecast(pt[hm], q[hm]), sc[hm]),
               "tail": _score(act[~hm], Forecast(pt[~hm], q[~hm]), sc[~hm]),
               "by_horizon": [metrics.wape(act[hm][:, h], pt[hm][:, h]) for h in range(horizon)],
               "by_origin": [{"origin": int(o), "wape": metrics.wape(act[hm & (orig == o)], pt[hm & (orig == o)])}
                             for o in origins],
               "calibration": [float((act[hm] <= q[hm][..., j]).mean()) for j in range(q.shape[-1])],
               "by_segment": _by_segment(data.segments, sid, hm, act, pt, min_segment_rows)}
        out["models"][m] = res
        flat[m] = (act, pt, hm)

    if challenger in out["models"]:
        candidates = list(baselines) if baselines is not None else [m for m in names if m != challenger]
        base_names = [m for m in candidates if m in out["models"]]
        if base_names:
            best = min(base_names, key=lambda m: out["models"][m]["head"]["wape"])
            tw = out["models"][challenger]["head"]["wape"]
            bw = out["models"][best]["head"]["wape"]
            act, _, hm = flat[challenger]
            errs = np.stack([np.abs(flat[m][0] - flat[m][1]).sum(axis=1) for m in base_names])
            oracle = float(errs.min(axis=0)[hm].sum() / np.abs(act[hm]).sum())
            tw_cov = out["models"][challenger]["head"]["coverage"]
            out["verdict"] = {
                "challenger": challenger,
                "best_baseline": best,
                "challenger_head_wape": tw,
                "best_baseline_head_wape": bw,
                "improvement_vs_best_baseline": _ratio(bw - tw, bw),
                "oracle_baseline_head_wape": oracle,
                "improvement_vs_oracle": _ratio(oracle - tw, oracle),
                "band_coverage_head": tw_cov,
            }
    out["seconds"] = round(time.perf_counter() - t0, 3)
    return out


def _by_segment(segments, sid, hm, act, pt, min_rows: int) -> dict:
    """Head-set WAPE per segment label, for segments with more than `min_rows` rows."""
    if segments is None:
        return {}
    labels = np.asarray(segments).astype(str)[sid]
    out = {}
    for lab in sorted(set(labels[hm].tolist())):
        msk = (labels == lab) & hm
        if msk.sum() > min_rows:
            out[lab] = metrics.wape(act[msk], pt[msk])
    return out
