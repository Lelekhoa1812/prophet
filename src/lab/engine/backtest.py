"""Walk-forward backtest: forecast each past origin using only data before it, score what happened."""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone

import numpy as np

from ..store import open_lab
from . import baselines, metrics
from .data import Demand, load_demand, load_expected, load_promo_covariate
from .forecast import Forecast, TimesFMEngine

MODELS_W = ["timesfm", "timesfm_cov", "ensemble", "ma4", "ma8", "seasonal_year"]
MODELS_D = ["timesfm", "timesfm_cov", "ensemble", "ma8", "seasonal_naive"]


def _score(actual, fc: Forecast, scale):
    return dict(wape=metrics.wape(actual, fc.point), bias=metrics.bias(actual, fc.point),
                pinball=metrics.pinball(actual, fc.q), coverage=metrics.coverage(actual, fc.q),
                mase=metrics.mase(actual, fc.point, scale))


def run_backtest(dataset: str, freq: str = "W", horizon: int = 4, n_origins: int = 26,
                 min_hist: int | None = None, models: list[str] | None = None, progress=None) -> dict:
    t0 = time.time()
    dem = load_demand(dataset, freq)
    models = models or (MODELS_W if freq == "W" else MODELS_D)
    season = 52 if freq == "W" else 7
    n_per = dem.qty.shape[1]
    min_hist = min_hist or (8 if freq == "W" else 28)
    last_origin = n_per - horizon
    origins = list(range(max(min_hist, last_origin - n_origins + 1), last_origin + 1))
    if not origins:
        raise ValueError(f"not enough history: {n_per} periods, need > {min_hist + horizon}")

    rev_total = dem.revenue.sum(axis=1)
    order = np.argsort(-rev_total)
    cum = np.cumsum(rev_total[order]) / max(rev_total.sum(), 1e-9)
    head_mask = np.zeros(len(dem.product_ids), dtype=bool)
    head_mask[order[: int(np.searchsorted(cum, 0.8)) + 1]] = True

    engine = TimesFMEngine.get() if any(m.startswith("timesfm") or m == "ensemble" for m in models) else None
    promo = load_promo_covariate(dataset, dem, extra=horizon) if "timesfm_cov" in models else None
    expected = load_expected(dataset, dem)
    if "noise_floor" not in models and expected is not None:
        models = models + ["noise_floor"]
    acc = {m: {"act": [], "pt": [], "q": [], "sid": [], "orig": [], "scale": []} for m in models}
    for k, o in enumerate(origins):
        hist = dem.qty[:, :o]
        active = np.where(hist[:, -min(26 if freq == "W" else 90, o):].sum(axis=1) > 0)[0]
        if len(active) == 0:
            continue
        ctxs = [hist[i] for i in active]
        actual = dem.qty[active, o:o + horizon]
        ds = np.array([np.abs(np.diff(c[-52:])).mean() if len(c) > 2 else np.nan for c in ctxs])
        cache = {}
        for m in models:
            if m == "timesfm":
                fc = cache["timesfm"] = engine.predict(ctxs, horizon)
            elif m == "timesfm_cov":
                covs = [promo[i, max(0, o - 512):o + horizon][None, :] for i in active]
                fc = engine.predict_cov(ctxs, horizon, covs)
            elif m == "ensemble":
                tf = cache.get("timesfm") or engine.predict(ctxs, horizon)
                ma = baselines.run("ma8", ctxs, horizon, season)
                fc = Forecast((tf.point + ma.point) / 2, (tf.q + ma.q) / 2)
            elif m == "noise_floor":
                p_ = expected[active, o:o + horizon]
                sd = np.sqrt(np.maximum(p_, 1e-6) * 3.0)   # lumpy demand: wide band around the true mean
                fc = Forecast(p_, np.clip(p_[..., None] + baselines.Z[None, None, :] * sd[..., None], 0, None))
            else:
                fc = baselines.run(m, ctxs, horizon, season)
            a = acc[m]
            a["act"].append(actual); a["pt"].append(fc.point); a["q"].append(fc.q)
            a["sid"].append(active); a["orig"].append(np.full(len(active), o)); a["scale"].append(ds)
        if progress:
            progress(k + 1, len(origins))

    out = {"dataset": dataset, "freq": freq, "horizon": horizon, "origins": len(origins),
           "first_origin": str(dem.periods[origins[0]].date()), "last_origin": str(dem.periods[origins[-1]].date()),
           "models": {}, "head_share_series": int(head_mask.sum()), "n_series": len(dem.product_ids)}
    best_pt = {}
    for m in models:
        a = acc[m]
        act = np.concatenate(a["act"]); pt = np.concatenate(a["pt"]); q = np.concatenate(a["q"])
        sid = np.concatenate(a["sid"]); orig = np.concatenate(a["orig"]); sc = np.concatenate(a["scale"])
        hm = head_mask[sid]
        res = {"all": _score(act, Forecast(pt, q), sc),
               "head": _score(act[hm], Forecast(pt[hm], q[hm]), sc[hm]),
               "tail": _score(act[~hm], Forecast(pt[~hm], q[~hm]), sc[~hm]),
               "by_horizon": [metrics.wape(act[hm][:, h], pt[hm][:, h]) for h in range(horizon)],
               "by_origin": [{"origin": str(dem.periods[o].date()),
                              "wape": metrics.wape(act[hm & (orig == o)], pt[hm & (orig == o)])} for o in origins],
               "calibration": [float((act[hm] <= q[hm][..., j]).mean()) for j in range(9)]}
        cats = {}
        for c in sorted(set(dem.category.values())):
            msk = np.array([dem.category[dem.product_ids[i]] == c for i in sid]) & hm
            if msk.sum() > 20:
                cats[c] = metrics.wape(act[msk], pt[msk])
        res["by_category"] = cats
        out["models"][m] = res
        best_pt[m] = (act, pt, sid, hm)

    # ship-bar maths: improvement of timesfm over the best single baseline (chosen on head WAPE)
    if "timesfm" in out["models"]:
        base = [m for m in models if not m.startswith("timesfm") and m not in ("ensemble", "noise_floor")]
        best = min(base, key=lambda m: out["models"][m]["head"]["wape"])
        tw = out["models"]["timesfm"]["head"]["wape"]; bw = out["models"][best]["head"]["wape"]
        # oracle: best baseline per series (harsh)
        act, _, sid, hm = best_pt["timesfm"]
        per = {}
        for m in models:
            pt = best_pt[m][1]
            e = np.abs(best_pt[m][0] - pt).sum(axis=1)
            per[m] = e
        stack = np.stack([per[m] for m in base])
        oracle_err = stack.min(axis=0)
        oracle = float(oracle_err[hm].sum() / np.abs(act[hm]).sum())
        out["verdict"] = {"best_baseline": best, "timesfm_head_wape": tw, "best_baseline_head_wape": bw,
                          "improvement_vs_best_baseline": (bw - tw) / bw if bw else None,
                          "oracle_baseline_head_wape": oracle,
                          "improvement_vs_oracle": (oracle - tw) / oracle if oracle else None,
                          "band_coverage_head": out["models"]["timesfm"]["head"]["coverage"],
                          "ship_bar": {"improve_min": 0.15, "coverage_range": [0.70, 0.90]}}
        v = out["verdict"]
        v["passes_error_bar"] = bool(v["improvement_vs_best_baseline"] is not None and v["improvement_vs_best_baseline"] >= 0.15)
        v["passes_coverage_bar"] = bool(0.70 <= v["band_coverage_head"] <= 0.90)
    out["seconds"] = round(time.time() - t0, 1)

    lab = open_lab()
    bid = f"BT-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    lab.execute("insert into backtests values (?,?,?,?,?)", [bid, dataset, datetime.now(timezone.utc),
                json.dumps({"freq": freq, "horizon": horizon, "n_origins": n_origins, "models": models}), json.dumps(out)])
    lab.close()
    out["backtest_id"] = bid
    return out
