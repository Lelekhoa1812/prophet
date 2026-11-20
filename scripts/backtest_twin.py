"""G13: walk-forward backtest of TimesFM 3 against the honest baselines on the synthetic twin.

Labels: every number this script prints is SYNTHETIC REPLAY. The series come from the synthetic generator, not from
client data. History is the recorded sales (censored by stockouts, as a planner would see them); the future four weeks
are the generator's true demand. The result is written under data/backtests/, which is git-ignored.

The 15 % bar is applied as written in the legacy audit: relative WAPE improvement over the best baseline on the head
set (80 % of volume). The bar is reported, not moved.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/backtest_twin.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from prophet_agent.forecast.backtest import BacktestData, run_backtest
from prophet_agent.forecast.baselines import moving_average, seasonal_naive, seasonal_year
from prophet_agent.forecast.types import Forecast
from prophet_agent.synth.generator import generate

SEED = 20261120
N_ENTITIES = 12
HISTORY = 156          # three years of weekly history
HORIZON = 4
SEASON = 52
BAR = 0.15
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "backtests"


def per_series(fn):
    """Lift a one-series baseline (series, h) -> (point, quantiles) to a predictor over all active series."""
    def predict(origin) -> Forecast:
        rows = [fn(c, origin.horizon) for c in origin.contexts]
        return Forecast(np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows]))
    return predict


def timesfm_predictor(origin) -> Forecast:
    from prophet_agent.forecast.timesfm import TimesFMEngine
    return TimesFMEngine.get().forecast_batch(origin.contexts, origin.horizon)


def main() -> int:
    ds = generate(seed=SEED, n_entities=N_ENTITIES, n_periods=HISTORY, horizon=HORIZON, freq="W")
    demand = np.concatenate([np.asarray(ds.observed, dtype=float), np.asarray(ds.truth_future, dtype=float)], axis=1)
    weights = np.asarray(ds.observed, dtype=float).sum(axis=1)

    models = {
        "timesfm": timesfm_predictor,
        "moving_average": per_series(lambda s, h: moving_average(s, h, 8)),
        "seasonal_naive": per_series(lambda s, h: seasonal_naive(s, h, SEASON, k=4)),
        "seasonal_year": per_series(lambda s, h: seasonal_year(s, h, SEASON)),
    }

    t0 = time.perf_counter()
    out = run_backtest(
        lambda: BacktestData(demand=demand, weights=weights),
        models,
        horizon=HORIZON,
        n_origins=26,
        season=SEASON,
        scale_window=SEASON,
        challenger="timesfm",
        baselines=["moving_average", "seasonal_naive", "seasonal_year"],
        head_share=0.8,
        progress=lambda k, n: print(f"  origin {k}/{n}", flush=True),
    )
    elapsed = round(time.perf_counter() - t0, 1)

    print("\nSYNTHETIC REPLAY: TimesFM 3 vs baselines, walk-forward, 26 origins, horizon 4 weeks")
    print(f"series {out['n_series']} (head set {out['head_share_series']}), origins {out['origins'][0]}..{out['origins'][-1]}")
    print(f"{'model':<16}{'WAPE all':>10}{'WAPE head':>11}{'WAPE tail':>11}{'bias all':>10}{'cov80 head':>12}")
    for name, res in out["models"].items():
        print(f"{name:<16}{res['all']['wape']:>10.3f}{res['head']['wape']:>11.3f}{res['tail']['wape']:>11.3f}"
              f"{res['all']['bias']:>10.3f}{res['head']['coverage']:>12.3f}")

    verdict = out.get("verdict", {})
    imp = verdict.get("improvement_vs_best_baseline")
    if imp is not None:
        status = "MEETS" if imp >= BAR else "DOES NOT MEET"
        print(f"\nbest baseline: {verdict['best_baseline']} (head WAPE {verdict['best_baseline_head_wape']:.3f})")
        print(f"TimesFM head WAPE {verdict['challenger_head_wape']:.3f}; improvement vs best baseline {imp:+.1%}")
        print(f"improvement vs per-series oracle {verdict['improvement_vs_oracle']:+.1%}")
        print(f"15 % bar: {status}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = OUT_DIR / f"twin_backtest_{stamp}.json"
    payload = {"label": "synthetic replay", "seed": SEED, "n_entities": N_ENTITIES, "history": HISTORY,
               "horizon": HORIZON, "bar": BAR, "seconds": elapsed, "result": out}
    path.write_text(json.dumps(payload, default=float, indent=1))
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
