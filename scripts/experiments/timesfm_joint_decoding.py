"""Measure whether TimesFM 3 joint (multivariate) decoding beats per-entity decoding on a real panel.

Question: on the same rolling origins and the same history, does one 2-D context holding related entities (decoded
jointly through the checkpoint's variate attention) forecast better than each entity as its own 1-D context? Does
use_symmetric_averaging help the 1-D arm?

Arms, all at every origin, all with the same history (every day before the origin):
  per_entity            one 1-D context per entity (what Prophet does today), through TimesFMEngine.forecast_batch
  joint                 all entities as one 2-D context (num_variates, context); one forecast row per entity
  per_entity_symmetric  1-D contexts with use_symmetric_averaging=True
  seasonal_naive        last 4 same-weekday values, band from its own residuals (forecast/baselines.py)
  naive                 last value, band from its own residuals (forecast/baselines.py, window 1)

Scoring per entity and origin: MASE (mean absolute error over the horizon divided by the in-sample mean absolute
seasonal difference of the history, season 7, lag-1 fallback), and whether the actual falls inside the 0.1 to 0.9
band (nominal 80%). Paired comparison: per-entity mean MASE over origins, differences between arms, a sign-flip test
(prophet_agent.forecast.global_gbm.sign_flip_test) and the count of entities where the first arm is lower.

Every number written to the JSON or printed is measured on the named panel, origins and horizon.

Usage (from the repository root, with the venv):
  PYTHONPATH=src .venv/bin/python scripts/experiments/timesfm_joint_decoding.py --panel beijing --smoke
  PYTHONPATH=src .venv/bin/python scripts/experiments/timesfm_joint_decoding.py --panel beijing
  PYTHONPATH=src .venv/bin/python scripts/experiments/timesfm_joint_decoding.py --panel electricity

The panels are git-ignored data. In a git worktree without data/, pass --data-dir pointing at the checkout that has it.
Output goes to <data-dir>/experiments/. Do not run the full measurement while a live mission is using the model.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prophet_agent.forecast.baselines import moving_average, seasonal_naive  # noqa: E402
from prophet_agent.forecast.global_gbm import sign_flip_test  # noqa: E402
from prophet_agent.forecast.timesfm import nonnegative_domain  # noqa: E402

PANELS = {
    "beijing": ("beijing_pm25_daily_12_stations.csv", 14),
    "electricity": ("electricity_daily_60_clients.csv", 7),
}
PANEL_SUBDIR = "datasets/panel_sample_2026-11-22"
SEASON = 7
MAX_MISSING_SHARE = 0.05  # an entity missing more than this share of calendar days is dropped
MIN_HISTORY = 4 * SEASON  # the first origin must leave at least this many days of history
ARMS = ("per_entity", "joint", "per_entity_symmetric", "seasonal_naive", "naive")
Q_LO, Q_HI = 0, 8  # quantile indices of the 0.1 and 0.9 bounds


def load_panel(path: Path, max_missing_share: float = MAX_MISSING_SHARE):
    """Long CSV (entity, timestamp, value) to aligned arrays on one daily calendar.

    Gap policy: the calendar is every day from the first to the last timestamp in the file. An entity whose first day
    is missing is dropped (nothing to carry forward). An entity missing more than max_missing_share of the calendar is
    dropped. Interior and trailing gaps are filled by carrying the last observed value forward. This is causal: a filled
    day never reads a later day, so no origin's history contains information from after it.

    Returns (entities, dates, X, log) with X shaped (n_entities, n_days).
    """
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        value_col = [c for c in reader.fieldnames if c not in ("entity", "timestamp")][0]
        obs: dict[str, dict[dt.date, float]] = {}
        for r in reader:
            obs.setdefault(r["entity"], {})[dt.date.fromisoformat(r["timestamp"])] = float(r[value_col])
    all_days = sorted({d for v in obs.values() for d in v})
    start, end = all_days[0], all_days[-1]
    n_days = (end - start).days + 1
    dates = [start + dt.timedelta(days=i) for i in range(n_days)]
    log: dict = {"calendar": {"start": start.isoformat(), "end": end.isoformat(), "days": n_days},
                 "dropped": {}, "filled_days": {}, "value_column": value_col}
    entities, rows = [], []
    for name in sorted(obs):
        y = np.array([obs[name].get(d, np.nan) for d in dates], dtype=float)
        missing = np.isnan(y)
        if missing[0]:
            log["dropped"][name] = "first calendar day missing"
            continue
        share = float(missing.mean())
        if share > max_missing_share:
            log["dropped"][name] = f"{share:.1%} of days missing (limit {max_missing_share:.0%})"
            continue
        last_seen = np.maximum.accumulate(np.where(~missing, np.arange(n_days), 0))
        rows.append(y[last_seen])
        entities.append(name)
        log["filled_days"][name] = int(missing.sum())
    if not entities:
        raise ValueError(f"no entity survives the gap policy in {path}")
    return entities, dates, np.vstack(rows), log


def origins_for(n_days: int, horizon: int, n_origins: int) -> list[int]:
    """Start index of each forecast window: the last n_origins non-overlapping blocks of horizon days.

    The history of the window starting at s is columns [0, s). The windows never overlap and the last one ends on the
    final day of the calendar.
    """
    if n_origins < 1 or horizon < 1:
        raise ValueError("need at least one origin and a horizon of at least one day")
    starts = [n_days - (n_origins - j) * horizon for j in range(n_origins)]
    if starts[0] < MIN_HISTORY:
        raise ValueError(f"the first origin leaves {starts[0]} days of history; at least {MIN_HISTORY} are needed")
    return starts


def mase_scale(history: np.ndarray) -> float:
    """Mean absolute seasonal difference (season 7) of the history. Falls back to lag 1 when that is zero or not finite.

    Returns NaN when neither is usable (a constant history), and the entity is then left out of the paired comparison.
    """
    for lag in (SEASON, 1):
        if history.size > lag:
            s = float(np.mean(np.abs(history[lag:] - history[:-lag])))
            if np.isfinite(s) and s > 0:
                return s
    return float("nan")


def mase_per_entity(actual: np.ndarray, pred: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """MASE per entity: mean absolute error over the horizon divided by the scale. Same definition as metrics.mase."""
    return np.abs(actual - pred).mean(axis=1) / scale


def finalise(point: np.ndarray, q: np.ndarray, histories) -> tuple[np.ndarray, np.ndarray]:
    """Post-processing applied to every TimesFM arm, copied from TimesFMEngine.forecast_batch.

    Quantiles are sorted. Point and quantiles are floored at zero only for histories that are non-negative measures
    (nonnegative_domain). predict_batch does no flooring itself, so without this the arms would not be comparable.
    """
    point = np.asarray(point, dtype=float)
    q = np.sort(np.asarray(q, dtype=float), axis=-1)
    nonneg = np.array([nonnegative_domain(h) for h in histories], dtype=bool)
    point = np.where(nonneg[:, None], np.clip(point, 0, None), point)
    q = np.where(nonneg[:, None, None], np.clip(q, 0, None), q)
    return point, q


class TimesFMForecaster:
    """Adapter over the shared TimesFMEngine. Each method returns (point (n, h), q (n, h, 9)) for the rows it is given.

    Reuses the engine's loaded weights, its context cap and its inference lock, so the arms run on the same checkpoint
    that Prophet serves. The lock is private to the engine; this script takes it because it calls predict_batch
    directly for the symmetric and joint arms.
    """

    def __init__(self, engine):
        self.engine = engine

    def per_entity(self, histories, horizon: int):
        out = self.engine.forecast_batch(histories, horizon)
        return out.point, out.q

    def per_entity_symmetric(self, histories, horizon: int):
        with self.engine._infer_lock:
            outs = list(self.engine.model.predict_batch(
                [np.asarray(h, dtype=np.float32) for h in histories], horizon=horizon,
                return_quantiles=True, use_symmetric_averaging=True))
        point = np.stack([np.asarray(o.forecast) for o in outs])
        q = np.stack([np.asarray(o.quantiles) for o in outs])
        return finalise(point, q, histories)

    def joint(self, matrix: np.ndarray, horizon: int):
        """One 2-D context (num_variates, context). predict_batch returns (num_variates, horizon) and quantiles
        (num_variates, horizon, 9); every row is a target because there are no covariates."""
        matrix = np.asarray(matrix, dtype=np.float32)
        with self.engine._infer_lock:
            (out,) = list(self.engine.model.predict_batch([matrix], horizon=horizon, return_quantiles=True))
        point = np.asarray(out.forecast).reshape(matrix.shape[0], horizon)
        q = np.asarray(out.quantiles).reshape(matrix.shape[0], horizon, -1)
        return finalise(point, q, list(matrix))


def run_origin(forecaster, X: np.ndarray, start: int, horizon: int, arms=ARMS, group_size: int = 0):
    """Forecast every arm from the same history X[:, :start] and return the actuals, per-arm forecasts and timings.

    group_size 0 puts every entity in one joint context. A positive value splits the entities, in panel order, into
    joint contexts of at most that size. No variate cap is in the MLX code (see the experiment doc), so the default is
    all entities.
    """
    histories = [X[i, :start] for i in range(X.shape[0])]
    actual = X[:, start:start + horizon]
    n = X.shape[0]
    groups = [list(range(n))] if group_size <= 0 else [list(range(i, min(i + group_size, n)))
                                                       for i in range(0, n, group_size)]
    forecasts, timings = {}, {}
    for arm in arms:
        t0 = time.perf_counter()
        if arm == "per_entity":
            point, q = forecaster.per_entity(histories, horizon)
        elif arm == "per_entity_symmetric":
            point, q = forecaster.per_entity_symmetric(histories, horizon)
        elif arm == "joint":
            pts, qs = [None] * n, [None] * n
            for g in groups:
                p_g, q_g = forecaster.joint(X[g, :start], horizon)
                for local, idx in enumerate(g):
                    pts[idx], qs[idx] = p_g[local], q_g[local]
            point, q = np.stack(pts), np.stack(qs)
        elif arm == "seasonal_naive":
            res = [seasonal_naive(h, horizon, SEASON) for h in histories]
            point, q = np.stack([r[0] for r in res]), np.stack([r[1] for r in res])
        elif arm == "naive":
            res = [moving_average(h, horizon, 1) for h in histories]
            point, q = np.stack([r[0] for r in res]), np.stack([r[1] for r in res])
        else:
            raise ValueError(f"unknown arm {arm!r}")
        timings[arm] = time.perf_counter() - t0
        forecasts[arm] = (np.asarray(point, dtype=float), np.asarray(q, dtype=float))
    scales = np.array([mase_scale(h) for h in histories])
    return {"actual": actual, "forecasts": forecasts, "timings": timings, "scales": scales}


def score_origin(result: dict) -> dict:
    """Per-entity MASE and 80% band hit rate for each arm at one origin."""
    actual, scales = result["actual"], result["scales"]
    out = {}
    for arm, (point, q) in result["forecasts"].items():
        inside = (actual >= q[..., Q_LO]) & (actual <= q[..., Q_HI])
        out[arm] = {"mase": mase_per_entity(actual, point, scales), "inside": inside}
    return out


def summarise(panel: str, horizon: int, origin_dates: list[str], entities: list[str], mase: dict, inside: dict,
              wall: dict, paired_pairs: list[tuple[str, str]], dropped: dict) -> dict:
    """Aggregate per arm and run the paired comparisons. mase[arm] is (n_entities, n_origins), inside[arm] is
    (n_entities, n_origins, horizon)."""
    provenance = f"measured on the {panel} panel, {len(origin_dates)} origins, horizon {horizon}"
    arms = {}
    for arm in mase:
        per_entity_mean = mase[arm].mean(axis=1)
        arms[arm] = {
            "provenance": provenance,
            "mean_mase_over_entities": round(float(np.mean(per_entity_mean)), 4),
            "median_mase_over_entities": round(float(np.median(per_entity_mean)), 4),
            "coverage_80_pooled": round(float(inside[arm].mean()), 4),
            "wall_seconds": round(wall[arm], 3),
        }
    paired = {}
    for a, b in paired_pairs:
        da = mase[a].mean(axis=1)
        db = mase[b].mean(axis=1)
        d = da - db
        test = sign_flip_test(d)
        paired[f"{a}_minus_{b}"] = {
            "provenance": provenance,
            "pairs": int(d.size),
            "mean_mase_difference": round(float(d.mean()), 4),
            "entities_where_first_arm_lower": int(np.sum(d < 0)),
            "entities_where_first_arm_higher": int(np.sum(d > 0)),
            "sign_flip": test,
        }
    per_entity = {name: {arm: round(float(mase[arm].mean(axis=1)[i]), 4) for arm in mase}
                  for i, name in enumerate(entities)}
    return {"provenance": provenance, "panel": panel, "horizon": horizon, "season": SEASON,
            "origin_starts": origin_dates, "entities_used": entities, "entities_dropped_at_load": dropped,
            "arms": arms, "paired": paired, "per_entity_mean_mase": per_entity}


def run(args) -> dict:
    fname, default_h = PANELS[args.panel]
    horizon = args.horizon or default_h
    n_origins, max_entities = args.origins, args.max_entities
    if args.smoke:
        n_origins, max_entities = 2, 4
    path = Path(args.data_dir) / PANEL_SUBDIR / fname
    entities, dates, X, log = load_panel(path)
    dropped = dict(log["dropped"])
    if max_entities and X.shape[0] > max_entities:
        for name in entities[max_entities:]:
            dropped[name] = f"not in the first {max_entities} entities (--max-entities)"
        entities, X = entities[:max_entities], X[:max_entities]
    starts = origins_for(X.shape[1], horizon, n_origins)

    # Scale problems are per origin, so an entity is kept only when every origin has a usable scale.
    scale_ok = np.ones(X.shape[0], dtype=bool)
    for s in starts:
        sc = np.array([mase_scale(X[i, :s]) for i in range(X.shape[0])])
        scale_ok &= np.isfinite(sc)
    for i, name in enumerate(entities):
        if not scale_ok[i]:
            dropped[name] = "constant history (no usable seasonal or lag-1 scale) at some origin"
    entities = [e for i, e in enumerate(entities) if scale_ok[i]]
    X = X[scale_ok]
    if X.shape[0] < 2:
        raise ValueError("fewer than two entities survive; a paired comparison needs at least two")

    from prophet_agent.forecast.timesfm import TimesFMEngine
    t0 = time.perf_counter()
    engine = TimesFMEngine.get()
    load_seconds = round(time.perf_counter() - t0, 2)
    forecaster = TimesFMForecaster(engine)

    n, k = X.shape[0], len(starts)
    mase = {arm: np.zeros((n, k)) for arm in ARMS}
    inside = {arm: np.zeros((n, k, horizon), dtype=bool) for arm in ARMS}
    wall = {arm: 0.0 for arm in ARMS}
    for j, s in enumerate(starts):
        result = run_origin(forecaster, X, s, horizon, group_size=args.joint_group_size)
        scored = score_origin(result)
        for arm in ARMS:
            mase[arm][:, j] = scored[arm]["mase"]
            inside[arm][:, j, :] = scored[arm]["inside"]
            wall[arm] += result["timings"][arm]
        print(f"origin {j + 1}/{k} start {dates[s].isoformat()}: done", flush=True)

    origin_dates = [f"{dates[s].isoformat()} to {dates[s + horizon - 1].isoformat()}" for s in starts]
    summary = summarise(args.panel, horizon, origin_dates, entities, mase, inside, wall,
                        paired_pairs=[("joint", "per_entity"), ("per_entity_symmetric", "per_entity"),
                                      ("joint", "per_entity_symmetric")],
                        dropped=dropped)
    summary.update({
        "mode": "smoke" if args.smoke else "full",
        "checkpoint": engine.checkpoint,
        "context_cap_points": engine.global_context,
        "model_load_seconds": load_seconds,
        "joint_group_size": args.joint_group_size,
        "gap_policy": log,
        "max_missing_share": MAX_MISSING_SHARE,
        "scale": "mean absolute seasonal difference (season 7) of history before origin; lag-1 fallback",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    })
    return summary


def print_table(summary: dict) -> None:
    print()
    print(f"{summary['provenance']} ({summary['mode']} run; {len(summary['entities_used'])} entities)")
    print(f"{'arm':<22}{'mean MASE':>11}{'median MASE':>13}{'cov 80%':>10}{'wall s':>10}")
    for arm, r in summary["arms"].items():
        print(f"{arm:<22}{r['mean_mase_over_entities']:>11.4f}{r['median_mase_over_entities']:>13.4f}"
              f"{r['coverage_80_pooled']:>10.3f}{r['wall_seconds']:>10.2f}")
    print()
    for name, p in summary["paired"].items():
        print(f"{name}: mean diff {p['mean_mase_difference']:+.4f} MASE, first arm lower on "
              f"{p['entities_where_first_arm_lower']}/{p['pairs']} entities, sign-flip p={p['sign_flip']['p_value']} "
              f"({p['sign_flip']['method']})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--panel", choices=sorted(PANELS), required=True)
    parser.add_argument("--origins", type=int, default=6, help="number of non-overlapping rolling origins")
    parser.add_argument("--horizon", type=int, default=None, help="default 14 (beijing) or 7 (electricity)")
    parser.add_argument("--max-entities", type=int, default=0, help="keep the first N entities; 0 keeps all")
    parser.add_argument("--joint-group-size", type=int, default=0,
                        help="split entities into joint contexts of at most this size; 0 uses one context")
    parser.add_argument("--smoke", action="store_true", help="2 origins and 4 entities, for a quick end-to-end check")
    parser.add_argument("--data-dir", default=str(REPO / "data"), help="directory holding datasets/ (git-ignored)")
    args = parser.parse_args(argv)

    summary = run(args)
    out_dir = Path(args.data_dir) / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"timesfm_joint_{args.panel}_{summary['mode']}_{stamp}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print_table(summary)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
