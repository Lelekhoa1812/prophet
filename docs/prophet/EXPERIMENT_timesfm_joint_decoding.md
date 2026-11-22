# Experiment: does TimesFM 3 joint decoding beat per-entity decoding?

Status: measurement script built and smoke-tested. The full measurement has not been run. No result in this document
is evidence yet.

Script: `scripts/experiments/timesfm_joint_decoding.py`
Tests: `tests/prophet_agent/experiments/test_timesfm_joint_decoding.py` (offline, stub forecaster, no model load)

## Question

On real related-entity panels, on the same rolling origins and the same history:

1. Does TimesFM 3 decoding of all related entities as one 2-D context (joint, variate attention across entities) forecast
   better than decoding each entity as its own 1-D context (what Prophet does today)?
2. Does `use_symmetric_averaging=True` improve the 1-D arm?

Panels: `beijing_pm25_daily_12_stations.csv` (12 stations, daily PM2.5, H = 14) and
`electricity_daily_60_clients.csv` (60 clients, daily kWh, H = 7). Both are real public data in the git-ignored
`data/datasets/panel_sample_2026-11-22/`.

## Method

**Panel and calendar.** The CSV is long (entity, timestamp, value). It is pivoted onto one daily calendar from the first
to the last timestamp. Gap policy:

- An entity whose first calendar day is missing is dropped (nothing to carry forward).
- An entity missing more than 5% of calendar days is dropped.
- Interior and trailing gaps are filled by carrying the last observed value forward. This is causal: a filled day never
  reads a later day, so no origin's history contains information from after it.

The script prints and records the dropped entities, the reason, and the number of filled days per entity.

**Rolling origins.** The last N non-overlapping windows of H days (default N = 6). Window j starts at
`n_days - (N - j) * H`. Its history is every column strictly before the start. The last window ends on the final day.

**Arms.** Each arm sees the same history at each origin.

| Arm | What it is | Code |
|---|---|---|
| `per_entity` | one 1-D context per entity, through `TimesFMEngine.forecast_batch` (Prophet today) | `TimesFMForecaster.per_entity` |
| `joint` | all entities as one 2-D context `(num_variates, context)`; one forecast row per entity. `--joint-group-size` splits them into chunks, default one chunk | `TimesFMForecaster.joint`, `run_origin` |
| `per_entity_symmetric` | 1-D contexts with `use_symmetric_averaging=True` | `TimesFMForecaster.per_entity_symmetric` |
| `seasonal_naive` | last 4 same-weekday values, band from its own residuals | `forecast/baselines.py` |
| `naive` | last value, band from its own residuals (`moving_average` with window 1) | `forecast/baselines.py` |

Every TimesFM arm goes through the same post-processing (`finalise`), copied from `TimesFMEngine.forecast_batch`. Quantiles
are sorted. Point and quantiles are floored at zero only for non-negative histories (`nonnegative_domain`). The MLX
`predict_batch` does no flooring of its own, so without this step the arms would not be comparable.

**Scoring.** Per entity and origin:

- MASE: mean absolute error over the horizon, divided by the in-sample mean absolute seasonal difference of the history
  (season 7). If that is zero or not finite, lag-1 is used. If both are unusable, the entity is dropped (constant history).
  This matches `forecast/metrics.py:mase`, computed per entity.
- Coverage: whether the actual falls inside the 0.1 to 0.9 band (nominal 80%), pooled over entities, origins and days.
- Wall time per arm, summed over origins.

**Paired comparison.** The pairs are entities. Each entity's MASE is averaged over origins, and the arms are differenced
per entity. The script reports, for `joint - per_entity`, `per_entity_symmetric - per_entity` and
`joint - per_entity_symmetric`:

- the mean difference,
- the count of entities where the first arm is lower,
- a two-sided sign-flip test (`prophet_agent.forecast.global_gbm.sign_flip_test`: exact up to 16 entities, then Monte
  Carlo with a fixed seed).

Every number in the output carries `provenance`: panel, number of origins and horizon.

## How 2-D contexts are passed (verified in the installed package)

Package: `.venv/lib/python3.12/site-packages/timesfm3/mlx/`. Paths below are relative to that folder.

1. **Input shape.** `timesfm3_forecaster.py:184-187` (docstring): a 1-D context gives `forecast (horizon,)`; a 2-D context
   `(num_variates, context)` is one multivariate series, and gives `(num_variates, horizon)`.
2. **The joint path is the general path.** The fast path at `timesfm3_forecaster.py:255-262` runs only for plain 1-D
   contexts without covariates, symmetric averaging, z-normalisation or padding. A 2-D context, and any symmetric run,
   goes to the general path at `:283-288`.
3. **The 2-D context is kept whole.** `:289` `tgt_orig = np.atleast_2d(...)` keeps `(u, ctx)`. `:328`
   `final = _series_logits(tgt, po, pf)` decodes it in one call. `:335` `_shape(final, tgt.shape[0], ...)` treats every
   variate row as a target, because there are no covariates.
4. **Outputs.** `:233-237` `_shape`: `forecast = tgt[..., median_idx]` gives `(num_target, horizon)`, and
   `quantiles = tgt` gives `(num_target, horizon, 9)`. `:246-248` only squeezes when the input was 1-D.
5. **Decode.** `model.py:111-136` (`decode`): `target (b, u, context)`. `:133-135` targets, past-only and past-future
   covariates are stacked on the variate axis. `:204` `ctx_vals = mx.concatenate(...)`; `:206` `num_variates`.
   `:258-266`: with no covariates, every variate is a target patch. `:275` one forward pass over all variates.
6. **Variate attention.** `transformer.py:133-137`: after sequence attention, attention runs across the variate axis at
   each patch index (`reshape(b*n, v, d)`). `transformer.py:120`: `var_attn` is built with `causal=False`, so it is
   non-causal across variates. `transformer.py:91-97`: the only mask is padding. Variates share a time index, so the
   entities must share one calendar, which the gap policy ensures.
7. **Normalisation is per variate.** `util.py:83-93` (`get_running_stats`) gives running stats per `(b, v)`, so each
   entity is normalised by its own history.
8. **No variate cap in the MLX code.** `configs.py:27-42` has no maximum-variates field. `configs.py:38` only switches
   variate attention on (default True). `timesfm3_forecaster.py:71` sets only the context cap (`max_context_length =
   15360`). The script therefore uses one joint context by default. If a cap appears later, use `--joint-group-size`.
9. **Symmetric averaging.** `timesfm3_forecaster.py:219-229`: decode the series and its negation, sort each, and return
   `(pos - neg[..., ::-1]) / 2`. It is implemented only in the general path (see 2).

Prophet's side, `src/prophet_agent/forecast/timesfm.py`:

- `:41-55` loads the checkpoint once (`CHECKPOINT = "google/timesfm-3.0-pytorch"`), and `TimesFMEngine.get()` is the
  process-wide singleton. The experiment reuses it, so it runs the same checkpoint with the same MLX backend.
- `:80` truncates to `global_context` (15,360 points). The histories here are about 1,400 days, far below the cap.
- `:81-82` runs `predict_batch` under `_infer_lock`. The script takes the same lock for its direct calls.
- `:89-93` applies the non-negative floor, which `finalise` copies.

Other source facts used:

- `src/prophet_agent/forecast/global_gbm.py:42-43, 333-357`: `EXACT_MAX_ENTITIES = 16`, `MC_DRAWS = 10_000`,
  `sign_flip_test`.

## Smoke run (a smoke run, not evidence)

Command (smoke only; this check loads the real model):

```
PYTHONPATH=src .venv/bin/python scripts/experiments/timesfm_joint_decoding.py --panel beijing --smoke --data-dir /Users/liamle/Downloads/prophet/data
```

Smoke means 2 origins (starts 2017-02-01 and 2017-02-15, H = 14) and the first 4 stations in name order
(Aotizhongxin, Changping, Dingling, Dongsi). Four entities and two origins are far too few to support any conclusion.
Measured on that run:

| Arm | Mean MASE | Median MASE | Coverage 80% | Wall s |
|---|---|---|---|---|
| per_entity | 0.4961 | 0.4940 | 0.804 | 0.90 |
| joint | 0.5078 | 0.5125 | 0.830 | 0.12 |
| per_entity_symmetric | 0.4769 | 0.4812 | 0.795 | 0.31 |
| seasonal_naive | 0.8381 | 0.8511 | 0.938 | 0.00 |
| naive | 1.2370 | 1.2403 | 0.955 | 0.00 |

Paired (smoke, 4 entities, exact sign-flip): joint minus per_entity +0.0117 MASE (joint lower on 1 of 4);
symmetric minus per_entity -0.0193 (lower on 4 of 4, p = 0.125, the smallest p a 4-entity exact test can give);
joint minus symmetric +0.0310 (lower on 0 of 4).

The smoke JSON is `/Users/liamle/Downloads/prophet/data/experiments/timesfm_joint_beijing_smoke_20261122T235551.json`.
The wall times are not comparable across arms. The arms run in a fixed order and the first MLX call for each new
input shape includes compilation, which is why `per_entity` shows the largest time.

## Full measurement

Run this only when no live mission is using the model. The coordinator runs it.

```
PYTHONPATH=src .venv/bin/python scripts/experiments/timesfm_joint_decoding.py --panel beijing --data-dir /Users/liamle/Downloads/prophet/data
PYTHONPATH=src .venv/bin/python scripts/experiments/timesfm_joint_decoding.py --panel electricity --data-dir /Users/liamle/Downloads/prophet/data
```

Defaults: 6 origins, H = 14 (Beijing) or 7 (electricity), all entities in one joint context. Output:
`<data-dir>/experiments/timesfm_joint_<panel>_full_<timestamp>.json`, git-ignored.

## Limits to read with any result

- Six origins give six windows per entity. Coverage is measured on at most 6 x H days per entity.
- Entities are not independent. They share calendar effects (seasons, weather, the same days). The sign-flip test treats
  per-entity averages as exchangeable, so its p-values are indicative, not exact for this design.
- Wall times include MLX compilation per input shape, and arms run in a fixed order. Treat timing as indicative.
- Joint decoding is measured with all entities in one context. Whether a related-entity subset does better is not tested.
- The `per_entity` arm batches the 1-D contexts into one `predict_batch` call. The MLX fast path decodes rows
  independently, so this equals decoding them one at a time. Not checked against a single-call run.
- The `_infer_lock` access is private to `TimesFMEngine`. The script uses it for the symmetric and joint calls.
- Checkpoint weights are non-commercial: research use only. Use these outputs for research, not for commercial decisions,
  client deliverables or paid services.

## Result (measured, 2026-11-23, real TimesFM 3 MLX, real public panels)

Six non-overlapping rolling origins per panel, history strictly before each origin, MASE per entity averaged over origins,
paired sign-flip test on per-entity means. JSON: `data/experiments/timesfm_joint_beijing_full_20261123T012301.json` and
`data/experiments/timesfm_joint_electricity_full_20261123T012317.json` (git-ignored).

| Panel | Arm | Mean MASE | Median MASE | 80% coverage |
|---|---|---|---|---|
| Beijing PM2.5, 12 stations, H = 14 | per entity (Prophet today) | 0.6473 | 0.6610 | 0.799 |
| | joint (all stations as one multivariate context) | 0.6762 | 0.6871 | 0.814 |
| | per entity, symmetric averaging | 0.6268 | 0.6334 | 0.827 |
| | seasonal naive | 1.3027 | 1.3245 | 0.593 |
| | naive | 1.8856 | 1.8989 | 0.801 |
| Electricity, 60 clients, H = 7 | per entity | 1.7033 | 1.6109 | 0.757 |
| | joint | 1.6907 | 1.6449 | 0.771 |
| | per entity, symmetric averaging | 1.7061 | 1.6361 | 0.761 |
| | seasonal naive | 2.0868 | 2.0794 | 0.678 |
| | naive | 2.2069 | 1.9822 | 0.843 |

Paired comparisons:

- Joint minus per entity: Beijing +0.0289 (joint lower on 2 of 12 stations, exact p = 0.0034); electricity -0.0126 (23 of
  60, Monte Carlo p = 0.81).
- Symmetric averaging minus per entity: Beijing -0.0205 (lower on 12 of 12, exact p = 0.0005); electricity +0.0029 (19
  of 60, p = 0.77).

Decision: joint decoding is not offered. It was worse on one panel and indistinguishable on the other. Symmetric averaging
stays a per-series candidate in control_lab, not a default: it helped on Beijing (about 3 percent) and did nothing on
electricity. Caveats: six origins per panel; stations in one city are not independent, so the per-entity tests overstate
the evidence; wall times include MLX compile and are not comparable across arms.
