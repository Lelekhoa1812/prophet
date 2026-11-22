# TimesFM 3 checkpoint and package: what the shipped artefacts declare (2026-11-22)

Method: REA `inspect_model_artifact` on the cached checkpoint (header only, no weights read, no inference), plus a read of
the installed `timesfm3` package source in `.venv/lib/python3.12/site-packages/timesfm3/`. Everything below is a file
declaration or source text. Nothing here measures forecast quality.

## Checkpoint header (observed)

- File: the Hugging Face blob behind `google/timesfm-3.0-pytorch` snapshot `43046b85…/model.safetensors`
  (1,322,898,824 bytes; header 54,920 bytes, header SHA-256 `dc9635b0…8e8d`; full-file SHA-256 not computed).
- REA evidence: parent `ev_4ad0fbe0…0d38`; tensor pages `ev_4f0c3c0d…fb2a` and `ev_99c61363…4a08`.
- 445 tensors, all F32, 330,710,976 parameters (derived from declared shapes).
- Model width 1280; attention head width 80, so 16 heads (inferred from `key_ln` shape 80 and 1280/80).
- Each layer has two attention blocks: `seq_attn` (along time) and `var_attn` (across variates), each with query, key,
  value and output projections and their own layer norms. Layers are numbered 0 to 19 in the tensor names (20 layers,
  matching the package comment).
- Input block `pre_transformer_resblock` maps 192 inputs to 1280. Output `output_head` maps 1280 to 576.
- No positional-embedding tensor is declared, so positions are not a learned table in this file (inferred).

## Package API (observed in source)

| Capability | Where | Prophet today |
|---|---|---|
| Context cap `max_context_length = 15360` | `timesfm3/mlx/timesfm3_forecaster.py:72` | Uses `global_context` (full context): matches |
| `predict_batch(contexts, horizon, past_only_covariates, past_future_covariates, return_quantiles, use_symmetric_averaging, make_positive, sort_quantiles, use_znorm, padding_mode)` | same file, line 168 | Calls it with 1-D contexts and `return_quantiles=True` only |
| A 2-D context `(num_variates, context)` is one multivariate series: the targets are decoded jointly | docstring at line 186 | Never used: each panel entity is forecast alone (`forecast_batch` stacks 1-D series in the batch axis) |
| `decode(target (b, u, context), past_only_covariates, past_future_covariates, target_mask, past_only_mask, past_future_mask, mask)`: targets, past-only and past-future covariates are stacked on the variate axis and mixed by variate attention | `timesfm3/mlx/model.py:111` | `forecast_controlled` uses it with one target, covariates and a target mask (`forecast/timesfm.py:85`) |
| `use_symmetric_averaging` (average with the sign-flipped series) | `predict_batch` | Never set |
| `use_znorm`, `padding_mode` ("none" or "edge") | `predict_batch` | Prophet applies its own z-normalisation in `forecast_controlled` |

## Implications to test, not conclusions

1. Joint decoding of related entities (stations, clients, SKUs in one family) as one multivariate target could use the
   variate attention the checkpoint carries. Whether it beats per-entity decoding is unknown and must be measured on the
   panels with `panel_backtest`-style rolling origins before it is offered.
2. `use_symmetric_averaging` is a cheap option to add to the control lab as one more candidate, kept only if a backtest on
   the same origins says it helps.
3. Past-only covariates from other entities (for example a neighbouring station's PM2.5) can be passed through the
   existing `past_only_covariates` path; the control lab already guards covariates with a backtest.
