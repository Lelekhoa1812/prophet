# 01. TimesFM 3 integration: what exists today and what an agent could control

Scope: read-only investigation of `/Users/liamle/Downloads/prophet` (lab code, docs, installed `timesfm` 3.0.2 distribution in `.venv`, cached checkpoint `config.json`). No data files were rewritten and no forecast or backtest was run. Line numbers are `file:line`. Paths under `src/lab/` are relative to the repo root. Library paths are under `.venv/lib/python3.12/site-packages/`, abbreviated `SP/`.

Labels used below: **[Verified]** = read in source or doc, with a citation. **[Verified, doc only]** = stated in a doc, not re-derived by running code. **[UNVERIFIED]** = hypothesis, not tested.

---

## 1. How TimesFM is loaded, shared, locked and cached today

**[Verified] Loader and checkpoint.** The only model the lab loads is the MLX backend of the 3.0 checkpoint.
- Checkpoint id: `TIMESFM_CHECKPOINT = "google/timesfm-3.0-pytorch"` (`src/lab/config.py:30`).
- Loader: `TimesFMEngine.__init__` calls `from timesfm3.mlx import TimesFM3Forecaster` and `TimesFM3Forecaster.from_pretrained(TIMESFM_CHECKPOINT)` with no kwargs (`src/lab/engine/forecast.py:25-30`). So `compile=True`, `max_context_length=15360`, `median_quantile_index=4`, and the HF defaults all apply (`SP/timesfm3/mlx/timesfm3_forecaster.py:56-77`).
- The cached checkpoint lives at `~/.cache/huggingface/hub/models--google--timesfm-3.0-pytorch/snapshots/43046b85.../` with `config.json` and `model.safetensors`. The CLAUDE.md note that the first load downloads from HF applies only on a cold cache.

**[Verified] Process-wide singleton.** `TimesFMEngine` is a class-level singleton.
- `_inst` and `_lock` are class attributes (`forecast.py:22-23`). `get()` takes `_lock` and constructs the engine once (`forecast.py:32-37`). Construction, including the weight load, runs while the lock is held.
- `_infer` is a second class-level `threading.Lock` (`forecast.py:39`). Every `predict` and `predict_cov` call holds it for the whole batch loop (`forecast.py:45`, `forecast.py:61`). Concurrent jobs therefore serialise on the model in one process.
- `api/main.py:106` reads `TimesFMEngine._inst` to report `loaded`. `api/main.py:501`, `:520` and `:532` call `TimesFMEngine.get().predict` directly, bypassing the router.

**[Verified] Routing singleton.** The routing layer is a second process-wide singleton.
- `_ROUTER: Router | None` (`src/lab/engine/models.py:222`). `get_forecaster()` lazily builds a default `Router()` (`models.py:225-231`). `set_router(r)` replaces it with no lock (`models.py:234-236`).
- The default router sends every series class to `timesfm` (`models.py:129-130`, `:134-135`). Nothing installs a learned routing table in a live path. `docs/GOAL_EVIDENCE.md:9` records this: "No code path called `set_router`". `docs/RUNBOOK.md:18` says to run `fit_routing` manually.
- **[Verified] Licence guard is off by default.** `Router.allow_non_commercial` defaults to `True` (`models.py:131`). The guard that swaps in the free fallback (`tsb`) runs only when it is `False` (`models.py:137-138`).

**[Verified] Call path into the model.** Two paths reach the model.
- `Router.predict` splits series by class, applies a `min_history` check, falls back to `ma4` for short series, then calls `TimesFM.predict` (`models.py:144-159`). `TimesFM.predict` calls `TimesFMEngine.get().predict(contexts, horizon)` and ignores `season` (`models.py:34-41`).
- `ForecastService` (`engine/state.py:144-255`) builds weekly matrices and calls `get_forecaster().predict` (`state.py:164`, `:284`, `:323`).
- Covariate calls bypass the router entirely and happen only in backtests (`engine/backtest.py:73-79`).

**[Verified] Context cap.** `TimesFMEngine.predict` and `predict_cov` truncate each context to its last 512 points (`forecast.py:41`, `:44`, `:56`, `:59`). The library default cap is 15360 (`SP/timesfm3/mlx/timesfm3_forecaster.py:71`). The lab overrides it with a much smaller window.

**[Verified] Batching.** `predict` and `predict_cov` slice contexts into chunks of 128 (`forecast.py:41-42`, `:46-47`, `:62`). The library's `per_core_batch_size` field (`SP/timesfm3/mlx/timesfm3_forecaster.py:63`) is never read by the MLX code path, so chunking is the lab's own loop.

**[Verified] Cache.** Forecasts are pickled per (dataset, kind, as-of, horizon) under `data/cache` (`state.py:19`, `:151-152`). Each entry stores a `fp` fingerprint and the history matrix. Both must match or the entry is recomputed (`state.py:115-141`). The fingerprint hashes route, model names, licence strings, `allow_non_commercial`, the checkpoint id and the fallback (`models.py:161-177`). It does not hash weights or model hyper-parameters. `docs/GOAL_CHANGELOG.md:9` and `docs/GOAL_AUDIT.md:53` record this as an accepted limit: "a retrained model under the same name needs a new name" (`models.py:166-167`).
- `data/cache` held 1,962 entries when counted (`ls data/cache | wc -l`).

**[Verified] Post-processing applied by the lab.** After the model returns, the lab clips to zero, sorts the nine quantiles, and wraps the result in a `Forecast` (`forecast.py:50-53`, `:67-68`). `QUANTILES` is a constant (`forecast.py:12`). The library's own `make_positive` and `sort_quantiles` flags are not used.

**[Verified] Other backends.** A torch backend ships in the same package at `SP/timesfm3/torch/`. Its device is `cuda` if available, otherwise `cpu` (`SP/timesfm3/torch/timesfm3_forecaster.py`, device selection around line 370-432). The lab does not use it.

---

## 2. Control knobs the installed library exposes

Installed distribution: `timesfm-3.0.2.dist-info` (`SP/timesfm-3.0.2.dist-info/METADATA`). It ships two packages: `timesfm3` (the 3.0 model with MLX and torch backends) and `timesfm` (a 2.5-style API with a different 200M checkpoint). The lab uses only `timesfm3.mlx`.

**[Verified by `inspect.signature` on the installed class]**

```
TimesFM3Forecaster.from_pretrained(pretrained_model_name_or_path: str, **kwargs) -> TimesFM3Forecaster
TimesFM3Forecaster.predict(context: np.ndarray, horizon: int,
    past_only_covariates=None, past_future_covariates=None, ts_id=None,
    return_quantiles=False, use_symmetric_averaging=False, make_positive=False,
    sort_quantiles=True, use_znorm=False, padding_mode="none") -> ForecastOutput
TimesFM3Forecaster.predict_batch(contexts: list[np.ndarray], horizon: int,
    past_only_covariates: list|None=None, past_future_covariates: list|None=None,
    ts_ids: list[str]|None=None, return_quantiles=False, use_symmetric_averaging=False,
    make_positive=False, sort_quantiles=True, use_znorm=False, padding_mode="none") -> Iterator[ForecastOutput]
TimesFM3Mlx.decode(target, horizon=0, past_only_covariates=None, past_future_covariates=None,
    target_mask=None, past_only_mask=None, past_future_mask=None, mask=None) -> mx.array (b, variates, h, q)
TimesFM3Mlx.from_pretrained(checkpoint_path="google/timesfm-3.0-pytorch", *, compile=True,
    cache_dir=None, revision=None, token=None, local_files_only=False, force_download=False)
```

**Constructor fields (`_ModelConfig`, `SP/timesfm3/mlx/timesfm3_forecaster.py:56-77`).** These are the only settings that `from_pretrained(**kwargs)` accepts.

| Field | Default | What it controls | Used by MLX code? |
|---|---|---|---|
| `checkpoint_path` | `google/timesfm-3.0-pytorch` | Weights source | Yes (`:113`) |
| `per_core_batch_size` | 4 | Batch size hint | No (only defined) |
| `median_quantile_index` | 4 | Which quantile is the point forecast | Yes (`:122-125`, `:236`) |
| `compile` | True | `mx.compile` of forward pass | Yes (`mlx/model.py:98-109`) |
| `max_context_length` | 15360 | Truncation of target and covariates | Yes (`:201`, `:309-316`) |
| `cache_dir`, `revision`, `token`, `local_files_only`, `force_download` | HF defaults | Hub download | Yes (`model.py:354-368`) |

**Architecture config (`TimesFM3MlxConfig`, `SP/timesfm3/mlx/configs.py:27-54`) and loader (`:56-83`).**
- Fields: `input_patch_len=32`, `output_patch_len=64`, `model_dims=1280`, `num_layers=20`, `num_heads=16`, `quantiles` (nine values 0.1..0.9), `use_variate_attention=True`, `use_stitching=True`, `use_linear_detrending=True`, `linear_detrending_threshold=0.5`, `value_clip=1e20`.
- `from_hf_config` maps `config.json` onto these fields (`configs.py:56-83`). It raises `NotImplementedError` if `use_frozen_running_stats` is true, because the MLX backend does not implement it (`configs.py:59-66`). The cached checkpoint sets it to `false`.

**Per-call flags (`predict_batch`, `SP/timesfm3/mlx/timesfm3_forecaster.py:168-341`).**
- `return_quantiles` (`:181`, `:236`, `:248`): returns the full 9-quantile array. The lab always sets this to True.
- `use_symmetric_averaging` (`:219-229`): decodes the series and its negation, sorts each, and averages the reversed negative run. This is a flip-symmetry device. **Not used by the lab.**
- `make_positive` (`:238-245`): clamps the forecast to zero when the input is non-negative. The lab clips itself instead (`forecast.py:50-51`).
- `sort_quantiles` (`:226-231`, `:276-277`): on by default.
- `use_znorm` (`:300-305`, `:330-333`): z-normalises each target and covariate row with its own mean and std, then un-normalises the target output. **Not used by the lab.**
- `padding_mode` (`:189-190`, `:322-326`): `"none"` or `"edge"`. `"edge"` extends the future covariate to a patch multiple by repeating its last value. **Not used by the lab.**
- `past_only_covariates`: accepted, but the lab never passes it.

**Model-level flags read from `config.json` (`SP/timesfm3/mlx/configs.py`).**
- `use_stitching` is parsed (`configs.py:39`, `:79`) but **never read** by the model. Patch stitching in `decode` is unconditional (`mlx/model.py:279`, `mlx/util.py:112-127`). **[Verified]**
- `use_variate_attention` is honoured (`mlx/transformer.py:116`).
- `use_linear_detrending` and `linear_detrending_threshold` are honoured (`mlx/model.py:210`, `:229`, `:281`, `:291`, `:316`).
- `value_clip` is honoured (`mlx/model.py:94`).
- Causal attention and RoPE are hard-coded (`mlx/transformer.py:115-120`). The checkpoint sets `causal_attention: true` and `use_rope_seq: true`, which match the hard-coded values, so this is consistent but not configurable.
- The MLX path always runs the iterative CPM RevIN refinement, because `horizon_cpm` is always passed (`mlx/model.py:268-275`). The checkpoint's `use_iterative_cpm_revin: true` agrees.

**Horizon and context rules.**
- `decode` requires `horizon > 0` (`mlx/model.py:143-144`). There is no maximum horizon in the MLX code. The number of decoded patches is computed from the horizon (`model.py:181-184`). Quality beyond the horizons seen in training is **[UNVERIFIED]**.
- `context_length` returns `input_patch_len` (32). It is a patch size, not a limit (`mlx/timesfm3_forecaster.py:127-129`).
- Contexts are left-padded with masked values to a whole number of patches (`mlx/model.py:148-162`).

**Quantile head.** Nine quantiles are fixed by the checkpoint (`config.json` `quantiles`). The lab hard-codes the same nine (`forecast.py:12`). There is no continuous-quantile head option in the MLX path.

**`timesfm` package (2.5-style), not used by the lab.** `SP/timesfm/configs.py` `ForecastConfig` has `max_context`, `max_horizon`, `normalize_inputs`, `per_core_batch_size`, `use_continuous_quantile_head`, `force_flip_invariance=True`, `infer_is_positive=True`, `fix_quantile_crossing`, `return_backcast`. `SP/timesfm/timesfm_2p5/timesfm_2p5_base.py:155-196` defines `forecast(horizon, inputs)`. Its `forecast_with_covariates` (`:198-`) fits a linear XReg model (`SP/timesfm/utils/xreg_lib.py:411`) with modes `"xreg + timesfm"` or `"timesfm + xreg"`. This is a different model, a different checkpoint (200M), and it is not wired into the lab.

**Torch backend differences.** `SP/timesfm3/torch/timesfm3_forecaster.py:38-100` exposes more fields: `input_transform`, `use_sdpa`, `use_iterative_cpm_revin`, `use_frozen_running_stats` (default False, supported in torch but rejected by MLX), `residual_block_config`, `transformer_config`, `device`. The torch `predict_batch` has the same signature as MLX (`:465-`). The lab does not use it, and the two backends' parity has **not** been checked here.

**Fine-tuning.** A search of the 3.0 package (excluding test files) for `finetun`, `fine_tun`, `optimizer`, `lora`, `backward(`, `.train(`, `grad(` and `adapter` returned no matches. **[Verified]** No training or fine-tuning hook is exposed. The checkpoint config has `"training": true` inside the torch transformer block (`config.json`), which is a model flag, not a hook.

---

## 3. "Event manipulation": what exists and how each piece works

There are four different mechanisms. They are not connected to one another, and only one of them touches the live forecast path.

### 3a. Known-future covariates (forecast side)

**[Verified] Wrapper.** `TimesFMEngine.predict_cov(contexts, horizon, future_covs)` (`forecast.py:55-68`).
- Each `future_covs[i]` has shape `(k, len(context_i)+horizon)` and is sliced to the last `len(context)+horizon` values (`forecast.py:60`).
- Each context is wrapped as a 1-row `(1, L)` array (`forecast.py:59`) and passed as `past_future_covariates` (`forecast.py:63`).
- The MLX library requires the covariate length to equal `context + horizon`. It infers the horizon from that length (`mlx/model.py:141-142`). The lab respects this rule.

**[Verified] How the library uses a future covariate.**
- The covariate is stacked as an extra variate after the target and past-only variates (`mlx/model.py:199-206`).
- On the horizon side it carries real future values, detrended with its own trend (`mlx/model.py:226-237`).
- It is marked as conditioning only, not as a patch to reconstruct. `patch_is_target` is 0 for covariate rows (`mlx/model.py:259-266`).
- Variate attention lets the target attend to it (`mlx/transformer.py:115-120`).
- The model applies its own RevIN normalisation over the stacked variates (`mlx/model.py:57-63`, `_forward_logits`), so the covariate's scale is partly absorbed internally. The lab adds no external normalisation: `use_znorm` is not set, so the covariate reaches the model as passed (`timesfm3_forecaster.py:300-305` applies only when the flag is set). How much the internal RevIN removes the scale effect is **[UNVERIFIED]**.
- Multiple covariate rows are supported by the library (`mlx/model.py:199-206`). The lab passes one row per series, and its own docstring says "The forecast wrapper takes one covariate channel per series" (`calendar_vic.py:67-68`). That limit is the lab's choice, not the library's.

**[Verified] Who uses it.** Only `engine/backtest.py`.
- `timesfm_cov`: planned promotions (`backtest.py:55`, `:73-75`).
- `timesfm_cal`: the Victorian calendar (`backtest.py:56`, `:76-79`).
- No live path (`models.py`, `state.py`, `agents/`) passes covariates. `Router` has no covariate method.

**[Verified] Promotion covariate.** `load_promo_covariate` (`engine/data.py:52-76`).
- Reads `truth_events` where `kind='PROMO'` (`data.py:59`). It converts each promotion to `(magnitude-1) × overlap/step` per period (`data.py:70-75`).
- The docstring says "Promotions are planned and known ahead, so they are legitimate future covariates" (`data.py:53-54`).
- **Caveat:** the source is the twin's truth table. CLAUDE.md states that only the twin has `truth_events` and `daily_outcomes`. A real tenant would need a promo calendar that does not exist in the lab. `source/staging.py:134` clears the table but does not populate it from staging. **[Verified, doc only for the staging population.]**
- **No result for the promo covariate is recorded in any doc.** A grep of `docs/` found no `timesfm_cov` outcome.

**[Verified] Calendar covariate.** `calendar_vic.py`.
- Hand-entered public holidays 2022 to 2026 (`calendar_vic.py:17-32`), school-holiday windows (`:36-42`), Easter run-up and Christmas run-up rules (`:49-62`), and the Cup day flag.
- `calendar_channel` sums the flags into one channel per period. It takes the max of the holiday, Christmas, Easter and Cup flags, adds the school flag, then averages over each period (`calendar_vic.py:65-74`, line 73 is the sum). **Observation [Verified by reading]:** the school flag is added on top of the max, so a day that is both a public holiday and in school holidays scores 2, not 1. This is not documented.
- The docstring says the lab result "is an optimistic upper bound" because the twin generates its effects from the same module (`calendar_vic.py:6-10`). The same covariate is shared by every series (`backtest.py:78`, comment).

### 3b. Injected shocks and truth events (twin side)

All of this is in `src/lab/twin/generator.py`, driven by `TwinConfig.shocks` (`generator.py:55-64`, `shocks: bool = True`).

- **Calendar effects by segment:** `CAL_EFFECTS` (`generator.py:40-52`), labelled "SYNTHETIC ASSUMPTIONS, not measured trade data" (`:41-43`). Applied as `prod(1 + (effect-1) × flag)` (`:180-185`). Category-specific holiday effects in `CAT_CAL` (`:204-211`).
- **Promotions:** about 6% of SKUs get one to three windows of 7 to 14 days, demand up 1.4 to 2.4× and price down 12% (`generator.py:187-201`). Each is recorded as a `PROMO` truth row (`:198-200`).
- **Shocks** (only when `shocks=True`, from `scored_from = 55% of days` onward, `generator.py:226-227`):
  - demand surge, ×1.8 to 3.2 for 4 to 8 days (`:234-240`);
  - demand collapse, ×0.15 to 0.4 for 7 to 14 days (`:242-247`);
  - customer lost, multiplier 0 from a day onward, drawn from the top 40 accounts (`:249-254`);
  - customer slowdown, ×0.35 for 14 to 27 days (`:256-260`);
  - supplier delay, extra lead of 3 to 6 days (`:262-267`), applied through `lead_extra` in `supply.py`;
  - supplier cost spike, ×1.10 to 1.28 persisting to the end (`:269-274`), applied through `cost_mult`, capped at 0.85 to 1.45 (`:313`);
  - sell price keyed below floor, ×0.62 to 0.80 for 5 to 19 days (`:276-281`);
  - over-buy of perishables, a forced PO of 3 to 5× normal cover (`:283-289`), consumed by the supply simulator (`supply.py:98-100`, `:173-175`).
- **Keying errors:** on random non-zero days, quantity multiplied by 10 or 100 (`generator.py:301-310`).
- **Demand realisation:** occurrence is Bernoulli on `1-exp(-mean)` with a lognormal quantity (`:292-299`). `expected` is kept as the latent mean (`:299`) and written to `truth_expected` (`build.py:56-61`).
- **Truth labels:** every event gets a sequential `event_id` (`:315-317`) and is written to `truth_events` (`build.py:51-55`). `sim/scorer.py:28` reads them to score agents.
- **Dead config:** `TwinConfig.extra: dict` (`generator.py:64`) is defined but **never read** by `build_world`. There is no scenario-injection hook that a caller can use without editing the generator. **[Verified]**

### 3c. Censoring (twin side)

- `censoring_mask` (`src/lab/twin/censor.py:18-31`). A stockout on day d marks orders on d+1 to d+lag as at risk. Each at-risk order line is dropped with probability `p_drop` (seeded, `censor.py:30`).
- Applied when writing the censored dataset (`src/lab/twin/build.py:15-18`). The censored run is `twin_censored`, with `p_drop=0.4`, `lag_days=2` (`build.py:86`).
- Latent demand stays in `daily_outcomes` and `truth_expected`, so the gap between recorded orders and truth is measurable (`censor.py:8-9`).
- The strength is an assumption, not a measurement (`censor.py:9-10`). `docs/GOAL_EVIDENCE.md:20` records a single seed and no sweep.

### 3d. Replay and supply simulator

- `replay(dataset, start, end, agents, params, policy, ...)` (`src/lab/sim/replay.py:25-26`). It steps days, builds one `ForecastService` per dataset (`replay.py:28`), and runs agents with `AgentContext(w, d, fs, params, lessons)` (`replay.py:48`). The only forecast-related input is the agent `params` dict.
- `supply.simulate(world, policy, policy_b, switch_idx, seed)` (`src/lab/twin/supply.py:82-84`) runs a day-by-day stock simulation. It takes a `Policy` callable (`:32`, `:82`) and can switch policy on a given day (`:120`). This is how the lab replays alternative buying rules on the same demand.
- `bau_policy` is a fixed heuristic buyer with a 4-week mean and safety stock (`supply.py:35-61`). It does not call TimesFM.

### 3e. Forecast-side manipulation that exists now

These are controls on the output of a forecast, not on the model:
- **Quantile sampling:** `distribution.draws` makes common random numbers with a one-factor Gaussian copula, default `rho=0.1`, `seed=11`, `n=1000` (`src/lab/engine/distribution.py:27-48`). `week_samples` interpolates quantiles with tapered tails at 0.6 (`distribution.py:51-68`).
- **Weekday shape:** `weekday_profile` shrinks each product's weekday share toward flat with `k=2` pseudo-weeks (`state.py:172-187`).
- **Daily split:** `daily_path` and `demand_samples` split the weekly forecast by weekday share (`state.py:189-248`).
- **Ensemble:** quantile averaging of TimesFM and ma8 (`models.py:91-105`). Used only in backtests (`backtest.py:80-83`).
- **Oracle controls (scoring only):** `noise_floor` uses the true expected demand (`backtest.py:84-87`, `data.py:79-100`). It is not a forecasting option.

---

## 4. What has been learned experimentally, with references

Sources: `docs/PLAN.md` (decisions and round results), `docs/DESIGN_REVIEW.md` (evidence on the twin), `docs/GOAL_EVIDENCE.md` (audit runs), `docs/GOAL_AUDIT.md`, `docs/RUNBOOK.md`, and `CLAUDE.md`. All numbers are on the synthetic twin unless stated. They are method demonstrations, not client results.

**4.1 Calendar covariate: no gain on the twin [Verified, doc only].**
- `PLAN.md:32`: over 52 weekly origins, WAPE is 21.5% with `timesfm_cal` against 21.0% for plain TimesFM. On the 28 origins whose horizon includes a holiday or run-up, it is 22.2% against 21.6%. Plain TimesFM beats the 8-week moving average on those weeks (21.6% vs 23.1%). Noise floor 18.5%.
- `DESIGN_REVIEW.md:12`: "Calendar covariate ... adds error on the twin. The effect is segment-specific and the history holds only one or two of each event."
- `PLAN.md:33`: the stated cause is that one covariate channel cannot learn per-segment magnitudes. The next candidate, a per-segment uplift layer applied to the forecast, is "Not built yet."
- `PLAN.md:36`: the twin's calendar effects come from the same module, so any gain is an upper bound.
- `PLAN.md:34`: the scored window covers Christmas 2025 only when `n_origins=52`. The 26-origin default misses Christmas.

**4.2 Promotion covariate: no result recorded.** `backtest.py:73-75` implements it. No doc records an outcome. **[UNVERIFIED as a result, not as code.]**

**4.3 The 15% improvement bar: status conflicting.** The bar is defined in `backtest.py:141` (`"improve_min": 0.15`) and `:143`.
- `CLAUDE.md` (Invariants): "The 15% improvement-over-best-baseline bar ... is documented and currently **not met** (7–10% on the twin)."
- `PLAN.md:35`: 7.3% on 52 origins and 9.5% on 26 origins. Both below the bar.
- `docs/GOAL_EVIDENCE.md:47-49` (audit run 20 Nov 2026, same 52 origins, head series, WAPE): twin TimesFM 22.9%, ma8 27.7%, gain **17.4%, "15% bar passed"**. twin_censored gain **13.5%, bar NOT passed**.
- `docs/GOAL_EVIDENCE.md:50` flags the conflict explicitly: "Treat the bar as dataset-dependent until reconciled."
- `docs/GOAL_AUDIT.md:60` says "about 17% only against one baseline on the twin".
- **Verdict: the documents disagree and the conflict is unresolved.** The CLAUDE.md invariant says "not met", and the newest audit says "passed" on the base twin. Any tool that reports "beats baseline" must show which baseline set, which origins and which dataset.

**4.4 Routing.**
- `PLAN.md:66`: steady series pinball loss TimesFM 0.114, TSB 0.130, ma8 0.136. Lumpy series TimesFM 0.341, TSB 0.348. The best free model costs about 14% accuracy on steady series and 2% on lumpy.
- `models.py:181`, `:218`: `fit_routing` keeps the default unless a challenger is at least 2% better (`margin=0.02`) and the class has at least 8 series-origins (`min_series=8`).
- `RUNBOOK.md:18` describes the manual procedure. No live path installs the result (see section 1).

**4.5 Calibration and the median.**
- `DESIGN_REVIEW.md:8`: share of actuals at or below each forecast level, weekly origins June 2025 to September 2026, horizon 4: q10 17.6%, q50 49.6%, q90 89.9%. Only the lower tail runs high.
- `DESIGN_REVIEW.md:9`: for big sellers 42% of actuals fall at or below the median (under-forecast). For small sellers 57% (over-forecast). Categories sit within a few points of 50%.
- `DESIGN_REVIEW.md:10`: a trailing multiplicative bias correction removes bias but worsens WAPE and pinball. Decision: keep the raw distribution.
- `PLAN.md:42`: the median under-forecasts weekly units by 6.7% but has the best WAPE (27.5%). The 60th percentile over-forecasts 2.2% (WAPE 28.1%). Decision: keep the median for expiry maths. **[Verified, doc only.]**
- `DESIGN_REVIEW.md:11`: decisions that add up demand use sampled distributions, not the median.

**4.6 Direct daily TimesFM is not used.** `PLAN.md:43`: for P-0079 direct daily output gives 0.6 to 1.2 units a day, against 3.4 from the weekly view. Daily numbers are derived from weekly forecasts (`state.py:189-217`). **[Verified, doc only.]**

**4.7 Baselines.** Implemented in `engine/baselines.py`: `ma4`, `ma8` (`:18-29`), `seasonal_naive` (`:32-45`), `seasonal_year` (`:48-59`). Backtests compare against `ma4`, `ma8`, `seasonal_year` (`backtest.py:25`). The "best baseline" is chosen on head WAPE (`backtest.py:122-125`). The oracle baseline per series is 5.0% better than TimesFM on the base twin (`GOAL_EVIDENCE.md:48`).

**4.8 Censoring.** `GOAL_EVIDENCE.md:20`: recorded orders over latent demand is 1.000 on the base twin and 0.973 when censored. Median forecast over latent in the top stockout fifth is 0.930 base against 0.882 censored. Share of weeks with median below latent is 0.518 base against 0.553 censored. Single seed, no sweep.

**4.9 Coverage.** 10 to 90 band coverage is 81.2% on the base twin and 80.8% on the censored twin (`GOAL_EVIDENCE.md:48-49`). The ship-bar coverage range is 70 to 90% (`backtest.py:141`, `:144`).

**4.10 Context length, horizon, flip symmetry, z-normalisation, stitching, padding.** No experiments are recorded in any doc or script that I could find. **[UNVERIFIED as untested.]**

---

## 5. Gaps and limits

**Library limits (verified in source)**
1. **No fine-tuning or training API** in the installed 3.0 package (section 2). Adaptation is limited to covariates, context selection and output post-processing.
2. **No runtime change of quantiles.** The nine quantile levels come from `config.json` and are fixed (`mlx/configs.py:37`).
3. **Covariates are conditioning only.** They are not forecast; the model does not reconstruct them (`mlx/model.py:259-266`). Their scale is raw unless `use_znorm` is set.
4. **Backend parity is not established here.** The MLX file has test functions named `test_parity_*` (`SP/timesfm3/mlx/timesfm3_forecaster_test.py:291-360`). I did not run them, so the MLX-torch equivalence is **[UNVERIFIED]**.
5. **Silent flags.** `use_stitching` is parsed but ignored (`mlx/configs.py:39`, `mlx/model.py:279`). `per_core_batch_size` is ignored (`mlx/timesfm3_forecaster.py:63`). `context_length` is a patch size, not a limit (`:127-129`).
6. **Rejected flag.** `use_frozen_running_stats=True` raises `NotImplementedError` on MLX (`mlx/configs.py:59-66`). A caller cannot use that checkpoint setting on MLX.
7. **Hard-wired architecture.** Causal attention and RoPE are fixed (`mlx/transformer.py:115-120`).
8. **Licence.** Weights are non-commercial (`PLAN.md:10`, `CLAUDE.md`). `allow_non_commercial` defaults to True (`models.py:131`).
9. **Horizon.** No maximum is enforced in MLX (`mlx/model.py:143-144`). Accuracy at long horizons is not tested in the repo **[UNVERIFIED]**.

**Lab limits (verified in source)**
10. **Only one covariate channel** is used per series (`calendar_vic.py:67-68`), and covariates are used only in backtests (`backtest.py:73-79`).
11. **Context capped at 512** (`forecast.py:41`, `:56`), against a library cap of 15360.
12. **Global mutable state.** `_ROUTER` is replaced with no lock (`models.py:234-236`). `_WORLDS` (`state.py:40`) and the profile and draw caches (`profile.py:16`, `distribution.py:41`) are plain dicts. Concurrent jobs can race on the router.
13. **Process-wide model lock.** All predictions serialise on `_infer` (`forecast.py:39`, `:45`, `:61`).
14. **Cache fingerprint excludes weights and hyper-parameters** (`models.py:161-173`).
15. **Promo covariate reads the twin's truth table** (`data.py:59`). It has no real-data source in the repo.
16. **Scenario hook is dead code** (`generator.py:64`).
17. **Clipping, not the library flag,** enforces non-negativity (`forecast.py:50-51`, `:67`).
18. **Staging has about 8 weeks** of orders (`PLAN.md:92`). No seasonal or covariate test is possible on real data yet.

---

## 6. Proposed agent-callable tool surface

Design rules, from the project's own guardrails (`CLAUDE.md`, `PLAN.md:106-111`): read-only against client data; no writes to FoodFlow; any change to what produces a forecast goes through a governed, reviewable step; autonomy capped at "prepares" (`CLAUDE.md`, agents section). Every output must carry the synthetic-versus-real label and the `fingerprint`.

| # | Tool | Inputs | Outputs | Safety and notes |
|---|---|---|---|---|
| 1 | `forecast.describe_model` | none | checkpoint id, backend, quantile levels, context cap, licence string, `allow_non_commercial`, `fingerprint()`, `describe()` | Read-only. Exposes `models.py:161-177`. |
| 2 | `forecast.predict` | dataset, product or series id list, as_of, horizon (default 4), `max_context` (1 to 512) | point and 9 quantiles per series and step (`Forecast`), plus `fp` | Read-only. Goes through `Router`. Clips and sorts. |
| 3 | `forecast.predict_with_covariates` | series list, horizon, named covariate rows `{name: values of length context+horizon}`, `max_context` | point and quantiles, plus covariate names used | Enforces the length rule (`mlx/model.py:141-142`). Caller must state which channels are known-future. Only promo and calendar sources are allowed. |
| 4 | `calendar.channel` | dates, step (1 or 7) | per-period calendar value and the flags | Read-only. Wraps `calendar_vic.calendar_channel`. Must carry the "2024 to 2026, approximate" warning. |
| 5 | `forecast.backtest` | dataset, freq, horizon, n_origins, model list | WAPE, pinball, coverage, bias, by-origin, ship-bar verdict with baseline name | Writes a row to the lab DB only (`backtest.py:147-151`). Must report which baseline set and origins were used, because of the 15% conflict (section 4.3). |
| 6 | `forecast.fit_routing` | dataset, models, origins, margin, min_series | proposed route table and per-class pinball table, not installed | Read-only with respect to the live router. Returns a proposal. |
| 7 | `forecast.set_route` | proposed route id or table | confirmation, new fingerprint | **Human approval required.** Changes what every forecast uses. Must refuse when `allow_non_commercial=True` for any client tenant (`models.py:131-138`). Must record the old and new fingerprint. |
| 8 | `distribution.demand_quantiles` | dataset, product, as_of, days ahead | expected, median, q10, q90 for the window, from `daily_demand_quantiles` (`state.py:250-255`) | Read-only. Shows the basis (median or expectation). |
| 9 | `distribution.shortfall_probability` | dataset, product, as_of, days ahead, position | E[shortfall], P(exceeds) (`distribution.py:81-93`) | Read-only. |
| 10 | `twin.censor_sweep` | p_drop list, lag list | recorded-to-latent ratio and forecast bias per setting | Twin only. Refuses staging. Writes to a named twin file only. |
| 11 | `twin.inject_scenario` | kind (surge, collapse, lost customer, cost spike), product or customer id, start, length, magnitude | new twin dataset with the event in `truth_events` | Twin only. **Not possible today:** `TwinConfig.extra` is unused (`generator.py:64`). Needs a real hook. Human approval before any run. |
| 12 | `replay.run` | dataset, window, agent list, params, economics | scorecard with `dollars_at_stake` and `dollar_basis` | Twin or staging. Persists only when `persist=True` (`replay.py:82-110`). Results are labelled synthetic or estimate. |
| 13 | `profile.tenant_profile` | dataset, as_of, weeks | client spend, quantity and margin percentiles (`profile.py:20-46`) | Read-only. Used to set client-relative thresholds. |

Not proposed: anything that changes weights, hyper-parameters or the checkpoint. The library exposes no fine-tuning path (section 5, item 1). Any change would be a new model name under the cache rules (`models.py:166-167`).

---

## 7. Hypotheses (all UNVERIFIED)

- **H1. Calendar covariate fails because one raw channel cannot express per-segment effect sizes.** Source: `PLAN.md:33`. Test: a per-segment uplift layer on the output, scored with the same walk-forward harness.
- **H2. Promotion covariate may help, because promotions are the only known-future event with a large effect in the twin.** Test: `timesfm_cov` on the 52-origin window, with promo windows inside the horizon scored separately. No result exists.
- **H3. Z-normalisation (`use_znorm`) may reduce the big-seller under-forecast.** Section 4.5 shows the bias pattern by volume. Scale is a plausible cause and `use_znorm` is the only library setting that changes it. Untested.
- **H4. Symmetric averaging (`use_symmetric_averaging`) may reduce variance on small series.** It is a flip-symmetry device the lab never tests. Untested.
- **H5. A longer context (up to 15360) may improve steady series.** The lab caps at 512 (`forecast.py:41`). Untested; costs and cache size would rise.
- **H6. The 15% conflict comes from the baseline set and origins, not the model.** `GOAL_EVIDENCE.md:50` proposes twin rebuild or different baseline set. The 7.3% figure uses 52 origins in PLAN and 17.4% in the audit. Reconcile by running both on the same frozen cache.
- **H7. Daily direct TimesFM is poor because the 3.0 model is tuned for longer strides.** Section 4.6 shows the effect. Untested.
- **H8. Concurrent routing swaps could mix cache entries.** The fingerprint should prevent it (`state.py:125`, `:155`). Not tested under threads. The risk is in `_ROUTER` (`models.py:222-236`).
- **H9. The `"edge"` padding mode may matter for future covariates near the horizon edge.** `mlx/timesfm3_forecaster.py:322-326`. Not used.
- **H10. The calendar channel may exceed 1 on overlap days** (section 3a observation). Untested whether this biases the calendar result.
