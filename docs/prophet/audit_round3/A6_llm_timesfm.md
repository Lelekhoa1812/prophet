# A6. How the LLM agents use TimesFM 3, and whether evidence reinforces or revises the narrative

Scope: read-only audit. No model was loaded, no API call was made, no git state changed. Labels: **measured** = read from
the gitignored live-run JSON files (`data/live_gates_*.json`); **code-read** = verified by reading source, not executed;
**web** = fetched or searched during this audit.

## Summary

1. The package is used correctly at the core: full context (up to 15,360 points), median = quantile index 4, quantiles sorted, native past-only and past-future covariates, and the masks used for ignored points. Prophet does not use the 2.x `forecast_with_covariates`/XReg API (it exists only in `timesfm` 2.5, not in the 3.0 MLX backend Prophet imports).
2. The largest defects are in the intermittent-demand path: the delivered point, band and calibration do not match what the backtest scored (A6-01, A6-02), and a flat TimesFM band reaches the delivered pack when the centre is a simple rule (A6-03).
3. Season and clock are not checked against the series: a 5-step weekly cycle on a business-day series was backtested as `season=7` while the health check found period 5 (A6-04). The delivery gate accepts any simple rule as centre, not the best one (A6-05).
4. Narrative is only partly coupled to evidence: the answers cite causes and labels (for example "mean of the last season", "weekly cycle confirmed") that the tools' own output contradicts (A6-06, A6-09).
5. Live evidence is thin: three completed or partial runs, none used covariates, none ran `control_lab` on the flagged series, and no treatment was measured end to end (A6-07, A6-10).

## 1. TimesFM 3 package capabilities versus Prophet usage

Installed: `timesfm` 3.0.2 (`.venv/lib/python3.12/site-packages/timesfm-3.0.2.dist-info`) and its `timesfm3` MLX/torch
backends. Prophet imports `timesfm3.mlx.TimesFM3Forecaster` (`src/prophet_agent/forecast/timesfm.py`).

| Capability | Package (verified, file:line) | Prophet usage | Assessment |
|---|---|---|---|
| Max context | `timesfm3/mlx/timesfm3_forecaster.py:71` (`max_context_length=15360`), `:132-134` (`global_context`), `:201` and `:309-316` (truncation). Torch: `timesfm3/torch/timesfm3_forecaster.py:32`. Not in checkpoint `config.json` (cached copy, no context field). | `forecast_batch` takes `x[-global_context:]`; `forecast_controlled` uses `min(context_points, global_context)`. | Used in full. The 15,360 cap is a code constant, not a trained property. The HF card gives no context length (web). |
| Output patch / stitching | `timesfm3/mlx/configs.py:31-32` (input 32, output 64). `mlx/model.py:179-182`: `extract_len=min(2p,64)=64`, overlap 32, more than one output patch only when horizon > 64. `model.py:277-279` stitch. | Horizon cap 512 (`timesfm_tool.py`). | Stitching starts above 64, not 128 as the skill says (A6-12). |
| Median and quantiles | Nine outputs, 0.1..0.9, median index 4 (`mlx/configs.py:37`, `mlx/timesfm3_forecaster.py:65,236`). Local `config.json` quantiles list matches. HF card: "median at index 4" (web). | Point = `q[:,4]`; band = indices 0 and 8. | Correct. |
| Quantile crossing | Sorting on by default (`timesfm3_forecaster.py:226-231`, `:277`). | `np.sort` in `forecast_batch` and `forecast_controlled`; no crossing count recorded. | Correct but silent: crossing rate unknown (A6-12). |
| Clipping at zero | `make_positive` (`:238-245`), per variate on the input. | Own clip: only when all non-ignored context is non-negative (`timesfm.py`). | Equivalent or better (ignores masked points). |
| Point-ignore masks | `mlx/model.py:117` `target_mask`, `:186-214` masked values excluded from detrend and zeroed; running RevIN stats use masks (`mlx/util.py:83-93`). | `ignore_mask` passed as `target_mask` in `_decode`. | Correct and leak-free in backtests (`Control.at_origin`). |
| Past-only covariates | `mlx/model.py:193-197`; conditioning only, not reconstructed (`:259-266`). Shape: `(v, context)`. | `control.past` stacked, sliced to context. | Correct. Never used in live runs (A6-10). |
| Past-future covariates | `mlx/model.py:199-203`, `:226-237` future slice of length context+horizon. README example (`dist-info/METADATA`, lines ~282-325). | `control.future` must equal history+horizon; checked in `build_control` and `forecast_controlled`. | Correct. Never used in live runs (A6-10). |
| Horizon padding for covariates | `mlx/model.py:238-245`: future covariates padded with masked zeros beyond horizon. `timesfm3_forecaster.py:322-327`: `padding_mode="edge"` repeats the last covariate so "the model sees a covariate for every decoded step" (README). | `_decode` calls `model.decode` directly; no `padding_mode`. | Untested lever (A6-10). |
| Symmetric averaging | `timesfm3_forecaster.py:219-229`. | Re-implemented in `forecast_controlled` (negates target, masks, covariates). Same formula. | Correct. |
| Z-normalisation | `timesfm3_forecaster.py:296-307` (target and covariates). | `forecast_controlled` z-scores target over usable points, then de-normalises. | Correct. |
| Linear detrending, RevIN | `mlx/model.py:288-317`, `mlx/util.py:42-53`. | Not controlled by Prophet (always on). | Fine; not tuned. |
| Frozen running stats | `mlx/configs.py:59-65` raises NotImplementedError if set; checkpoint sets False. | n/a | No issue for the public checkpoint. |
| Flags in 2.5 API | `timesfm/configs.py:42-59`: `force_flip_invariance`, `infer_is_positive`, `fix_quantile_crossing`, `normalize_inputs`. | Not used (2.5 backend not imported). | Not in the 3.0 MLX API; no loss. |
| XReg / `forecast_with_covariates` | Only `timesfm/timesfm_2p5/timesfm_2p5_base.py:198-245` (2.5 torch/flax, needs `return_backcast`, `jax`, `scikit-learn`). Not in `timesfm3`. | Not used. | Correct not to use: different model, different extras. The brief's premise is 2.5-only. |
| Compile and throughput | `mlx/model.py:98-109` `mx.compile` per input shape; `mlx/util.py:83-93` Python loop over patches inside the compiled graph. | Backtests group origins of different lengths. | Unmeasured risk: recompiles per length and long unrolled graphs at 15,360 points. Measure before relying on long contexts. |
| Licence | HF card: "released under the TimesFM Non-Commercial License v1.0" (web). `timesfm` dist METADATA says 3.0 weights are non-commercial. | Documented in CLAUDE.md. | Consistent with CLAUDE.md; no change proposed here. |

Other checks made:
- Health detector: `forecast/health.py` relies on `sandbox/prophet_lib.dominant_period` (`prophet_lib.py:1075-1093`, ACF peaks with no measured false-positive rate). Health thresholds (`SHAPE_FLAT=0.5`, `LEVEL_FLAT=0.30`) are heuristics, not calibrated on TimesFM outputs.
- Full context: yes. Median: yes. Quantile indices: yes. Intermittent handling: not model-consistent (A6-01).

## 2. Live evidence (measured, read-only)

- `data/live_gates_wti_rep4.json` (completed, 20 trading days): backtest verdict `no_clear_difference`, `best_baseline=mean_last_season` (MASE 6.66), TimesFM MASE 7.28, naive 7.27, range coverage 0.729. Delivered `forecast_report centre=naive`. Health `flat_expected` (random-walk test: consistent). `calibrate_intervals` recommended `keep_raw`. The answer's accuracy paragraph was appended by code and is consistent with the stored numbers.
- `data/live_gates_retail_rep4.json` (partial, 14 days): series is business-day indexed (504 points). Backtests and forecast used `season=7`. Health: `period=5`, `strength=0.47`, `shape_ratio=0.14`, verdict `flat_unexpected`. Calls 52 and 53 delivered `centre=naive` with the flat band; the answer quotes the figures and labels them flat_unexpected. No `control_lab` and no `calibrate_intervals` ran.
- `data/live_gates_retail_rep3.json` (partial): same dataset, season 7, health `ok`, backtest `timesfm_better` against `seasonal_naive`. Series preparation differed from rep4, so the verdict is not stable across runs.
- Panels (`data/live_panel_*_rep1.json`): not reviewed in detail. Their structure matches the same tools; no additional TimesFM defects were found in the sampled structure.

## 3. Findings (ranked by impact)

### A6-01. Intermittent demand: delivered point, band and backtest score three different things
- **Severity:** High. **Evidence:** code-read. `tools/timesfm_tool.py:46-58` sets `shown = expected_volume` (TSB mean) and replaces the quantiles with TSB quantiles. `forecast/rolling.py:196-238` scores `timesfm` as TimesFM's median (`cfc.point`) and its coverage from TimesFM's quantiles (`cfc.q`). The verdict, `timesfm_range_coverage_80` and `timesfm_vs_best_baseline` therefore describe a different forecast from the one delivered.
- **Root cause:** the audit-14 fix moved the point and band in the tool but not in the backtest.
- **Proposed fix:** when `is_intermittent`, score the delivered object: in `rolling_backtest`, take the point from TSB expected volume (the same function) and the band from the TSB quantiles, and report both `timesfm_median` and `timesfm_expected` scores. Make the verdict and coverage use the delivered object. Label the backtest `point_basis` in the result.
- **Files:** `src/prophet_agent/forecast/rolling.py`, `src/prophet_agent/tools/timesfm_tool.py`.
- **Regression test:** new `tests/prophet_agent/test_intermittent_consistency.py`: for a synthetic intermittent series, `timesfm_forecast` point lies inside its own band for every step, and `backtest_series` coverage equals coverage of the delivered band recomputed from the same windows.
- **Effort:** M (1 to 2 days). **Risk:** medium (changes verdicts on lumpy series; re-run any stored lumpy backtests).

### A6-02. `calibrate=true` on intermittent demand reinstates the band that audit item 14 removed
- **Severity:** High. **Evidence:** code-read. `tools/report_tool.py:286-287` runs `_calibrated` after `timesfm_run`. `_calibrated` (`:151-171`) replaces `fc["quantiles"]` with `calibrate()`'s TimesFM-based band, around TimesFM's median. `forecast/calibrate.py` has no intermittent branch (grep: none). For lumpy series the point is TSB expected demand, so the calibrated band can exclude the point.
- **Root cause:** `_calibrated` assumes the band is TimesFM's; the lumpy path breaks that assumption and nothing checks it.
- **Proposed fix:** in `_calibrated`, if `fc` has `median_point`, skip calibration and return `interval_calibration: {applied: false, reason: "intermittent demand: band is TSB's; calibration is not defined for it"}`. Add the same guard in `_centre` and `_blend_centre` (they already skip blends for intermittent). Longer term, calibrate the TSB band against the TSB backtest (A6-01).
- **Files:** `src/prophet_agent/tools/report_tool.py`, `src/prophet_agent/forecast/calibrate.py`.
- **Regression test:** `tests/prophet_agent/test_report_intermittent_band.py`: `forecast_report(calibrate=true)` on a lumpy series keeps every point inside its band.
- **Effort:** S (under 1 day). **Risk:** low.

### A6-03. Flat TimesFM band reaches the delivered pack when the centre is a simple rule
- **Severity:** High. **Evidence:** measured. `data/live_gates_retail_rep4.json` evidence [52] and [53]: `centre=naive`, health `flat_unexpected` (period 5, shape ratio 0.14), answer quotes "80% range £810k–£1.57m" built on that band. **Code-read:** `runtime/contracts.py:480-488` `delivered_health` returns `centre_rule:<rule>` for any pack with a rule centre, so `health_repaired` (`:647-654`) and the checklist item (`:551-554`) pass. The band is still TimesFM's and still flat, and `_centre` (`report_tool.py:196-227`) only shifts it.
- **Root cause:** the health gate was written for the centre line, but the band is what the flat shape corrupts (flat median implies a band around the wrong level and width).
- **Proposed fix:** keep the health gate on the band regardless of centre. Make `delivered_health` return the TimesFM health for the band, and have the answer say the band is unrepaired when verdict is `flat_unexpected`. Block finish unless the forecast is repaired (see A6-07) or the answer states the band is not reliable.
- **Files:** `src/prophet_agent/runtime/contracts.py`, `src/prophet_agent/tools/report_tool.py` (note in pack), answer template.
- **Regression test:** `tests/prophet_agent/test_health_gate_rule_centre.py`: a pack with centre `naive` and health `flat_unexpected` fails `health_repaired`.
- **Effort:** S to M. **Risk:** medium (more runs will stall until repaired; budget impact).

### A6-04. Season and clock are not checked against the data; the weekly cycle is mis-specified on a business-day series
- **Severity:** High. **Evidence:** measured. Retail rep4 backtests at `season=7` (evidence [18], [25]) and forecasts at season 7 (forecast calls) on 504 business-day points. Health on the same series reports `period=5` (forecast record, `health.period`). The answer says "weekly (7-day) confirmed". Rep3 on the same dataset passes `season=7` and gets verdict `timesfm_better`, showing that the verdict depends on the prepared series, not only the method. **Code-read:** `tools/timesfm_tool.py:115-134` `_typical_step` takes the median of day gaps; for business days the median is 1 day, so `series_step` reads 1.0 for calendar and business-day series alike. Nothing compares `season` with the health period or with the series' weekday pattern.
- **Root cause:** `season` is an unchecked free parameter; the step is a calendar-day median that cannot express weekday spacing.
- **Proposed fix:** (a) in `rolling_backtest` and `timesfm_forecast`, when `season` is given and `dominant_period` differs from it by more than 10 percent, return a `season_check` warning and run the backtest at both; (b) detect business-day spacing (share of gaps of 3 days on Fridays) and report `series_step` as "business day" with its weekly cycle in steps; (c) `prepare_series` should record the period in steps.
- **Files:** `src/prophet_agent/tools/timesfm_tool.py`, `src/prophet_agent/forecast/rolling.py`, `src/prophet_agent/forecast/health.py`, `src/prophet_agent/sandbox/prophet_lib.py` (prepare report).
- **Regression test:** `tests/prophet_agent/test_season_clock_check.py`: a business-day series with a 5-step cycle produces `season_check` for `season=7`, and `series_step` names business days.
- **Effort:** M. **Risk:** low to medium (new warning fields; some planner prompts need the new field).

### A6-05. The delivery gate accepts any simple rule as the centre, not the best one
- **Severity:** High. **Evidence:** measured. WTI rep4: backtest `best_baseline=mean_last_season` (MASE 6.66), TimesFM 7.28, centre delivered `naive` (MASE 7.27). The answer calls naive "the appropriate benchmark", which the backtest does not support. **Code-read:** `runtime/contracts.py:560-566` only fires for verdict `baseline_better` (WTI is `no_clear_difference`, so it did not fire). `_baseline_gate` (`:596-608`) blocks only when the centre is TimesFM's; any rule passes.
- **Root cause:** the rule checks presence of a rule, not which rule the backtest preferred.
- **Proposed fix:** in `_baseline_gate`, when the centre is a rule, require it equals `best_baseline` (or the blend the backtest adopted) unless the answer says why. Also fire the delivery gate when `best_baseline` beats TimesFM by any margin, not only on `baseline_better`, and show the margin in the answer.
- **Files:** `src/prophet_agent/runtime/contracts.py`, prompt `planner.v22.md` (section "Which centre line to present": make the rule explicit: the best MASE, not the verdict).
- **Regression test:** `tests/prophet_agent/test_centre_gate.py`: a backtest with `best_baseline=mean_last_season` and centre `naive` fails `baseline_compared`.
- **Effort:** S. **Risk:** low.

### A6-06. "mean_last_season" with no season is a 4-point mean, and the reports describe it as the last season
- **Severity:** Medium. **Evidence:** code-read and measured. `forecast/rolling.py:57-58`: `_mean_last` returns `x[-max(m,4):].mean()`. With `season=1` (WTI), the best baseline is the mean of the last 4 prices. `tools/report_tool.py` `RULE_TEXT` and the answer say "the average of the last season". WTI answer: "best simple baseline (mean_last_season) MASE 6.66" with that label.
- **Root cause:** the `max(m, 4)` floor was added for stability and the name was not changed.
- **Proposed fix:** rename the rule to `mean_last_k` with k = max(season, 4) reported in the result, and make the text say "the mean of the last k points"; keep `mean_last_season` as an alias only when season ≥ 2.
- **Files:** `src/prophet_agent/forecast/rolling.py`, `src/prophet_agent/tools/report_tool.py`, `src/prophet_agent/runtime/contracts.py` (`BASELINE_NAMES`).
- **Regression test:** `tests/prophet_agent/test_baseline_labels.py`: with season 1 the result states k=4 and the text does not say "season".
- **Effort:** S. **Risk:** low (labels change in stored results; old runs keep their names).

### A6-07. A flagged forecast is not repaired: the health `try` text omits the detected period, and no lab was run on it
- **Severity:** Medium. **Evidence:** measured and code-read. Retail rep4 health (period 5, verdict `flat_unexpected`): the cause text names the period (`forecast/health.py:155-156`), but the fix text does not: "check how the series was prepared ... run backtest_series to see whether a seasonal baseline beats it" (`health.py:157-158`, branch starts at `:153`). The fix never says which period to pass. No `control_lab` call exists in that run, and no call passes `period=5`. `control_lab` accepts `period` (`tools/lab_tool.py:52-53`).
- **Root cause:** the advice is generic; the detected period is in the record but not in the instruction.
- **Proposed fix:** in `assess`, when `period` is set and the verdict is `flat_unexpected`, the `try` text names it: "run control_lab with period=<p> and backtest_series with season=<p> on the same series". Add a contracts item: `flat_unexpected` requires a `control_lab` (with the period) before finish.
- **Files:** `src/prophet_agent/forecast/health.py`, `src/prophet_agent/runtime/contracts.py`.
- **Regression test:** `tests/prophet_agent/test_health_try_names_period.py`: a 5-cycle history with a flat forecast yields a `try` containing `period=5`.
- **Effort:** S. **Risk:** low.

### A6-08. Configured backtests are selected on the windows they then score, with no multiplicity control, and a no-op context counts as a treatment
- **Severity:** Medium. **Evidence:** code-read. `backtest_series` with `ignore_ranges`, `context_points`, `znorm`, `symmetric_averaging` (`forecast/rolling.py:183-186`) scores the configuration on the same origins where the planner chose it. `control_lab` uses Holm correction (`forecast/lab.py` `_holm`), but direct backtests do not. `check_controls` (`runtime/contracts.py:436-444`) passes on any gain above zero and warns otherwise. Retail rep4 used `context_points=504` on a 504-point history: `configured_vs_default_pct=0.0`, so the treatment was a no-op but still counted as tested.
- **Root cause:** explicit controls skip the lab's selection safeguards, and no-op values are not detected.
- **Proposed fix:** (a) when `context_points >= len(history)` or `ignore_ranges` is empty, reject with "no-op: the default already uses all points"; (b) add `selection: {optimistic: true, configurations_tried: n}` to configured backtests and a warning when more than three configurations are tried; (c) require a `control_lab` run for `ignore_ranges` (lab currently cannot take explicit ranges, per the comment at `contracts.py:434-435`): add a `ranges` candidate to the lab.
- **Files:** `src/prophet_agent/forecast/rolling.py`, `src/prophet_agent/runtime/contracts.py`, `src/prophet_agent/forecast/lab.py`.
- **Regression test:** `tests/prophet_agent/test_controls_noop.py`: `context_points` equal to history length is rejected or flagged as no-op.
- **Effort:** M. **Risk:** low to medium.

### A6-09. Narrative claims are not checked against the tools' own output
- **Severity:** Medium. **Evidence:** measured. WTI answer: "five level shifts ... correspond to known oil-market regimes"; the scan requires a cause or a test for each (planner.v22 step 5), and the run records no test, only causes from memory and a partial web search (which failed for the IEA and OPEC sources, per the evidence). Retail answer: "weekly (7-day) confirmed" while the health record shows period 5. WTI answer: "naive is the appropriate benchmark" against a backtest where `mean_last_season` is better (A6-05).
- **Root cause:** the answer is generated by the model; the deterministic accuracy paragraph is appended, but claims in the body are not checked.
- **Proposed fix:** add an answer check in `contracts.py`: every named period, season, level-shift count and "best benchmark" in the answer must equal a stored value (health period, scan count, backtest `best_baseline`), and every causal attribution must cite a ledger entry with a test. Fail the finish with the mismatched item.
- **Files:** `src/prophet_agent/runtime/contracts.py` (answer grounding), `src/prophet_agent/prompts/planner.v22.md` (section F, item 31).
- **Regression test:** `tests/prophet_agent/test_answer_season_grounding.py`: an answer stating "7-day cycle" when health says period 5 fails the grounding check.
- **Effort:** M. **Risk:** medium (more finish refusals; needs tuning on the live runs).

### A6-10. Covariates are never used or measured in live runs, and the horizon padding lever is untested
- **Severity:** Medium (value left on the table; impact not measured). **Evidence:** measured (all three runs: `future_covariates=null`, `past_covariates=null`) and code-read. `mlx/model.py:238-245` pads future covariates with masked zeros beyond the horizon; `padding_mode="edge"` (`timesfm3_forecaster.py:322-327`) is the package's answer to "the model sees a covariate for every decoded step". `forecast/timesfm.py` `_decode` never sets it, and `control_lab`'s `with_covariates` candidate is the only covariate test, and it runs only when a driver is supplied.
- **Root cause:** the padding mode was not considered when covariates were added (the 3.0 path is `_decode`, not `predict_batch`).
- **Proposed fix:** (a) add `padding_mode` to `Control` (default `none`) and a lab candidate `covariates_edge_pad`; (b) in the retail and WTI cases, the planner should build at least one known-ahead driver (the Christmas closure, or the public holiday calendar) and measure it in `control_lab`. Report the measured gain or say it was not tested.
- **Files:** `src/prophet_agent/forecast/timesfm.py` (`_decode`), `src/prophet_agent/forecast/control.py`, `src/prophet_agent/forecast/lab.py`.
- **Regression test:** `tests/prophet_agent/test_covariate_padding.py`: with a known-ahead covariate and horizon 20, the decoded forecast with `edge` padding equals the forecast from a history whose covariate is extended by its last value (shape and sign checked, not a numeric target).
- **Effort:** M. **Risk:** low (opt-in lever).

### A6-11. `timesfm_forecast` silently accepts `season` and `centre`
- **Severity:** Low. **Evidence:** code-read. `tools/timesfm_tool.py:25-27` puts `season` and `centre` in `ALLOWED_ARGS` with the comment "accepted and ignored". Whether the planner passed `season` to `timesfm_forecast` in the live runs was not re-checked in the final pass; if it did, the record gives no sign it was ignored, so the planner could reasonably believe it applied.
- **Root cause:** accepted for call compatibility, with no echo.
- **Proposed fix:** echo `ignored_arguments: ["season"]` in `config` and in the observation; or refuse them when the tool is called directly and only accept them through `forecast_report`.
- **Files:** `src/prophet_agent/tools/timesfm_tool.py`, `src/prophet_agent/tools/calibrate_tool.py` (same pattern).
- **Regression test:** `tests/prophet_agent/test_ignored_args_echo.py`.
- **Effort:** S. **Risk:** low.

### A6-12. Skill and prompt text is out of step with the code and the package
- **Severity:** Low. **Evidence:** code-read and web. (a) `skills/timesfm-control-and-flat-forecasts.md` item 3 says stitching starts "above 128 steps"; the code stitches above 64 (`mlx/model.py:179-182`, extract_len 64, overlap 32). (b) Item 6 cites "an open upstream issue reports flat repeated outputs on one hardware setup". Web: issue #328 concerns TimesFM 2.5 torch on H100, open, no fix, no maintainer confirmation; it does not concern the 3.0 MLX backend. (c) Item 4 says to "sort or flag" quantile crossing; the code sorts silently and counts nothing. (d) `planner.v22.md` says "TimesFM reads up to 15,360 points" as a model property; it is a code constant (A6-08 context).
- **Proposed fix:** correct item 3 to "above 64 steps (one output patch)"; reword item 6 to name the 2.5 issue and say the 3.0 MLX path has no recorded report; record `quantile_crossings` in the config; state the 15,360 cap as Prophet's cap.
- **Files:** `src/prophet_agent/skills/timesfm-control-and-flat-forecasts.md`, `src/prophet_agent/forecast/timesfm.py`, `src/prophet_agent/prompts/planner.v22.md`.
- **Regression test:** `tests/prophet_agent/test_quantile_crossing_count.py` (crossings counted on a synthetic crossing input).
- **Effort:** S. **Risk:** none.

## 4. What was not verified

- Final citation check (grep and sed on the source, this pass) confirmed: `timesfm3` `model.py:111, 179-180, 278`; `timesfm3_forecaster.py:71, 132-134, 148, 189-190, 201, 226-231, 238, 276, 318-322`; `rolling.py:57, 159, 197-198`; `contracts.py:480-486, 560, 585, 647-654`; `timesfm_tool.py:27, 40-58`; `report_tool.py:151, 196, 230, 270, 280-287`; `health.py:23, 149-158`. Not re-checked in this pass: `timesfm3/mlx/configs.py` line numbers (cited as 31-32, 37, 59-65), `model.py:98-110, 186-214, 238-245, 288-317` and `mlx/util.py` line numbers, `contracts.py:403-444`, `planner.v22.md` section numbers, and `rolling.py:183-186, 196-238`. Treat those citations as approximate until checked.
- A6-07 correction: the health cause text does name the period; only the fix text omits it.
- The model was not loaded, so none of the "code-read" defects has been reproduced numerically. A6-01 and A6-02 should be confirmed with a read-only snippet that calls only `tools/timesfm_tool.py` helpers on a synthetic intermittent series, once a time slot allows a model-free check.
- Whether TimesFM's flat retail forecast is a model failure or a health false alarm is not established. The evidence contains no `control_lab` with `period=5`.
- MLX compile cost at 15,360 points and per-length recompilation are unmeasured (A6 risk note in section 1).
- Training context length of the checkpoint is not stated on the HF card (fetched via a summary tool; the card's model details list patch sizes only), so the 15,360 cap is not confirmed as a trained limit.
