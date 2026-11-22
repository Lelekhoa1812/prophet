# A2: advanced maths, statistics, DS and ML/DL correctness (round 3)

Read-only audit. Nothing was committed, no API on port 8010 was called, and the test suite was not run. Scratch scripts sit in the session scratchpad (`a2_checks.py`, `a2_boot2.py`). Labels: **measured** = run by me on synthetic input or read from a stored result JSON; **inferred** = read from code or formula and not run; **quoted** = taken from the earlier report and not re-checked here.

## Summary

1. The backtest and calibration choose the season on the whole series, so test windows leak into the MASE scale and the seasonal baseline (A2-01, measured).
2. The quantile GBM band is uncalibrated: its nominal 80% band covered 37% on the 12-station Beijing run and 74% on electricity (A2-02, measured from stored JSON).
3. The paired bootstrap that gates "timesfm_better" has a null false-positive rate of about 0.09 at 6 origins against a nominal 0.05, and `panel_backtest` names a winner with no test at all (A2-06, A2-07, measured).
4. Calibration and dispersion maths are wrong in two places: the fixed cross-period copula (rho = 0.1) gives a 7-day total only 0.45 times the random-walk spread, and changepoints() reports annual seasonal steps as regime changes (A2-04, A2-05, measured).
5. The scoring is proper only in a truncated form: no WIS or CRPS, MAE is used to score mean forecasts, and MinT reconciliation is built but not wired to any tool (A2-08, A2-09, A2-10).

## Findings, ranked by impact

### A2-01: Season chosen on the full series leaks test windows into the backtest (major)

- **Evidence**: `src/prophet_agent/forecast/rolling.py:167` calls `choose_season(x, freq_hint, dates)` on the whole series `x`, and `rolling.py:65` (`pick_season`) does the same when no season is given. `src/prophet_agent/forecast/calibrate.py:237` does `pick_season(x, None)` on the full series too.
- **Counter-example (measured)**: a random-walk history of 200 points, then 56 points with a weekly pattern. History only gives season 1. The full series gives season 7 with autocorrelation 0.34 and reason "daily data: the one-step changes repeat every 7 periods".
- **Root cause**: model selection (which lag is the MASE denominator and which seasonal naive is scored) reads the windows it is then scored on. The season changes every origin's scale and the seasonal baseline, so the verdict is optimistic or pessimistic depending on the window.
- **Proposed fix**: choose the season from `x[:origins[0]]` (the first origin's history) in `rolling_backtest` and `calibrate`. Signature: `pick_season(x_hist: np.ndarray, given: int | None) -> int`, called with the first-origin history.
- **Files touched**: `forecast/rolling.py`, `forecast/calibrate.py`.
- **Regression test**: `test_season_uses_first_origin_only`. Build the counter-example series, run `rolling_backtest` with and without the appended weekly pattern, and assert the reported `season_used` is 1 in both cases. Fails before (gives 7 for the full series), passes after.
- **Effort**: S. **Risk**: low. Changes reported seasons on some series; say so in the result text.

### A2-02: Global GBM 10 to 90 band is uncalibrated and under-covers (major)

- **Evidence**: `forecast/global_gbm.py:130` fits `HistGradientBoostingRegressor(loss="quantile", max_iter=200, early_stopping=False)` with default regularisation. `global_gbm.py:175-180` returns the quantile outputs with no calibration step. Stored results (measured, from `data/live_panel_beijing_rep1.json` and `data/live_panel_electricity_rep1.json`): `coverage_80.global_gbm` = **0.37** (12 stations, 14-day horizon, 6 origins), **0.587** (3 entities, same settings), **0.740** (electricity, 60 entities, 7-day horizon). TimesFM on the same runs: 0.797 and 0.757.
- **Root cause**: the three quantile models are fitted on overlapping training rows (one row per origin and horizon step) with no out-of-time validation, so the bands are too narrow. The panel tool reports `coverage_80` but never acts on it.
- **Proposed fix**: split-conformal (or CQR, Romano, Patterson and Candès 2019, arXiv 1905.03222) on out-of-time origin scores per entity, reusing the score form in `calibrate._ratios` and `fit_scales`. Signature: `global_forecast(panel, horizon, season=None, quantiles=..., seed=0, calibrate_origins: int = 6) -> dict` returning `scale` per horizon third in addition to the bands.
- **Files touched**: `forecast/global_gbm.py`, `tools/panel_tool.py` (report the calibrated coverage).
- **Regression test**: `test_gbm_band_calibrated_on_noisy_panel`. Synthetic panel of 20 entities with heavy-tailed noise, then assert held-out 80% coverage within ±0.05 after calibration. Fails before (observed under 0.6 on this construction, inferred from the stored results), passes after.
- **Effort**: M. **Risk**: short panels leave too few origins; the calibrated band then falls back to raw with a stated reason.

### A2-03: Two-sided imputation leaks future values into the history (major)

- **Evidence**: `sandbox/prophet_stats.py:320` (`_candidate`, linear) uses `s.interpolate(method="linear", limit_area="inside")` when `causal=False`. `impute()` defaults to `causal=False` (`prophet_stats.py:340`). The Beijing answer text says gaps were "filled by linear interpolation" before the panel run; the order in the run is not confirmed in the stored log.
- **Counter-example (measured)**: series of 80 points with a gap at 50 to 55. Editing only index 56 changes the filled values at 50 and 51 that a backtest origin at 52 would read. Values before the edit: 35.459 and 35.551. After: 41.173 and 46.979. The maximum change inside `x[:52]` is 11.43.
- **Root cause**: imputation is run on the full series, before the origins are cut.
- **Proposed fix**: in any backtest, impute per origin with `causal=True` on `x[:origin]`, or refuse two-sided imputation in a backtest. Signature: `impute(x, method, period=None, max_gap=None, causal=True)` as the default for backtest callers, with `causal=False` allowed only for description.
- **Files touched**: `sandbox/prophet_stats.py`, the sandbox tool that runs the backtest.
- **Regression test**: `test_backtest_imputation_is_causal`. Edit index 56, rebuild the origin-52 history through the backtest path, and assert the history is byte-identical. Fails before, passes after.
- **Effort**: S. **Risk**: low for forecasts, since the fill differs only at gap positions near origins.

### A2-04: Fixed cross-period copula understates the spread of totals (major)

- **Evidence**: `forecast/distribution.py:36` default `rho=0.1`, with the docstring at lines 8-9 saying the correlation of errors between periods is "about 0.09" measured on the synthetic twin. That constant is applied to every series.
- **Counter-example (measured)**: a random-walk centre with 7 daily steps. The true sd of the 7-day total error is **11.85** (theory sqrt(140) = 11.83). `Draws.make(rho=0.1)` with per-period normal marginals gives an sd of **5.30**, which is **0.45** times the truth. For a random walk the implied correlation between periods 1 and 7 is 0.378, and between 4 and 5 is 0.894, not 0.1.
- **Root cause**: cumulative errors in price-like or random-walk-like series are strongly correlated across horizons. A constant correlation inherited from a different data-generating process is wrong for them.
- **Proposed fix**: replace the single-rho copula with a correlation matrix `C` estimated from standardised backtest errors across origins (fallback to the random-walk structure `C_hh' = sqrt(min(h,h')/max(h,h'))` when there are fewer than about 20 origins). Signature: `Draws.make_corr(n: int, corr: np.ndarray, seed: int) -> Draws` using a Cholesky factor.
- **Files touched**: `forecast/distribution.py`, and the decision tools that call `draws`.
- **Regression test**: `test_total_spread_matches_random_walk`. Simulate a random walk, pass its implied correlation to `make_corr`, and assert the 7-day total sd is within 5% of 11.83. Fails before (0.45 ratio), passes after.
- **Effort**: M. **Risk**: the estimated correlation is noisy with few origins; the random-walk fallback keeps the default safe.

### A2-05: Changepoint search reports seasonal steps as regime changes (major)

- **Evidence**: `sandbox/prophet_stats.py:538-648`. `changepoints()` searches the raw series with default `min_size=20` and penalty `3*log(n)*var` (line 579). It has no seasonal adjustment and no autocorrelation correction. Its own report says "diagnostic_only", but the Beijing answer presents the detections as "real regime changes".
- **Counter-example (measured)**: four years of daily values with a winter step (+0.8) every year and no regime change. White noise gives **5 changepoints** at indices 80, 320, 445, 685, 810. AR(1) with phi 0.7 gives 5 as well. Without seasonality, AR(1) noise gives 0.
- **Beijing check (measured, on the stored CSV `data/datasets/815e21c808444717b58f27087a9a6655/beijing_pm25_daily_12_stations.csv`, gaps linearly interpolated, defaults)**: Dongsi, Tiantan and Wanshouxigong each show a cluster 2016-11 to 2017-01 (for example Tiantan 2016-11-25, 2016-12-17, 2017-01-08 with after/before ratios 1.42, 1.89, 0.34). The segment 2016-11-25 to 2017-01-08 lasts about 44 days, which is a winter episode, not a regime. Huairou, Nongzhanguan and Shunyi produce **0** changes, so the reported 1 January 2014, 2015 and 2016 shifts for those stations do not reproduce with these defaults. The reported ratios (×2.05, ×0.44) also differ from the detector's (1.42, 0.34). I did not find the exact settings the run used, so this is a mismatch to explain, not proof of error.
- **Root cause**: the search does not remove the seasonal profile and does not require segments to be longer than a seasonal cycle, although the docstring says short changes are transients.
- **Proposed fix**: remove the phase median (as `lab.robust_flags` does) before the search, require `min_size >= period` when a period is known, and set the penalty from the residual variance with an autocorrelation correction (Killick, Fearnhead and Eckley 2012, PELT, JASA 107(500), not re-checked online). Signature: `changepoints(x, min_size=20, penalty=None, max_changes=5, period=None)`.
- **Files touched**: `sandbox/prophet_stats.py`, the Beijing-type analysis prompts or skill snippets that cite level shifts.
- **Regression test**: `test_changepoints_ignore_annual_step`. The synthetic annual-step series above must return 0 changes after the fix, and a true level shift of 3 sd must still be found. The first assertion fails before, both pass after.
- **Effort**: S to M. **Risk**: fewer detections on genuinely seasonal data, which is the intent.

### A2-06: Paired bootstrap is anti-conservative at small origin counts (major)

- **Evidence**: `forecast/paired.py:30-35` resamples the paired differences i.i.d. and reads the percentile interval. `forecast/rolling.py:73-76` uses it for the verdict `timesfm_better` (requires `lo > 0` and at least 60% wins).
- **Counter-example (measured, no true difference)**: the verdict rate under the null is **0.089** at 6 origins, **0.086** at 8 and **0.068** at 12, against a nominal one-sided 0.05. An exact sign-flip test over the same 6 origins gives **0.046** (exact level 3/64 = 0.047).
- **Root cause**: percentile bootstrap of the mean of 6 to 12 values undercovers; the verdict is a one-sided test at 5% in effect.
- **Proposed fix**: for k <= 12 origins use the exact sign-flip test (2^k sign patterns, one-sided p), and for longer origin sets a stationary bootstrap with mean block length tied to the horizon (Politis and Romano 1994, JASA 89(428), DOI 10.1080/01621459.1994.10476870). Signature: `paired_gain_test(base, cand, *, method="signflip" | "stationary", seed, resamples, block_len=None) -> dict` returning `p_one_sided`, `gain`, `lo`, `hi`.
- **Files touched**: `forecast/paired.py`, `forecast/rolling.py` (`_verdict`), `forecast/combine.py` (`_beats`), `forecast/lab.py` (`_decide`).
- **Regression test**: `test_signflip_null_rate`. Simulate 2000 null comparisons at k = 6 and assert the one-sided rejection rate is at most 0.06. The bootstrap version fails at 0.089 before, the exact test passes after.
- **Effort**: S. **Risk**: the exact test has low power at 6 origins (smallest one-sided p is 1/64), so some real wins become `no_clear_difference`. That is the correct direction for a gate.

### A2-07: `panel_backtest` names a winner with no statistical test (major)

- **Evidence**: `forecast/global_gbm.py:286-287` takes `best_overall = min(models, key=overall)` and reports it with no interval, while the single-series `rolling.py` gives a paired bootstrap verdict. The electricity result (measured, `data/live_panel_electricity_rep1.json`) reports `best_overall: "blend"` with overall MASE blend 1.6967, TimesFM 1.7033, seasonal naive 1.7227 and GBM 1.8478, and wins TimesFM 23 of 60, seasonal naive 17, blend 15.
- **Evidence gap (measured)**: the stored `per_entity` list has 30 rows, not the 60 entities the summary covers. Recomputed from those 30 rows, the mean of the 30 rows is TimesFM 1.7209, not the reported 1.7033, so the stored evidence cannot reproduce the summary.
- **Paired bootstrap on the 30 stored rows (measured)**: TimesFM minus blend = +0.052, 90% interval [-0.011, +0.126], TimesFM lower on 16 of 30 rows. The interval includes zero, so the lead of blend over TimesFM is not established on this evidence.
- **Root cause**: no paired test between the reported best and the alternatives; ranking on means alone.
- **Proposed fix**: report `best_overall_test` for the best model against each alternative, using A2-06's paired test over entity-level MASE, and use the rolling verdict vocabulary (`timesfm_better`, `baseline_better`, `no_clear_difference`). Also store every entity's row. Signature: `panel_backtest(..., ) -> dict` gains `"best_overall_test": {...}` and `"per_entity"` must list all entities.
- **Files touched**: `forecast/global_gbm.py`, `tools/panel_tool.py`.
- **Regression test**: `test_panel_winner_requires_test`. Two entities with MASE differing by 0.5% must yield `no_clear_difference`, not a named winner. Fails before, passes after.
- **Effort**: S. **Risk**: panel answers become more hedged; that is intended.

### A2-08: MAE scores mean-type forecasts, so the median wins on volume (major)

- **Evidence**: `forecast/rolling.py:198` builds `tsb_expected` from `expected_volume` (`forecast/tsb.py:21-33`), which returns a mean. `rolling.py:232-233` scores every forecast with MAE/MASE, which is minimised by the median (Gneiting 2011, JASA 106(494), from memory, not re-checked online). `tsb_expected` is also in `SIMPLE` (`rolling.py:38`), so the "best baseline" is chosen with the same bias.
- **Counter-example (measured)**: lumpy demand, occurrence p = 0.2, size 10. The median forecast 0 has MAE **2.017** with volume bias -100%. The expected-volume forecast 2.0 has MAE **3.210**, bias -0.8%, and MSE 16.10 against 20.17 for the median.
- **Root cause**: the loss used for selection does not match the functional being forecast.
- **Proposed fix**: score each forecast with the loss for its functional: MAE/MASE for quantile or median forecasts, and MSE/RMSSE plus volume bias for mean forecasts. Signature: `_score(forecast_kind: Literal["median","mean"], err) -> float`, with `FUNCTIONAL = {"timesfm": "median", "tsb_expected": "mean", ...}`.
- **Files touched**: `forecast/rolling.py`, `forecast/tsb.py`.
- **Regression test**: `test_mean_forecast_scored_with_mean_loss`. The lumpy counter-example must rank the expected-volume forecast above the median on MSE and volume bias. Fails before (MAE ranking), passes after.
- **Effort**: S. **Risk**: the baseline ranking changes on lumpy series; expected and reported.

  Note (inferred, not re-checked): the code calls this "TSB", but the size is a 52-point rolling mean of non-zero sizes, not TSB's exponentially smoothed size with its own beta (Teunter, Syntetos and Babai 2011, EJOR 214(3)). Rename it or implement the smoothing.

### A2-09: No proper interval score is used; scoring is a truncated pinball (major)

- **Evidence**: the only proper rule is the mean of the nine-level pinball at `forecast/metrics.py:19-25`, which is a discrete approximation of CRPS over the central 80% of levels. It is used at `rolling.py:240` and `lab.py:401`. Nothing computes CRPS, WIS or the energy score (`grep` over `forecast/` and the sandbox statistics modules found none).
- **Gap**: the nine quantiles already contain four central intervals, so the weighted interval score is available at no cost (Bracher, Ray, Gneiting and Reich 2021, arXiv 2005.12881, PLOS Computational Biology 17(2) e1008618, verified). WIS is proper and approximates CRPS.
- **Proposed fix**: add `wis(actual, q)` with alpha in {0.2, 0.4, 0.6, 0.8}, intervals (q[0], q[8]), (q[1], q[7]), (q[2], q[6]), (q[3], q[5]), and median q[4]. Formula: WIS = (1/(K + 0.5)) * (0.5·|y − m| + Σ_k (α_k/2)·IS_{α_k}), with IS_α = (u − l) + (2/α)(l − y)·1{y<l} + (2/α)(y − u)·1{y>u}, and K = 4. Signature: `wis(actual: np.ndarray, q: np.ndarray) -> float` in `forecast/metrics.py`.
- **Files touched**: `forecast/metrics.py`, `forecast/rolling.py`, `forecast/lab.py`.
- **Regression test**: `test_wis_matches_closed_form`. A perfect median with zero spread at actual y gives WIS = 0.5·|y−m|/4.5 (K = 4). Fails before (function missing), passes after.
- **Effort**: S. **Risk**: none to existing verdicts unless WIS is wired into `_decide`.

### A2-10: MinT reconciliation is built but not wired to any tool (major)

- **Evidence**: `forecast/reconcile.py` implements MinT with Schäfer-Strimmer shrinkage. I checked the algebra: the whitener satisfies Q'Q = W^-1, P solves `(QS)P = Q` by least squares, the shrinkage intensity uses the unbiased variance of the sample correlations, and every output is S·P·base, so it is coherent by construction. `grep` shows no caller in `src/prophet_agent/tools/` or `runtime/`. The reference is Schäfer and Strimmer (2005), SAGMB 4(1), DOI 10.2202/1544-6115.1175 (verified). The formulas were not compared with the paper's text.
- **Quoted (not re-checked)**: MinT reduced RMSE by about 5.2% against bottom-up's 5.3% on correlated ETS series, while bottom-up raised RMSE by 90.4% on mixed lengths (Steinmeister and Pauly, arXiv 2409.18550, via the earlier report). This is why the choice must be backtested.
- **Proposed fix**: add a `reconcile_hierarchy` tool that takes a summing matrix S and the base forecasts from `panel_backtest`, uses in-sample one-step residuals for `mint_shrink`, and reports the chosen method against bottom-up on the same origins. Signature: `reconcile(S, base, residuals=None, method=...)` already exists; the new piece is the tool wrapper and a backtest comparison.
- **Files touched**: new tool module under `src/prophet_agent/tools/`, registration in `tools/__init__.py`.
- **Regression test**: `test_reconciled_output_is_coherent`. For a two-level hierarchy, assert `reconciled == bottom @ S.T` and that a coherent base returns itself. Fails before (no tool), passes after.
- **Effort**: M. **Risk**: few origins make the residual covariance noisy; the code already falls back to WLS-structural below 3 rows.

### A2-11: Season detector has no multiplicity control and false-positives on short series (minor)

- **Evidence**: `forecast/season.py:23-24` uses a fixed threshold 0.3 for every n, and the no-hint path takes the maximum over many lags with no correction (`season.py:95-109`).
- **Measured**: white-noise changes, no hint: false season rate **0.080** at n = 60, **0.020** at n = 120, **0.000** at n = 240 (150 trials each). With a daily hint, 0.000 at all three sizes.
- **Proposed fix**: threshold = z(1 − α/(2K)) / sqrt(n) with K the number of candidate lags and α = 0.05, floored at 0.3. Signature: `choose_season(values, freq_hint=None, dates=None, alpha=0.05)`.
- **Files touched**: `forecast/season.py`.
- **Regression test**: `test_season_false_positive_rate_small`. 300 white-noise series at n = 60 must give a non-1 season at most 0.05 of the time. Fails before (about 0.08), passes after.
- **Effort**: S. **Risk**: real short-series seasons are missed more often.

### A2-12: Window-level Wilson and pooled conformal ranks overstate effective sample size (minor, inferred)

- **Evidence**: `forecast/calibrate.py:208-219` (`_wilson_share`) takes the point-level coverage share `p` and the window count `n` and treats them as a binomial. `AUDIT_2026-11-22.md` item 16 records this as fixed (windows as the sample size); the call I read still passes the point-level share, so confirm the fix in `calibrate.py` before relying on this finding. A share averaged over windows is not a binomial proportion over independent windows, so the interval is heuristic. `fit_scales` (`calibrate.py:65-74`) ranks all points of a horizon third as if independent, although points inside a window share a series and origin.
- **Labelling**: `calibrate.py:23-29` and `aci_scales` (`calibrate.py:154-198`) describe the method as DtACI, but the code picks the expert with the lowest recent pinball loss (argmin) and does not run the weighted mixture with fixed share of Gibbs and Candès (2022, arXiv 2208.08401). The label is wrong.
- **Proposed fix**: report the Wilson or Clopper-Pearson interval on the per-window hit counts (windows as Bernoulli units), or a t-interval on per-window coverage fractions with k − 1 degrees of freedom. Relabel `aci_scales` as "expert-selection ACI" unless the mixture is implemented.
- **Files touched**: `forecast/calibrate.py`, `tools/calibrate_tool.py` (text).
- **Regression test**: `test_wilson_uses_window_hits`. For 3 hits out of 4 windows the interval must equal the Wilson interval for 3/4, not for the point share. Fails before, passes after.
- **Effort**: S. **Risk**: wider reported intervals at small k, which is the honest direction.

### A2-13: Seasonal and moving-average bands widen with sqrt(h) (minor, inferred)

- **Evidence**: `forecast/baselines.py:21` widens every baseline band as `Z·sd·sqrt(h)`. That is correct for a random walk (moving average) but not for seasonal naive (`baselines.py:52`), whose h-step error repeats one season and does not accumulate.
- **Proposed fix**: for seasonal naive, use the residual sd of the lag-season difference per phase, with no sqrt(h) growth past one season. Signature: `seasonal_naive(series, h, season, k=4) -> (point, band)` with `band_mode="phase"`.
- **Files touched**: `forecast/baselines.py`.
- **Regression test**: `test_seasonal_band_constant_past_one_season`. On an exactly periodic series with noise, the band width at horizons 1 and 3 seasons out must match. Fails before, passes after.
- **Effort**: S. **Risk**: low.

## Checked and found sound

- Variance-ratio z-statistic (`randomwalk.py:27-40`): the heteroskedasticity-robust form is correct. Measured false positive on true random walks, n = 300: **0.047** (nominal 0.05).
- Sidak-adjusted Chow-Denning critical value (`randomwalk.py:62`): correct two-sided form.
- Schäfer-Strimmer intensity and the variance of the sample correlations (`reconcile.py:84-101`): the algebra matches the sum-of-squares form I checked by hand; the paper text was not compared.
- ACI score rank `ceil((n+1)(1-alpha))` and the miscoverage update (`calibrate.py:145-198`): match Gibbs and Candès (2021, arXiv 2106.00170).
- Causal flags in `lab.robust_flags` (`lab.py:183-211`): the trailing level uses only earlier points and the scale uses only history before the origin. No future leak found.
- Control `at_origin` (`control.py:77-84`): cuts past covariates at the origin and future covariates at origin + horizon.
- TimesFM point: the installed `timesfm3` forecaster returns the median quantile as `forecast` (`timesfm3/mlx/timesfm3_forecaster.py:236`). The batch and controlled paths agree, so no inconsistency.
- Symmetric averaging formula (`timesfm.py:131-133`): matches the library's `(pos - neg[..., ::-1]) / 2`.

## Missing methods a principal forecaster would expect

Each entry gives the source and the measured or quoted effect where one exists. Items marked "from memory" were not re-verified online.

1. **Weighted interval score for model selection** (A2-09). Bracher et al. 2021, arXiv 2005.12881 (verified). No Prophet effect yet; use it to rank candidates alongside pinball.
2. **Exact sign-flip or stationary-bootstrap paired tests** (A2-06). Politis and Romano 1994, JASA, DOI 10.1080/01621459.1994.10476870 (verified). Measured: null rate 0.089 falls to 0.046 with the exact test at k = 6.
3. **Conditional coverage test for interval violations.** Christoffersen 1998, "Evaluating interval forecasts", International Economic Review 39(4), 841-862, DOI 10.2307/2527341 (verified). Tests whether violations are independent of each other, which the ACI and window-based bands need; the test is missing from `calibrate`. Effect: not measured.
4. **Probability integral transform uniformity check.** Diebold, Gunther and Tay 1998 (from memory). Uses the full nine-quantile distribution and is cheap; not present. Effect: not measured.
5. **Unconditional coverage test.** Kupiec 1995 proportion-of-failures test (from memory). Would replace the Wilson heuristic in A2-12 with a test on hit counts.
6. **Calibrated quantile regression for panel bands.** Romano, Patterson and Candès 2019 CQR, arXiv 1905.03222. Quoted in the earlier report: CQR average length 1.41 against 2.24 for split conformal at about 90% coverage in the authors' 11-dataset study. Relevant to A2-02.
7. **MinT reconciliation as a tool.** Wickramasuriya, Athanasopoulos and Hyndman 2019, JASA 114(526). Quoted in the earlier report: MinT RMSE -5.2% against bottom-up -5.3% on correlated ETS, and bottom-up +90.4% on mixed-length series (Steinmeister and Pauly, arXiv 2409.18550). Relevant to A2-10.
8. **Horizon-specific combination weights and simple-average benchmarks.** Wang, Hyndman, Li and Kang 2023, arXiv 2205.04216 (quoted in the earlier report): the simple average often beats learned weights because of weight-estimation error. Supports keeping the equal blend as the default and tying any per-horizon weights to many origins.
9. **Volatility-scaled intervals for prices.** Locally weighted conformal, Angelopoulos and Bates, arXiv 2107.07511 (quoted). Prophet has an EWMA variant already (`calibrate.ewma_sigma`); GARCH (Bollerslev 1986, from memory) needs far more data than an 8 to 20 origin backtest offers.
10. **Random-walk-consistent cross-horizon dependence** (A2-04). Derived here from the random-walk structure: implied correlation sqrt(min/max). Measured: the constant 0.1 gives 0.45 times the true total spread.
11. **Fractional and long-memory checks for rates.** ARFIMA (Hosking 1981, from memory). Not a priority: the variance-ratio test already gates the random-walk case, and no Prophet series needed it.

## Process notes

- Two Bash calls were blocked by an auto-mode hook that rewrote their input. I used the Read tool and a fresh read-only command instead. The blocked calls did not change any finding.
- The electricity stored result has 30 of 60 entity rows. The finding A2-07 depends on that, and the full 60-row result should be checked before any claim about the electricity winner.
