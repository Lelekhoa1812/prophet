# A4: Principal-analyst method audit (round 3)

Scope: whether Prophet works the way a principal data analyst would. Evidence is the six live runs in `data/live_gates_retail_rep3/rep4.json`, `data/live_gates_wti_rep3/rep4.json`, `data/live_panel_beijing_rep1.json`, `data/live_panel_electricity_rep1.json`, plus their ledgers under `data/sandbox/<run_id>/out/ledger.json` and the step traces. Intended method is `docs/prophet/ANALYST_PLAYBOOK.md`, the skills in `src/prophet_agent/skills/`, `prompts/planner.v22.md`, `evaluator.v9.md`, `composer.v6.md`, and `runtime/contracts.py` and `runtime/goal.py`. All figures below are measured from those files unless marked as a reading of code. Retail rep3 ran planner.v21 and WTI rep3 ran planner.v21, so those two are not strictly comparable with v22 runs.

## Summary

1. The one mechanical gate that matters most (a forecast passes if a check ticks) is being satisfied without the analysis being sound: a centre switch clears the health item and the baseline gate, and the answers still go out.
2. The worst defect is a clock error: in retail rep4 the forecast was dated 2011-11-07 to 2011-11-24 when the data runs to 2011-12-08; the "last value" used is the 2011-12-08 value.
3. The planner states assumptions in its rationales (Mon to Fri trading, Saturday closure, exchange holidays as gaps) before testing them, and then keeps them when the data contradicts them.
4. No run asked the person a single question (0 ask_user across six runs), though every run had an unresolved provenance or calendar question that only the person could settle.
5. Ledgers and answers disagree on what was done to the data in at least four places, and no check reconciles them; decisions are templated, untested and not hypotheses in form.

## Live answer reviews

**Retail rep3 (net sales, 14 days, calendar grid): grade C-.** Good instincts: calendar grid, the Christmas closure identified, and an honest "yearly seasonality not testable (603 of 730 days)". Failures: the 14-day total of £472,815 includes a negative point (-£3,606) on 2011-12-17, which is a Saturday; the answer calls it a "model artefact" when it is the imputed Saturday closure learned by the model. The ledger says "keep negative values" while the answer says the three negative days were "clipped to 0". Zero imputation and spike replacement were never tested through the treatment gate (rule 9). Health is reported as "ok" while two of 14 points are impossible.

**Retail rep4 (same data, 14 days, Mon to Fri assumed): grade D.** The forecast window (2011-11-07 to 11-24) is wrong: the series index ends 2011-11-04 but its last values are the 2011-12-07 and 2011-12-08 rows (81,417.78 is the raw 2011-12-08 value). The "last observed value £81,418" is therefore a value four weeks after the stated forecast origin. The ledger states "all missing are weekends... zero observed weekend days" after the run itself found 99 Sundays with sales. The answer delivers a `flat_unexpected` forecast with the centre switched to naive, though planner rule 10 says this "never reaches the answer". It names "Prophet" as a method (the word appears 107 times in the trace); no Prophet model exists in this stack. The table row "Prophet with changepoints: tested by sub-agent" is not supported: the sub-agent's seven steps only read files, and its own report gives a 14-day total of £809.6k against the main £1.14m. The Sunday trading question is left as a caveat, not asked.

**WTI rep3 (20 trading days): grade C-.** Correct calendar (2,498 trading-day rows, no interpolation), and it reads the range rather than the line for a random walk. Failures: the centre (90.76) is the mean of the last four closes, not the last close (89.53), although the answer says the backtest used season=1 (which gives the last close); the window is undeclared. The answer says both "about 8% more accurate" and "not distinguishable" about the same rule. The final ledger write was an empty `ps.Ledger()` (2 bytes, `[]`), so the scan-decision check still failed. Five level shifts are "kept as real regime changes" with no test.

**WTI rep4 (20 trading days, status met): grade C.** Strong on the stance: states the random walk, names naive as the appropriate benchmark, gives dated primary-source context and a list of what was not obtained. Failures: 109 exchange-holiday rows on a business-day grid were filled by interpolation (the playbook says holidays are not gaps); the naive centre (MASE 7.27) is worse than the backtest's best rule, mean_last_season (6.66), and the gate passed anyway; "one more" level shift is unnamed; the calibration choice (kept raw, backtest coverage 73%, held-out raw 88%) has no stated rule; a quoted EIA passage is well over the 15-word quote limit in the organisation's copyright rule.

**Beijing PM2.5, 12 stations (14 days, panel): grade D.** Broad coverage and a real panel backtest (TimesFM MASE 0.65 against 0.84 for the best rule, measured). Failures: the headline "single best method for all 12 stations" while Wanliu and Wanshouxigong have no forecast quantiles (four failed extraction attempts) and six stations have no calibration. Level shifts at 2014-01-01, 2015-01-01 and 2016-01-01 (calendar boundaries) and a three-station step on 2016-11-25 and 2017-01-08 (x2.05 then x0.44) are "treated as real regime changes (no documented unit change)". Per-station outlier counts of 92 to 128 (6 to 9% of 1,461 rows) are kept without the 2% stop rule. The ledger decision text is identical across all 12 stations. No hypothesis about the winter heating season (which starts mid-November in northern China and overlaps the 2016-11-25 step) was stated or tested. Nongzhanguan's calibrated band is said to be adopted, but the table shows raw bands.

**Electricity, 60 clients (7 days, pooled test): grade F.** No forecast, no total, no accuracy figure reached the answer. 233 steps. About four or five identical `panel_backtest` re-runs happened because the result was clipped. The auto researcher searched US grid operators for a dataset of unknown provenance, with three rounds and no usable number. Zero shares of 25 to 69% in every client (a censoring question) were recorded as "treat zeros as observed for now" and never put to the person. The answer tells the person the analysis "kept repeating the same step" and lists run ids.

## Findings (ranked by impact)

### A4-01: Business-day clock misaligned after dropping rows (critical)
- Severity: critical (wrong dates and wrong anchor value shown to the person).
- Evidence: retail rep4 forecast dates 2011-11-07 to 2011-11-24 (`forecast.json`, `plan:d0:25` and later). Series index last five dates 2011-10-31 to 2011-11-04; series last five values `[81651.84, 57751.32, 54228.37, 75076.22, 81417.78]`, which are the raw 2011-12-05 to 2011-12-08 rows (inspect_data: last=2011-12-08, unique=738). 504 weekdays from 2009-12-01 end on 2011-11-04, so 504 non-null rows were placed positionally on a weekday index after the 99 Sunday rows were removed. The answer's naive centre (£81,418) is the 2011-12-08 value. WTI and Beijing use the same pattern of freq-B reindexing but their clocks match their sources.
- Root cause: the planner asserted "trading-day series (Mon to Fri only)" in its rationale at `plan:d0:17` before checking day-of-week counts; `pl.build_series` with `freq='B'` on a frame with dropped rows assigns values by position; nothing checks the prepared series' last date against the source's last date.
- Proposed fix: (a) in `prophet_lib.prepare_series` and `build_series`, align by timestamp, never by position; raise if rows are dropped without an explicit index; (b) new contract check `series_clock_matches_source` (block): last index date equals last non-null source date, and the count of non-null rows equals the count of rows in the prepared series; (c) planner step 2 passage: "Before you drop any row or choose a trading calendar, count non-null values by day of week. A day with values is a trading day unless you can name why not."
- Files touched: `src/prophet_agent/sandbox/prophet_lib.py` (prepare_series, build_series), `src/prophet_agent/runtime/contracts.py` (new check), `src/prophet_agent/prompts/planner.v23.md` (step 2).
- Test: offline fixture (synthetic, labelled as such) with Sunday sales and a Saturday gap; assert the prepared index's last date equals the source's last date and the check fails on a positional build. Live check: rerun retail goal with the same CSV; the forecast dates must start 2011-12-09.
- Effort: M. Risk: low to medium; changes the prepared series for any frame that currently drops rows (re-run the existing panel tests).

### A4-02: Impossible forecast values reach the total and are explained away (critical)
- Severity: critical.
- Evidence: retail rep3 point forecasts include -3,606 (2011-12-17, Saturday) and 2,477 (2011-12-10, Saturday); the total of £472,815 is the sum of the 14 points including the negative one (I summed the table). The answer calls both "model artefacts" and keeps health "ok". History has zero negatives after clipping. Saturdays were imputed to 0 in the same run (ledger: "Impute missing as 0"), so the model learned the closure and then produced a non-zero, negative value on a day it had been told is closed.
- Root cause: no non-negativity floor for a non-negative measure, no structural-zero rule for closed days, and the health verdict does not test plausibility against the domain floor. The imputation (zero for closure) was not tested through `backtest_treatment` (planner rule 9).
- Proposed fix: (a) a `floor` in the series metadata for non-negative measures (from the scan's `values.negative` finding); `timesfm_forecast` and `forecast_report` clip point, q10 and q90 at the floor and report the count clipped; (b) a closed-day rule: a weekday with zero in at least 95% of history is forecast as zero, or tested against the model via `control_lab`; (c) new check `forecast_within_floor` (block): no point or band below the floor unless recorded as clipped.
- Files touched: `src/prophet_agent/runtime/contracts.py` (check_forecast), `src/prophet_agent/tools/` (timesfm forecast and forecast_report), `src/prophet_agent/prompts/planner.v23.md` (forecast window bullet), skill `timesfm-control-and-flat-forecasts.md`.
- Test: offline unit: a forecast with a negative point and a non-negative history must fail `forecast_within_floor`; after clipping it passes and reports the count. Live: rerun retail rep3 and confirm no negative in the delivered table.
- Effort: S to M. Risk: clipping can hide genuine returns; the answer must say when a floor was applied.

### A4-03: The centre and health gates accept any rule, so a worse rule clears the checklist (high)
- Severity: high.
- Evidence: WTI rep4 delivered naive as centre (MASE 7.27) with mean_last_season the best rule (MASE 6.66); status `met`. Retail rep4 delivered naive (£81,418 per day) with tsb_expected as best rule and health flat_unexpected; checklist item "forecast health is not flat_unexpected" ticked.
- Root cause (reading of code): `_baseline_gate` in `contracts.py` (around lines 585 to 610) sets `on_timesfm = False` whenever the centre is any rule, so no comparison against the best rule is made. `delivered_health` returns `centre_rule:<name>` when a rule is the centre, which the progress item treats as done, although the band still comes from the broken TimesFM shape.
- Proposed fix: the gate requires the centre rule to equal `best_baseline`, or to be `naive` with a random-walk verdict from the variance-ratio test, and the answer must say which exception applied. Health exemption applies only to the line, never the band: when the centre is a rule and health is flat_unexpected, the band must be repaired or built from the rule's own residuals.
- Files touched: `src/prophet_agent/runtime/contracts.py` (_baseline_gate, delivered_health, progress), `src/prophet_agent/prompts/evaluator.v10.md` (new centre paragraph), planner rules 9 and 12.
- Test: offline unit: `_baseline_gate` with centre naive and best mean_last_season must return ok=False unless random-walk exception applies; with random-walk exception and explanation text it passes. Live: rerun WTI rep4 goal; expect not_met until the centre is mean_last_season or the random-walk path is recorded.
- Effort: S. Risk: blocks a defensible naive centre for random walks unless the exception is written in.

### A4-04: Ledgers, answers and series disagree about treatments, and nothing reconciles them (high)
- Severity: high.
- Evidence: retail rep3 ledger "Keep negative values as genuine returns-heavy days" vs answer "clipped to 0". Retail rep4 ledger "Zero observed weekend days" vs its own later finding of 99 Sundays. Beijing ledger decision "Outliers kept as real pollution spikes; ... (no documented unit change)" repeated verbatim for all 12 stations. Electricity ledger "Level shifts ... will not ignore these ranges as they represent real regime changes" with no test. WTI rep3 and rep4 disagree on whether gaps were filled (none vs 109).
- Root cause: ledger entries are written in the cell that found the fact and never revised; the composer writes from the draft and the evidence summary, not from the ledger; the contract checks count stages and check that ids are present, not that the verbs (kept, clipped, filled, imputed, replaced) match the answer.
- Proposed fix: (a) `ps.Ledger` gets `supersedes=<index>` and `revise(...)` so a later cell corrects an entry; (b) new check `ledger_answer_consistent` (warn, then block): each treatment verb in the answer's "what was done" section must match a non-superseded ledger entry for the same series or measure; (c) composer.v7 receives the ledger's decision list as a table and must cite the index of each decision.
- Files touched: `src/prophet_agent/sandbox/prophet_stats.py` (Ledger), `src/prophet_agent/runtime/contracts.py` (new check), `src/prophet_agent/prompts/composer.v7.md` (new), planner step 5.
- Test: offline fixture with a ledger saying "keep" and an answer saying "clipped"; assert the check fails. Live: rerun retail rep3; the ledger and answer must agree on negatives.
- Effort: M. Risk: false positives on wording; start as warn.

### A4-05: `ps.Ledger()` in a later cell overwrites the mission ledger with an empty one (high)
- Severity: high (silent data loss in a saved decision record).
- Evidence: WTI rep3, cell `plan:d0:9`, created `ledger = ps.Ledger()` and saved; the saved `out/ledger.json` of run d908f0a5 is `[]`. The five stage entries in run 9de19bc9 are still there, but the newest copy is empty and the check takes the maximum over all saved ledgers (`ledger_stages` and `_saved_ledgers` in `contracts.py`), so the empty file is invisible to the checks and to later carries. The planner prompt step 5 teaches the pattern `ps.Ledger().add(...)` and `save`.
- Root cause: `Ledger.__init__` starts empty and `save` overwrites; the contract reads the best ledger, not the newest one.
- Proposed fix: `ps.Ledger.open(path)` that loads an existing file (default in examples); `save` refuses to write fewer entries than the file it replaces unless `replace=True`; the check reads the newest ledger for the mission and reports entry counts.
- Files touched: `src/prophet_agent/sandbox/prophet_stats.py`, `src/prophet_agent/runtime/contracts.py` (_saved_ledgers), `src/prophet_agent/prompts/planner.v23.md` (step 5).
- Test: offline: two cells, the second doing `Ledger()` then `save`; expect a refusal or merged file with 6 entries. Live: rerun WTI rep3; the final ledger must hold all decisions.
- Effort: S. Risk: low.

### A4-06: No question to the person across six runs, and a stuck loop hands the problem to a fresh analyst (high)
- Severity: high.
- Evidence: `questions` is empty in all six runs; `action=ask_user` count is 0 in every trace. Retail rep4 repeated a Sunday-observation probe many times before stopping, and the stuck detector delegated to an analyst, who did not settle it. Unresolved questions that only the person can answer: Sunday trading, sign of negative sales (returns or errors), censoring zeros of 25 to 69% in electricity, client region, and whether "14 days" means trading or calendar days (rep3 and rep4 answered it differently).
- Root cause: planner rule 21 says ask only when the data cannot tell, and the planner judges that the data can tell and keeps probing; the stuck handler delegates to a fresh analyst rather than asking; the evaluator's "data not settled" rule is only checked when the evaluator is reached.
- Proposed fix: planner rule: after two probes that do not settle a provenance question (calendar, sign, censoring, units, region), ask_user once with options. In `goal.py`, a stuck step whose subject is provenance escalates to ask_user instead of a fresh analyst. Add an evaluator check that a provenance question in the ledger with no decision is not_met.
- Files touched: `src/prophet_agent/runtime/goal.py` (stuck handling, around the `stuck:` delegation), `src/prophet_agent/prompts/planner.v23.md` (rule 21), `src/prophet_agent/prompts/evaluator.v10.md`.
- Test: offline scripted run with an ambiguous weekday fixture (synthetic, labelled as such): expect ask_user within three steps. Live: rerun retail rep4 goal; expect a question on Sunday trading.
- Effort: M. Risk: over-asking; limit to provenance subjects.

### A4-07: Clipped observations cause repeated tool calls and failed extractions (high, cost)
- Severity: high (cost and time).
- Evidence: `goal.py` clips each observation at `OBS_CHARS = 4000`. Electricity re-ran `panel_backtest` about five times because the per-entity table was clipped. Beijing tried to extract Wanliu and Wanshouxigong quantiles about four times, each returning empty stdout. Retail rep4 built `forecast_report` several times with the same purpose.
- Root cause: the planner sees a clipped result with no pointer to the full file, so it re-runs the call; a sandbox print of an empty filter returns empty stdout without an error, which the failure logic does not count.
- Proposed fix: tools return a compact summary and the saved file path with row count ("[clipped: 60 of 60 rows; full table at run_id/out/backtest_panel.json]"); planner rule: "never re-run a tool to see more output; read the saved file with a small print." Treat an empty stdout after a filter as a failed step for the repeat counter. Make `forecast_report` idempotent on identical arguments.
- Files touched: `src/prophet_agent/runtime/goal.py` (_clip, _view, repeat handling), `src/prophet_agent/tools/` (panel_backtest, forecast_report), planner rules 16 and 17.
- Test: offline: a 60-row backtest result produces an observation with the pointer and no more than 4,000 characters; two identical forecast_report calls return the cached result. Live: rerun electricity; the panel_backtest count should be 1.
- Effort: S to M.

### A4-08: Outside research scoped without knowing the data's region or provenance (high)
- Severity: high.
- Evidence: electricity researcher was started at step 1 (`auto:d0:1`) with a brief covering ERCOT, CAISO, ISO-NE, FERC and NERC for a file named only as 60 clients (MT_001 to MT_060). It ran three rounds with no usable figure (`plan:d1.a1:0` to `d1.a1:13`). WTI rep4 ran 13 web steps, most timing out, before the planner gave up. The planner prompt step 7 says research starts "once the data has been inspected", but the trigger in the trace fired before the region was known.
- Root cause: the research brief is written from the goal text, not from the data's provenance; no bound on research rounds without a usable number.
- Proposed fix: research starts only after inspect_data returns a region or unit field, or after ask_user has answered; the brief quotes the region; cap at two rounds with no dated figure, then record "not obtained" and move on.
- Files touched: `src/prophet_agent/runtime/goal.py` (auto research trigger), `src/prophet_agent/prompts/planner.v23.md` (step 7).
- Test: offline with stubbed web tools returning nothing: assert at most two rounds and a not-obtained entry. Live: rerun electricity goal with the region question answered.
- Effort: M.

### A4-09: Level shifts are called real regimes without a test or a cause that the data supports (medium-high)
- Severity: medium-high (drives the range and the centre).
- Evidence: Beijing: shifts on 2014-01-01, 2015-01-01 and 2016-01-01 for three stations (calendar boundaries) and a three-station step on 2016-11-25 (x2.05) and 2017-01-08 (x0.44), both in the final four months before the origin. Electricity: shift on 2012-01-01 in many of 60 clients, "likely a systematic change ... represent real regime changes". WTI rep4: five shifts "correspond to known oil-market regimes" with one unnamed. Ledger decisions say "no documented unit change". The changepoints skill (step 6) requires a walk-forward test of options before keeping a shift.
- Root cause: the planner treats absence of a documented unit change as evidence of a real regime; the scan does not flag breaks on a calendar boundary or shared by several entities on the same date; no backtest of truncate-vs-flag was run in any run (the string `backtest_treatment` appears 71 times in the electricity trace and 4 in the Beijing trace, with no visible test of a level shift).
- Proposed fix: scan adds `on_calendar_boundary` and `coincident_entities` to each changepoint finding; a shift with either flag is labelled "recording hypothesis" and must be tested (truncate vs flag vs none) before it is kept; planner step 9 and the changepoints skill get a Beijing-style example (heating season as a named alternative hypothesis, to be tested).
- Files touched: `src/prophet_agent/sandbox/prophet_scan.py` (changepoint findings), `src/prophet_agent/skills/changepoints-and-regimes.md`, `src/prophet_agent/prompts/planner.v23.md`, `src/prophet_agent/runtime/contracts.py` (shift decision check, warn).
- Test: offline synthetic panel with a common 1 January step across three entities: assert both flags set and the check requires a tested decision. Live: rerun Beijing and record the decision per shift.
- Effort: M.

### A4-10: Hypotheses and decision framing are missing from every run (medium-high)
- Severity: medium-high (the method the playbook calls "hypotheses, not habits" is not visible).
- Evidence: none of the six ledgers has an expectation field; no run recorded a falsifiable expectation ("the level shifted in 2024", "there is a weekly cycle") as planner step 4 requires. None of the six final answers states the decision the forecast serves or which error costs more (planner step 1; checked by reading the answer texts). The mission checklist (`MIN_LEDGER_STAGES = 4`) counts stages with a finding and a decision, so a ledger of descriptions passes.
- Root cause: the check is on stages, not on tests; framing is a prompt instruction with no field.
- Proposed fix: `ps.Ledger.add` accepts `expected` and `outcome` (confirmed, refuted, not tested); a warn-level check `hypotheses_tested` requires at least two entries with `expected` and an outcome; the framing (decision, horizon unit, costlier error) becomes a required first line of the brief and is shown to the composer.
- Files touched: `src/prophet_agent/sandbox/prophet_stats.py` (Ledger), `src/prophet_agent/runtime/contracts.py` (progress item and check), `src/prophet_agent/prompts/planner.v23.md` (steps 1 and 4), `src/prophet_agent/prompts/composer.v7.md`.
- Test: offline: a ledger without expected or outcome fields fails the warn check; with two tested hypotheses it passes. Live: rerun retail rep4 and check the ledger.
- Effort: M. Risk: warn only at first, so no run is blocked by it.

### A4-11: Outlier share rule not applied (medium)
- Severity: medium.
- Evidence: Beijing per-station Hampel outliers 92 to 128 of 1,461 rows (6.3% to 8.8%); retail rep4 STL outliers at 9.6%. The outliers skill says stop above about 2% and treat as noise or a broken feed. Beijing kept all as real spikes with the same ledger text; retail rep4 wrote "STL outliers at 9.6% are high; use robust" and then did not use robust.
- Root cause: the share is printed but no gate reads it; the decision text is a template.
- Proposed fix: a warn-level contract check reads each series' flagged share from the scan; above 2% the planner must state the share, the cause for a sample of flagged points (for example by month), and the test of a winsorised copy.
- Files touched: `src/prophet_agent/runtime/contracts.py`, `src/prophet_agent/sandbox/prophet_scan.py` (outlier share in the findings), skill `outliers-and-level-shifts.md`.
- Test: offline: 9% flagged fixture gives the warn; live: rerun Beijing and inspect the decision text.
- Effort: S.

### A4-12: Baseline set not matched to the demand class, and an undeclared rule window (medium)
- Severity: medium.
- Evidence: retail rep4 used `tsb_expected` (an intermittent-demand rule) as the best simple rule for a series with ADI = 1.0 (its own scan figure). WTI rep3: the "mean_last_season" centre equals the mean of the last four closes (90.757, measured from `datasets/.../eia_wti_spot_daily_2016_2026.csv`), not the last close (89.53, which season=1 implies). The window of the rule is not printed in any answer.
- Root cause: the candidate baselines are not filtered by demand class; the rule's window is an internal parameter not returned in the result.
- Proposed fix: backtest candidate set selected by class (TSB and Croston only when ADI > 1.32 or sales in fewer than three in four periods, per the zeros skill); backtest results print each rule's window; the answer names the window.
- Files touched: the backtest tool in `src/prophet_agent/tools/`, `src/prophet_agent/forecast/` (baselines), `src/prophet_agent/prompts/composer.v6.md` (name the window).
- Test: offline unit: a continuous series returns no TSB candidate; mean_last_season reports window 1 when season=1. Live: rerun WTI rep3; the centre must equal the stated window's mean.
- Effort: S.

### A4-13: The clock and horizon metadata are wrong or missing, and calendar versus trading days is decided silently (medium)
- Severity: medium.
- Evidence: both retail and WTI rep3 answers say "the series step is not recorded, so the request cannot be converted". The series step is known (typical_step 1 day in inspect_data; B frequency in WTI). The horizon warning is produced by `_series_step` in `goal.py` (around line 1090), which reads `config.series_step` that the forecast record does not carry. Retail rep3 read "14 days" as calendar days; retail rep4 read it as 14 trading days (about 20 calendar days) and says so only in a caveat; no ask_user.
- Root cause: the forecast record omits the step; the calendar question is treated as an assumption rather than a question.
- Proposed fix: `timesfm_forecast` and `forecast_report` record `series_step` and `calendar` in the config; when the requested unit and the series' unit differ by more than 20% in horizon, planner must ask (folded into A4-06).
- Files touched: `src/prophet_agent/tools/` (forecast result), `src/prophet_agent/runtime/goal.py` (_series_step consumer), planner step 2.
- Test: offline: a forecast record for a daily series carries series_step 1 day and the horizon check passes. Live: rerun retail and WTI; the caveat must disappear.
- Effort: S.

### A4-14: "Tested" and "computed" claims that no run did, and a delegated report the planner did not reconcile (medium)
- Severity: medium (credibility of the answer).
- Evidence: retail rep4 table: "Excluding first week / winsorizing outliers: Tested; did not resolve flat forecast shape" and "Prophet with changepoints: Tested by sub-agent". Trace: no call to `control_lab` or `backtest_treatment` for those treatments (grep count 0 for both in that trace); the sub-agent's seven tool steps only read files (its report claims Prophet changepoints and a £809.6k total). The sub-agent report and the main answer disagree and the answer does not mention it. Beijing: "single best method for all 12" while two stations have no forecast. WTI rep3: "about 8% more accurate" and "not distinguishable" in one answer.
- Root cause: the composer grounds numbers (figures must appear in evidence) but not method claims; there is no check that a claimed test has a tool call in the evidence.
- Proposed fix: a grounding check for method verbs ("tested", "checked", "compared", "adopted") that requires a matching tool call id in the evidence; sub-agent reports are compared with the main answer's figures, and a disagreement blocks the finish. Composer rule: a count or claim about a station, series or test must cover every item in the list, or say which are missing.
- Files touched: `src/prophet_agent/runtime/contracts.py` (grounding), `src/prophet_agent/prompts/composer.v7.md`, `src/prophet_agent/runtime/goal.py` (sub-agent merge).
- Test: offline: an answer claiming a tested winsorising without a matching call fails the check; a sub-agent total that differs from the answer total is flagged. Live: rerun retail rep4 and confirm the table rows are supported or removed.
- Effort: M.

### A4-15: Budget-driven finishes and repeated pack builds waste the mission (low to medium)
- Severity: low to medium.
- Evidence: the planner's rationale "tool budget used" appears 16 times across the six JSON files (retail rep4 four, Beijing four, retail rep3 two, WTI rep3 two, WTI rep4 two, electricity two; these are occurrences and may repeat the same event), and each finish was refused and the run continued. This is a code path (`_decide_plan`, `out_of_calls`) that emits a finish, so the planner's own finish is never a judgement. Versions were mixed: retail rep3 and WTI rep3 ran planner.v21, the rest v22, so comparisons across these six runs need that caveat.
- Root cause: the budget signal and the finish action share a code path; out-of-calls is handled by forcing a finish rather than composing from evidence.
- Proposed fix: when out of calls, compose from the evidence with a fixed message, not a planner finish; record the planner version in the answer's evidence; keep one forecast_report per distinct argument set.
- Files touched: `src/prophet_agent/runtime/goal.py` (_decide_plan and budget handling).
- Test: offline: out-of-calls produces a compose step with no planner finish. Live: count finish actions in a rerun.
- Effort: S.

## Notes on the method comparison

- The playbook's order (provenance, clock, scan, hypotheses, funnel, prepare, outside context, baselines, treatments, forecast, range, deliver) was followed in shape by the WTI runs and the Beijing run, and was broken at step 2 (clock) and step 9 (treatments) in retail.
- The baseline gate (A4-03) and the health exemption together let the checklist turn green with a broken band; this is the most important single fix after A4-01 and A4-02.
- Beijing and electricity show the cost of unclipped results and loops (A4-07, A4-08); the electricity run produced nothing a person could use.
- Interval language is honest in most answers ("read the range, not the line"); the weak spots are the unsupported "single best" headline and "about 8% more accurate" beside "not distinguishable".

## Outside practice used (cited)

- Hyndman and Athanasopoulos, Forecasting: Principles and Practice, 3rd ed., regression chapter, on day-of-week and trading-day effects for daily data: https://otexts.com/fpp3/regression.html (used for A4-01 and A4-13: treat weekday and trading-day effects as modelled inputs, not as data cleaning).
- Hyndman and Koehler (2006), "Another look at measures of forecast accuracy", International Journal of Forecasting 22(4): https://www.doi.org/10.1016/J.IJFORECAST.2006.03.001 (MASE definition; supports the playbook's use of MASE against naive, and A4-12's window disclosure).
- Teunter, Syntetos and Babai (2011), "Intermittent demand: linking forecasting to inventory obsolescence", European Journal of Operational Research 214(3): https://dx.doi.org/10.1016/j.ejor.2011.05.018 (TSB is designed for intermittent demand; used for A4-12). The search returned abstracts and secondary summaries only, so the claims above are limited to what those sources state.

## Caveats

- Live answers were read as written; no live run was repeated, and no API call was made on port 8010.
- Offline fixtures proposed in the tests are synthetic and must be labelled as synthetic replay in any reported figure.
- Grades are judgements against the stated intended method, not against an external benchmark.
