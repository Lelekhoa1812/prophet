# A3: Research coverage and mission planning (audit round 3)

Scope: `src/prophet_agent/research/`, `tools/research_tools.py`, `runtime/goal.py` (research trigger, budget, stuck and repeat rules), `runtime/contracts.py` (checklist, research and baseline gates), `prompts/planner.v22.md`. Evidence: six measured snapshots in `data/` (read-only). Labels: "measured" = read from a snapshot or code path; "inferred" = reasoned from code, not reproduced live.

## Summary

1. The lead planner's tool allowance is spent by its own researcher and panel analysts. Sub-agent calls are counted in `_calls_used`, so 8 of 20 mission-part finishes were forced "tool budget used" (measured), and the harness's allowance raise does not reach the planner's view.
2. External research is shallow and disconnected from the data. The WTI runs read 3 to 4 usable primary pages, lost a dated OPEC+ event that was fetched but never passaged, invented an "accessed" date, and never tied any source to the five detected level shifts (measured, code search confirms no link).
3. The answers contradict their own measured backtests in places: WTI rep4 says no model beats naive while `mean_last_season` (MASE 6.66) beats naive (7.27), and the Beijing answer labels a blend figure as a "Seasonal Mean" baseline.
4. The planner repeats itself. Identical calls recur (electricity `panel_backtest` 3x identical; retail 6 identical cells; Beijing analysts reload data 5 to 6 times). Stuck-detection fires only after 16 idle calls. 15 of 20 finish attempts were rejected by the evaluator, 7 of them with the claim "All checklist items are satisfied".
5. Planning lacks the principal-analyst frame: no run recorded a decision frame or falsifiable hypotheses (0 of 6), the checklist has no hypothesis, entity-coverage or non-negativity item, and the gates accept a non-best centre line. Fix order: A3-01, A3-03, A3-02, then A3-06 and A3-07.

## Per-run table (calls by stage)

Counts include sub-agent calls. "Lead" = top-level calls; "sub" = calls inside sub-agents. Stage tags are a heuristic on tool name and, for `sandbox_python`, on code keywords (scan = `prophet_scan`/`scan`+`findings`; cleaning = clean/impute/Ledger/interpolate/fillna; sandbox_other = the rest). "Repeats" = calls with identical tool and arguments. "Errors" = result `ok:false`, `error`, or HTTP note. "Forced" = finishes with rationale "tool budget used". "Rejected" = finishes followed by an evaluator `not_met`.

| Run | Status | Lead | Sub | inspect | scan | cleaning | sandbox_other | backtest | forecast | calibrate | research | report | skill | repeats | errors | forced fin. | rejected fin. | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| retail rep3 | partial | 27 | 0 | 1 | 9 | 8 | 3 | 2 | 1 | 0 | 0 | 2 | 1 | 2 | 4 | 1 | 2 | 27 |
| retail rep4 | partial | 49 | 7 | 1 | 7 | 12 | 28 | 2 | 0 | 0 | 0 | 5 | 1 | 7 | 16 | 2 | 2 | 56 |
| WTI rep3 | partial | 9 | 12 | 1 | 4 | 0 | 0 | 1 | 1 | 1 | 12 | 1 | 0 | 0 | 2 | 1 | 3 | 21 |
| WTI rep4 | completed | 13 | 13 | 1 | 4 | 1 | 2 | 1 | 1 | 1 | 14 | 1 | 0 | 0 | 4 | 1 | 4 | 26 |
| Beijing panel | partial | 22 | 57 | 5 | 39 | 4 | 12 | 8 | 0 | 8 | 0 | 1 | 2 | 13 | 15 | 2 | 3 | 79 |
| Electricity panel | partial | 15 | 76 | 5 | 54 | 0 | 11 | 7 | 0 | 0 | 12 | 0 | 2 | 21 | 13 | 1 | 1 | 91 |

Notes on the table (measured):
- The user's premise "WTI rep4 called web_research once" undercounts. The lead made 1 `web_research`; the auto-started researcher made 8 `web_research` and 5 `web_fetch`. Research is 14 of 26 calls.
- Research never ran in retail (no trigger keyword in the goal) or Beijing PM2.5 (no keyword). It ran in electricity, where the researcher's queries targeted US grid operators (ERCOT, ISO-NE, PJM, NYISO) with no evidence of region.
- Forced finishes: 8 across six runs (retail3 1, retail4 2, WTI3 1, WTI4 1, Beijing 2, electricity 1).
- Rejected finishes: 15 of 20 finish attempts.

## Findings (ranked by impact)

### A3-01: Sub-agent calls consume the lead planner's tool allowance
- **Severity:** Critical (cost, premature finish, wasted sub-agent work)
- **Evidence:** `runtime/goal.py:282-291` (`_calls_used` adds sub-agent evidence). `goal.py:1194` sets `out_of_calls` from that count against `cfg.tool_budget`. Snapshot brief says `tool_calls: 20` (`contracts` step). Measured: WTI rep4 forced finish at evidence 8 with calls_used 20 after a 13-call researcher (lead made 6 calls). WTI rep3 forced at 11 (calls_used 20). Beijing forced at 5 (calls_used 45) and 24 (66). Electricity forced at 18 (85). `_harness_allowance` (`goal.py:967-983`) raises the store budget but not the `cfg.tool_budget` the planner reads.
- **Root cause:** Two budget accountings. The store-level allowance is protected for harness sub-agents; the planner's own count is not. The lead is starved and forced to finish, then `REPAIR_CALLS` (goal.py:1344-1353) grants calls back after the evaluator rejects, so the cycle wastes calls.
- **Proposed fix:** Count only lead-agent calls in `_calls_used` for the lead's `out_of_calls` and `_budget_note`. Give harness sub-agents their own cap (e.g. sub_steps times specs) and report their spend separately. Keep one global hard cap as a safety stop.
- **Files:** `src/prophet_agent/runtime/goal.py` (`_calls_used`, `_budget_note`, `_harness_allowance`, line 1194).
- **Regression test:** history with a researcher sub-agent of 13 tool calls and 3 lead calls, `tool_budget=20`: `out_of_calls` is False and the budget note reports 17 left. Plus: a sub-agent exceeding its own cap is stopped.
- **Effort:** S (half a day). **Risk:** Total cost rises; mitigate with the global cap and a per-mission sub-agent cap.

### A3-02: Researcher briefing is lost and then re-run; checklist ticks on any attempt
- **Severity:** High
- **Evidence:** WTI rep4 researcher (evidence 1): `answer: ""`, `reason` says "research briefing is incomplete", no deliverable returned. Lead then re-ran `web_research` at evidence 10 (planner step `plan:d0:9`: "The research sub-agent failed to gather primary-source evidence... I need to run web_research myself"). Same pattern WTI rep3 (answer ""), electricity (answer "" in researcher, reason says "no current regime level figures ... for any ISO/RTO region"). The WTI rep4 finish rationale then claims "A researcher ... returned primary-source evidence from EIA, IEA, OPEC+, CFTC" (evidence 12), which the researcher's own reason contradicts. `contracts.py:509-519` `researched()` ticks the checklist for any researcher run, including checkpoint with empty answer. WTI rep3's final answer does not mention outside context at all.
- **Root cause:** The sub-agent's report is taken from `answer` (empty) rather than a structured briefing; the gate counts attempts, not usable findings; the answer is not required to state a failed research item (the item text allows "say plainly why" but the tick already passed).
- **Proposed fix:** (a) Return the sub-agent's finish `rationale`/last memo into `answer` when `answer` is empty. (b) `researched()` returns True only when a research call produced at least one dated, passaged source or the answer states the failure. (c) Block a lead `web_research` duplicate of a briefing already in the evidence, unless the lead names the gap it fills.
- **Files:** `runtime/goal.py` (`_delegate` result shaping), `runtime/contracts.py:509-519` and checklist item at 544-547, `prompts/planner.v22.md` step 7.
- **Regression test:** researcher with empty answer and zero passaged sources -> `researched()` False; finish with research item todo is refused; WTI-style history yields a final answer containing a "not obtained" line.
- **Effort:** M. **Risk:** More research blocked on empty results; acceptable.

### A3-03: Answers contradict their own backtest; the centre gate accepts any non-TimesFM rule
- **Severity:** High (wrong accuracy claim to the user)
- **Evidence:**
  - WTI rep4 backtest (evidence 5): `best_baseline: mean_last_season` MASE 6.6628; `naive` 7.2703; `timesfm` 7.2783. Final answer: "no model tested beats the naive forecast on accuracy", and the centre is `naive` (evidence 11). Measured contradiction.
  - Gate: `contracts.py:561-566` `recentred = any(centre != "timesfm")` passes naive. `contracts.py:596-608` checks only "not TimesFM". Prompt A12 requires the best baseline.
  - Same goal, WTI rep3: evaluator forced `mean_last_season` three times (evidence 9, 12, 15). So identical inputs produced different centres.
  - Beijing answer: "best baseline (Seasonal Mean) 0.84". Measured overall (evidence 13): blend 0.8408, seasonal_naive 1.2748, naive 1.8877, global_gbm 1.2277, timesfm 0.6477. No "Seasonal Mean" exists in the output; 0.84 is `blend`, which includes TimesFM.
- **Root cause:** The gate tests the form of the centre, not its identity against `best_baseline`. The accuracy prose is generated by the model, and no check compares named baselines and numbers with the stored backtest.
- **Proposed fix:** (a) Require `centre == best_baseline` (with the same season) or a stated reason in the answer that the evaluator accepts. (b) Extend `grounding.py` so each named baseline and its figure must match the backtest record. (c) Exclude `blend` from "simple baseline" language.
- **Files:** `runtime/contracts.py:561-608`, `runtime/grounding.py`, `runtime/goal.py` accuracy paragraph (line ~759-770).
- **Regression test:** history with backtest best `mean_last_season` and forecast_report centre `naive` -> baseline item not done; answer sentence "no model beats naive" with measured `mean_last_season < naive` -> grounding failure.
- **Effort:** S-M. **Risk:** Some legitimate centre choices need an explicit reason; allow it.

### A3-04: Ledger records causes that no tool tested; promised tests dropped
- **Severity:** High (unsupported explanations presented as findings)
- **Evidence:** WTI rep4 ledger stage `outliers_level_shifts` (snapshot step `act:d0:4`): "Level shifts correspond to known oil market regimes: COVID crash/recovery, post-COVID demand surge, 2022 energy crisis, 2024 demand concerns." No tool output supports this. The scan breaks were 2020-02-25, 2020-12-15, 2021-06-01, 2024-08-30 and one unnamed; 2022 is not among them. The same entry says "Will test ignore_ranges for 2020-04-20 ... in control_lab"; no `control_lab` or `ignore_ranges` call appears in the run. The final answer repeats "correspond to known oil-market regimes" and lists "one more" date, unfinished.
- **Root cause:** `ps.Ledger().add` takes free text; nothing requires a source id, a tool result, or a test for a causal decision. The scan message asks "is there a known event" and the planner answers from memory.
- **Proposed fix:** For `changepoints.*` findings, the ledger decision must cite either a research source id with a date inside ±N days of the break, or a test id (control_lab or backtest with the break as a driver); otherwise the decision must say "unexplained, kept as observed". Reject ledger entries whose decision names a future test and no later stage runs it (check at finish).
- **Files:** `runtime/contracts.py` (ledger stage validation, `ledger_stages`), `tools/research_tools.py` (source ids), `prompts/planner.v22.md` step 5.
- **Regression test:** ledger entry with "correspond to known ... regimes" and no source id -> item todo; promised `control_lab` absent -> finish refused.
- **Effort:** M. **Risk:** More ledger friction; acceptable for causal claims only.

### A3-05: Exchange closures are interpolated into a trading-day series, and two runs disagree
- **Severity:** High (data treatment contrary to the prompt; affects the forecast's inputs)
- **Evidence:** Prompt A2: "A trading-day series is forecast on its observed rows: weekends and exchange holidays are not gaps to fill." Scan finding `calendar.business_days` says "do not fill w..." (WTI rep4 evidence 2). WTI rep4 ledger `coverage_gaps` (`act:d0:4`): "109 gaps in 102 runs filled by straight-line interpolation" on a B-frequency index (2607 periods vs 2498 observed). WTI rep3 answer: "Calendar: trading-day (B) frequency; 2,498 observations ... no gaps filled." Same CSV, different treatment, both reported as correct.
- **Root cause:** `pl.prepare_series` reindexes to business days by default; the scan knows the calendar is trading-days but nothing carries that into preparation; the ledger treats interpolation as the default cure.
- **Proposed fix:** When the scan's calendar finding is `business_days`, prepare on observed dates (no reindex) and record "holidays: not gaps". Add a checklist check that the prepared series' index equals the observed index for trading-day data unless the planner justifies a reindex in the ledger.
- **Files:** `prophet_lib` `prepare_series` (outside this audit's tree; report it to the owner), `runtime/contracts.py` (check), `prompts/planner.v22.md` step 6.
- **Regression test:** WTI-like CSV with weekday gaps at holidays -> prepared index equals observed dates; interpolated count 0.
- **Effort:** M. **Risk:** Some gap-based methods assume regular steps; TimesFM takes the observed series, so the risk is low.

### A3-06: Research trigger and tool family do not match; electricity and PM2.5 are handled by accident
- **Severity:** High (wasted budget; wrong sources; missing research where it matters)
- **Evidence:**
  - `runtime/goal.py:99-103` `EXTERNAL_GOAL` includes `electricity`, `energy`, `retail`, `prices?`, `weather`, `traffic`, `tariffs?`, `tourism`.
  - `tools/research_tools.py:113` and `:350`: the `family` enum is oil, gas, gold, fx, rates, equities, inflation. Electricity run, sub-agent evidence 1: first call `family: "electricity"` -> `tool.failed` ("family must be one of ...", events seq 24066). 
  - Electricity researcher then issued 12 `web_research` calls: 96 query strings, 45 distinct (53% repeats); `site:iso-ne.com load forecast` and two similar queries 6 times each; 6 of 12 calls returned 0 new sources. Its briefing: "no current regime level figures ... for any ISO/RTO region". The data are 60 anonymised clients (`kwh_daily_sum`, entity `MT_xxx`); no region is given.
  - Beijing PM2.5 (weather-driven) and retail net sales (UK, Christmas closures) did not trigger research; the trigger is goal-text keyword only.
- **Root cause:** Keyword trigger on goal wording, not on the series' drivers; a tool enum that cannot serve the triggered families; no region/provenance check before research.
- **Proposed fix:** Decide the trigger from the inspected data and the planner's driver hypothesis (one short classification step, stored), not the goal text. Give `web_research` a free-form `topic` or per-family registries (weather, electricity, retail calendars), and drop `family` as a required enum. Before research, the researcher must state the data's region or provenance; if unknown, it asks or reports "region unknown".
- **Files:** `runtime/goal.py:99-103, 423-427` (`_harness_plan`), `tools/research_tools.py:107-113, 350`, `research/primary.py`.
- **Regression test:** electricity goal without region -> no ISO/RTO queries; `family="electricity"` accepted or mapped; Beijing PM2.5 with weather hypothesis triggers research.
- **Effort:** M. **Risk:** Classification adds a step; keep it to one call.

### A3-07: Research fetch budget and passage selection lose dated events and dates
- **Severity:** High (answer says "not obtained" for an event the tool had)
- **Evidence:**
  - `tools/research_tools.py:218` `chosen = ranked[:fetch_top]`: no backfill. WTI rep4 research (evidence 10, fetch_top 6): 3 search-page failures (HTTP 403; two "no readable text") consumed the cap.
  - S6 (tier 3, date 2026-11-20, title "OPEC+ Meeting 29 November 2026: Time and What to Expect") was fetched but returned 0 passages: `research/passages.py:152` drops every passage with score 0, so the page is read and yields nothing. The answer says "Not obtained: OPEC+ Dec 2026 meeting date". The 29 November date, inside the 20-day horizon, never reaches the answer.
  - S2 (STEO) passage contains "The current STEO forecast was released November 10", yet `date: None, date_source: none`. The answer then cites "EIA STEO page, accessed 2026-11-20"; no tool output contains an access date (inferred: planner-supplied).
  - S3 (RWTCD, 2026-11-18): clipped to a table header; the same page was fetched 3 times with different queries (researcher evidence, `cached: true`), and the EIA data download link on the page was not followed.
  - `web_fetch` max_chars 60000/100000 requested by the researcher; schema caps at 20000 (`tool.failed`, WTI rep3 and rep4).
- **Root cause:** Fetch count is fixed before failures are known; passage selection is query-match-only; dates are read from metadata only; the planner is not told which pages failed in a way it can act on.
- **Proposed fix:** (a) Fetch in ranked order until `fetch_top` successful pages, with a hard cap on attempts. (b) When no passage matches, return the first 320 characters and the title as a low-score passage. (c) Extract in-text dates ("released <Month D>", "<Month D>, <YYYY>") into `date` with `date_source: text`. (d) Follow a page's own data-download link when the page holds a table. (e) Clamp requested `max_chars` to the schema and say so.
- **Files:** `tools/research_tools.py:190-241`, `research/passages.py:140-159`, `research/fetch.py` (`_date_of`), `runtime/grounding.py` (dates in answers must appear in a tool output).
- **Regression test:** fetch_top=2 with 2 failing candidates and 2 good ones -> 2 successes returned; title-only page -> passage present; "released November 10" -> date 2026-11-10 text source.
- **Effort:** M. **Risk:** More fetches per call; cap attempts at 2x fetch_top.

### A3-08: Research is not connected to the data it is meant to explain
- **Severity:** Medium-High (research decorates the answer)
- **Evidence:** Code search: `grep changepoint|level_shift|series|horizon src/prophet_agent/research/` returns nothing. WTI rep4 research ran at evidence 10, after backtest (5), TimesFM (6), calibrate (7); the forecast centre chosen at evidence 11 cites the backtest and random-walk test only. The RWTCD page's 2026-11-18 print date was not compared with the last observed value (2026-11-17, 89.53). The research briefing's scheduled events inside the horizon were not matched to the forecast window. The five detected breaks were not compared with any dated event in the sources.
- **Root cause:** The researcher's deliverable is a free-text briefing; the data side never reads it; no step asks "does a dated source match a break or a level?".
- **Proposed fix:** Add a checklist item "research reconciled with the data": for each dated event in the briefing inside or before the data window, state whether a detected break falls within ±N days; for each level quoted, compare with the last observed value and report the gap. Store the result in the ledger (stage `external_check`).
- **Files:** `runtime/contracts.py` (checklist, line ~544), `runtime/goal.py` (researcher deliverable spec, `_harness_plan`), `tools/research_tools.py` (structured `events` output).
- **Regression test:** briefing with an event 2024-08-30 and a detected break 2024-08-30 -> reconciled; a quoted level 90 vs last 89.53 -> gap printed.
- **Effort:** L. **Risk:** Matching is loose; report matches as "consistent with", not causal.

### A3-09: Premature finishes with false completion claims; the evaluator loop is the main cost
- **Severity:** Medium-High
- **Evidence:** 15 rejected of 20 finishes (measured). 7 rejected finishes claimed "All checklist items are satisfied" while items were open: retail3 evidence 18 (negative forecast, evaluator 19), WTI3 evidence 8 and 14, WTI4 evidence 12, 15, 18, Beijing 32 (open: entity coverage, evidence 33). Finish claims false in content: WTI rep4 evidence 12 asserts researcher evidence from IEA, OPEC+, CFTC that the briefing says was "not obtained". Beijing final claims "TimesFM ... for all 12 stations" and the answer body discloses "10/12 (Wanliu, Wanshouxigong missing)".
- **Root cause:** The finish rationale is free text and is not checked against the checklist. The evaluator's feedback names the failing check but the planner's next finish repeats the same template.
- **Proposed fix:** A finish is refused when its rationale contains "satisfied" or "all checklist" while any item is todo (deterministic string check, cheap). The finish must list each checklist item with its status and evidence step. The evaluator's `missing` text is passed back as a single failing item id, not prose.
- **Files:** `runtime/goal.py` (finish handling near lines 1399-1410), `runtime/contracts.py` (finish check).
- **Regression test:** finish with "All checklist items are satisfied" and an open item -> refused without calling the evaluator.
- **Effort:** S. **Risk:** Low.

### A3-10: Identical calls repeat before the stuck rule fires
- **Severity:** Medium
- **Evidence:** Identical code cells (measured, `sandbox_python` code hash): retail rep4 evidence 4-9 (six cells, same code, same output); retail3 2 repeats; retail4 7; Beijing 6; electricity 15. Identical non-sandbox calls: electricity `panel_backtest` evidence 21-23 (three identical calls after an error); electricity `web_research` 10 repeats; Beijing `inspect_data` 5 and `backtest_series` 5 (across analysts); retail4 `forecast_report` 2 identical. Beijing analysts each run "Load the saved panel" or "Load the data" 3 to 6 times at start (sub-agent evidence 1-4). Stall rescue fires only after `STUCK_CALLS = 16` idle calls (`goal.py:544`), by which time 16 calls are spent; electricity stall at evidence 16, Beijing at 22.
- **Root cause:** No cache or dedupe of identical (tool, args) with unchanged results; the stuck threshold is large relative to the 20-call part.
- **Proposed fix:** If the same (tool, args) returns the same result as before, return the earlier result with the note "same as call N; change the question or stop", without a new tool count. Lower the stuck threshold to 4 idle calls on identical signatures. Sub-agents receive the lead's stored outputs (run ids) in their brief, so they do not reload.
- **Files:** `runtime/goal.py` (`STUCK_CALLS`, `_harness_plan`, tool dispatch), `runtime/store.py` (idempotent ledger already exists; reuse it for read calls).
- **Regression test:** three identical `panel_backtest` calls -> second and third do not count against budget and return the cached result with the note.
- **Effort:** S. **Risk:** Some tools are meant to be re-run after a file change; key the cache on run ids.

### A3-11: Repeated failures are not probed; tool names are called as Python modules
- **Severity:** Medium
- **Evidence:** Retail rep4 evidence 11-14: four consecutive failures, three with `TypeError: list indices must be integers or slices, not str` before any probe (prompt rule 17: "After two failures of the same kind, probe the object"). Evidence 16-17: same `KeyError` ("no key 'outlier_pct'") twice. `ModuleNotFoundError: No module named 'backtest_series'` in retail3 and retail4 (evidence 41, 45); `AttributeError: module 'prophet_stats' has no attribute 'backtest_series'` and `'panel_backtest'` in Beijing and electricity; `ImportError: Import tabulate failed` twice in electricity (environment).
- **Root cause:** `_budget_note` (goal.py:579-590) acts on streaks of 3, not 2; the helper catalogue does not say that tools are not importable; the environment gap is not surfaced as a single fix.
- **Proposed fix:** Trigger the probe instruction at a streak of 2 of the same error class. Add to the sandbox error handler: when the module name equals a tool name, reply "backtest_series is a tool, call it with call_tool, not from Python". Report missing packages once to the planner as an environment fact.
- **Files:** `runtime/goal.py:579-590`, sandbox tool error text (`tools/sandbox_tool.py`), `prompts/planner.v22.md` step 15.
- **Regression test:** two `TypeError`s of the same class -> budget note includes the probe instruction; `import backtest_series` -> tool-name hint.
- **Effort:** S. **Risk:** Low.

### A3-12: Panel entity coverage is not a checklist item
- **Severity:** Medium
- **Evidence:** Beijing: 12 stations in the panel; `forecast.json` holds 10 (Wanliu and Wanshouxigong missing, evaluator evidence 33); calibration for 6 of 12 (evidence 33). The finish at evidence 32 says the 14-day forecasts were produced "for all 12 stations". The checklist (`contracts.py:534-566`) has no per-entity item, so the only signal was the evaluator's free-text check.
- **Root cause:** Panel coverage is never counted against the panel's entity list.
- **Proposed fix:** Checklist item "every panel entity has a forecast and a calibration" computed from `forecast_report` series names and `calibrate_intervals` calls against `inspect_data` entity list.
- **Files:** `runtime/contracts.py:534-566`, `tools/report_tool.py` (series names in result).
- **Regression test:** panel of 3 with 2 forecasts -> item todo naming the missing entity.
- **Effort:** S. **Risk:** Low.

### A3-13: No decision frame or hypotheses are recorded
- **Severity:** Medium (planning quality, the principal-analyst habit the prompt asks for)
- **Evidence:** Prompt steps 1 (frame: decision, horizon, error cost) and 4 (two or three falsifiable hypotheses in the ledger). Measured: first rationale in all six runs is an inspect step with no frame; no ledger stage named "hypotheses"; 0 of 6 runs recorded a hypothesis that a later stage confirmed or refuted. Two later steps mention "expectation" in passing (WTI rep3 step 8/11, Beijing step 11/21), not as hypotheses.
- **Root cause:** Neither frame nor hypotheses is in the checklist, so the planner treats them as optional.
- **Proposed fix:** Add checklist items "frame recorded" (a ledger stage `frame` with decision, horizon, error cost) and "two or more hypotheses recorded and each marked confirmed or refuted before finish". Ask the frame question in the first step.
- **Files:** `runtime/contracts.py` (checklist, `ledger_stages`), `prompts/planner.v22.md` steps 1 and 4.
- **Regression test:** finish without a frame stage -> refused; hypothesis left untested -> refused.
- **Effort:** S-M. **Risk:** Adds one to two cells per mission.

### A3-14: Forecast health and non-negativity are not enforced before the answer
- **Severity:** Medium (delivers flagged forecasts)
- **Evidence:** Retail rep3: forecast point -3,606 on day 9 and day-to-day swings (evaluation evidence 19); final status partial. Retail rep4: the final answer headlines a 14-day total of £1.14m and a centre of £81,418 per day while the health verdict is `flat_unexpected`; prompt step 10 says `flat_unexpected` "never reaches the answer". Retail rep4 called `forecast_report` five times (evidence 26, 42, 49, 52, 53); prompt step 12 says once. Retail rep4 total "1.14" is in `unverified`.
- **Root cause:** The health gate (`contracts.py` `delivered_health`) blocks only the finish, and a finish reached through the budget path (forced) is not checked; the non-negativity check is judgement, not a deterministic check.
- **Proposed fix:** Deterministic checks: a series with no negatives in the data must not produce negative point forecasts (clip or re-run with a stated reason); `flat_unexpected` blocks forecast_report answers unless the answer's first sentence says it is flat. Count forecast_report calls and refuse a second unless the centre changed.
- **Files:** `runtime/contracts.py` (`delivered_health`, checklist), `runtime/goal.py` (answer composition), `tools/report_tool.py`.
- **Regression test:** negative forecast on a non-negative series -> item todo; second `forecast_report` with the same centre -> refused.
- **Effort:** M. **Risk:** Clipping can bias; require the planner to state it.

## What the sources say about how strong research agents plan (for comparison)

- Anthropic's multi-agent research write-up describes a lead agent that plans, saves the plan so it survives context truncation, and gives each sub-agent "an objective, an output format" and task boundaries. It reports vague delegation caused duplicated searches. Source: https://www.anthropic.com/engineering/multi-agent-research-system (fetched).
- The same post sets effort scaling in the prompt: simple fact-finding with 3-10 tool calls, comparisons with 2-4 sub-agents of 10-15 calls, complex research with more than 10 sub-agents. Prophet's 20-call part budget is inside that range for the lead, but sub-agents are charged to the lead (A3-01), so the effective lead budget is much smaller.
- The post encodes primary-over-secondary source heuristics and a separate citation step. Prophet's tiering (`rank.py`) does this partly; the missing piece is per-claim attribution to a dated source (A3-04, A3-07, A3-08).
- The post reports about 4x tokens for agents and 15x for multi-agent systems versus chat. Prophet's electricity mission spent 91 calls; the research and panel fan-out is where the spend went, which argues for A3-01 and A3-10 before more fan-out.

Sources: https://www.anthropic.com/engineering/multi-agent-research-system; https://www.anthropic.com/news/multi-agent-research-system (search result, not fetched).

## Not verified

- The forced-finish mechanism is verified from code and snapshot counts; I did not replay a live run.
- `prophet_lib.prepare_series` was not read (outside the assigned tree); A3-05's file reference is to be confirmed by its owner.
- Whether the OPEC+ page text really lacks the 29 November date was not checked against the live page; the title in the search result is the basis.
