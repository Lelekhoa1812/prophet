# Prophet: mission and completion criteria

Status as of 2026-11-20. Every claim below is labelled with its evidence:
**[LIVE]** a real NIM or TimesFM call or a live browser or API check made in this repo, **[TEST]** an offline test in
the gate (`scripts/check.sh`), **[DOC]** a document or research note, **[TODO]** not started or not verified.

Gate at this status: `scripts/check.sh` 155 passed, 1 skipped (the real-weights test); `scripts/check.sh --real` adds
2 passed (TimesFM 3 with cached weights).

## Mission

Turn the legacy `timesfm-erp` lab into **Prophet**: a domain-independent forecasting agent that plans, calls tools
(including TimesFM 3), asks the user structured questions, produces charts and documents, and keeps its state durable
across crashes. Read-only against any ERP. The legacy lab was deleted in commit 9cf4c01 on the user's instruction of
2026-11-20. The TimesFM 3 licence position is in ARCHITECTURE.md §5.

## Completion criteria

Each criterion is done only when its check passes and its evidence is recorded in this file.

| # | Criterion | Check | Status |
|---|---|---|---|
| G1 | NIM transport implements NIM.md §4.2 row by row | `scripts/check.sh` (transport tests in `tests/prophet_agent/test_nim_transport.py`) | **[TEST]** in the gate; one test per row |
| G2 | Every configured key answers a 1-token chat call | `PYTHONPATH=src .venv/bin/python -m prophet_agent.nim.probe keys` | **[LIVE]** 6 of 6 slots 200 on 2026-11-20 |
| G3 | Each chain model answers a schema-bound claim prompt, thinking off | `... -m prophet_agent.nim.probe models` | **[LIVE]** 5 of 5 valid JSON in two runs; GLM 91–100 s |
| G4 | Same, thinking on | `... -m prophet_agent.nim.probe models --thinking` | **[LIVE]** 5 of 5 valid JSON on a trivial prompt; does not test heavy reasoning (ARCHITECTURE.md §3.1) |
| G5 | Key failover works against the live API | invalid key first; it must be retired and a real key must serve | **[LIVE]** 403 retired, real key served (2026-11-20) |
| G6 | Durable runtime: SQLite WAL event log; a crash mid-step resumes from the last completed step with no duplicate side effects | `tests/prophet_agent/test_runtime_durability.py` and `test_crash_mid_loop_resumes_without_repeating_decisions_or_tools` | **[TEST]** |
| G7 | Goal loop: planner, tool use, evaluator returns met / not met / impossible; hard budgets; stall detection | scripted-brain tests for each outcome and budget, including the stored mission status | **[TEST]** for all outcomes. **[LIVE]** met in one iteration on 2026-11-20 (mission `1a1639de…`). Two live defects fixed after the run (below) |
| G8 | Structured questions (AskUserQuestion-style) validate the schema, pause the run, and resume from the answer | validator and pause/resume tests; UI round-trip | **[TEST]** validator and pause/resume. UI round-trip verified with a scripted brain. [TODO] live NIM `ask_user` not exercised in the final pass |
| G9 | TimesFM 3 tool: forecast with quantiles and a recorded config | `scripts/check.sh --real` and a live planner mission | **[LIVE]** `google/timesfm-3.0-pytorch` on MLX; config recorded (global context 15360, context used 24, nine quantiles). Input was a synthetic series, so "real series" is not yet met |
| G10 | Sandboxed Python for analytics: subprocess with `-I`, scrubbed env, timeouts, rlimits, output manifest with sha256 | `tests/prophet_agent/sandbox/test_sandbox.py` | **[TEST]**. Labelled "guardrail, not isolation". The memory cap is not enforced on macOS |
| G11 | Charts and documents (PNG/SVG, XLSX, DOCX, PDF) produced from the sandbox and listed in a manifest | live forecast-pack mission; downloads re-hashed | **[LIVE]** 5 files from mission `1a1639de…`; each download's sha256 matched the manifest (2026-11-20) |
| G12 | Chat UI with artefact panel, Dune-meets-space identity, works at 375 px with no horizontal scroll | browser check with screenshots and console | **[LIVE]** 375 px: no horizontal overflow (scroll width 375), artefacts card renders. 1280 px: three columns, fan chart and file list render. Console not re-read after the last fix |
| G13 | TimesFM evaluation on the synthetic twin vs baselines, reported honestly against the 15 % bar | `scripts/backtest_twin.py` (written to `data/backtests/`, git-ignored) | **[LIVE]** SYNTHETIC REPLAY. The 15 % bar is **not met**: see the result below |
| G15 | Provider routing never shows a rate-limit, overload, truncation or schema error to the person: key and model rotation, quota groups, diagnosed repair | transport tests (73) and live NIM and Gemini pressure runs | **[TEST]** per branch. **[LIVE]** Gemini 3.5 Flash and NIM completed hourly-load missions with no provider error shown (2026-11-21). One live failure found and fixed: Google answers a bad key with HTTP 400, which killed a Gemini mission; the key is now retired |
| G16 | The agent works like a principal analyst: funnel, atomic breakdown, cause-led treatment, leakage-safe cleaning, deterministic checks before "met" | `docs/prophet/ANALYST_PLAYBOOK.md`; `runtime/contracts.py`; `sandbox/prophet_stats.py`; 11 skills | **[TEST]** 74 contract, 39 stats and 17 skill tests. **[LIVE]** NIM read playbooks mid-mission and repaired its own steps (r8 runs). Whether any cleaning step helps TimesFM is **unmeasured** (see below) |
| G17 | The agent can test, not assume, what helps TimesFM: a leak-free paired experiment over input treatments with a zero baseline and pinball loss | `forecast/lab.py`, `tools/lab_tool.py` (`control_lab`), `ps.causal_canary` | **[TEST]** 44 lab tests with a fake engine; 47 helper tests. **[LIVE-MEASURED]** real TimesFM 3 on six real public series (below) |
| G18 | Typed, parallel, resumable sub-agents with own notes; replay makes no model calls | `tests/prophet_agent/test_subagents.py` | **[TEST]** scoping, brief, notes, parallel replay with 0 brain calls, resume, errors, unverified flag. **[LIVE]** pending |
| G19 | Self-hosted web research (search, fetch, rank, quote-verify) available to the agent | `tests/prophet_agent/research/` (110 tests) and a live fetch | **[TEST]** offline. **[LIVE]** ddgs returned results through the bing, google and yahoo backends; `auto` often fails here |
| G20 | Prediction intervals are calibrated on held-out windows and never narrowed | `tests/prophet_agent/forecast/test_calibrate.py`; real TimesFM 3 on three real series | **[TEST]** 6. **[LIVE-MEASURED]** 80 % band coverage on held-out windows: quakes 0.762 to 0.821 (applied), CO2 0.856 to 0.867 and WTI 0.808 to 0.808 (kept raw) |
| G14 | No key or secret reaches logs, errors, transcripts or git | redaction tests plus a whole-tree scan before each commit | **[TEST]** redaction; **[LIVE]** whole-tree scan on 2026-11-20 (below) |

### Evidence for G13 (synthetic replay, 2026-11-20)

- Data: the synthetic generator, seed 20261120, 12 entities, 137 weekly series, three years of history, a 4-week
  horizon. History is the recorded sales (censored by stockouts, as a planner would see them). The future four weeks
  are the generator's true demand.
- Method: walk-forward, 26 origins (131 to 156). Head set = series carrying 80 % of volume (54 series).
- Head-set WAPE: TimesFM 3 **0.908**; best baseline `moving_average` 0.976; `seasonal_naive` 1.342; `seasonal_year` 1.616.
- Improvement over the best baseline: **+6.9 %**. Improvement over the per-series oracle: **−10.0 %**.
- The 15 % bar is **not met**. TimesFM 3 under-forecasts on this set (bias −0.25 across all series). The 80 % band
  covers 81.7 % of head-set actuals, close to nominal.
- Output: `data/backtests/twin_backtest_20261120-205503.json`. Not committed (`data/` is git-ignored).

### Evidence for G15 and G16 (2026-11-21; live runs are real public series through the real API, forecasts by TimesFM 3)

- Hourly UCI electricity load, NIM: 8 windows of 48 h, TimesFM MASE 1.385 against seasonal naive 2.779, WAPE 5.74 % against 11.54 %,
  80 % band held 75 % of the time. Gemini 3.5 Flash earlier gave the same MASE figures but missed the clock-change defects that
  NIM found (three spring-forward rows), with only 3 tool calls. That depth gap is what planner.v11, the skills tool and the
  finish gate target.
- Health check calibration on the daily CO2 record (13 holdouts, 60-day horizon, full history, TimesFM 3): the shape ratio read
  flat in 7 of 13, yet TimesFM's MAE was below seasonal naive's in 13 of 13. A window of a sixth of a cycle cannot show shape.
  The verdict is now "check" for horizons under a quarter of the cycle. Before the fix a NIM mission looped on a false
  flat_unexpected and ended partial.
- The finish gate sends a forecast with no baseline backtest or no prior data inspection back to the planner (twice at most), then
  labels the answer partial. It is covered by three goal-loop tests.
- Live pressure matrix on the final code (Gemini 3.5 Flash, real public series, real TimesFM 3, no provider error shown to the
  person in any run): hourly load, retail units and WTI price all ended `done_verified` with the ledger gate passed (elapsed
  148 s, 124 s, 253 s). Daily CO2 on Gemini ended `done_verified` earlier in the session (161 s). NIM completed load, CO2 and WTI
  as `partial_unverified`: the answers were sound, the label came from derived figures and, before the fix below, a ledger count
  that ignored a cell that failed after saving. NIM is slow: 898 s, 1183 s and one retail run still going at the 1800 s harness limit.
- Three routing defects found only by running live, each now a failing test first: all keys resting past one attempt's budget raised a
  failure (three Gemini missions died); Google's HTTP 400 for a bad key was blamed on the model (one mission died); keys of an
  account-wide quota were paced separately.
- Open: NIM planner efficiency (20+ analysis cells before the backtest on retail). The finish gate and ledger make the work thorough;
  they do not make it short. Candidate fix: a cell budget per funnel stage.
- Grounding now accepts a figure marked as approximate ("about", "~", "roughly") within one percent of a printed figure. A derived
  figure that is not marked, and not printed, still leaves the answer partial; planner rule 57 tells the planner to print it.
- Open: no measurement yet of whether smoothing, outlier replacement or imputation improves a TimesFM backtest. The
  research found none. Each step must earn its place on the synthetic twin with a no-cleaning arm.

### Evidence for G17 (measured, real TimesFM 3 on real public series, 2026-11-22)

Method: `control_lab`, 8 rolling origins per series, origins two horizons apart, last 700 points, MASE and pinball loss paired
against the default, bootstrap 90 % interval on folds. Adopt needs MASE gain of at least 2 %, a win rate of at least 60 %, an
interval above zero and no worse pinball loss. Each series had 5 to 8 treatments tested on the same folds.

- Retail net sales (h14), COVID cases (h14), quake counts (h14), CO2 (h30), pollen (h14): **no treatment adopted**. Replacing
  or ignoring flagged outliers never beat the default. Shortening the context (2x, 4x, 8x horizon, one season) was
  **harmful** on retail, CO2, pollen and quakes (2x only), and never helped. z-normalisation was harmful on CO2 and pollen.
- WTI spot (h20): **symmetric averaging adopted** (7 of 8 wins). One adoption in about 40 treatment tests is what chance alone
  would give at these thresholds, so treat it as a lead to re-check on newer data, not a rule.
- Reading: on these series the default TimesFM input was never beaten by cleaning. This is the first measurement of the
  question the research could not answer; it is six series, 8 folds each, so it does not show that cleaning never helps.
  Thresholds (2 % gain, 60 % wins) are proposals to calibrate.
- Reproduce: scratch script in the session scratchpad; results in `data/lab_real_results.json` (git-ignored).

### Evidence for G15 to G17, live round 2 and 3 (real public series through the real API, 2026-11-22)

Same four scenarios on the build before and after the round-2 fixes (sandbox carry, tolerant scan, progress-gated auto-continue,
`Pair`, batched compaction, `controls_tested`, planner.v13). Wall time and tool calls are as the mission reported them.

| Scenario, planner | Before | After |
| --- | --- | --- |
| CO2 daily (h30), NIM | checkpoint after two auto-continues, about 2,400 s, about 50 tools, empty answer, same crash repeated | `partial_unverified`, 351 s, 19 tools, checklist 6 of 6, answer with scan findings, ledger and backtest (unverified: three derived figures) |
| WTI (h20), NIM | not run | `partial_unverified`, 451 s, 15 tools, 6 of 6 (one derived interval half-width unverified) |
| Quakes (h14), NIM | not run | `done_verified`, 491 s, 17 tools, 6 of 6 |
| Retail (h14), NIM | `partial_unverified`, 1,162 s, 39 tools, four unverified figures | `partial_unverified`, 711 s, 16 tools, 7 of 7 (one derived figure unverified) |
| Retail, Gemini | `done_verified`, 50 to 130 s | `done_verified`, 110 s, 9 tools |

Reading: the missions finish in minutes with the checklist complete. The remaining `partial_unverified` results come from the
grounding gate refusing a figure the model derived itself (a half-width, a ratio); the gate is working as designed, and the answer
says so. Gemini finishes fastest with the fewest analysis cells (6 to 9 tools); it did not call `control_lab`, which is correct
when it forecasts with the default. Six series on two providers is a small sample; no run here shows the agent matches a human
analyst.

### Evidence, live round 3 (real public series through the real API, TimesFM 3, 2026-11-22)

Three missions on the code with auto-carry and planner.v17 (WTI 20 trading days, EUR/JPY 30 business days, Beijing PM2.5
for 12 stations). Measured from the ledger:

- Sandbox cells: 30 of 71 failed (42 %). None failed on a missing carried file (round 2: about 15 of 24 WTI failures).
  Leaders: change-point field names (`mean_before`, `jump`, `at`) 13, `TimedeltaIndex.abs()` 6, `report['outliers']`
  after `ps.hampel` 5. All three now work (aliases, flags reachable from the report, `abs` added).
- WTI resent one broken line 8 times between other calls; the in-a-row counter never reached the fresh-analyst hand-over.
  Recurrence within the last 10 calls of the tool now also counts.
- Harness delegation fired in all three: a researcher for WTI and EUR/JPY, four parallel panel analysts for Beijing.
- Outcomes: none finished with a forecast. Beijing's analysts spent the part's allowance (20 calls) before the lead
  backtested; EUR/JPY ticked 8 of 8 checklist items and was stopped by the runtime, which counted the researcher's calls,
  before the lead knew it was out; WTI spent its calls on analysis cells. Harness sub-agents now get their own allowance
  and a short-calls note names the open items.
- Variance-ratio test on the bundled files (measured): WTI and EUR/JPY consistent with a random walk (max |z| 1.26 and
  1.86 against 2.49), daily CO2 and quake magnitudes mean-reverting (max |z| 10.1 and 23.6). It now appears in every
  backtest and in the computed accuracy paragraph.

### Evidence, live round 4 (same three missions, 2026-11-22)

- Sandbox cells: 11 of 63 failed (17 %, from 42 % in round 3). WTI ran 4 cells with none failing.
- WTI and EUR/JPY each delivered a pack with a naive centre line and TimesFM's band, after a backtest and calibration
  (WTI checklist 7 of 7, EUR/JPY 7 of 8). Both still ended `partial`:
  - WTI: the treatment check demanded a control_lab test of `ignore_ranges`, which control_lab does not accept; the
    planner re-sent the pack three times and stall detection stopped it. A configured backtest_series (which scores
    default TimesFM on the same windows) now counts as that measurement.
  - EUR/JPY: every ledger cell failed on `report['outlier_indices']` (4) or `finding['title']` (2), so no ledger was saved.
    Both names now resolve.
- Beijing: the harness's four panel analysts ran inside their own allowance (the fix worked), but the lead re-ran
  profiling 9 times because it saved scans per station (`out/scan_<station>.json`) and the "scan saved" item accepted only
  `out/scan.json`. Per-entity scan files now count.
- Two missions asked for the playbook `profiling_a_new_dataset` (the tool description's own words); names now resolve to
  the nearest playbook.

### Evidence, live rounds 5 to 8 (real public series through the real API, TimesFM 3, 2026-11-22)

| Round | WTI | EUR/JPY | Beijing (12 stations) | Retail | Sandbox cells failed |
|---|---|---|---|---|---|
| 5 | completed, verified (19 calls) | partial: two figures no tool printed | checkpoint, 40 of 97 cells failed | not run | 52 of 129 |
| 6 | partial: stalled on a refused `centre` | partial: the removal note re-inserted 175.3 | checkpoint, profiling loop | not run | 17 of 82 |
| 7 | completed, verified (27 calls) | partial: three derived percentages | checkpoint, 2 of 74 failed, ledger never written | not run | 13 of 107 |
| 8 | not run | partial: two derived figures | checklist 7 of 7, 3 of 12 stations packed before the budget ran out | checkpoint: 17 pack builds held by TimesFM's flat line | 34 of 94 |

What each round changed is in the commit log. The measured pattern: market series now finish in about 20 calls with a naive
centre line, TimesFM's calibrated band, primary-source context and the variance-ratio finding; "partial" there is now
always an answer figure that no tool printed, which is the grounding check working. Panels are the open problem: the
pack was one series per call (a panel pack is being added), and the retail run showed that a rule-centred pack was held
by the health of TimesFM's own line, which the pack did not deliver (fixed: health is judged on the delivered line).

### Evidence, live rounds 9 to 13 (real public series through the real API, TimesFM 3, 2026-11-22)

| Round | WTI | EUR/JPY | Beijing (12 stations) | Retail | Quakes | Mauna Loa CO2 |
|---|---|---|---|---|---|---|
| 9 | completed, verified (16 calls) | not run | completed, verified: one panel pack, 12 stations (58 calls) | checkpoint: carried a forecast.json the pack did not write | not run | not run |
| 10 | not run | partial: evaluator asked for paid consensus data | not run | partial: quoted a 14-day total no tool printed | completed, verified (20 calls) | not run |
| 11 | not run | completed, verified (34 calls) | not run | partial: stopped by the first stall at 11 calls | not run | completed, verified (12 calls) |
| 12 | partial: one figure (0.9999) | not run | completed, verified (56 calls) | failed: replay diverged on resume | not run | not run |
| 13 | completed, verified (21 calls) | partial: two outside figures no page held | not run | partial: pack delivered after two resumes, ten derived figures unprinted | not run | not run |

Every scenario except retail has completed with every figure verified at least once; WTI four times. Each "partial"
since round 9 is the grounding check refusing a figure that no tool or fetched page printed, which is the check doing
its job, except round 12's retail failure: a resumed run took a different path than the stored one because the harness's
choice depended on the call allowance. That is fixed (the loop follows the step the ledger holds at each position) and
round 13's retail resumed twice without diverging. Sandbox cells failed: 13 of 107 in round 7, 31 of 82 in round 13 (27
of them in retail, 12 on `if report['outliers']:`, which now reads as "any outlier").

### Evidence, live rounds 14 to 16 (retail grounding, real UCI Online Retail II daily sales through the real API, TimesFM 3, 2026-11-22)

Measured on one mission per round; a single run is not a rate.

| Round | Change under test | Retail result | Calls | Seconds |
|---|---|---|---|---|
| 13 | none (baseline) | partial: 10 figures unmatched (range totals, "27.9k"-style means, rounded stats) | 73 | 1,883 |
| 14 | `pl.result` helper; scaled figures ("27.9k") matched to the printed value; `planner.v19`, `composer.v5` | completed, verified | 14 | 280 |
| 15 | pack prints `total_range_80`; `planner.v20`, `composer.v6` | partial: four figures were copying slips (557,050 for the printed 557,051.21; 352,702 for 352,602.3; 849,585 for 843,784; one more) | 23 | 481 |
| 16 | large-figure copying slips corrected to the single printed value within 0.05 percent, with a note in the answer | completed, verified | 9 | 140 |

- WTI re-run in round 15 as the regression check: completed, verified, 31 calls.
- What each change fixes: the planner derived totals, shares and ratios in its head and the grounding check refused them
  (right to). Printing them with `pl.result` makes them evidence. A person writing "27.9k" for a printed 27,912 is rounding,
  not inventing, so the check reads scale words. The planner also summed the per-day q10 and q90 into a range for the 14-day
  total (329,049 to 851,044); that is the widest case, not the range. The pack now prints both bounds (days miss
  independently, days miss together) and the answer states both.
- What it does not prove: one passing run per round. Round 15 shows the composer can still miscopy a printed figure; round 16
  passed after the snap correction, which only repairs slips under 0.05 percent on figures of 1,000 or more that have a
  single printed non-integer neighbour. A figure that is merely unprinted is still refused and still makes the answer partial.
- The independent-days bound assumes a normal shape per day and zero correlation between days; the together bound assumes
  perfect correlation. The true range lies between. Neither has been calibrated against backtested totals.

### Pre-existing helpers that are not causal (found with `ps.causal_canary`, 2026-11-22)

`ps.changepoints` and `ps.isolation_outliers` are fitted on every row passed, so a flag at time t depends on later rows. Their
reports now carry `diagnostic_only` and a leak warning. The earlier claim that `ps` is causal by default holds for hampel,
smooth, impute, derivative_features and the new helpers only.

### Evidence for the live planner defects fixed on 2026-11-20

- Stored mission status followed the goal outcome, not "completed" for every finished program. Covered by a test that
  asserts the mission row reads `stalled`.
- The goal and the tool list were lost once the evidence grew past the observation budget, because the JSON was key-sorted
  and clipped as one string. The goal now sits in its own section, and each observation is clipped on its own. Covered
  by a regression test.
- The planner now sees each tool's argument names. Before, it passed the forecast output as `forecast` to the report
  tool.

### Evidence for G14 (whole-tree scan, 2026-11-20)

- No value from the git-ignored `.env` appears in any file to be committed (count 0).
- `NIM.md` (kept out of the commit) contains no key-shaped string.
- Four test files contain key-shaped placeholders such as `nvapi-LEAKED…` and `nvapi-AAAA…`, used to prove redaction.
  They are not live keys.

## Working rules for this mission

- Use Haiku sub-agents for research workers and bounded reviews; keep design and integration in the main thread.
- No access to the user's SharePoint (standing instruction, 2026-11-20).
- Keep the flow going until everything is shipped and pushed end to end. Use the question tool only for decisions the
  user must make (standing instruction, 2026-11-20).
- Push to `origin/main` after each green gate. The user's 2026-11-20 instruction to ship and push end to end is the
  authorisation.
- Label every figure as synthetic replay, real staging, or measured. Never present synthetic numbers as observed results.
