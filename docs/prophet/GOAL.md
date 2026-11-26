# Prophet: mission and completion criteria

Status as of 2026-11-20; test counts re-collected on 2026-11-23. Every claim below is labelled with its evidence:
**[LIVE]** a real NIM or TimesFM call or a live browser or API check made in this repo, **[TEST]** an offline test in
the gate (`scripts/check.sh`), **[DOC]** a document or research note, **[TODO]** not started or not verified.

Collected on 2026-11-23 with `pytest --collect-only -q`: `scripts/check.sh` (`-m "not timesfm"`) collects 1997 tests (1972 before the r6l tests were added), and
`scripts/check.sh --real` adds the 4 `timesfm`-marked tests. The 2026-11-20 result of 155 passed and 1 skipped has not been
re-run; the next full gate run records the current pass and skip split.

## Mission

Turn the legacy `timesfm-erp` lab into **Prophet**: a domain-independent forecasting agent that plans, calls tools
(including TimesFM 3), asks the user structured questions, produces charts and documents, and keeps its state durable
across crashes. Read-only against any ERP. The legacy lab was deleted in commit 9cf4c01 on the user's instruction of
2026-11-20. The TimesFM 3 licence position is in ARCHITECTURE.md §5.

## Locked decision: licence wording lives only in documents (2026-11-22)

The owner decided, permanently, that licence and research-use wording (for example "research use only", "non-commercial")
appears only in Markdown documents: README.md, NOTICE.md, CLAUDE.md and docs/. It must not appear in UI components,
agent prompts, skills, contracts, API responses, tools or code strings. Anyone who clones the project reads the README
themselves. Enforced by `tests/prophet_agent/test_licence_wording_guard.py`, which scans the app and fails on any hit.
This settles the conflict recorded on 2026-11-22 (evaluator wording against CLAUDE.md): the evaluator prompt no longer
mentions licence terms. The legal position itself is unchanged and stays in ARCHITECTURE.md §5.

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
| G15 | Provider routing never shows a rate-limit, overload, truncation or schema error to the person: key and model rotation, quota groups, diagnosed repair | transport tests (75 in `test_nim_transport.py`, collected 2026-11-23) and live NIM and Gemini pressure runs | **[TEST]** per branch. **[LIVE]** Gemini 3.5 Flash and NIM completed hourly-load missions with no provider error shown (2026-11-21). One live failure found and fixed: Google answers a bad key with HTTP 400, which killed a Gemini mission; the key is now retired |
| G16 | The agent works like a principal analyst: funnel, atomic breakdown, cause-led treatment, leakage-safe cleaning, deterministic checks before "met" | `docs/prophet/ANALYST_PLAYBOOK.md`; `runtime/contracts.py`; `sandbox/prophet_stats.py`; 18 skill playbooks | **[TEST]** 87 contract tests (`test_contracts.py`), 88 stats tests (`sandbox/test_prophet_stats.py`) and 19 skill-tool tests (`tools/test_skill_tool.py`), collected 2026-11-23. **[LIVE]** NIM read playbooks mid-mission and repaired its own steps (r8 runs). Whether any cleaning step helps TimesFM is **unmeasured** (see below) |
| G17 | The agent can test, not assume, what helps TimesFM: a leak-free paired experiment over input treatments with a zero baseline and pinball loss | `forecast/lab.py`, `tools/lab_tool.py` (`control_lab`), `ps.causal_canary` | **[TEST]** 48 lab tests with a fake engine (`forecast/test_lab.py`, collected 2026-11-23). The "47 helper tests" figure from 2026-11-20 could not be matched to a file and is removed. **[LIVE-MEASURED]** real TimesFM 3 on six real public series (below) |
| G18 | Typed, parallel, resumable sub-agents with own notes; replay makes no model calls | `tests/prophet_agent/test_subagents.py` | **[TEST]** scoping, brief, notes, parallel replay with 0 brain calls, resume, errors, unverified flag. **[LIVE]** pending |
| G19 | Self-hosted web research (search, fetch, rank, quote-verify) available to the agent | `tests/prophet_agent/research/` (238 tests, collected 2026-11-23) and a live fetch | **[TEST]** offline. **[LIVE]** ddgs returned results through the bing, google and yahoo backends; `auto` often fails here |
| G20 | Prediction intervals are calibrated on held-out windows and never narrowed | `tests/prophet_agent/forecast/test_calibrate.py`; real TimesFM 3 on three real series | **[TEST]** 17 (collected 2026-11-23). **[LIVE-MEASURED]** 80 % band coverage on held-out windows: quakes 0.762 to 0.821 (applied), CO2 0.856 to 0.867 and WTI 0.808 to 0.808 (kept raw) |
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
- `NIM.md` is tracked in the repo (owner decision: keep it tracked) and contains no key-shaped string.
- Seven test files contain key-shaped placeholders such as `nvapi-LEAKED…` and `nvapi-AAAA…`, used to prove redaction.
  They are not live keys.

## Working rules for this mission

- Use Haiku sub-agents for research workers and bounded reviews; keep design and integration in the main thread.
- No access to the user's SharePoint (standing instruction, 2026-11-20).
- Keep the flow going until everything is shipped and pushed end to end. Use the question tool only for decisions the
  user must make (standing instruction, 2026-11-20).
- Push to `origin/main` after each green gate. The user's 2026-11-20 instruction to ship and push end to end is the
  authorisation.
- Label every figure as synthetic replay, real staging, or measured. Never present synthetic numbers as observed results.

## Evidence, repeat runs of retail and WTI (2026-11-22)

Three repeats of each live scenario, run in sequence against the API (127.0.0.1:8010) with the same prompts and sandbox
code. Each run wrote `data/live_<scenario>_None_rep<n>.json`. Labelled **measured** (real NIM and TimesFM calls, not
synthetic). Three runs per scenario is a small sample, not a rate.

| Run | Scenario | Status | Stop reason | Seconds | Tool calls | Unverified figures |
|---|---|---|---|---|---|---|
| rep 1 | retail | completed | done_verified | 390 | 14 | none |
| rep 1 | WTI | partial | partial_unverified | 921 | 32 | `886,887`, `91.23` |
| rep 2 | retail | completed | done_verified | 541 | 19 | none |
| rep 2 | WTI | completed | done_verified | 340 | 14 | none |
| rep 3 | retail | completed | done_verified | 1042 | 45 | none |
| rep 3 | WTI | partial | partial_unverified | 721 | 23 | `0.001`, `12.5` |

Read-out: retail completed 3 of 3; WTI completed 1 of 3, and both partial WTI runs stopped on figures that no tool result
contained. Two runs (retail rep 3, WTI rep 1) also used one auto-continue.

Code state during the repeats. Sandbox files (`prophet_*.py`) were read from disk at each run, so runs after the tranche 1
commit (`14f8a9b`) used it. The API process held the code loaded at its start, so the tranche 2 and 3 fixes below are not
in these numbers. Prompts and skills were read on each call; the licence scrub landed after the last run.

## Tranche status after the six-dimension audit (2026-11-22)

Fixed, each with a regression test (see [AUDIT_2026-11-22.md](AUDIT_2026-11-22.md) for the full table):
items 1 (sub-agent answers are not evidence), 2, 3, 5, 6, 12 (tranche 1); 4 (baseline gate covers the delivered horizon and
blocks a TimesFM loss on its own centre), 7 (call count includes sub-agents), 8 (outside-context heading at any level),
9 (research trigger covers retail, prices, electricity, weather), 11 (truncated free text is continued), 19, 20, 22
(wording).

Fixed in the second audit tranche (2026-11-22), each with a regression test: items 13 (numeric store IDs are entity keys),
14 (lumpy band brackets the TSB expected-demand point), 15 and 16 (calibration labelled optimistic; Wilson on windows), 17
(scan blockers and majors need a ledger decision), 18 (the treatment gate needs a lab on this horizon and length that tested
the treatment), 21 (centre provenance wording), 23 (catalogue and holiday defaults), 24 (sub-agent step budget from the type).

Behaviour changes to check in live runs: sub-agent step budgets fall (general 14 to 12, verifier 14 to 10); hand-tuned
forecasts now need a `control_lab` on the same horizon and length; blocker and major scan findings need a ledger decision.
None of items 14, 17, 18 or 24 has been measured in a live run. A live run needs a restarted API and real NIM and TimesFM calls.

Open: item 10 (the plan horizon is not compared with the requested horizon; needs the series step recorded in the forecast
record), and the build items (MinT reconciliation, the equal-weight blend as a centre rule, and a live `panel_backtest` on a
real panel).

Fixed (2026-11-22, `fx-clean-text`, commit 5e9ad9d): a continuation keeps the one real space at a word-boundary cut and the
line break at a line-boundary cut. A mid-word cut still joins directly. Not verified against the live model's continuation
spacing, which needs a live run that cuts a reply.

Licence wording: locked as a permanent rule (2026-11-22). Licence and research-use wording lives only in Markdown documents;
`tests/prophet_agent/test_licence_wording_guard.py` enforces it. The prompt and skill scrub is done. `evaluator.v10` is the
current evaluator prompt, and the older evaluator versions and two skills no longer carry the wording.

## Round 3 and 4 (2026-11-22 to 2026-11-23)

Round 3 merged eight slices (`PLAN_2026-11-22_round3.md`) and the person's decisions R1 to R4: questions only for
high-risk decisions with a `high-risk:` rationale and no fixed count (replaces MAX_ASKS = 8); per-notebook data sharing;
deterministic context editing (stubs, the model writes the memo only at part boundaries); `.env` unchanged. Round 4
merged six slices (`PLAN_2026-11-23_round4.md`). Offline gate after round 4: 1,760 passed, 1 skipped, 0 failed.

Gate decision (D3): the panel cost guard counts model fits and training rows instead of time. A wall-clock bound and then a
single-thread CPU bound both failed only under concurrent load (11.7 to 275 s wall, 44 to 65 s CPU) and passed alone.

### Live round on the round-3 build (measured, real NIM and TimesFM 3 through the API, 2026-11-23)

Six round-4 builders ran during the panel missions (load average 30 to 70), so those timings are under load.

| Mission | Status | Stop reason | Seconds | Tool calls | Questions | Model compactions |
|---|---|---|---|---|---|---|
| retail rep 5 | partial | partial_unverified | 676 | 22 | 0 | 0 |
| WTI rep 5 | completed | done_verified | 931 | 40 | 0 | 0 |
| Beijing panel rep 3 | partial | stalled | 1,006 | 74 | 0 | 0 |
| electricity panel rep 3 | partial | stalled | 3,318 | 92 | 0 | 0 |

Earlier build for comparison: retail rep 4 took 1,683 s and 49 calls (partial); electricity rep 2 stopped at a checkpoint
after 5,406 s and 93 calls; Beijing rep 2 ended partial on an unrecorded scan decision.

Root causes found in these runs and fixed (each with a regression test):

- Retail: the answer stated a "likely range" of 547k to 867k that no tool printed (the pack printed 606,217 to 808,160 with
  days independent and 345,811 to 1,098,329 with days together). Grounding stopped it correctly. Planner v25 tells the
  model to quote the pack's `table` and `headline` (A7-02, A7-03 built in round 4).
- Panels: `read_observation` paged a summary whose middle the sandbox had already cut, and sub-agents paged the lead's
  history instead of their own. Analysts re-ran cells to see the cut output until stall detection stopped the mission. The
  full result is now kept beside its summary and paged; each sub-agent has its own pager; grounding reads the full result.
- Beijing: the level-shift text stated the ten-times unit-change rule without the measured ratio, and the answer reported
  "median jumps of 10x or more" for breaks of 1.05 to 1.34 times, then stopped without a forecast. The finding now leads
  with the measured ratio, and v25 says a major finding never stops the forecast by itself.

Observed working as designed: no questions were asked in any mission (R1), no model-written compaction between part
boundaries (R3), retail called `ps.changepoints(..., period=7)` (planner v24), WTI completed verified.

TimesFM 3 joint decoding: measured and not offered; see `EXPERIMENT_timesfm_joint_decoding.md`.

### Live round on the round-4 build (measured, real NIM and TimesFM 3 through the API, 2026-11-23)

The API process started at commit 8d4d433, so 7441589, d899d1c and 77de3ae (lab `ignore_ranges`, the `flat_expected`
headline, `band_source`) were not loaded in this round. Tool calls count the lead's and the sub-agents' together.

| Mission | Status | Stop reason | Seconds | Tool calls | Questions |
|---|---|---|---|---|---|
| retail rep 6 | completed | done_verified | 360 | 13 | 0 |
| WTI rep 6 | completed | done_verified | 736 | 40 | 0 |
| Beijing panel rep 4 | partial | stalled | 691 | 67 | 0 |
| electricity panel rep 4 | failed | NIM call out of time | 901 | 60 | 0 |

Against the round-3 build: retail went from partial (676 s, 22 calls) to verified (360 s, 13 calls); WTI stayed verified
and ran faster (931 s to 736 s). Beijing built a forecast pack this time (rep 3 stopped with none), but for 3 of 12
stations. Retail rep 6 quoted the pack's total and range exactly as printed.

Root causes found in these runs and fixed (round 5, each with a regression test):

- Electricity: at 899 s of the 900 s allowance a planner call started with a 1 s budget (round-4 A5-06 floored the cap at
  1 s); the transport ran out at once and the mission ended `failed` after 49 delegated calls, with nothing kept for the
  person. A call with under 45 s left, or one cut short by the allowance, is now a checkpoint the person continues.
- Electricity: three calls named files their runs never wrote, one with a run id no tool had returned. The error now says
  the run does not exist, or lists the files it did write.
- Beijing: one analyst's file of 3 stations fed every backtest and the pack. The panel check took its expected count from
  those results (3 of 3); `inspect_data` showed 12. The expected count now comes from the data's grouping column too,
  any short result carries a `coverage_note` with the fix, and a repeated call carries it. After the evaluator refused
  the finish for 3 of 12, the same backtest had been sent three more times and the run stalled.
- All missions: `contracts.check_answer` (C3a, A6-09) had tests but no caller, so no answer claim was ever checked live.
  It now runs after the answer is written; a blocking claim goes back to the composer once, then the answer leaves
  partial. Replayed over the 169 stored live answers: 9 would have blocked at first, 6 of them reader misreads (fixed:
  "365 days" against 364, a horizon read as a cycle, "105 cycles", `season=252`, "best" matched to a distant rule). The
  3 left are true mismatches, including the two A6-09 cases. New check: a total the pack withheld is not stated.
- The pack's total range came as two bounds (days independent, days together). The backtest now measures the variance
  ratio R of the horizon total's standardised error (leave-one-out for its own check), and the computed accuracy
  paragraph gives one 80 % range from it. Synthetic check: a random walk gives R of 5.4 to 5.8 at horizon 8; independent
  noise gives 1.1 to 1.3.

### Live round on the round-5 build, and the first business-user missions (measured, 2026-11-23)

| Mission | Status | Stop reason | Seconds | Tool calls | Questions |
|---|---|---|---|---|---|
| Beijing panel rep 5 | partial | could not close a gap | 811 | 83 | 0 |
| electricity panel rep 5 | partial | stalled | 1,352 | 100 | 0 |
| retail, business user | partial | figures unmatched | 435 | 29 | 0 |
| WTI, business user | partial | could not close a gap | 698 | 35 | 1 card (2 questions) |

Beijing forecast all 12 stations (rep 4: 3) and electricity answered (rep 4: failed). The business-user goals were written as
a person would ask, with no method words: "Sales feel like they've been slipping lately. Is that real, or just the usual
ups and downs? And what should I expect for next month?" and "I need a number for next month's fuel budget. What will
crude cost on average next month, and how sure are you? Keep it short, I'm not a data person."

Root causes and fixes (each with a regression test):

- Beijing: `scan_decisions_recorded` named three undecided level shifts; the planner read the scan six times, read the
  ledger expecting a dict (it is a list), and finished three more times. The check now gives the cell to run.
- Electricity: a list in `source.name` crashed ("unhashable type: 'list'"); one client's pack (MT_001, 1,390) was
  answered as the 60-client total; the stall rescue gave a backtest item to an analyst with no backtest tool;
  `panel_backtest` did not tick the baseline item. Each is fixed (subset by list, single-entity coverage note, modeler
  rescue with `panel_backtest`, panel backtest counts).
- Retail business user: the answer was a forecast report and never said whether the slip was real. The questions in the
  goal now reach the evaluator (met only when each is answered from the evidence), the composer and the planner. It also
  named "seasonal naïve" as best (the accent hid it from the benchmark check), and the code-appended measured total range
  was flagged as unverified; both fixed.
- WTI business user: asked one card (which month, how many barrels; the second is not needed for an average price). The
  person answered "All of December 2026"; the pack ran 21 trading days to 16 December. A calendar month named in the goal
  or an answer, or "next month", is now a span the pack must reach (block). The answer averaged the daily bounds by hand
  ($78.90 to $103.42, unmatched); the pack and the measured range now carry the per-step mean and its range.
- The measured range of the total held the actual total in 7 of 12 windows on retail (target about 10 of 12): it is
  reported as measured, and its own check says so in the answer.

The live driver no longer calls answer-now on an open question (the API answers it with 409, by design: answer-now is a
checkpoint action). It answers the card as the person would.

### Live reruns on the round-5 fixes (measured, 2026-11-23)

| Mission | Build | Status | Stop reason | Seconds | Tool calls |
|---|---|---|---|---|---|
| retail, business user rep 2 | 20738c5 | completed | done_verified | 667 | 45 |
| WTI, business user rep 2 | 20738c5 | completed | done_verified | 425 | 25 |
| Beijing panel rep 6 | 20738c5 | partial | stalled | 1,794 | 112 |
| electricity panel rep 6 | 20738c5 | partial | stalled | 1,063 | 72 |
| Beijing panel rep 7 | f4dae02 | **completed** | **done_verified** | 1,247 | 95 |
| electricity panel rep 7 | f4dae02 | failed | NIM stream cut in a sub-agent | 685 | 64 |
| retail, business user rep 3 | c5e5c66 | partial | figures unmatched | 688 | 42 |
| electricity panel rep 8 | c5e5c66 | partial | could not close a gap | 2,330 | 138 |

Beijing rep 7 is the first panel mission to finish verified: all 12 stations, each with a verdict (panel MASE 0.736 for
TimesFM against 1.018 to 1.431 for the simple rules). Electricity rep 8 survived two time checkpoints by itself.

Root causes and fixes in these runs:

- Rep 6: the coverage note never fired. `data_entity_count` read the full `inspect_data` profile, but the history holds
  the compact view; its tests had used the full shape. Fixed and replayed on the stored histories (fires at Beijing step
  36, 3 of 12, and electricity step 10, 15 of 60; silent on Beijing rep 5, which covered all 12). Carrying several
  analysts' `out/scan.json` failed five times on a name clash; the second file is now mounted under its run's prefix.
- Electricity rep 7: "nim stream cut 4 times" inside one analyst failed the mission. Any NIM error other than missing keys
  is now a `model_service` checkpoint; nothing after the failed call was stored, so continuing replays and retries it.
- Retail business user rep 2 answered "The slip is real: the model captures a downward drift". A forecast's slope is not
  evidence of a change. New `ps.change_test` ranks the latest change among every earlier change at the same lag, by date.
  Measured on the real retail data: the last 28 days are +10.3 % against the same weeks a year earlier and 48 % of earlier
  changes were as large (p = 0.48), so the live answer was wrong. Counted in rows instead of dates, 364 trading days reach
  back 14 months and read "+113 %, real rise", which is why it takes dates. Rep 3, with the helper in the planner's hands,
  answered "Is the slip real? No: +12 % year on year, p = 0.312".
- Retail rep 3 stayed partial: the backtest ran 20 steps and the pack 43, so no measured range applied and the answer
  mixed the two bounds. The measured ratio now carries to another horizon at the same average correlation between days.
- Electricity rep 8: `forecast_report` refused 60 series (panel cap 50) and crashed on a per-series centre dict with
  `seasonal_naive` (KeyError). The cap is 100 and the dict builds. Open, in build: delivering the pooled model's forecast
  (`panel_backtest` can rank it best) and a total across the panel's entities with its range.

### Business users and panels after the pooled centres (measured, 2026-11-23)

| Mission | Build | Status | Stop reason | Seconds | Tool calls |
|---|---|---|---|---|---|
| electricity panel rep 9 | bc5d421 | partial | could not close a gap | 1,202 | 119 |
| retail, business user rep 4 | bc5d421 | partial | stalled | 225 | 20 |
| retail, business user rep 5 | 2bf706d | partial | scan decisions unrecorded | 390 | 21 |
| electricity panel rep 10 | bc5d421 API, 2bf706d tree | **completed** | **done_verified** | 1,532 | 103 |

Electricity rep 10 is the first electricity mission to finish verified: all 60 clients, a pooled-model panel backtest
(TimesFM and the pooled blend tied at mean MASE 1.70, sign-flip p = 0.80; seasonal naive 1.93) and a 7-day panel total.
The API ran bc5d421 for the goal loop; sandbox cells import helpers from disk on each run, so they ran the newer tree.

Root causes and fixes:

- Retail rep 4 asked "what should I expect for next month?". `FORECAST_GOAL` matched none of the business phrasings, so the
  mission had no checklist, no stall rescue and no forecast. It now matches plain future questions (next month or quarter,
  coming weeks, what will X cost, how much will we sell). Rep 4 also answered "is that real?" by comparing two raw 60-day
  windows; a question about whether a change is real now points the planner to `ps.change_test`. Rep 5 used it (+10.3 %,
  p = 0.484) and built the forecast with the measured total range (held 9 of 12 windows).
- Electricity rep 9 ended with eight scan findings undecided and an answer that was only that gap. The decision recipe took
  the first six rows, all one id over six clients, so it covered one id of eight. It is now one entry per id with the series
  folded in. A run that stops short of met without a draft now asks for every established result, with the gap last.
- Retail rep 5's repair cell guarded each ledger add on `scan["calendar"]`, a key the scan file does not have; both adds
  were skipped and the saved ledger kept the same sha256. Beijing rep 5 and electricity rep 9 guessed other layouts.
  `ps.Ledger.decide(finding_id, decision)` now reads the stage, scope and measure from the saved scan, an unknown id raises
  with the real ids, and saving an opened ledger with no new entry prints a note.
- Rep 5 wrote "the recent dip is not statistically real" over a measured rise. The composer now corrects a premise the
  evidence contradicts before giving a verdict.
- Electricity rep 10's answer said the blend was the centre line; the pack drew TimesFM for all 60 clients, because
  `forecast_report` refused centre "blend" on a panel twice. On a panel "blend" is now the pooled blend (`global_blend`). The
  answer also called the sum of the clients' 10 to 90 % bounds the likely range and said it assumed independent errors;
  summing bounds assumes every client misses the same way at once. The computed accuracy paragraph now names the drawn
  centre per series and both panel-total bounds with what each assumes.

### Reruns on the ledger and panel-centre fixes (measured, 2026-11-23)

| Mission | Build | Status | Stop reason | Seconds | Tool calls |
|---|---|---|---|---|---|
| retail, business user rep 6 | 4d530e1 | partial | stalled | 450 | 26 |
| electricity panel rep 11 | 4d530e1 | partial | stalled | 1,412 | 105 |
| retail, business user rep 7 | f669929 | **completed** | **done_verified** | 270 | 16 |
| electricity panel rep 12 | f669929 | partial | stalled | 511 | 68 |

Retail rep 7 used `led.decide` and `ps.change_test`, opened by correcting the premise ("not real: +10.3 %, p = 0.484") and
finished in 270 s.

- Retail rep 6 printed the scan's findings in eleven cells and its rescue analyst in fourteen more; no decision was saved.
  The open ledger item now ends with the `led.decide` cell for the undecided findings, and the rescue brief carries it.
- Electricity rep 11's panel backtest ranked TimesFM first (overall MASE 1.331) and the pooled model fourth (1.539); the
  pack drew the pooled model for all 60 clients, and its total was 3.08 million kWh against rep 10's 4.38 million.
  `panel_centre_matches_backtest` now blocks a single panel rule more than 2 % behind the backtest's best.
- Retail rep 7 opened with "an 80 % range of 1,257,168 to 2,610,558", the bound if every day misses the same way at
  once, above a measured range of 1,358,450 to 2,142,805, and passed. `answer_total_range_named` now blocks a bound given as
  the range without what it assumes; replayed, it blocks rep 7 and electricity rep 10 and passes the other stored answers.
- Electricity rep 12's only backtest covered one analyst's 15 clients, and the lead re-ran an entity-listing cell three
  times. The coverage note now gives the cell that saves all 60 (checked on the real file). Its answer also called MASE
  "MAE"; now blocked (see below).

### Answer checks on ranges and metrics, and the next reruns (measured, 2026-11-23)

| Mission | Build | Status | Stop reason | Seconds | Tool calls |
|---|---|---|---|---|---|
| WTI, business user rep 3 | efab5f9 | completed | done_verified | 360 | 24 |
| electricity panel rep 13 | efab5f9 | **completed** | **done_verified** | 886 | 75 |
| WTI, business user rep 4 | 5c6968e | **completed** | **done_verified** | 330 | 23 |
| retail, business user rep 8 | 5c6968e | **completed** | **done_verified** | 435 | 28 |

Electricity rep 13 is the second verified electricity mission: the pack drew the pooled blend the backtest ranked (centre
"blend" now maps to it), and the answer gave the 7-day total with both bounds named. Its first line still calls the
pooled blend a "per-client blend".

- WTI rep 3 answered "what will crude cost on average next month" with one day's band ($77 to $110) as the range of the
  average. Replay found the same in reps 1 and 2 (rep 2: $75 to $110 against a measured 80.66 to 100.85), so those
  "verified" answers overstated the uncertainty of the monthly average. `answer_average_range` now blocks an average goal
  whose answer lacks the measured range of the average. Rep 4 gave $80.44 to $100.62, the measured range.
- MASE scores reported as "MAE": replaying the last 120 stored answers found electricity reps 4 and 12 and one Beijing run,
  with no false hit. `answer_metric_named` blocks it unless a cell printed an MAE of its own.

### Panels on the full set of answer checks (measured, 2026-11-23)

| Mission | Build | Status | Stop reason | Seconds | Tool calls |
|---|---|---|---|---|---|
| Beijing panel rep 8 | ad98df0 | partial | could not close a gap | 1,307 | 88 |
| electricity panel rep 14 | ad98df0 | **completed** | **done_verified** | 1,112 | 99 |
| Beijing panel rep 9 | 5eae779 | **completed** | **done_verified** | 1,382 | 106 |

Electricity has now finished verified three times running (reps 13 and 14 on the new checks; rep 14 drew the pooled blend,
the best overall at mean MASE 1.697). Beijing rep 8 had a correct forecast but ended partial: each station's health sat in
the pack and its backtest scores in panel_backtest, and the planner could not join them. The loop now attaches both,
side by side, to the panel pack's observation, and the pack reports its horizon, context and model. Rep 9 finished
verified with them. It also wrote that Huairou had a `flat_expected` verdict while the pack gave it `ok`;
`answer_series_health_matches` now blocks that (one hit in 150 stored answers, this one).

### Levels are not summed (measured, 2026-11-23)

| Mission | Build | Status | Stop reason | Seconds | Tool calls |
|---|---|---|---|---|---|
| retail rep 9 | 25fa8df | **completed** | **done_verified** | 225 | 13 |
| WTI rep 9 | 25fa8df | partial | figures unmatched | 541 | 33 |
| WTI rep 10 | 56916c8 | partial | could not close a gap | 436 | 32 |
| Beijing panel rep 10 | 56916c8 | **completed** | **done_verified** | 871 | 77 |

WTI rep 9 gave "the 80% range for the 20-day total is $1,644 to $1,987" of an oil price, and Beijing reps 8 and 9 a 14-day
panel total of PM2.5 concentrations (12,222 ug/m3). A price or a concentration is a level: its sum over days or stations
means nothing. The computed accuracy paragraph produced those totals itself, so the figure check passed them, and Beijing
rep 9 and older WTI runs were marked verified with them. `measure_is_level` reads the data's column names and the pack
title (a flow name such as sales, kWh or a daily sum wins). For a level the paragraph now gives the 80% range of the average
over the horizon and no panel total, and `answer_level_not_summed` blocks a stated total (7 hits in 150 stored answers, all
WTI or Beijing; none on a flow goal). WTI rep 10 gave the average's range (82.18 to 99.34) and Beijing rep 10 no total.

WTI rep 10 still ended partial: the evaluator's reason named the unsupported figures, the "Not settled" section copied it,
and the answer failed the figure check on the figures it was describing. The gap is now written without figures. The
slips themselves were real (drift MASE 7.29 written as 7.12; an outlier share of 2.0 % written as 2.1 %).

### All six scenarios verified on their latest run (measured, 2026-11-23)

| Mission | Build | Status | Stop reason | Seconds | Tool calls |
|---|---|---|---|---|---|
| WTI rep 11 | 6b527f7 | **completed** | **done_verified** | 255 | 21 |
| electricity panel rep 15 | 6b527f7 | partial | false block (bound wording) | 1,066 | 72 |
| electricity panel rep 16 | 4b5df07 | **completed** | **done_verified** | 1,127 | 111 |
| retail, business user rep 9 | 4b5df07 | **completed** | **done_verified** | 270 | 22 |

Latest run per scenario: retail rep 9, retail business user rep 9, WTI rep 11, WTI business user rep 4, electricity
rep 16 and Beijing rep 10, all done_verified. This is one run each, not a rate; repeat rounds measure it.

Electricity rep 15 was blocked by `answer_total_range_named` for "if clients' errors cancel out", a correct name for the
independent bound. Replaying 160 stored answers found two more false blocks ("if they compound", "(perfectly
correlated)"). The accepted wording for each bound is wider now; six hits remain on the stored answers, five plainly a single
bound given as the 80 % range and one borderline ("under that assumption", naming the previous sentence's assumption).
Gate: 1861 passed.

### Repeat round of the six scenarios on build 4b5df07 (measured, 2026-11-23)

The API process (pid 90241) started at 08:48 on 4b5df07 and served the whole round, so 6e734df and 66777be were not
exercised live. One run per scenario, real NIM and TimesFM 3 through the API. Synthetic and real staging data are not
mixed here; every figure comes from the mission's own evidence.

| Scenario | Mission | Status | Stop reason | Seconds | Tool calls |
|---|---|---|---|---|---|
| retail | 0ae0d417 | **completed** | done_verified | 1,201 | 27 |
| WTI | c85260a9 | **completed** | done_verified | 240 | 17 |
| WTI business user | 6b5796f3 | **completed** | done_verified | 1,246 | 29 |
| retail business user | 50074f99 | partial | partial_unverified | 555 | 26 |
| electricity panel | 35c194a1 | partial | partial_unverified | 991 | 87 |
| Beijing panel | 82fb34d9 | partial | partial_unverified | 1,772 | 109 |

Round result: 3 of 6 done_verified on this build. Both checkpointed missions resumed and finished: 0ae0d417 is retail
(the earlier note had the labels swapped) and 6b5796f3 is WTI business, so the automatic resume after the model-service
outage works. The three partials stopped with an unverified answer. Causes, read from each mission's stored answer:

- **retail business user (50074f99):** contract `answer_cycle_matches_health` blocked it. The answer stated a 3-step
  cycle from the backtest flag (strength 0.25), but the health check measured a 6-day period. A real contradiction that
  the check caught; the answer used the backtest's 3-step figure instead of the health record's.
- **electricity panel (35c194a1):** the analysis could not close its gap. The answer reported the global_gbm panel
  total and its 80 % range, but per-client MASE for global_gbm against naive, seasonal_naive and TimesFM was never
  computed. The backtest reports gave overall means only. The agent spent 87 calls and did not write the per-client
  table.
- **Beijing panel (82fb34d9):** four figures (198, 203, 215, 191) in the answer did not match the analysis output, so
  the answer was marked unverified after 109 calls and 46 iterations, the most of any mission in the round.

Causes, found in the stored missions and fixed in f903603 (gate 1,869 passed):

- **Cycle (retail business user):** the backtest's season check measured 3 steps on 273 points, the health record 6 steps on
  633 points. The contract said "state the measured period" and the rewrite kept 3. The contract now names the number to
  write (6 steps) and says not to use the season check for it.
- **Figures (Beijing):** 198.77, 203.71, 215.63 and 191.58 were written as 198, 203, 215 and 191, cut down rather than
  rounded, so the matcher found no source. A whole number that is the cut-down part of a printed decimal is now rounded to
  it, and only when exactly one rounding fits.
- **Per-client comparison (electricity):** the panel_backtest result held a MASE for all 60 entities, but the planner's
  view showed the first 30 rows, so the evaluator called the comparison missing. The view now counts, over all entities,
  where each model beats naive (`beats_naive_on`, `beats_naive_out_of`).

### Rerun of the three partial scenarios on f903603 (measured, 2026-11-23)

| Scenario | Mission | Status | Stop reason | Seconds | Tool calls |
|---|---|---|---|---|---|
| retail business user | 1aa2106e | **completed** | done_verified | 270 | 24 |
| electricity panel | f62b473c | **completed** | done_verified | 811 | 73 |
| Beijing panel | 3c379255 | partial | partial_unverified | 856 | 85 |

Two of three verified. Beijing did not hit the figure check this time. Its checklist was 8 of 8, but it repeated identical
calls (inspect_data five times with the same arguments, read_observation pages three times each) and the stop reason is
"the analysis kept repeating the same step", so the answer rests on earlier findings. This is a stall, not a number
error, and it is the next thing to look at.

The fixes are checked by unit tests and by these runs, one run each, so this is not a rate. Verified across the two
rounds: retail, WTI and WTI business user in round 6 (build 4b5df07), then retail business user and electricity panel on
the rerun (build f903603). Beijing panel is still partial.

### Repeat round 6c of the six scenarios on build f84cce2 (measured, 2026-11-23)

Build f84cce2 holds the seeded-analyst fix (5551f4a) and the blend-wording check (f84cce2). One run each. Every series is
synthetic. Mission ids are from data/prophet.sqlite.

| Scenario | Mission | Status | Stop reason | Seconds | Tool calls |
|---|---|---|---|---|---|
| retail | 8cea16c6 | completed | done_verified | 166 | 14 |
| WTI | c8213ced | completed | done_verified | 555 | 30 |
| WTI business user | 364a288f | completed | done_verified | 166 | 21 |
| retail business user | cf7f4aa6 | partial | partial_unverified | 150 | 15 |
| Beijing panel | 97ab893f | partial | partial_unverified | 976 | 100 |
| electricity panel | c1154fd1 | partial | partial_unverified | 1,592 | 108 |

Round result: 3 of 6 done_verified. The three partials each stopped on a contract check that the rewrite did not clear.
Causes, read from each stored answer (not yet fixed; see the next step in this section):

- **retail business user (cf7f4aa6):** `answer_cycle_matches_health` blocked it. The answer claimed a 260-step cycle from
  the backtest's same-period test (+150.8 %, p = 0.016), but the health record measured 5 steps. The f903603 wording named
  the number to write, and the rewrite still kept the backtest figure.
- **Beijing panel (97ab893f):** `answer_total_range_named` blocked it. The answer gave 4,082 to 26,035 as the total's
  range, which is the bound if every station misses the same way at once. The check asks for both bounds, each with what
  it assumes.
- **electricity panel (c1154fd1):** `scan_decisions_recorded` blocked it. The scan had blocker or major findings,
  censoring.stockout_runs and schema.missing_column, with no ledger decision. The answer was written without recording
  them.

Build f84cce2 gate: scripts/check.sh 1,877 passed, 1 skipped, 4 deselected.

### Repeat rounds 6d and 6e on builds 3f047a5 and ad15f53 (measured, 2026-11-23)

Round 6d (build 3f047a5) ran the retail business user and Beijing together, then electricity. Round 6e (build ad15f53) ran
the same order. Every series is synthetic. Mission ids are from data/prophet.sqlite.

| Round | Scenario | Mission | Status | Stop reason | Seconds | Tool calls |
|---|---|---|---|---|---|---|
| 6d (3f047a5) | retail business user | f1cc71cb | partial | partial_unverified | 149 | 14 |
| 6d (3f047a5) | Beijing panel | 3581d07d | partial | partial_unverified | 1,388 | 117 |
| 6d (3f047a5) | electricity panel | 8f4ada17 | completed | done_verified | 706 | 77 |
| 6e (ad15f53) | retail business user | cae00201 | completed | done_verified | 263 | 17 |
| 6e (ad15f53) | Beijing panel | d861b74e | partial | partial_unverified | 931 | 99 |
| 6e (ad15f53) | electricity panel | 29acb1c0 | partial | partial_unverified | 991 | 99 |

Causes of the four partials, read from the stored missions, and the fix for each:

- **Retail business user (f1cc71cb, 6d):** the change test was called with a lag shorter than its recent window, so the two
  windows overlapped. The tool returned a "not enough history" note that did not name the cause, the planner repeated the same
  call, and the run stopped as a stall. Fixed in ad15f53: `change_test` raises a ValueError that names the fix when lag is
  less than recent. Not re-run on a stall case since the fix.
- **Beijing panel (3581d07d, 6d):** the ledger was saved as out/ledger_full.json, which the check did not read, so the
  analysis_ledger check counted 0 stages, and the scan's changepoints.level_shift had no decision. The run also repeated the
  same ledger cell 117 times with new carry ids, so the stall signature never matched. Fixed in ad15f53: LEDGER_FILE accepts
  out/ledger*.json, and the stall signature ignores carry ids. The next Beijing run (d861b74e) stored only one failure, the
  level-sum check, so the ledger check passed on that run.
- **Beijing panel (d861b74e, 6e):** the answer gave an average daily PM2.5 of 873 µg/m³ (12,222 ÷ 14). 12,222 was the sum of
  station totals, which forecast_report emitted as a panel total for a level title. A level has no meaningful sum.
  answer_level_not_summed blocked it. Fixed at source in b68ae46: report_tool gives panel_total None for level titles
  (is_level_name).
- **Electricity panel (29acb1c0, 6e):** the answer named global_gbm as the centre, following the ledger, while forecast_report
  drew global_blend for every client. The reason the run stopped on was that the centre the answer named did not match the
  pack. Fixed in b68ae46: _centre_named_wrong blocks a centre claim that names the other rule. A ranking sentence that names
  a rule is not a centre claim.

Both fixes in b68ae46 are checked by unit tests and by the replay of the stored answers. The live rerun is in the next section.
Known open item: the Beijing answer still gives a per-station 14-day total for a level measure (PM2.5), because only the panel
total is withheld at source.

### Round 6f on b68ae46 and round 6g on 785a42f (measured, 2026-11-23)

Round 6f (build b68ae46) ran Beijing and electricity together, then retail business. Every series is synthetic.

| Round | Scenario | Mission | Status | Stop reason | Seconds | Tool calls |
|---|---|---|---|---|---|---|
| 6f (b68ae46) | Beijing panel | 355ea1d1 | completed | done_verified | 1,081 | 83 |
| 6f (b68ae46) | electricity panel | 14d115d3 | completed | done_verified | 815 | 80 |
| 6f (b68ae46) | retail business user | e6caf00e | partial | partial_unverified | 389 | 23 |
| 6g (785a42f) | Beijing panel | d1a1d077 | partial | partial_unverified | 1,350 | 82 |

- **Beijing and electricity (6f)** finished done_verified on b68ae46. The electricity run is the first on the centre check.
- **Retail business user (e6caf00e, 6f):** the forecast came back with dates null. `_future_dates` returns nothing for an
  irregular history (sales on some days only), so the forecast window could not be stated, and the run then repeated several
  failing sandbox steps. Not fixed yet: the forecast does not record the last history date, so the answer cannot give the window.
- **Beijing panel (d1a1d077, 6g):** the figure check blocked the answer, which is correct. The panel view gave each station's
  first-step point but no first-step band, so the answer's day-1 and day-14 bands were not the stored ones (Aotizhongxin
  day-1 band written 8.8 to 131.2; stored 4.5 to 114.2). Fixed in this commit: the view now gives q10_step_1 and q90_step_1.
  The answer also gave a 14-day average per station and no per-station total, which is the level fix working.

### Round 6h: Beijing rerun on cf1f0cc (measured, 2026-11-23)

| Scenario | Mission | Status | Stop reason | Seconds | Tool calls |
|---|---|---|---|---|---|
| Beijing panel | 0137871e | partial | partial_unverified | 4,242 | 103 |

The answer was written: a 14-day table per station with the centre line and an 80 % range for each day, and no station total.
The run still stopped as a stall, "the analysis kept repeating the same step". After the forecast, the planner read pages 1 to 8
of step 21's observation, read pages 2 to 8 again, and then ran two sandbox cells with the same code that printed nothing and
wrote no file. The stall rule saw two identical outcomes and stopped. Fixed in this commit: a cell that prints nothing and
writes no file now carries a note that says so and asks for the figures to be printed or saved in the next cell.

The day-by-day table does not match the stored forecast. The stored forecast_report (run 685b8c6123cb479ea1749b7a6619465d)
gives Aotizhongxin day 1 as 40.7, band 4.5 to 114.2; the answer gives 52, band 12 to 132. The figure check passed the table
anyway, so a number is matched to some figure elsewhere in the history rather than to its own station and day. Open gap: the
table was not grounded to its forecast, and this answer must not be used as a verified Beijing forecast.

### Table grounding: a station's day figures must be its own (fixed 2026-11-23, build after cf1f0cc)

The figure check matches a number against every number in the evidence, so a table cell is "grounded" by any matching
figure anywhere. Replaying the stored answers with a per-station, per-day check found the same fault in several runs,
including two that were marked done_verified:

- **355ea1d1 (Beijing, round 6f, done_verified):** the row for Wanshouxigong carried Tiantan's figures (day 14 75.3, band 21.8 to
  156.1; the forecast gives Wanshouxigong 67.0 on day 1). The figure check passed it.
- **0137871e (Beijing, round 6h, partial):** Aotizhongxin day 1 52 (12 to 132); the forecast gives 40.7 (4.5 to 114.2).

Cause: the panel view gave each station only its first and last steps, so the planner had no per-day figures to copy.
Fixed: the panel view now gives every day of each station as "point (lower to upper)", for horizons up to 31 days
(`by_day`), and a new block check, answer_table_matches_forecast, reads each table row against its station's own forecast
and blocks the first mismatch. Replayed on the stored answers: the 6h and 355ea1d1 tables are blocked; the 744a6a8c and
d1a1d077 tables pass. Blocking mismatches in other stored answers (c69a6e80, 032eea79, 68d70b8d, 82fb34d9, d861b74e) are
real mismatches, not checked by hand one by one.
Gate: 1,916 passed, 1 skipped, 4 deselected.

### Round 6i: a wrong station day cell is set to its forecast (fixed 2026-11-23, build after a1c81fd)

Beijing 80bbc853 (round 6h rerun, partial, answer_table_matches_forecast): 111 cells in eight stations' rows were not the stored
forecast. Guanyuan day 2 was 124.0 (57.5 to 225.2); the forecast view gave 129.1 (59.2 to 226.5). The view was not the cause:
the planner's observation carried every station's day (`by_day`), and the wrong cells were copied wrong at composition. One
rewrite was given the same evidence and copied them wrong again, so the answer left partial.

Fixed: `contracts.snap_tables` sets each station's day and average cell to the forecast's own value at the printed decimals,
keeping the cell's markup and separators, and leaves a cell alone when it already matches, holds a band header, or is not a
single number or a point with its band. It runs after `snap` in `goal.py`, so the claims check and any repair see the corrected
table. Replayed on the stored 80bbc853 answer: 111 cells corrected, the table then passes (168 cells match), and text outside
the table is unchanged. Tests: `tests/prophet_agent/test_r6i_table_snap.py` (11), with the table tests (22 passed in total).

Live rerun on this build, e14af4bd (Beijing, REP=2): partial, stop reason `partial_unverified`, "the analysis kept repeating the
same step". It stopped at 56 calls, all sandbox_python analysis with the same code, before any `forecast_report`. So the table
correction was not exercised live. The 80bbc853 failure remains the only live evidence; the fix is verified by the replay and
the tests, not by a live run.

Live rerun on 7ab9fbe, a7efe739 (Beijing, REP=3): done_verified, 1,156 seconds, 89 tool calls, no contract failures. The
answer's table (day 1 and day 14 per station, average over 14 days) matches its stored forecast on all 24 checked cells, and
the answer carries one correction note covering 16 figures. The note combines figure and table corrections, so the table's
share of the 16 is not counted separately. This is the first live run of round 6i that reached a verified answer.

Gate: 1,927 passed, 1 skipped, 4 deselected.

### Round 6j: repeated sandbox cells and guessed schemas (fixed 2026-11-23, build after 7ab9fbe)

The stalled Beijing run e14af4bd (partial, 56 calls, "kept repeating the same step") was traced to two causes:

- **Repeated cells.** The repeat rule answered an identical call with a pointer only when no sandbox cell had run since. A
  cell that ran in between forced a rerun, though `sandbox_python` depends only on its code and on immutable `carry` folders.
  Twelve identical cells ran in a row, and the stall rule stopped the mission. Measured across stored missions, identical
  sandbox calls were common (e.g. 97 cells with 83 distinct arguments). Fixed: `PURE_TOOLS` exempts `sandbox_python` from the
  rerun clause; read tools that depend on written files keep it.
- **Guessed schemas.** A written JSON file was handed to the planner as a bare path. The scan is `findings` and `summary`, but
  the planner tried `per_entity` three times and failed with KeyError. Fixed: `observe` now names each written JSON file's
  top-level keys and their kinds (`json_shape`, up to 12 keys, files up to 5 MB).

Tests: `tests/prophet_agent/test_r6i_repeat_and_shape.py` (8). Live rerun on this build, d4af2aaf (Beijing, REP=4): done_verified,
1,126 seconds, 66 tool calls. Its 39 sandbox calls all had distinct arguments (against 63 calls with 5 repeats in a7efe739), the
shape description reached the planner in 10 observations, and the final table matches its stored forecast on all 168 checked
cells. One run does not prove the stall is gone in every case; it is the open check for the next rounds.

Gate: 1,935 passed, 1 skipped, 4 deselected.

### Round 6k: the forecast window is named in every observation (fixed 2026-11-23, build after 008917e)

Live retail rep9 (e6caf00e) reached done_verified, but its answer gave the forecast total without the period it covered. The
forecast did carry its dates: all five `forecast_report` calls wrote `out/forecast.json` with date lists matching their
horizons (30, 38, 39, 40 and 54 steps), and the table had a date column. Nothing named the window, so the planner had to
infer it from the table's first and last rows.

- **Forecast report.** `observe()` now adds `window` (first, last, steps) for a single series, and each panel line carries
  its own `window`. Fixed in `report_tool.py` by the `_window` helper.
- **Raw TimesFM.** `timesfm_forecast` returns `future_dates` and `window` when the history's dates are known and match its
  length, and the compact view the planner reads shows the window (rep8 had answered with 20 values and no period).
  Fixed in `timesfm_tool.py` by `_attach_window`.

Tests: `tests/prophet_agent/test_r6j_report_window.py` (2) and `tests/prophet_agent/test_r6j_timesfm_window.py` (5), with the
retail date tests (8). Commits 1c0882c and e147381. Gate: 1,950 passed, 1 skipped, 4 deselected. A live rerun on this build
has not been done; it is the open check for the next round.

### Round 6k triage: the high-severity audit items (2026-11-23)

Five high-severity items from the audit were checked against the code before any change. Each was reproduced first.

- **Repeated dates read as hourly** (`forecast/season.py`). A daily series with each date repeated was classified as hourly,
  because zero-length gaps pulled the median below one day. Gaps are now taken between distinct dates. Commit e811fdd.
- **Delivered seasonal line not the scored one** (`tools/report_tool.py`). The delivered seasonal rule measured its season on
  the whole history; the backtest measures it before its first test window. The two disagreed in 40 of 40 probes (season 7
  delivered, season 1 scored). The rule now asks `backtest_season`. Commit 444f5eb.
- **Measured counts hidden as identifiers** (`runtime/grounding.py`, verified). "Customers 1,200" and "Orders 3,000" were
  blanked whole, so the figure was never checked. A list separator now needs a space, and a number that runs into a decimal
  or thousands group is a value. Commit 373c888.
- **Lost lease recorded as a tool error** (`runtime/goal.py:1177`). The lease is renewed inside `Ctx.call`, and `_act`
  caught every exception as tool evidence, so the loop kept writing after another runtime took the mission. `LeaseLost` now
  propagates. Commit 9dd443c.
- **dir_fd bypass of the sandbox guard** (`sandbox/runner.py:83`). The audit hook checked a relative name against the run
  folder, but the kernel resolves it against the descriptor's folder: `os.remove("keep.txt", dir_fd=fd)` removed a file
  outside the run (reproduced). Each `dir_fd` call now resolves its real folder first. Commit 5c2e12c.

Tests: `test_r6k_season_duplicate_dates.py` (4), `test_r6k_delivered_season.py` (3), `test_r6k_identifier_values.py` (8),
`test_r6k_lease_lost_propagates.py` (2), `sandbox/test_r6k_dir_fd.py` (4). Three of the dir_fd tests fail on the old runner.

Not done in this round: the lease fence on store writes (`store.py:409-437`, writes do not check the owner, so a takeover
between renewal and write is unfenced), the contract gaps (benchmark passes without `best_baseline`, `contracts.py:2063`; table
cells under numeric headers unchecked, `contracts.py:1561`), the remaining sandbox hardening items, and the medium and low items
in the forecast, API and docs audits.

Gate: 1,971 passed, 1 skipped, 4 deselected.

### Round 6l: lease fence, contracts, API, sandbox, goal loop, forecast and docs (fixed 2026-11-23, merged to main b2d9249)

Work was split across parallel worktrees, each reproduced before it changed code, and merged to main with the gate re-run
on the combined tree.

- **Lease fence on store writes** (`runtime/store.py`, `runtime/runner.py`, commit 6ddbe8f). Writes now check the lease owner
  inside the write transaction, so a runtime whose lease was taken over is refused. `run()` returns `lease_lost` instead of
  letting `LeaseLost` escape. Eleven tests in `test_r6l_lease_fence.py`.
- **Contracts** (`runtime/contracts.py`, commits 85df180, 551b73e, 1127637). A benchmark claim with no backtest best baseline
  is blocked; a table day cell and a bare day-number header are checked.
- **API** (`api/app.py`, commits aa122cc, c8ee863, bdc986a, 4233fe6, 5faea06). Streamed upload bodies are capped; key strings
  are bounded at 256 characters; a failed mission start is marked failed; redaction covers Bearer tokens and common key shapes.
  The licence-wording guard is widened and baselines 13 dataset labels in `scripts/fetch_datasets.py` pending an owner decision.
- **Sandbox** (`sandbox/runner.py`, commits 6a54362 to d6210e9). Manifest and listing stay inside the run folder; path checks
  fold case; SQLite databases must sit in the run folder and URIs are refused; name lookups are blocked; captured output keeps
  its first and last 40,000 bytes; `source_covariates` and `run_dates` stay inside `out/`; a repeated identical append is a
  no-op; the profile is read from `out/profile.json`. Repeated appends are now a no-op by design.
- **Goal loop** (`runtime/goal.py`, `runtime/grounding.py`, commits 1dbf6ea to 3e0e604). A two-call A, B, A loop with unchanged
  outcomes is a stall; sub-agent figures are checked against their own tool results, not the brief; the snap tolerance is
  capped at 1,000 units; method-claim markers match by word, so "asymmetric" is not "symmetric"; claims are re-checked after
  the repair pass.
- **Forecast** (`forecast/baselines`, `artifacts`, commits 5271158, a5862e2, ec0fbfd). Seasonal naive averages the last k
  same-phase values, so its band widens by sqrt(1 + 1/k) and it no longer runs out past k seasons. Coverage on a synthetic
  period-4 probe at the 80% nominal rose from 0.70 to about 0.78 (synthetic; the residual gap is sigma estimated from 48 points).
  A cut PDF note now says so. Any stored backtest figures for seasonal naive are stale and need regenerating. The season bar
  was checked and is already a Bonferroni bound with a 0.3 floor; no change.
- **Docs** (`README.md`, `docs/prophet/ARCHITECTURE.md`, `docs/prophet/GOAL.md`, commits d006699, d9879bd, 4ddf3a3, 2cc5ce8).
  README outbound-call claim corrected; demotion codes, playbook count (18) and planner version corrected; stale counts
  re-collected; 15% bar locked by a regression test. NIM.md is tracked (owner decision: keep it tracked).

Not done in this round: DNS rebinding in `research/fetch.py` (the host is resolved three times and the connection is not pinned
to the validated address; the fix sits outside the sandbox agent's files); other JSON routes still read unbounded chunked
bodies; the widened `redact()` also masks name=value text in sandbox output; the baseline band is still labelled 80% in
backtest; the untested modules `nim/probe.py`, `api/panels.py`, `api/sharing.py`, `api/bundle.py` and `api/__main__.py`.

Gate on the combined tree: 2,112 passed, 1 skipped, 4 deselected, 0 failed (372 s).

### Round 6m: merged sub-agent branches, API write atomicity, lease fence and shutdown (fixed 2026-11-23, main 3989713)

Each branch was reproduced before it changed code, tested on its own worktree, merged to main and pushed. The full gate was
re-run on the combined tree after the last code change.

- **Top-level modules** (`settings.py`, `notebooks.py`, `mcp_server.py`, `config.py`, `datasets.py`, `mentions.py`, merged as
  85edb9d). MCP errors, timeouts and argument checks are explicit; settings keys are masked and written atomically; a stored
  `.env` value keeps no trailing note; a damaged dataset metadata file no longer breaks the upload list; `make_card` no longer
  loops forever on a blank title; a notebook version of 0 names version 0 (it was treated as "no version"); a mention always
  stays a readable link. Six test files, `test_r6m_top_*.py`.
- **Redaction false positives** (`nim/errors.py`, merged as c6025d9). A bare `token=` masks only in a URL query, as `key=` already
  does; `api_key`, `access_token`, `auth_token`, `secret` and `password` mask everywhere. Real key shapes still mask. The 7
  strict-xfail tests in `test_r6m_redaction_false_positives.py` now pass with no xfail marker.
  **Trade-off for the owner:** a `Bearer` or `sk-` value with no digit (for example `Bearer abcdefgh`) now prints unmasked.
- **PUT /api/settings is all-or-nothing** (`api/app.py`, a93ae9b). The route dry-runs the whole save with `settings.check_save`
  before any write. A valid limit sent with a malformed key was saved before the key was refused; a regression test fails
  without the change and passes with it.
- **Answer and cancel paths under the lease fence** (`api/service.py`, merged as edce96a). `answer()` resumes an answer that
  lands while its run is pausing, and treats `LeaseHeld` as "being run". `cancel()` closes a mission only when no live lease
  is held, in one IMMEDIATE transaction. A notebook's provider rule is checked before an answer is recorded. Eleven tests in
  `test_r6m_service_{answer,cancel,leases}.py`.
- **Outside writes and stale owners** (`runtime/store.py`, f847276). `_fence` refused an outside write (cancel, budget) whenever
  the database named another runtime as lease owner, even after that lease had lapsed. It now refuses only while a foreign
  lease is live. `test_r6m_fence_stale_owner.py` (2 tests) fails under the old condition.
- **Shutdown stops the orphan sweeper** (`api/service.py`, `api/app.py`, 3989713). The sweeper was a daemon thread with no
  stop path, so it kept reading a store its owner had finished with; tests logged `orphan sweep failed: ProgrammingError`.
  `MissionService.close()` stops and joins it, and the app lifespan calls `close()` on shutdown. `test_r6m_service_close.py`
  (2 tests).

Gate on the combined tree at 3989713: 2,483 passed, 1 skipped, 4 deselected, exit 0 (307 s). No orphan-sweep warnings.

Known and not fixed in this round. Owner decisions are marked.

- **Research** (`research/fetch.py`): the DNS lookup is not bounded by the deadline; gzip and brotli expand before the byte cap;
  a robots.txt 5xx fails open; robots is not paced; the throttle sleep sits outside the deadline.
- **Forecast**: seasonal naive under-covers random walks (synthetic, about 0.47); `choose_season` returns NaN and misaligns
  dates; baselines propagate NaN; the DOCX shows small values as 0.00; "8 in 10" is hard-coded in `report_tool.py` and
  `contracts.py`. Stored backtest, lab and TSB figures for the moving-average, seasonal-year and seasonal-naive bands are stale
  and need regenerating.
- **Runtime**: tool started, done and failed events are written in separate writes; argument-refusal classification is loose.
  `archive()` does not check the lease. A provider rule that changes between `answer()`'s pre-check and `start()` can still
  strand an answer.
- **Tools**: consecutive identical appends are dropped silently; `goal.py` `_ARG_REFUSAL` charges refusals; `read_observation`
  cannot page past 200k; `research_tools.py` has an undefined `Page` annotation.
- **NIM** (NIM.md): the AWS Secrets Manager key source (§8) is not built; §5 rule 7 is not built; a 413 does not demote (pinned
  by a test, to be reviewed).
- **Sandbox**: the darwin memory cap is absent; `os.stat` and `os.path.exists` raise no audit event; `shm_open` and
  `RLIMIT_NPROC` are not restricted.
- **Goal loop and grounding**: 409 identical-output sandbox cells with different code; count nouns are blanked as identifiers;
  a date-only question changes the replay; the no-backtest block is wider than its audit item; `_drop_row_indexes` ignores rows
  without outer pipes; `sandbox_tool` stderr run paths.
- **Synthetic generator** (top-level audit): `synth/generator.py` crashes when `n_items` is under 8; `synth/censor.py` mishandles
  a non-integer latent; synthetic source labelling is incomplete; usage behaviour lacks tests. `.gitignore` lacks `.env.*`.
- **Owner decision (2026-11-23)**: keep the redaction rule as merged. A `Bearer` or `sk-` value with no digit prints unmasked.
- **Worktrees**: `agent-a20d6af35060836c5`, `agent-a8a099c3eda9246a3`, `agent-a7cf04b87537a34d9` and `agent-a36045868ea93098d`
  are locked by the Claude Code app process (PID 35675). All four branches are merged. Removal awaits the owner's go-ahead.
- **Shutdown**: `MissionService.close()` stops the sweeper; the owner of the `Store` still closes the database, which the
  service does not do. That is the next step.

## Round 7: open-data path and one measured weather run (2026-11-23 to 2026-11-24, main 201b490)

Round 7 had two parts: review the whole system for gaps, and let the deep-research path fetch, store and check open data
directly. Each change was reproduced or measured before it changed code, committed on its own, and the full gate was run on
the final head.

- **Open-data store** (`opendata/store.py`, `opendata/sources.py`, dd7e016 and e625848). The default request now asks for CSV.
  Open-Meteo answers JSON when no format is given, and the store used to accept that JSON as a one-column table. A body that
  starts like JSON is now refused as CSV. The stored file is the table alone; the location block Open-Meteo prints before it
  is kept as text in provenance, with `raw_sha256` and `raw_bytes` for the body as fetched.
- **Unverified SDMX type refused** (`opendata/fetch.py`, f3ab917). The `application/vnd.sdmx.data+csv` type was on the sniff
  list before any live check. The live ECB endpoint answers `text/csv`, so the unverified type is removed and a test pins it.
- **Bundle hashes have one shape** (`api/bundle.py`, ab81f8a). Every `hashes.json` entry is an object with `sha256`. A copy
  that differs from the bytes read also carries `sha256_original`. Three test files were updated to the shape.
- **Settings export is capped** (`settings.py`, 4c9da47). Each table is limited to 50,000 rows and the export names every table
  it cut, including `events`, which each mission also writes. `test_settings.py` covers it.
- **Measured weather run** (`scripts/weather_open_data_check.py`, `docs/prophet/WEATHER_OPEN_DATA_RUN.md`, 201b490). Berlin
  daily maximum temperature, 1,780 points from the Open-Meteo archive, retrieved 2026-11-23, with no missing values. The
  check refuses gaps. Results with the annual season set to 365, which is the fair comparison: TimesFM 3 MASE 0.752 against
  0.951 for seasonal naive, a 20.9% gain (90% interval 9.2% to 33.2%), better in 10 of 12 origins, sign-flip p = 0.0095. The
  tool's error-correlation check gives a variance ratio of 5.2, so the origins are not independent and p is indicative only.
  The forecast health check returned `check`, not `ok`, and the report says so.
- **The automatic-season result is not quoted.** The tool found no season and used 1, because its search tests lags up to 14
  only. That gives a weak simple rule and an inflated headline (MASE 1.50 against 2.14). The report states this and uses the
  annual-season figures.
- **Licence wording** sits only in `docs/prophet/` (the report and `DATASET_LICENCES.md`, key `open_meteo`). The script and the
  source code refer to the source by key. The licence guard passes.

Gate on 201b490: 2,692 passed, 1 skipped, 4 deselected, exit 0 (315 s). The gate on dd7e016 earlier passed with 2,688.

Owner decisions in this round:

- Stock: proceed to automatic extraction without asking again. No compliant open stock source has been wired in yet, so stock
  series still come from a user upload.
- Download: automatic download from any source where the tool can, without asking permission each time.
- Open-Meteo: research use, labelled in the Markdown documents only. Not in agent prompts, contracts, tools or the UI.

Known and not fixed in this round:

- **Automatic season detection** tests lags up to 14 only, so daily series with an annual cycle get season 1, a weak simple
  rule and an inflated gain. The weather report records it. The fix is a wider lag search with a guard for short series.
- **Planner transcription** (`tools/series_io.py`, about lines 93 to 98): numbers the planner copies from web sources into
  `values` have no provenance check. A value can reach a backtest with no link to the page it came from.
- **Research crawl** (`research/fetch.py`, the same items as Round 6m): the DNS lookup is not bounded by the deadline; gzip
  and brotli expand before the byte cap; a robots.txt 5xx fails open; robots is not paced; the throttle sleep sits outside the
  deadline.
- **Web typecheck** was not run. There are no `node_modules`, and installing them was not authorised.
- **Round 6m leftovers** stand as listed above. The worktrees still locked by the Claude Code app are unchanged.

## Round 7 follow-up: automatic season detection (fixed 2026-11-24)

- **Cause.** The season rule scored one-step changes only. The search did test lag 365 (an earlier report said it tested
  lags up to 14, which was wrong). The annual cycle in daily temperature is about a thousandth of the day-to-day noise in
  the changes, so the lag-365 autocorrelation was -0.02, and the rule returned season 1.
- **Fix** (`forecast/season.py`). When the changes test finds nothing, the detrended levels are scored by the share of
  variance that phase means explain at each candidate lag. The bar is the 95% quantile of the same score on random walks of
  the same length, floored at 0.3, so price-like random walks stay at season 1.
- **Measured.** Berlin, no season given: season 365, level strength 0.84 against a bar of 0.30. False-season rate on random
  walks: 3% at n = 60 (the existing test allows 5%), 0% at n = 600 and n = 1612. Annual cycle at noise 3: found in 20 of 20.
- **Weather run.** The automatic backtest now reproduces the hand-set annual run: TimesFM MASE 0.752 against 0.951 for
  seasonal naive, a 20.9% gain, 10 of 12 origins. The earlier automatic headline (1.50 against 2.14) is withdrawn.
- **Tests.** `tests/prophet_agent/forecast/test_season_levels.py` (6 tests, including the backtest default on an annual
  series). The gate on the change: 2,698 passed, 1 skipped, 4 deselected, exit 0.
- **Open.** The health check picks its period with `dominant_period` (sandbox library), which gives 356 for the same series
  against 365 for the backtest. It is not reconciled, because the sandbox uses that function too.

## Round 8: sub-agent branches merged, robots and CSV decisions, seasonal band (2026-11-24, main d63644f)

Owner decisions taken through questions on 2026-11-24. Each is recorded in the code or tests where it applies.

- **robots.txt is no longer honoured.** `HONOUR_ROBOTS = False` in `research/fetch.py`. The robots code in `netguard.py` is
  kept and covered by tests that opt in with `monkeypatch`. Commit 4f5ac15. This is an outward-facing crawling-policy
  change: a disallowed path is now fetched, and robots.txt is not asked. Compliance review is recommended. Authentication
  and CAPTCHA protections are unchanged (never bypassed).
- **CSV text cap raised to 180,000 characters** (`CSV_MAX_TEXT_CHARS`, CSV media only). Plain text keeps 60,000. Commit 4f5ac15.
  Tests: `tests/prophet_agent/research/test_fetch_robots_off_and_csv_cap.py` (4 tests).
- **Seasonal naive default band is now `band_mode="measured"`** (commits 691bc9a and d63644f). Each lead's spread is the rule's
  own past h-step errors at the same season position, using complete lags only; a lead with fewer than `MIN_PAIRS` (6)
  errors falls back to the phase band. The `phase` and `sqrt` modes are unchanged. The default value of `band_mode` changed.
  - Measured, synthetic, 1,000 replications on each of seeds 7 to 9, 200-point history, horizon 14, season 7, k 4, nominal
    0.80: random walk 0.74 to 0.77 (phase band 0.48 to 0.50, sqrt band 0.83 to 0.85); period 7 with noise sd 2, 0.79 to 0.80.
  - **Not met:** the random-walk coverage is still about five points under its label. The random-walk test window of
    0.70 to 0.85 records that shortfall; it is not a claim that the band is calibrated on random walks.
  - Tests changed: the width-equality assertion became a 15% tolerance on mean width (sampling noise between leads of one
    phase, measured 5.48 against 5.23). The k=1 against k=4 width-ratio test now asks for `band_mode="phase"`, since the
    sqrt(1 + 1/k) factor belongs to that band.
  - **Stale stored figures, not regenerated:** lab, backtest and twin results that use the default seasonal naive band
    (`lab.py` around line 384; `scripts/backtest_twin.py`). The seasonal-naive band figures earlier in this file (around
    lines 928 to 930) are stale for the same reason. Regenerate before quoting any of them.
- **Merge note.** The seasonal branch was cut from an older commit. Only its two commits were applied, not the branch.
  One conflict in `forecast/baselines.py` was resolved by keeping `require_finite` (the NaN refusal from a4f8e91) and the
  branch's `BAND_MODES` check.
- **Gate on the merged tree (d63644f):** 2,808 passed, 4 skipped, 4 deselected, exit 0 (`scripts/check.sh`, offline).
- **Open defects, not started in this round:** paging past 200k in `goal.py run_tool`; checkpoint status not documented
  in API.md; `iteration_budget` and `budget_exhausted` never produced; refusal-loop bound; planner continuing after
  "network access blocked"; `provenance_label` missing from timesfm, report and lab tool outputs; health-check period
  disagreement with the backtest season (356 against 365); 30-step cycles invisible to `choose_season`; `prophet_lib`
  `dominant_period`; the Linux sandbox path is not verified on macOS; 5xx robots closed for the process lifetime (now
  moot while robots is off); web typecheck not run.
- **Not verified:** the Linux sandbox path; the web typecheck; the four named workflows have not been re-run against the
  new build.

## Mission 9: multi-signal data acquisition and deep research (opened 2026-11-26)

**Why.** Two live missions on 2026-11-26 failed to find their data. Nvidia (9e3408a4): Yahoo answered 401, Stooq answered
HTML, which `download_web_data` refuses, and the planner then saved a Plotly Apple sample as `NVDA_daily.csv` and asked the
person for a file. EPL (56465803): the only path to the table was ranked `web_fetch` passages clipped at 20,000
characters, so 11 of 20 clubs were found. About 60 sandbox cells re-parsed the same passage. Two "Adblocker detected"
pages were counted as fetched, and the stall rule fired at step 63 of 196. The existing `extract_page_tables` toolchain was
never called.

**Owner decisions (2026-11-26, asked through questions).**
- Price data: keyless public endpoints (for example Yahoo chart JSON, Nasdaq historical JSON, Stooq), on by default, each
  result tagged with its source and endpoint. No keyed price provider. Revised the same day after the research report:
  the owner accepts the endpoints' terms and keeps FRED as it is. Data-source terms live in `DATASET_LICENCES.md` and other
  Markdown docs only, never in tool outputs, prompts, briefs or chat.
- Search: no keyed search API. Harden the ddgs cascade instead.
- JavaScript pages: add Playwright Chromium rendering. Pages behind bot checks or CAPTCHAs stay refused; they are never
  bypassed.
- Charts: read embedded chart data first (SVG, Highcharts, Plotly, Vega, Chart.js, ECharts). Digitise chart images with a
  NIM vision model, and label every digitised value as read from an image.
- Reverse engineering (REA): WebPlotDigitizer, DePlot and MatCha model files, packaged deep-research agents (GPT
  Researcher, STORM), and table extraction packages (Docling, Camelot, pdfplumber).
- Is Agentic: not used. eve: reference only; its patterns are adopted in the Python runtime, not a port.

**Success criteria.** Each needs evidence recorded below, with each figure labelled measured, synthetic or source claim.
- **M9-1 Prices.** The Nvidia goal, unchanged, obtains at least two years of NVDA daily closes from a keyless endpoint, passes
  an entity check (the file is NVDA, not another ticker), backtests against the simple rules and forecasts two months with
  a range. Final status completed (met). A file whose ticker does not match its name is refused.
- **M9-2 Tables from pages.** The EPL goal, unchanged, lands all 20 clubs of the current table as a dataset from HTML or a
  rendered page. The planner reaches a points-path forecast without re-parsing passages. Final status completed (met).
  No sandbox cell re-parses the same passage more than twice.
- **M9-3 Charts.** An embedded chart (Highcharts, Plotly or SVG) on a page becomes a series that matches the embedded values
  exactly. A synthetic chart image is digitised with a median absolute error under 2% of the axis range, and the values
  are labelled as read from an image.
- **M9-4 Text signals.** Dated events found in news text (earnings, fixtures, policy decisions) become an event table that a
  forecast tool can take as covariates, each event with its source URL.
- **M9-5 Search.** On a fixed panel of 20 data-discovery queries, at least 15 return a relevant non-registry hit with ddgs
  alone. Registry hits are labelled as registry, never presented as search results.
- **M9-6 Blocked pages.** Interstitials (adblock, consent, bot check, CAPTCHA, login walls) are detected and returned as
  not ok with a reason. None is counted as a fetched page.
- **M9-7 Harness.** The stall rule catches a near-duplicate re-parse loop within 4 repeats. A malformed ask_user call is
  repaired or explained in one step. The planner is given today's date, so it does not guess a data end.
- **M9-8 Open research panel.** A panel of at least 8 open research missions across finance, sport, science, economics and
  energy, run through the API: at least 6 finish completed (met), none uses fabricated data, and every figure carries a
  provenance label.
- **M9-9 Gate.** `scripts/check.sh` passes on main after every merged batch.

**Evidence.** Added below as each batch lands.
