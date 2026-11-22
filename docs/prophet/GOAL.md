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
`tests/prophet_agent/test_licence_wording_guard.py` enforces it. The prompt and skill scrub is done. `evaluator.v9` is the
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
