# A7: Real-user audit (round 3)

Read-only audit of what six kinds of real user would try with Prophet today, what it gives them, where they get stuck or
misled, and what would fix it. Nothing was run: no API calls to port 8010, no dev servers, no npm, no git changes.

## Method and labels

- **[M] Measured**: read from the saved mission and sandbox files: `data/live_gates_{retail,wti}_rep{3,4}.json`,
  `data/live_panel_{beijing_rep1,beijing_rep2,electricity_rep1}.json`, the `data/sandbox/<run_id>/out/` folders named in
  each finding, and the 58 mission files in `data/*.json` that carry events and a status. Arithmetic was recomputed from
  the saved numbers.
- **[I] Inferred**: read from code and not reproduced live. Every claim tagged [I] needs a live or unit check before it is
  treated as fact.
- Mission duration in [M] is the span from the first to the last stored event, not the wall clock from submission.
- Scope limit: I did not open the web app in a browser, so accessibility and layout are not assessed beyond what the code
  shows (see the end).
- External references are listed at the end. I did not fetch them in this session; they are for orientation only.

### Measured baseline across the 58 stored missions [M]

| Measure | Value |
|---|---|
| Missions with events and a status | 58 |
| `completed` with `done_verified` | 30 |
| `partial` with `partial_unverified` | 28 (48%) |
| Missions where the agent asked the person any question (`questions`, `answered_questions`) | 0 |
| Duration, first to last event: median | 557 s (about 9 min) |
| Duration, first to last event: maximum | 1,756 s (about 29 min), a second-part run |
| Default budget per part | `max_seconds` 900; `auto_continue` 2 parts without the person (service.py:34, 303-313) |

---

## Persona walkthroughs

### P1. CFO or business owner: "next quarter's revenue with a range" from a messy Excel export

1. Uploads the workbook: `POST /api/notebooks/{nid}/sources` (app.py:335). The profiling card (SourcesPanel.tsx) lists
   sheets, header rows and issues, which is a good first step.
2. Sends the turn: "Forecast next quarter's net sales with a range" (`POST .../turns`, app.py:374). Quarter is not a
   parameter. The planner must turn it into days or weeks itself, and nothing says which it chose.
3. Measured retail run (rep3) answers "£472,815" as the 14-day total. The sum of its 14 daily points is £472,816 [M], so
   the arithmetic is right. The same goal rerun (rep4) answers "£1.14 million" [M]. The business owner cannot tell which to book.
4. Two ranges are given with no recommendation: "£390,667 to £554,963 if daily errors cancel" and "£192,095 to £794,892
   if they compound" [M]. The second is about four times wider, and nothing says which a CFO should use.
5. The headline total includes day 9 at **minus £3,606** [M]. Its own reason says the forecast still has "implausible
   day-to-day swings", but the answer body says "TimesFM verdict: ok" (Turn.tsx:389-395 shows the body first).
6. The turn ends `partial` (`partial_unverified`) [M]. The UI copy "This is a partial answer" sits above the table, so
   the number is still the first thing read.
7. Cleaning choices are taken silently: 135 missing days imputed as zero in rep3, but not in rep4 [M]. The CFO is never asked.
8. Good looks like: one headline (expected total, likely range, and the two sums labelled), a fixed quarter definition
   the person can change, a negative-day policy asked once, and a total that is withheld when a health check fails.

### P2. Demand planner: 300 SKUs, intermittent sales, promotions

1. Uploads one long file of SKU by date: fine up to 50 MB (API.md:258). Quick forecasts take at most 10 files.
2. Asks for "all SKUs, with intermittent handling". The panel path exists (`panel_tool.py`) and intermittent demand uses
   TSB (`timesfm_tool.py` `is_intermittent`, `_one`). Both are planner-only; no route exposes them.
3. The only per-entity output is prose. The 12-station Beijing run wrote a 12-row table into the answer, which took
   976 to 1,756 s and 65 to 107 tool calls [M]. A 300-row table in chat is not usable, and the budget would likely be
   hit first [I].
4. Results.tsx draws one `ForecastChart` from one `forecast` object (`ArtifactRun.forecast`). A panel run's 12 series
   in `out/forecast.json` do not reach the page, so a planner cannot sort, filter or find the SKU that broke [M for the
   file, I for the UI].
5. Promotions and holidays need a known-ahead covariate of the right length. That machinery exists (`drivers_tool.py`,
   `future_covariates`), but the API has no field for it (`TurnIn`, app.py:95-98). A planner can only hope the planner
   finds the file.
6. Intermittent SKUs: a band that is not centred on the point can mislead. The round 2 fix centres it on TSB, but the
   answer does not show the expected volume next to the range [I, from the audit file's #14 status].
7. Good looks like: a SKU table (point, range, class, verdict) with sort, filter and CSV, an exceptions list (SKUs
   with a failed check), and a promotion calendar attached once to the notebook.

### P3. Quantitative analyst: FX or oil, distrusts black boxes, wants backtests, controls, reproducibility

1. Asks "forecast WTI for 20 trading days with a range, compare with a naive rule". Two reruns of the same goal and
   dataset (sha256 identical) gave two centres: the mean_last_season rule at **90.76** (rep3, `partial`) and the naive
   rule at **89.53** (rep4, `met`) [M].
2. The rep4 answer says "no model tested beats the naive forecast on accuracy" [M]. Its own numbers show
   mean_last_season MASE 6.66 against naive 7.27 (the naive figure is in rep3's text) [M]. So a rule did beat naive by
   about 8%, and the chosen centre was the worse one. The status is still `met` with `done_verified` [M].
3. The rep3 answer on the same goal failed `baseline_compared` and asked for the centre to be switched [M]. Nothing in
   the UI shows that the two runs disagreed.
4. The backtest is prose, with no per-origin table, coverage per horizon, or the CI definition. Coverage is given as
   "73% of outcomes (nominal 80%)" [M].
5. Reproducibility: the forecast is deterministic. Two different missions on the same data gave identical per-station
   points (for example Aotizhongxin day 14 = 75.2, 10th to 90th = 24.3 to 157.0) [M]. That is a strength, but no
   answer says so, and the analyst cannot download the code (see P4).
6. Only display extensions are served (app.py:100, 288-304), so `backtest_*.json` and `program.py` are not downloadable.
7. Good looks like: a Methods panel per run (model, checkpoint, context, horizon, centre rule and why, backtest table,
   code hash), a rerun button that pins the same decisions, and a diff between two runs.

### P4. Data scientist: steer the model (covariates, holdouts, centre rule), rerun with changes, export code

1. Wants to set the centre rule, holdout windows, season and covariate role. `MissionIn` and `TurnIn` take only
   `goal`, `dataset_ids` and `model` (app.py:63-67, 95-98). Every steer goes through goal text, which is ambiguous.
2. The UI has Continue and Answer now at a checkpoint (Turn.tsx:253-281, api.ts:193-194). They add a part with a fresh
   allowance; they do not change the plan. Notebook instructions (2,000 characters, app.py:85) are the only persistent steer.
3. An MCP client has no way past a checkpoint. `mcp_server.py` treats `checkpoint` as not final (line 24). `_wait`
   returns on it (line 75), and `_answer_text` gives no next action for it (lines 53-67). There is no `continue` or
   `answer_now` tool [I, from code; not run live].
4. Export code: the sandbox writes `program.py` and `lib/` into `data/sandbox/<run_id>/` [M]. The artefact route
   serves only `out/` with display extensions (app.py:262-305). The code, the input snapshot and the manifest's package
   versions are not reachable through the API [M for the folder layout, I for the route].
5. Rerun with a change: there is no "same mission, one change" action. The person must restate the goal and hope the
   planner makes the same earlier choices. The rep3/rep4 split shows it does not [M].
6. Good looks like: a Settings or Assumptions panel that becomes the planner's fixed decisions, a "download
   reproducibility bundle" that has code, lib hashes, input hash, package versions and checkpoint, and an MCP
   `continue` tool so agent users have the same controls.

### P5. Operations manager: hourly electricity load with daylight saving

1. Uploads hourly load for a site or several clients, and asks for the next 48 hours with a range.
2. The scan checks the clock: `time.dst_shift` is raised as **major** for repeated or missing hours (prophet_scan.py:126,
   586-614) [I, from code]. The contract then blocks the forecast until the ledger records a decision that names the
   finding id (contracts.py:349-369) [I].
3. Only the planner can write that ledger decision. The person has no screen to say "treat the repeated hour as DST and
   sum it", or "use the local clock, the grain is 30 minutes". The source card has no timezone or DST setting [I].
4. TimesFM's tool arguments are `values, source, horizon, season, centre` plus controls (timesfm_tool.py:27). There is no
   frequency or timezone argument. The hourly season candidates (24 and 168, season.py:28) depend on the planner
   passing the right grain [I].
5. The measured electricity run (60 clients, daily) ended `partial`: "The analysis kept repeating the same step", with no
   7-day forecast and no total. The MT_030 to MT_060 backtest was clipped out of the answer [M].
6. The person is left with a `partial` status and no number, and no control to fix the cause.
7. Good looks like: a per-source time setting (timezone, DST policy, grain) applied in the loader before the scan, shown
   on the source card; a scan finding that offers the person a choice in plain words; and a repeat-step stall that
   names the step and offers a narrower question.

### P6. Enterprise IT and risk reviewer: audit trail, access, data handling, reproducibility, cost, failure modes

1. Audit trail: the SQLite event log, steps, questions, notebooks and sources can be exported as JSON
   (`GET /api/settings/export`, settings.py:456-462). The export is everything, including every tool result, and is
   not marked as containing the person's data.
2. Access control: none. The API is single-user, loopback only, and refuses browser writes from non-loopback Origins
   (API.md:13, 262). That is a fair local-tool stance, but there is no identity, role or per-notebook permission to
   point to in a risk review.
3. Data handling: the planner's context and tool results (column names, sample values, sample dates) go to the chosen
   model provider. Providers include NVIDIA NIM, Claude, OpenAI and Gemini (nim/providers.py:136 onward). The model
   picker shows no notice (ModelSwitcher.tsx), and no per-notebook "keep this data local or in this provider" control
   exists [I, provider list from code; sample values from the live evidence lists].
4. Secrets: `.env` on disk is mode 0644 [M]. The app writes its own `.env` at 0600 (settings.py:445), so the existing
   file was made some other way. Keys are not printed, per the design, but the file is readable by other local users.
5. Retention: uploads and missions persist until "delete all data" (settings.py:464-474), which is all-or-nothing. There
   is no time-based retention or per-notebook delete of files in the answer path [I].
6. Cost: `/api/usage` gives tokens per model and window, with ceilings set in Settings (app.py:491-497). A run's cost is
   not shown before it starts, and auto-continue can add two more 900-second parts without the person (service.py:303-318) [I].
7. Failure modes are well handled in copy: `outcome_unknown` avoids repeating a side effect (Turn.tsx:189-196), and a
   resumed run replays its ledger. The gap is that `partial` mixes unrelated causes (A7-04).
8. Reproducibility: determinism is real and measured (P3 item 5). A reviewer still cannot get the package versions or the
   code from the API, so the claim cannot be checked from outside.
9. Good looks like: a one-page data-handling statement in the app, a per-notebook provider permission, a
   reproducibility bundle, and a cost estimate and cap per run shown before it starts.

---

## Design principles

1. **Decisions are data, not prose.** Each analysis choice (cleaning rule, centre rule, season, holdout, time setting,
   horizon) is a typed decision with a source (person, default or planner), stored on the notebook and reused on rerun.
2. **Ask before a material default.** A missing-versus-zero, negative-value, DST or level-shift choice that changes the
   number is asked once, not decided silently. The skip path must say what it will assume.
3. **Numbers come from artefacts; words come from the model.** Tables and totals are rendered from `out/forecast.json`
   and the workbook. The model writes the interpretation, and every figure in the text is checked against the table.
4. **Status means something.** `met` only when every check passes. `partial` names the failed check in plain words.
   The same condition gets one status everywhere (`stalled` or `partial`, not both).
5. **Plain language first, method on demand.** Headline in decision terms ("expected total", "likely range", "checks
   passed"). MASE, CRPS, variance ratios and the planner's own labels go in an expandable Methods section.
6. **Show time and spend before they happen.** Estimate range and cap per run, no unattended extra parts unless the
   person has set it, and progress from the runtime's own checklist.
7. **Every output reproducible from a bundle.** Code, input hash, package versions, checkpoint, decisions and
   the run's manifest, downloadable, with a note that a rerun should match.
8. **Checkpoints are controls on every surface.** Continue, answer now, change a decision and cancel are available in the
   UI, the HTTP API and the MCP bridge, and documented in API.md.
9. **Data leaves the machine only with a visible, per-notebook permission.**

Licence and research-use wording stays in Markdown documents. None of the findings below adds any to the UI, prompts,
tools or API responses (the locked rule).

---

## Findings (ranked by impact)

Severity: H = a wrong or misleading answer can reach the person unflagged; M = the person is stuck or the claim cannot be
checked; L = wording or hygiene. Effort: S under a day, M a few days, L about a week, in this codebase.

### A7-01: Rerunning the same goal gives different answers, with different cleaning and centre choices

- **Severity:** H. **Effort:** L. **Risk:** medium (changes the planner's context and the turn API).
- **Evidence [M]:**
  - Retail, same file and goal (sha256 `327e35a6...`), rep3 vs rep4: rep3 imputes 135 missing days as 0 and clips three
    negative days to 0. rep4 imputes nothing and keeps two negative days, and the two runs name different dates. The
    14-day total is £472,815 (TimesFM centre) in rep3 and £1.14 million (naive centre) in rep4.
  - WTI, rep3 vs rep4: centre 90.76 (mean_last_season) vs 89.53 (naive).
  - Beijing, rep1 vs rep2: TimesFM panel MASE 0.65 (rep1) vs 0.94 (rep2). Level shifts differ: rep1 lists
    Tiantan, Wanliu and Wanshouxigong, while rep2 lists a Wanliu shift in 2014 and two Dongsi shifts.
  - Rep2's outlier rate runs from 5.8% to 100% of points flagged. Full 100% flagged is an implausible scan result that
    went into the answer unquestioned.
- **Root cause [I]:** each turn's planner chooses cleaning, centre and shift treatment again. The notebook keeps
  "decisions already made" only inside a mission (planner prompt, line 141), not across turns or reruns. The ask path
  does not fire (A7-05).
- **Proposed fix:**
  - Add `decisions` to the notebook: `PATCH /api/notebooks/{nid}` accepts `decisions: [{key, value, source}]`, with keys
    such as `missing_days`, `negative_values`, `centre_rule`, `season`, `level_shift_policy`.
  - Pass the list to the planner context as fixed, and reject a conflicting choice in the ledger unless it is changed
    through the notebook.
  - Show the decisions on the notebook and in the answer's Methods section, with their source.
- **Files touched:** `src/prophet_agent/api/app.py` (`NotebookPatch`, turn creation), `src/prophet_agent/api/notebooks.py`
  (or `notebooks.py` in `prophet_agent/`), `src/prophet_agent/runtime/goal.py` (planner context), `src/prophet_agent/prompts/planner.v22.md`,
  `web/src/components/NotebookView.tsx`, `web/src/components/SourcesPanel.tsx`.
- **How to test:**
  - API test: set `missing_days=zero` and `centre_rule=naive`, run the turn with a stub planner, and check that the
    `timesfm_forecast` call and `forecast_report` config match the decisions.
  - Live check, after the 8010 missions finish: rerun retail with the same decisions twice and compare the totals and
    cleaning counts. They should match to the penny.

### A7-02: Answer tables and totals do not match the forecast file they cite

- **Severity:** H. **Effort:** M. **Risk:** low.
- **Evidence [M]:**
  - Beijing rep2 answer (sandbox run `767561e9...`, the run it cites 13 times): the "14-day centre total" column reads
    1,174 for Aotizhongxin, 965 for Gucheng and 797 for Wanshouxigong. The file gives 1,079.5, 1,173.7 and 1,180.6.
    The answer's totals run in a neat descending order, the file's do not.
  - The "Step 1 range" column is identical to the "Step 14 range" column for all 12 stations. The file's day-1 band for
    Aotizhongxin is 4.5 to 114.2, while the answer gives 24.3 to 157.0, which is day 14.
  - The same mission's `unverified` list names exactly these figures ("1,098", "28.1", "142.5", "1,052"...). The system
    flagged them and the answer still printed them as a table.
  - Turn.tsx:389-395 renders the answer body in full and shows the unverified list in a small caption ("Not found in
    the results: ..."). The wording is also unclear about what "not found" means.
- **Root cause [I]:** the model writes the table as text. The grounding check (`runtime/grounding.py`) tests that a figure
  appears in the evidence, not that it belongs to the right row and column, and it does not gate the table.
- **Proposed fix:**
  - Render every forecast table in the answer from `out/forecast.json` (and the workbook) by a deterministic function.
    The model writes the sentence around it and may only refer to rows by name.
  - Grade each numeric cell against its source row, not against the evidence as a whole.
  - In the UI, show partial answers with the unverified cells highlighted, and reword the caption to "Could not be
    traced to the results: ...".
- **Files touched:** `src/prophet_agent/tools/report_tool.py` (table renderer), `src/prophet_agent/runtime/grounding.py`,
  `src/prophet_agent/runtime/contracts.py`, `web/src/components/Turn.tsx` (lines 389-395), `web/src/components/Markdown.tsx`.
- **How to test:**
  - Unit test on a fixture copied from `out/forecast.json` of run `767561e9...`: every table cell must equal its source
    value within rounding, and the Step 1 and Step 14 columns must differ.
  - Regression: replay the rep2 answer text through the grounding check, and confirm it is flagged.

### A7-03: Headline totals include negative or implausible days, and the health verdict contradicts the status

- **Severity:** H. **Effort:** S. **Risk:** low.
- **Evidence [M]:**
  - Retail rep3: the 14-day total of £472,815 includes day 9 at −£3,606. The answer then describes the point forecast as
    "model artefacts" to "read the range" for. The total is still the headline.
  - The same answer says "TimesFM verdict: **ok**". The `reason` field says the gap is not closed: "the forecast still
    produces negative point predictions ... implausible day-to-day swings" (rep3 `answer.reason`).
  - Retail rep4 gives "flat_unexpected" for the same goal, with a different centre.
- **Root cause [I]:** the health check result is written into the text but does not gate the total or the headline.
  `forecast/health.py` verdicts are not carried into the status.
- **Proposed fix:**
  - When the health verdict is not `ok`, or any point is negative for a series that cannot be negative, the total is
    not the headline. Show the range and the reason first.
  - Ask the person whether negatives are returns (keep) or errors (floor at zero), since net sales can be negative.
  - Make the verdict a chip on the answer (`health: ok / flat_unexpected / negative_points`) and not free text.
- **Files touched:** `src/prophet_agent/forecast/health.py`, `src/prophet_agent/tools/report_tool.py`,
  `src/prophet_agent/runtime/contracts.py`, `web/src/components/Turn.tsx`.
- **How to test:**
  - Unit test: a series with a −3,606 day and `health=flat_unexpected` yields no headline total, and the status is `partial`.
  - UI test: the verdict chip renders from the answer's structured field, not from the text.

### A7-04: The status mixes causes, and a `met` answer can contradict its own backtest

- **Severity:** H. **Effort:** M. **Risk:** medium (more answers become `partial`).
- **Evidence [M]:**
  - WTI rep4 is `met` / `done_verified`, yet its text says "no model tested beats the naive forecast", while its own
    backtest figures show a rule 8% more accurate than naive (A7-01 item 2). The `baseline_compared` contract did not
    fail, so the check does not test the claim the text makes.
  - Electricity (`partial`, "kept repeating the same step") gives no forecast at all, while `Turn.tsx:168-171` maps the same
    kind of condition to `stalled`, with different copy.
  - All 28 partial missions carry `partial_unverified`, whether the cause was a contract failure, a repeat, or a
    clipped output.
- **Root cause [I]:** `_baseline_gate` (contracts.py) checks that a backtest exists at the delivered horizon, not that the
  centre matches the best rule or that the text agrees with the numbers. The stop reason is a single label.
- **Proposed fix:**
  - Make `met` require: every contract passes; the centre is the best backtest rule, or the answer states why not with
    the two MASE values; and the text claims no more than the numbers show.
  - Split `partial_unverified` into named reasons (`contract_failed`, `figures_unverified`, `repeated_step`,
    `budget_part_ended`), each with its own copy and next step.
  - Map repeated-step stalls to one status everywhere.
- **Files touched:** `src/prophet_agent/runtime/contracts.py` (`_baseline_gate`), `src/prophet_agent/runtime/goal.py`
  (stop reason), `src/prophet_agent/api/service.py` (status mapping), `web/src/components/Turn.tsx` (lines 150-200).
- **How to test:**
  - Unit test: a centre of naive when mean_last_season has 8% lower MASE and no stated reason must not be `met`.
  - Unit test: each stop reason maps to one status and one copy string.

### A7-05: The agent almost never asks, so material choices are made silently

- **Severity:** H. **Effort:** M. **Risk:** medium (more interruptions, which the notebook decisions can reduce).
- **Evidence [M]:** none of the 58 stored missions asked a question. The planner prompt says to ask when "a missing-versus-zero
  choice ... matters" (planner.v22.md:139-141), and the retail run did impute 135 zeros with no question. The question
  card offers "Skipped: use your best judgement" (QuestionCard.tsx:16), which hands the choice back silently.
- **Root cause [I]:** the rule lives in a prompt, and nothing checks it. The scan already raises the right findings
  (`scan_decisions_recorded`), so the data for a question exists.
- **Proposed fix:**
  - Add a harness rule: when the scan has a blocker or major finding in a material category (missing-versus-zero,
    negative values, level shift, DST, intermittency) and no notebook decision covers it, the mission waits for a
    question before the forecast step. Reuse `check_scan_decisions` (contracts.py:349-369) to raise the question.
  - Ask once per notebook, then reuse the answer (A7-01).
  - The skip option says the assumption it will use, in plain words, and that assumption is recorded.
- **Files touched:** `src/prophet_agent/runtime/contracts.py`, `src/prophet_agent/runtime/goal.py`,
  `src/prophet_agent/runtime/questions.py`, `src/prophet_agent/prompts/planner.v22.md`, `web/src/components/QuestionCard.tsx`.
- **How to test:**
  - Replay test: a scan fixture with missing days and a negative value must end `waiting_user` with a question, not a forecast.
  - Live check: the retail goal, with decisions empty, should ask about missing days and negative values before forecasting.

### A7-06: Steering is limited to goal text; the MCP bridge cannot continue a checkpoint

- **Severity:** H. **Effort:** L. **Risk:** medium (API surface).
- **Evidence:**
  - `MissionIn` and `TurnIn` have only `goal`, `dataset_ids` and `model` (app.py:63-67, 95-98) [M, from code].
  - `timesfm_tool.py` takes `horizon`, `season`, `centre` and controls, but no route passes them from the person [I].
  - The MCP bridge has no `continue` or `answer_now` tool, and `checkpoint` is not final (mcp_server.py:24, 53-75) [I, not run live].
  - Measured: P3 item 1, the rep3/rep4 WTI split, is the same user intent with no way to fix the centre [M].
- **Root cause [I]:** the API was built for the planner. The person's controls were added to the UI later and not carried
  to the API docs or the MCP surface.
- **Proposed fix:**
  - Add an optional `assumptions` object to `TurnIn`: `horizon`, `centre_rule` (enum), `season`, `holdout_windows`,
    `known_ahead` (`[{source, column, role}]`), `time` (`timezone`, `dst_policy`, `grain`). Validate it with a Pydantic
    model and write it as decisions (A7-01).
  - Add MCP tools `prophet_continue` and `prophet_answer_now`, and document them in API.md.
  - Document `continue`, `answer-now`, `resume` and `settings/export` in API.md, which currently omits them.
- **Files touched:** `src/prophet_agent/api/app.py`, `src/prophet_agent/api/service.py`, `src/prophet_agent/mcp_server.py`,
  `API.md`, `web/src/components/Composer.tsx`, a new `web/src/components/AssumptionsPanel.tsx`.
- **How to test:**
  - API test: `assumptions.centre_rule=naive` must appear in the `timesfm_forecast` arguments in the ledger.
  - MCP test: a mission at `checkpoint` returns a `next` action and `prophet_continue` moves it to `running`.

### A7-07: Latency expectations are wrong, and runs spend more time and tokens than the person sees

- **Severity:** H. **Effort:** M. **Risk:** low for copy; medium for the budget change.
- **Evidence:**
  - API.md:128 says "Typical turns take 20 seconds to a few minutes". Turn.tsx:375 says "Longer requests can take a few
    minutes". Measured median is 557 s and the maximum 1,756 s across 58 missions [M].
  - The default lets a mission start two more parts by itself (service.py:34, 303-318). Each part is 900 s by default
    (`max_seconds`). Measured runs reached the second part [M].
  - The budget the person can set is per model per window (BudgetEditor.tsx). The per-part time and steps are global
    Settings. There is no per-mission time or spend cap.
  - The runtime emits `mission.progress` with `checklist_done`, `checklist_total`, `elapsed_s` and `next`, and no UI
    reads them (grep of `web/src` finds nothing for these keys) [M].
- **Root cause [I]:** copy and budget were set before the runs were measured, and progress was built but not shown.
- **Proposed fix:**
  - Change the copy to the measured range ("usually 5 to 15 minutes, up to about 30") and update API.md.
  - Add `budget.auto_continue` to `MissionIn` and `TurnIn`, default 0 for the UI. Show "Continue to keep going" at the
    checkpoint, as Turn.tsx already does.
  - Show progress from `checklist_done/total` and `next`, with elapsed time, in a live region that is not announced every tick.
  - Add a per-mission time cap in the composer, and show the estimated spend before the run.
- **Files touched:** `API.md`, `web/src/components/Turn.tsx` (line 375 and the progress area), `src/prophet_agent/api/app.py`
  (`MissionIn`), `src/prophet_agent/api/service.py` (lines 34 and 303-318), `web/src/components/Composer.tsx`.
- **How to test:**
  - API test: `auto_continue=0` stops at the first checkpoint and does not start a second part.
  - UI test: a running mission shows `checklist_done/total` and the `next` text from the event payload.

### A7-08: Developer instructions leak into the user-facing reason

- **Severity:** M. **Effort:** S. **Risk:** low.
- **Evidence [M]:**
  - WTI rep3 `reason` contains "record each one with ps.Ledger().add(stage, finding, decision) and put the finding id in
    the text". This comes from `contracts.py:366-369`, and `Turn.tsx:392` renders the reason verbatim.
  - Beijing rep1 `reason` says "sandbox read shows they are NOT FOUND" and "calibrate_intervals runs", which are
    internal tool names.
  - Beijing rep1 `reason` says Wanliu and Wanshouxigong are missing from `out/forecast.json`. That file has all 12
    series [M], so the message is also false.
- **Root cause [I]:** contract and stop messages are written for the planner, and the same string is shown to the person.
- **Proposed fix:**
  - Split each message into `user_reason` (plain words, shown) and `agent_hint` (for the planner, never shown).
  - Add a test that fails if `user_reason` contains a code token (`ps.`, `()`, a tool name).
- **Files touched:** `src/prophet_agent/runtime/contracts.py`, `src/prophet_agent/runtime/goal.py`,
  `src/prophet_agent/api/service.py` (serialisation), `web/src/components/Turn.tsx`.
- **How to test:**
  - Unit test: a grep over `user_reason` values for code tokens and tool names returns nothing.
  - Regression: rerun the rep3 WTI contract with the new messages.

### A7-09: Panel and multi-entity results have no in-app view

- **Severity:** M. **Effort:** M. **Risk:** low.
- **Evidence:**
  - `out/forecast.json` carries all 12 Beijing series with dates, point, q10 and q90 [M]. The page gets only one
    `ForecastChart`, built from one `forecast` object (Results.tsx; `app.py:262-285`) [I].
  - The workbook and the grid PNG are the only per-entity outputs in the UI [M, file list].
  - Forty-plus SKUs or 60 clients would need the same view. The answer tries to carry it as text (A7-02).
- **Root cause [I]:** the artefact listing is built for one forecast per run.
- **Proposed fix:**
  - Add a `ForecastTable` component that reads `out/forecast.json` (a new route, or the existing
    `/api/artifacts/{run}/out/forecast.json`), with sort, filter, a per-row expander for the chart, and CSV download.
  - Show the exceptions first: rows with a failed check or a health flag.
- **Files touched:** `web/src/components/Results.tsx`, a new `web/src/components/ForecastTable.tsx`,
  `src/prophet_agent/api/app.py` (listing).
- **How to test:**
  - UI test on the Beijing run `767561e9...`: 12 rows, sortable, with values equal to the file.
  - At 375 px: the table scrolls inside its region, with no page-level horizontal scroll.

### A7-10: Time settings (timezone, DST, grain) cannot be set, so scan blockers have no decision path

- **Severity:** M. **Effort:** M-L. **Risk:** medium.
- **Evidence [I, from code; not run on a DST fixture]:**
  - The scan rates a DST shift as major (prophet_scan.py:613). The contract then blocks until a ledger decision exists
    (contracts.py:349-369).
  - Only the planner writes the ledger. TimesFM takes no frequency or timezone argument (timesfm_tool.py:27).
  - Hourly season candidates (24, 168) depend on the grain being right (season.py:28).
- **Root cause [I]:** the contract is correct to block, but the only resolver is the planner, and the person has no
  place to state the time rule.
- **Proposed fix:**
  - Add a per-source `time` setting: timezone, DST policy (`sum_repeated_hour`, `keep_as_is`, `drop_repeated`), grain.
    Apply it in the loader, before the scan.
  - Show it on the source card (`SourcesPanel.tsx`), and let the scan blocker be resolved by choosing a policy, which
    writes the ledger entry.
- **Files touched:** `src/prophet_agent/sandbox/prophet_lib.py` (loader), `src/prophet_agent/sandbox/prophet_scan.py`,
  `src/prophet_agent/api/app.py` (`SourcePatch`), `src/prophet_agent/runtime/contracts.py`, `web/src/components/SourcesPanel.tsx`.
- **How to test:**
  - Fixture test: a two-hour-per-year DST series, scanned with and without the policy. Without it, the scan is blocked
    with `time.dst_shift`. With it, the scan passes and the ledger holds the decision.
  - Live check later: a 48-hour electricity forecast end to end.

### A7-11: No reproducibility bundle, and the code cannot be exported

- **Severity:** M. **Effort:** M. **Risk:** low (the code may hold sample data, so check it first).
- **Evidence:**
  - Determinism is real: identical per-station forecasts across two missions on the same data [M].
  - Each run folder has `program.py`, `lib/`, `in/` and `manifest.json` [M]. The manifest has `limits`, `outputs`,
    `returncode`, `stdout` and `stderr`, with no package versions and no input hash [M].
  - The artefact route serves only `out/` with display extensions (app.py:100, 288-304) [M for route, I for effect].
  - No answer tells the person that outputs are deterministic, and no bundle exists.
- **Root cause [I]:** reproducibility was built for the runtime's own resume, and not for the person's audit.
- **Proposed fix:**
  - Add `GET /api/missions/{id}/repro` returning a zip: each run's `program.py`, `lib/*.py` with sha256, the input
    sha256 from the dataset record, `manifest.json`, the package versions (`importlib.metadata` for timesfm, numpy,
    pandas, lightgbm), the checkpoint id and the decisions in force.
  - Check the bundle for key-shaped strings before it is served (`nvapi-` and similar).
  - Add the download to Results.tsx, with a line: "Rerunning the same code on the same input should give the same figures".
- **Files touched:** `src/prophet_agent/api/app.py`, `src/prophet_agent/api/service.py`, `src/prophet_agent/sandbox/runner.py`
  (manifest fields), `web/src/components/Results.tsx`.
- **How to test:**
  - API test: the bundle's input sha256 equals the dataset's sha256, and the bundle has no `nvapi-` string.
  - Live check: run the bundle's `program.py` with `python -I` in a fresh directory, and compare `forecast.json`
    sha256 with the original.

### A7-12: Data leaves the machine without a visible, per-notebook permission; retention is all-or-nothing

- **Severity:** M. **Effort:** M. **Risk:** low.
- **Evidence:**
  - Providers include NVIDIA NIM, Claude, OpenAI and Gemini (nim/providers.py:136 onward) [M, from code].
  - Tool results with column names and sample values go into planner context (the evidence lists in the live files
    show samples such as "e.g. ['MT_001', ...]") [M].
  - The model picker shows no notice of where the data goes (ModelSwitcher.tsx) [M, from code].
  - `.env` on disk is 0644 [M]. The app writes its own at 0600 (settings.py:445), so the on-disk file predates that rule.
  - Retention: "delete all data" is the only removal (settings.py:464-474). The export has every tool result and is not
    marked as containing the person's data (settings.py:456-462) [I, from code].
- **Root cause [I]:** the design says "nothing leaves except NIM" (API.md:5), which is no longer true with four providers.
- **Proposed fix:**
  - Show, next to the model name, "Your column names and sample values are sent to <provider>".
  - Add a notebook `allow_external_model` flag, enforced in turn creation (422 if the chosen provider is external).
  - Chmod 0600 `.env` on startup, and warn if it is looser.
  - Add a retention setting (delete uploads older than N days) and a per-notebook delete.
  - Mark the export as containing the person's data, with an option to drop sample values.
- **Files touched:** `src/prophet_agent/nim/providers.py`, `src/prophet_agent/api/service.py`, `src/prophet_agent/settings.py`,
  `src/prophet_agent/api/app.py`, `web/src/components/ModelSwitcher.tsx`, `API.md` (line 5 is out of date).
- **How to test:**
  - API test: with `allow_external_model=false`, a turn using a non-NIM provider returns 422.
  - File test: after startup, `.env` mode is 0600.

### A7-13: Headline answers use specialist vocabulary, and the chart wording overstates calibration

- **Severity:** M. **Effort:** S. **Risk:** low.
- **Evidence [M]:**
  - Retail rep3 headline and body use "centre line", "P10-P90", "ADI=1.0, CV²=0.274", "shape ratio 1.05,
    vs-last-value 3.06", and "Lo-MacKinlay variance ratio" in the WTI run. A CFO cannot act on these.
  - Results.tsx reads each band as "8 in 10 outcomes are expected between ...". The Beijing run shows the bands are
    not all calibrated the same way: coverage 79.7% in rep1, 83.5% for TimesFM in rep2, and a "calibrated band" for
    Nongzhanguan only (rep1 answer).
- **Root cause [I]:** the answer is written for the analyst, and the chart text is generic.
- **Proposed fix:**
  - Two layers in the answer: a headline in decision terms (expected, likely range, the check that passed or failed),
    then Methods with the technical terms, collapsed by default.
  - In the chart caption, state the band's source: "raw TimesFM band" or "calibrated band (held-out coverage X%)".
- **Files touched:** `src/prophet_agent/prompts/planner.v22.md` (answer section), `web/src/components/Results.tsx`,
  `web/src/components/Turn.tsx`.
- **How to test:**
  - Lint: the first paragraph of a finished answer must not contain a listed technical term (for example MASE, ADI, CV²).
  - UI test: the chart caption names the band's source from the artefact's `config`.

---

## Prioritised feature and UX list

P0 is before the next live campaign. P1 is the next release. P2 follows.

| Pri | Item | Finding | Surface |
|---|---|---|---|
| P0 | Withhold the headline total when health is not ok; no negative totals without a question | A7-03 | answer, status |
| P0 | Render every table from the artefacts; highlight unverified cells | A7-02 | answer, Turn.tsx |
| P0 | Status = contracts passed; split the partial reasons | A7-04 | status, contracts |
| P0 | Plain-language `user_reason` with no code tokens | A7-08 | reason text |
| P0 | Correct the latency copy and API.md | A7-07 | copy, API.md |
| P1 | Notebook decisions (cleaning, centre, season, shift, time) shown and reused | A7-01, A7-05 | notebook, planner |
| P1 | Ask before a material default; skip states its assumption | A7-05 | question card |
| P1 | `auto_continue` off by default in the UI; per-mission time cap; show progress from the checklist | A7-07 | composer, Turn.tsx |
| P1 | Assumptions panel (horizon, centre, season, holdout, known-ahead, time) | A7-06, A7-10 | composer, sources |
| P1 | Per-notebook external-model permission and provider notice | A7-12 | model picker, service |
| P2 | ForecastTable for panel runs with CSV | A7-09 | results |
| P2 | Reproducibility bundle with a rerun check | A7-11 | results, API |
| P2 | MCP `continue` and `answer_now`; document the missing routes in API.md | A7-06 | MCP, API.md |
| P2 | Two-layer answer with Methods collapsed; plain-language vocabulary | A7-13 | answer |
| P2 | Retention setting and per-notebook delete | A7-12 | settings |

## Design-level UX requirements (for the list above)

- **Controllability:** pause at a checkpoint; continue or answer now; change one decision and rerun from the planner step
  that uses it (not from scratch); edit assumptions before a run; answer questions in the UI and through MCP.
- **Transparency:** each figure labelled measured (from the run's files), derived (arithmetic on measured figures) or
  assumed (a decision the person did not make). The evidence trail shows the code step behind each figure.
- **Consistency across reruns:** decisions reused; a rerun with unchanged inputs and decisions must match to the
  stated rounding. Report any difference with the decision that caused it.
- **Latency:** show a range before the run, progress during it, and a clear line at each checkpoint on what is next and what it costs.
- **Error states:** each `partial` or `stalled` names its cause, the figures that are not traceable, and one next step.
- **Accessibility (from code only):** the progress area should be a live region that updates on meaningful change, not
  every poll; unverified figures should not rely on a small caption alone. Reduced motion is handled (index.css:88).
  Tables are focusable regions (Markdown.tsx:180). The chart is hidden from screen readers and described in its
  figcaption (Results.tsx). A browser check is still needed before any accessibility claim.

## Latency expectations (measured, not benchmarked)

- Stored missions (n = 58): median 557 s, maximum 1,756 s, first to last event.
- Default budget: 900 s per part, with two automatic parts after the first, so up to about 45 minutes before a
  checkpoint needs the person.
- Suggested copy: "Usually 5 to 15 minutes, and up to about 30 for panels." The figures come from one machine and one
  set of models, so they should be re-measured before they are published.

## Not assessed

- Browser behaviour, layout at 375 px, contrast, and keyboard flow were not tested. The findings above are from code and
  saved files.
- Live API behaviour was not exercised. Items tagged [I] need a live or unit check.
- The "Outside context" sections in the WTI answers cite external sources. I did not check them.

## External references (orientation, not fetched in this session)

- NIST, AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework (for the audit and risk
  vocabulary in P6).
- Hyndman and Athanasopoulos, Forecasting: Principles and Practice, prediction intervals and forecast evaluation:
  https://otexts.com/fpp3/prediction-intervals.html (for the band wording in A7-13).
