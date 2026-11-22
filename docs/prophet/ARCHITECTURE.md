# Prophet: architecture decisions

Labels: **[V-docs]** vendor or primary documentation read by a research worker (summaries, not full reads),
**[V-live]** observed in a live call here, **[R]** research-note synthesis needing a check, **[H]** hypothesis.
Research notes: `research_notes/Durable agent runtime for Prophet/` and `research_notes/Agent platform study/`.

## 1. Runtime: build a small durable loop; adopt no framework for the core

**Decision.** Write our own loop on SQLite in WAL mode: an append-only event log, a step ledger and a tool-call ledger
with idempotency keys, a single-writer lease per mission, and cooperative cancel and pause. Keep the NIM transport
ours (`src/prophet_agent/nim/`).

**Why.** No framework fits a single-user, local, Python/FastAPI/SQLite app without adding a server process:
- Temporal, Restate and Inngest each need a separate server [R].
- LangGraph's checkpointers give durability, but its graph model would own the loop shape [R].
- The Claude Agent SDK resumes sessions only and has no base-URL option, so it cannot point at NIM [R, from the runtime
  study].
- OpenAI Agents SDK: approval pauses and `RunState` serialisation are the right pattern, but approval does not
  guarantee exactly-once execution, so idempotency keys are required either way [V-docs].
- DBOS Transact (MIT, in-process, SQLite default, step-level checkpoints) is the closest library fit if we later
  prefer not to write recovery code [V-docs, PyPI 3.6.0 on 2026-11-17].

**Rules carried over from the research.** Steps are at-least-once, so every side-effecting tool call carries an
idempotency key and records its result before the step is marked done. Replay reads the ledger, never re-asks the
model for a completed step.

## 2. Context and caching

**Decision.** Prompt layout is fixed: system and tool schemas first (sorted keys, no timestamps), then the mission
brief, then a pointer to durable state files, then the append-only transcript, then volatile data last. Keep the
primary model fixed per mission, because a fallback to another model loses the prefix cache. Keep full transcripts on
disk so a summary can always be checked against the source.

**Compaction.** Clear old tool outputs in batches first, because observation masking roughly matched LLM summarisation
on solve rate and both halved cost in the JetBrains study (arXiv 2508.21433) [V-docs, SWE-bench setting, not ours].
LLM summaries lengthened trajectories by up to 15 % [V-docs]. Summarise with a structured handoff at about 60–70 % of
the input budget. The 60–70 % figure is our design proposal, not a sourced threshold [H].

**NIM caching.** NIM.md reports `cached_tokens` of 32 and 48 on Laguna and 0 on Ultra. Our live calls returned
`cached_tokens: 0` for Ultra, Super-VL, Nano-Omni and GLM on a ~50-token prompt, in two runs each. Laguna reported 0 on
its first call and 48 on the identical repeat [V-live, n=2 per model]. So hosted prefix caching exists on at least
Laguna and is model dependent. NVIDIA's public KV-reuse documentation covers self-hosted containers only, so the
hosted block size and TTL are unverified [V-docs]. We will not depend on caching for cost until the primary model
(Ultra) shows a hit on a longer prompt [TODO].

## 3. NIM transport and model chain

**Decision.** Use the NIM.md §4 contract, with four fixes over the reference client: a wall-clock budget bounds every
HTTP request (F1); acquisition is non-blocking across keys (F2); Retry-After is honoured and capped at 120 s (F3); 401/403
retire a key for 10 minutes rather than forever (F4); a model is demoted only on 404/410/413/503 or on an empty answer
that did not stop for length (F5); usage is kept (F6).

**Live results (2026-11-20).**

| Check | Result |
|---|---|
| 1-token call per key | 6 of 6 slots 200 (0.38–0.74 s) |
| Invalid key first | 403, key retired, real key served (0.85 s) |
| Claim schema, thinking off (two runs) | all 5 chain models valid JSON both times; Ultra 1.18 s then 5.36 s, Super-VL 0.79 s then 0.78 s, Nano-Omni 1.20 s then 0.84 s, Laguna 0.96 s both, GLM 100.17 s then 90.86 s |
| Claim schema, thinking on (one run) | all 5 valid JSON; Ultra 2.95 s, Super-VL 2.37 s, Nano-Omni 2.06 s, Laguna 0.90 s, GLM 73.72 s; see §3.1 |
| GLM latency | 74–100 s in three samples against NIM.md §2's 9–11 s; consistent, not an outlier [V-live, n=3] |

GLM's 74–100 s is the reason it stays last in the chain. Its `reasoning_tokens` were 5–6, so the delay is not
reasoning time; it looks like queueing or time to first token, which NIM.md §2 also flags for gateways. A call that
reaches GLM can therefore use a large share of the default 240 s budget.

### 3.1 Thinking-on probe

Run on 2026-11-20 with the same trivial claim prompt, thinking on, `max_tokens` 8192. All five models returned valid
schema JSON. Reasoning tokens used: Ultra 33, Super-VL 97, Nano-Omni 119, Laguna 0 (no thinking mode), GLM 6.
**This does not validate NIM.md's 25–60 s thinking-on latency.** The prompt is tiny and the models reasoned for a
handful of tokens, so thinking-on looked as fast as thinking-off (2–3 s). That latency claim needs a prompt that forces
real multi-step reasoning, and it is still [TODO].

## 4. Package name

**Decision needed.** The legacy audit proposed `src/prophet/`. This work uses `src/prophet_agent/`, because a package
called `prophet` collides with the Meta forecasting library on PyPI that many engineers will have installed. Options:
keep `prophet_agent`, or choose a new short name. The user decides.

## 5. TimesFM 3: licence gate

**Blocking for any client data.** The TimesFM 3.0 weights are under the "TimesFM Non-Commercial License v1.0". Permitted
use is research not tied to revenue or production. Results may not be used in commercial decision-making, client
deliverables or paid services [V-docs, via a summarising fetch; the raw licence text needs a direct read]. Code is
Apache-2.0. Weights up to 2.5 are Apache-2.0 [V-docs].

**Consequence.** Prophet's TimesFM tool stays R&D-only. Before any client use, either obtain legal sign-off, use
TimesFM 2.5 (Apache-2.0), or use the statistical baselines. This goes to Bill or Ly for a decision. It also matches the
repo's own rule in CLAUDE.md.

**Model facts (model card, Google blog, PyPI) [V-docs].** 330M parameters; decoder-only; patches of 32 in and 64 out; nine
quantiles from the 10th to the 90th percentile. Maximum context and horizon are not published. The repo uses the full
global context of 15,360 (the legacy 512 cap was a lab choice and is removed); that is a setting, not a model limit. Covariates are supported (past-only and
past-future) but there is no independent evidence of covariate skill, and our own calendar covariate gave no gain on
the twin [R]. No training hook is in the 3.0 package; a LoRA example exists in the GitHub README with no stated version
[V-docs, unverified for 3.0].

**Routing.** Intermittent series go to TSB or Croston, not TimesFM [R]. Test covariates with and without, using a
rolling-origin backtest scored with MASE, CRPS or WQL, pinball loss and coverage [R].

## 6. Decisions

Recorded on 2026-11-20 from the user's instructions. Items marked *open* still need a person's call; none blocks the
gate.

1. **Legacy `src/lab` and the `timesfm-erp` package: deleted.** The user instructed deletion of the old stacks, the
   legacy model and tuning, and their tests. Done in commit 9cf4c01.
2. **Package name: `prophet_agent`, kept.** Not separately approved. *Open:* rename later if preferred (§4).
3. **TimesFM 3 licence: the full official model is in use.** The user stated that the project is not commercialised and
   that they have full right to use the official TimesFM 3 model. Recorded as their decision. The licence's research-use
   terms still bind any commercial, client or paid use, so the gate in §5 stays. *Open:* confirm in writing (Bill or Ly)
   before any client use.
4. **`NIM.md` and `.env`.** `NIM.md` stays uncommitted; the whole-tree scan found no key-shaped string in it. `.env` is
   git-ignored and was never committed, but it is mode 644. *Open:* the user decides whether to tighten it to 600.
5. **Commit and push: authorised.** The user's instruction to keep going until everything is shipped and pushed end to
   end is the approval. Commits are local and then pushed to `origin/main`.

## 7. Live results, 2026-11-20

- Forecast-pack mission end to end: planner (NIM) → `timesfm_forecast` (TimesFM 3, MLX) → `forecast_report` → five
  files, each re-hashed against the manifest. Evidence in GOAL.md G9 and G11.
- Two planner defects were found in the live run and fixed: the goal was lost from the evaluator's prompt once the
  evidence grew, and the planner did not see tool argument names. Both are covered by tests.
- Synthetic-twin backtest, walk-forward, 26 origins: TimesFM 3 is +6.9 % better than the best baseline on head-set WAPE,
  against a 15 % bar. The bar is **not met**. Evidence in GOAL.md G13. The numbers are synthetic replay, not observed
  results.

## 8. Answer quality, charts, large data and rendering (2026-11-20)

Decisions taken after reading how comparable agents and chat consoles are built. The notes behind them are kept locally
in `research_notes/` (git-ignored). Source for every claim was public documentation or published source read in this
session; the reverse-engineering (REA) tools were not connected, so no shipped binary was analysed.

- **Final answers are composed in a free-text call.** Schema-bound decoding drops newlines inside strings (measured live:
  0 line breaks against 11 in plain text), which produced one-paragraph answers. The evaluator decides whether the goal is
  met; `composer.v1` writes the Markdown from the evidence. [verified live]
- **Every figure is checked against the evidence.** `runtime/grounding.py` compares each number in the answer with the
  tool output at the precision written; one repair pass, then a partial answer that lists what could not be verified.
- **Streaming and timeouts.** Every NIM request already streamed. What changed: the read timeout is per chunk (idle), and
  the wall-clock budget is per logical call, so a long answer is not cut while tokens are arriving.
- **Code generation discipline.** `planner.v3`: one step per cell, typed loaders (`pl.load`), print every transformation's
  effect, a bounded repair loop with a probe after two failures, a second-route check on numbers that matter, a final
  `RESULT name: value` cell, and a visible step budget.
- **Charts.** `pl.chart` is the single route: a title that states the finding, units and period in the subtitle, bars from
  zero, direct end labels instead of a legend, Okabe-Ito colours, at most six lines or fifteen bars, nested forecast bands
  with the origin marked. It checks overlap and clipping, and writes the plotted numbers to a CSV beside the image so a
  quoted figure can be traced. Charts an analysis draws are shown inline in the console.
- **Large files.** `pl.size_report` and `pl.query` run DuckDB over CSV or Parquet where the file sits, with a 1 GB memory
  cap, spill to a temp folder, access limited to the named files, and a row cap on what returns. Totals come from the full
  file; samples are for shape only and must be disclosed. Forecasts are made at the level the decision needs, up to 100
  named series per TimesFM call.
- **Rendering.** `react-markdown` with GFM and `rehype-sanitize`: raw HTML is not parsed, links open with
  `rel="noopener noreferrer"`, tables scroll and right-align numeric columns, snippets have a copy button, and Mermaid
  loads on demand in its `strict` security level, falling back to its source when it cannot draw. Answers stored as one
  line are re-broken where list markers are visible.
- **Not met, stated plainly.** The 15 % improvement bar over the best baseline is still not met on the synthetic twin.

## 9. Analyst harness, finish gate and routing (2026-11-21)

Decisions, with the evidence that forced each. Full reasoning: `reports/Principal analyst agent harness design.md`.

1. **Skills are read on demand, not pasted into the planner.** Eleven playbooks in `src/prophet_agent/skills/` are served by the
   `analyst_skill` tool (list, then read one). The planner prompt stays at rules 1 to 57; detail loads when a step needs it.
2. **`ps` (prophet_stats) sits beside `pl` in the sandbox.** Every `prophet_*.py` module is copied into each run. `ps` is causal
   by default (trailing windows, one-sided imputers) and reports every change it makes. A cleaning step is kept only if
   `ps.backtest_treatment` says it helped on origins that never saw the data it was judged on.
3. **"Done" is a check on the evidence.** `runtime/contracts.py` runs when the evaluator says met: data inspected before the forecast
   that stands, forecast compared with baselines, forecast health repaired, deliverables present, and an analyst ledger of at
   least four stages, each with a finding and a decision. A blocking failure returns to the planner twice at most, then the
   answer leaves labelled `partial_unverified` with the failing check names. The result carries a typed `stop_reason`.
4. **The brief is data.** The goal, what counts as done, what must not change (including the 15 % bar) and the budget are stored
   in the `contracts` step, so a summary cannot drop them.
5. **Quota groups.** A provider whose limit belongs to the account (OpenRouter) shares one pacing window and one cooldown across its
   keys. Other providers keep per-key windows. Four keys do not buy four buckets where the vendor says they do not.
6. **Routing never reports a rate limit.** Found live and fixed: all keys resting past one attempt's budget raised a failure
   (now waited out); Google answers a bad key with HTTP 400 (the key is retired, not the model); two unusable strict-schema replies
   switch the model to plain JSON mode before it is dropped.
7. **The shape check cannot judge a short window.** Measured on 13 CO2 holdouts, "flat" fired on 7 where TimesFM beat seasonal naive on
   all 13. Under a quarter of a cycle the verdict is `check` and the backtest decides.
8. **Grounding accepts marked approximations.** "About 44,500" matches a printed figure within one percent; an unmarked figure still
   has to match to its last written digit, and a range endpoint is never an approximation.

Not decided by evidence (labelled): the 2 % improvement bar of `backtest_treatment`, the four-stage ledger minimum, the 1 % tolerance and
the quarter-cycle window are proposed defaults to calibrate on synthetic replay. Whether smoothing, outlier replacement or imputation
improves a TimesFM backtest is unmeasured.

## 10. Durable goal mode, learned from Claude Code's `/goal` (2026-11-22)

Source: `docs/prophet/goal_mode_rea.md` (mechanism read from the shipped Claude Code binary with REA) and the research report
`reports/Principal analyst agent harness design 2.md`. Decisions:

1. **A derived mission checklist rides in every planner prompt.** `contracts.progress` lists what a forecast needs (inspect, ledger
   stages, baseline comparison, forecast, health, requested files) from the evidence alone, marks the first undone item, and is
   never stored, so it cannot drift and a resumed run sees the same list. It is the feature-list device of Anthropic's long-running
   harness and Manus's todo recitation, built from our own deterministic checks. It is shown for forecast goals only.
2. **Compaction keeps the asks.** `compactor.v2` writes fixed sections (ASKS copied unchanged, FOUND with step references, DONE AND
   DECIDED, ERRORS, OPEN, SUB-AGENTS, NEXT with a quote), and a memo cut to fit removes its middle, never its head.
3. **Progress is an event.** One `mission.progress` event per step (step, tool calls, elapsed, checklist done of total, next item).
4. **A mission keeps going by itself.** The service starts the next part when a part ends at a checkpoint, up to
   `AUTO_CONTINUE_PARTS` (2, a proposal) per mission or `budget["auto_continue"]`; after that the person chooses Continue or
   Answer now. Stored steps replay, so nothing is asked of the model twice.
5. **Not adopted, and why.** `/goal`'s judge reads the whole transcript and has no deterministic floor; ours reads a compacted
   view, so the deterministic checks stay the authority and the evaluator's verdict stays advisory. `/goal`'s eight-block override
   is not copied: `MAX_REJECTED_FINISHES` and `MAX_CONTRACT_BLOCKS` end a loop with a labelled partial answer instead.
6. **Auto-continue needs progress.** A further automatic part starts only if the checklist advanced during the last one
   (`budget["auto_progress"]`); otherwise a `mission.auto_continue_stopped` event is written and the person decides. A mission
   that is not moving gains nothing from more of the same.
7. **Earlier results are carried, not rebuilt.** Each `sandbox_python` run starts with an empty `out/`. `carry=[{run_id, file}]`
   mounts earlier outputs read-only at `in/carried/<name>`. Live evidence: a mission re-ran its whole pipeline five times to get a
   file back before this existed.
8. **Helpers explain a wrong read.** `(value, report)` helpers return a `Pair` (a tuple) whose text-key read answers from the report
   or names what the function returns; `sc.scan` accepts `entity_col` and returns a result readable as `res.findings`. Hand-tuning
   TimesFM (context, ignored points, znorm, symmetric averaging) is blocked by the `controls_tested` contract until `control_lab`
   has run. Compaction folds at least three observations at a time.

## 11. Sub-agents v2, self-hosted web research and interval calibration (2026-11-22)

Sources: `reports/Prophet research agents and datasets.md` (deep research), `docs/prophet/agent_platforms_rea.md` and
`docs/prophet/subagents_architecture.md` (Claude Code, Cursor and others read from shipped binaries with REA). REA setup note: the MCP server
needed Node 24 and the system Node was 26, so it runs through `npx -y -p node@24 -p rea-agents@latest -- rea ...`; the MCP tools appeared only
after a Claude restart. Decisions:

1. **Sub-agents are typed, briefed, scoped and isolated** (`runtime/agents.py`). Types `general`, `researcher`, `analyst`, `modeler`,
   `verifier` each carry a tool allowlist, a step budget and a role line. A delegation carries `task`, `deliverable` and `constraints`; the child
   starts with a fresh history and memo, so the parent's context is not copied (Claude Code's `Task` tool, Codex's workers). Depth is 1: a child
   cannot delegate. Up to 4 run in parallel (`tasks`), 24 per mission. A child can be resumed by id and continues from its own steps.
2. **Hand-back is a report with evidence.** The child's answer returns files as `run_id:path` and figures; a figure not present in the
   child's own tool results is flagged `unverified` deterministically, never by the model's word.
3. **Each agent has notes** (`agent_notes` table, scopes `main`, `a1`, `a2` ...; `notes` tool). A child writes only to its own scope and can
   read the main notes, which is how a long mission keeps decisions outside the context window.
4. **Replay stays stable under parallelism.** A parallel child numbers its steps from a reserved block (`Ctx.branch`), so the same program asks
   for the same step numbers on every run. A replay of a finished delegation makes zero model calls (`tests/prophet_agent/test_subagents.py`).
5. **Web research is self-hosted** (`research/`, tools `web_search`, `web_fetch`, `web_research`): ddgs with a backend cascade, a disk cache,
   an SSRF and robots guard with per-host delay, trafilatura then bs4 extraction, source credibility tiers, BM25-lite passage ranking and
   quote verification. Two skills (`web-research-playbook`, `market-and-macro-research`) give the fan-out plan: market, policy, news, base rates,
   then known-ahead drivers. Research output is evidence for drivers and context, never a forecast.
6. **Intervals are calibrated, never narrowed** (`forecast/calibrate.py`, tool `calibrate_intervals`, `forecast_report(calibrate=true)`):
   split-conformal scaling of the 10-90 band per horizon third from past windows, validated on held-out windows; applied only when clearly
   closer to 80 percent. The point forecast is never changed.
7. **Planner v14** adds rules 65 to 69: delegate independent work with a brief, research external drivers for market series, keep notes,
   go atomic before modelling, calibrate the range.

Not adopted: nested delegation (cost and loss of control), a shared mutable workspace between children (children write to their own run
folders and hand files back), and a model-written verdict on a child's figures.

## 12. Audit of 2026-11-22 and what changed (harness-led delegation, computed trust, one procedure)

Six read-only audits (code and data, maths, research, analyst ideology, agent runtime, TimesFM coupling; notes in
`research_notes/Prophet audit 2026-11-22/`, git-ignored) scored the slices 18 to 35 out of 100 against a principal data scientist.
Measured from the ledger: 527 of 982 sandbox cells failed (54%); one delegation in 1,983 planner decisions; stall detection never
fired on repeated failures because every run id differs; answers misquoted calibration and coverage, cited a web figure no page
contained (EUR/JPY "IMF 162.34", the agent's own segment mean), and delivered TimesFM when a simple rule had won. Decisions:

1. **The harness delegates, not only the model** (`goal._harness_plan`). A market or macro goal gets a researcher once the data has been
   inspected; a long panel gets up to four parallel analysts on slices before any forecast; five identical failures hand the failing
   cell to a fresh-context analyst. Each fires once and is stored as an `auto:` step, so replay is stable. The planner rules alone did
   not produce delegation.
2. **Repeats are seen.** The stall signature ignores run ids and file lists; eight identical failures end the loop with a labelled
   partial answer; three quote the failing line in the planner note.
3. **Figures a reader relies on are written by code.** An accuracy paragraph (backtest MASE against the best simple rule, verdict,
   backtest and held-out coverage) is appended to every answer from the stored results. Outside-context lines whose figures appear in
   no fetched page are removed and the removal is said. The composer sees both before it writes.
4. **A state card** (newest file of each kind with its run id, backtest so far, sub-agent reports) is computed every step, so
   compaction cannot lose where the work stands.
5. **One operating procedure** (`planner.v16`, 32 rules from 70): frame the decision, read the series' own clock, scan, hypotheses,
   funnel, prepare, outside context, baselines, one treatment gate (control_lab), forecast, range, deliver. `composer.v4` writes in
   decision order.
6. **The simple rule can be the centre line.** `backtest_series` reports wins, a paired bootstrap interval and a verdict over 12
   origins; when the verdict is `baseline_better`, the checklist asks for `forecast_report(centre='naive'|'drift')`, which charts the
   rule with TimesFM's band recentred.
7. **Intervals.** Calibration is judged by distance to target, with a second pass and an EWMA-volatility conformal band used only when
   it repairs a too-narrow band; held-out coverage carries a Wilson interval. Never narrowed.
8. **Sandbox API is discoverable.** The tool spec carries a signature catalogue generated from the code; guessed argument names are
   aliased or refused with the real signature; pandas answers polars method names; panels are never summed silently (`pl.panel`).
9. **Research quality.** Recency-aware ranking, corrected credibility tiers, metadata-only publication dates, a primary-source registry
   per series family fetched before search, `exclude_urls`, and coverage metrics per call.
10. **Trading calendars.** Forecast dates skip the holidays of the exchange calendar the history follows (inferred from the data).

Research behind the next steps: `reports/Foundation forecaster control and intervals.md` (random-walk gate and return target for prices,
DtACI online adjustment, equal-weight blends, global GBM for panels, reconciliation).
