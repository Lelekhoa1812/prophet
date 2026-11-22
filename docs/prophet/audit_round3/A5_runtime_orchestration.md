# A5. Agentic runtime and orchestration (audit round 3)

Scope: sub-agent orchestration, delegation, goal planning and execution, budgets, compaction, replay and durability, stall handling, evaluator loop, harness auto-plans, error recovery. Code: `src/prophet_agent/runtime/goal.py`, `runner.py`, `store.py`, `agents.py`, `src/prophet_agent/api/service.py`, `src/prophet_agent/nim/transport.py`, tool observers in `tools/sandbox_tool.py` and `tools/panel_tool.py`. Measured from six live snapshots in `data/` (read-only). The API on port 8010 was not called and no git state was changed.

Labels: **measured** = computed from snapshot events, steps and stored observations; **inferred** = reasoned from code, not reproduced; **external** = fetched from the URL given.

## Summary

1. Two-thirds of model time is spent waiting for the planner and about a quarter on LLM-written compaction. Model calls account for 79% of wall time across the six runs (measured); the planner alone accounts for 44%.
2. The planner is blind to most of what its tools print. 163 of 295 tool observations (55%) were clipped at the planner's 4,000-character view, and 133 of 198 sandbox stdout blocks lost their printed tail (measured). Repeated identical cells follow directly from this.
3. Budgets are kept in two ledgers that disagree. Sub-agent calls count against the planner's allowance but not against the runtime's, so the lead can be forced to finish before it starts (Beijing run, step 2, measured).
4. Five of six runs ended `partial`; the evaluator refused 26 of 42 finishes or stalls, and 11 of those cite output the lead could not see (measured). Only one run reached `met`.
5. The highest-leverage fixes are deterministic (no model call): one budget ledger, a ledger-backed observation view with paging, deterministic context editing in place of LLM compaction, and a pre-call dedupe. All four keep replay determinism because they read stored results rather than re-asking the model.

## Measured profile of the runs

Source: `data/live_gates_{retail,wti}_rep{3,4}.json`, `data/live_panel_{beijing,electricity}_rep1.json`. Wall time is first-to-last event timestamp. Step time is the gap from the previous `step.done` (includes lease and queue time). Model calls are `plan`, `eval*`, `compose*`, `compact`, `repair`.

| Run | Status | Wall (s), segments | Model calls (time, share of wall) | Tool calls (act time) | Compaction (time, share) | Sub-agents (tool calls, share of tools) | Evaluator | Repeated identical calls |
|---|---|---|---|---|---|---|---|---|
| retail rep3 | partial | 715; 503, 212 | 41 (697 s, 97%) | 27 (19 s) | 7 (281 s, 39%) | none | 3 of 3 not_met | 2 |
| retail rep4 | partial | 1677; 544, 758, 375 | 80 (1642 s, 98%) | 56 (36 s) | 15 (588 s, 35%) | 1 stuck analyst (7, 13%) | 3 not_met, 1 met | 7 |
| wti rep3 | partial | 511; 418, 93 | 40 (389 s, 76%) | 21 (122 s) | 4 (90 s, 18%) | 1 research (12, 57%) | 3 not_met, 3 met | 0 |
| wti rep4 | completed (met) | 796; 533, 263 | 46 (511 s, 64%) | 26 (285 s) | 5 (110 s, 14%) | 1 research (13, 50%) | 3 not_met, 3 met | 0 |
| beijing rep1 | partial | 1212; 350, 862 | 126 (820 s, 68%) | 74 (392 s) | 16 (237 s, 20%) | 4 panel + 1 stuck (52, 70%) | 5 not_met, 7 met | 9 |
| electricity rep1 | partial | 1756; 922, 613, 220 | 138 (1177 s, 67%) | 91 (578 s) | 21 (422 s, 24%) | 1 research, 4 panel, 1 stuck (76, 84%) | 6 not_met, 5 met | 21 |

Derived (measured):

- Totals over six runs: wall 6,667 s; model time 5,235 s (79%); planner 2,905 s (44%); compaction 1,728 s (26%); evaluator, compose and repair the remainder.
- Model calls per tool call: 1.4 to 1.9 (retail rep4 80/56; beijing 126/74; electricity 138/91).
- Compactions per tool call: 0.19 to 0.27. A compaction costs 14.8 to 40 s mean per call.
- Auto-continues: 1, 2, 1, 1, 1, 2 (`mission.auto_continue`). Segment 1 of electricity ran 922 s against a 900 s `max_seconds` allowance.
- Sub-agents: 14 started in total. Their own evaluator verdicts: 5 met, 5 not_met, 4 ended in `evalstall` with no numbered verdict (inferred from step names).
- Tool failures (`tool.failed`): 9 in total, mostly argument-validation errors (`web_research` family, `web_fetch` max_chars, `forecast_report` centre, `backtest_series` missing file, `calibrate_intervals` scope).
- Evaluator: 42 calls; 23 `not_met` plus 3 `evalstall`; 11 of the refusals cite clipped, missing or not-printed output (reason text).
- Finishes: 3, 3, 6, 6, 9, 6 per run. Three of six runs finished before forecasting had started or while a backtest was still open (e.g. beijing `plan:d0:2`, retail `plan:d0:18`).
- Stall rescue and stall: `evalstall` appears 1 to 3 times per run; no `stall_rescue` appeared in the snapshots because it is kept only in history, not in events.

Beijing segment 1, traced from events (measured): the harness delegated four panel analysts at step 1 (`auto:d0:1`). Their 44 tool calls took 346 s. At `plan:d0:2` the progress event reads `tool_calls: 45`, the planner was forced to `finish` with rationale "tool budget used", the evaluator said "No forecasting has been performed yet", and the part checkpointed. The lead had made one tool call of its own. The planner's allowance was 20 and the sub-agents spent it all.

## Findings, ranked by impact

### A5-01. Two budget ledgers: sub-agent calls starve the lead
- **Severity:** High.
- **Evidence:** `goal.py:282-292` (`_calls_used` adds sub-agent calls recursively) and `goal.py:1194` (`out_of_calls` compares that count with `cfg.tool_budget`). `goal.py:967-983` (`_harness_allowance`) raises only `ctx.rt.budget["max_tool_calls"]`, never `cfg.tool_budget`. `goal.py:674` (`_decide_plan` forces `finish` when `out_of_calls`). Run: beijing rep1, events seq 24407 to 24412 (`mission.progress` tool_calls 45, `plan:d0:2` finish "tool budget used", `mission.status` checkpoint, `mission.auto_continue`). Electricity: 76 of 91 tool calls were sub-agent calls (measured).
- **Root cause:** the planner and the runtime use different counters for the same allowance. The harness comment (`goal.py:967`) says "four panel analysts used the whole part's allowance", so the runtime side was patched; the planner side was not.
- **Proposed fix:** one function `budget_used(mission_id)` that the runtime guard, the planner note and the forced-finish check all call. Harness-started delegations are charged to a separate `delegation` budget with an explicit cap (`MAX_AGENTS` already caps count). The planner's remaining allowance excludes that budget. Store the allowance per segment as a `control:` step (see A5-07) so a replay reads the same value.
- **Replay determinism:** the allowance is read from a stored step, so replay takes the same branch. The counter is the ledger count, which is already replay-stable.
- **Files:** `runtime/goal.py` (`_calls_used`, `_harness_allowance`, the `out_of_calls` line), `api/service.py` (`_next_part`, `continue_mission` top-up).
- **Regression test:** new `tests/prophet_agent/test_goal_budget.py::test_harness_delegation_does_not_force_lead_finish` with a fake brain that delegates four analysts that each spend 10 calls against a lead allowance of 20. Assert the lead still gets a planner step with `out_of_calls` False and that the runtime and planner counts agree.
- **Effort:** S (half a day).
- **Risk:** higher total spend per part. Bounded by `MAX_AGENTS` and `max_seconds`. Cost is the reason the allowance exists, so the new cap must be explicit in the brief.

### A5-02. The planner cannot see most of what its tools print
- **Severity:** High.
- **Evidence:** `goal.py:92` (`OBS_CHARS = 4000`), `goal.py:183-186` (`_clip` cuts the canonical sorted-key JSON at 4,000 characters). `sandbox_tool.py:20-22, 75-80` (`observe` keeps head and tail of stdout up to 3,500 characters, with the note "Do not re-run this cell"). Measured on stored observations (act steps, which hold the post-`observe` result): 163 of 295 tool observations exceed 4,000 characters (retail rep4 35 of 56; beijing 34 of 74; electricity 60 of 91). For 198 sandbox observations, 133 have their stdout end beyond the view and 73 have stdout entirely hidden (approximate string match, measured).
- **Root cause:** sorted keys put `stdout` after `outputs`, `run_id` and `stderr_tail`, so the clip removes the printed summary that `observe` deliberately keeps. The clip is also blind to priority: it truncates whatever comes last. Consequences seen in the snapshots: sandbox cells re-run identical code (retail rep4 7 times; beijing 5); a sub-agent wrote "All profiling attempts produced clipped stdout" (electricity `composestall:d1.a5:12`); the evaluator wrote "output was clipped" (electricity `eval:d0:14`).
- **Proposed fix:** (a) render an observation as text in priority order: status line, then stdout head and tail with a per-field budget, then outputs; (b) store the full observation in the ledger under a reference (run id plus tool-call key) and expose `read_observation(ref, offset, limit)` as a ledger read. It is not a tool call, is not charged to the budget, and reads stored data, so replay is unaffected; (c) say in the clip marker how to page.
- **Replay determinism:** the renderer is a pure function of the stored result. `read_observation` returns stored text, not fresh tool output.
- **Files:** `runtime/goal.py` (`_clip`, `_evidence_lines`, a new `_render_observation`), `tools/sandbox_tool.py` (`observe`), `tools/panel_tool.py` (`observe` already has `per_entity_note`; keep), `runtime/store.py` (ref lookup via `tool_results`).
- **Regression test:** `tests/prophet_agent/test_observation_view.py::test_stdout_tail_visible_under_clip` (a 9,000-character stdout with a summary last: assert the summary line appears in the rendered view) and `::test_read_observation_pages_stored_result` (paging returns the stored text, adds no tool row).
- **Effort:** M (1 to 2 days).
- **Risk:** a larger planner prompt. Mitigate by keeping the head and summary and paging on demand. Measure the prompt length in the next live run.

### A5-03. LLM compaction is the second-largest cost and runs too often
- **Severity:** High.
- **Evidence:** `goal.py:93` (`VIEW_CHARS = 16000`), `goal.py:95` (`MIN_FOLD = 3`), `goal.py:688-713` (`_compact`, one `ctx.step("compact:...")` per fold, calling the writer model). Measured: 1,728 s of 6,667 s (26%) across six runs; retail rep4 15 compactions, 588 s (35% of wall); about 0.2 compactions per tool call.
- **Root cause:** with clipped observations of about 4,000 characters, the 16,000-character view overflows every three to four calls, so the memo is rewritten by a model roughly every fourth call. The memo is also rewritten in full each time (cap 8,000 characters), so each fold re-reads the previous memo.
- **Proposed fix:** deterministic context editing, in the style of the Anthropic tool-result clearing strategy (external: `https://platform.claude.com/docs/en/build-with-claude/context-editing`, which describes `clear_tool_uses` with `trigger`, `keep` and `clear_at_least`, applied server-side with no extra model call). Here: keep the newest `KEEP_RECENT` observations verbatim and replace older tool bodies with one-line stubs built from stored fields (tool, args summary, status, run id, the figures in `state card`). Keep the decisions and the state card as now. Retain the LLM memo only as an optional fallback behind a flag, and only when a stub view still exceeds a hard limit.
- **Replay determinism:** the stub is a pure function of history, so no `compact:` step is needed. If a stored `compact:` step exists from an earlier run, its memo is still read (keep the code path for old ledgers).
- **Files:** `runtime/goal.py` (`_compact`, `_view`, `_evidence_block`), `runtime/goal.py` tests.
- **Regression test:** `tests/prophet_agent/test_context_editing.py::test_stubs_are_replay_stable` (same history gives the same view twice; resumed run gives the same view) and `::test_state_card_survives_stub` (run ids and file names still appear after clearing).
- **Effort:** M.
- **Risk:** the planner loses detail from earlier observations. The state card and decisions list already carry what compaction preserved, so the loss is smaller than it looks. Watch the evaluator refusal rate (A5-04) after the change.

### A5-04. Finishes are premature, and the evaluator judges a view the lead cannot see
- **Severity:** High.
- **Evidence:** finish decisions with rationale: retail `plan:d0:18` ("All checklist items are satisfied") refused with "negative point predictions"; beijing `plan:d0:2` ("tool budget used") refused with "No forecasting has been performed yet"; retail rep4 `plan:d0:20` refused with "No forecast has been produced yet". Evaluator: 26 of 42 calls ended in refusal or stall; 11 cite clipped or missing output (measured from reason text). `goal.py:1407-1431` (checkpoint and `MAX_REJECTED_FINISHES` path).
- **Root cause:** the planner may choose `finish` at any step, and the evaluator is a call to the same model over the same clipped view (A5-02). Nothing ties `finish` to the checklist that `contracts.progress` already computes. Real defects (negative forecasts) are caught, which is useful, but many refusals are about the view.
- **Proposed fix:** (a) the planner's `finish` is accepted without an evaluator call only when `contracts.progress` shows every item done or the allowance is spent; otherwise the harness records "finish refused: items open: ..." as an observation and does not call the evaluator, which saves a model call per premature finish; (b) the evaluator input includes a deterministic digest of the stored results (figures and run ids), not only the rendered view.
- **Replay determinism:** the gate reads the same derived checklist from the ledger. The refusal is an ordinary history entry, so replay reproduces it.
- **Files:** `runtime/goal.py` (`_run` finish branch around lines 1395-1431), `runtime/contracts.py` (`progress`).
- **Regression test:** `tests/prophet_agent/test_finish_gate.py::test_premature_finish_skips_evaluator` (fake brain returns finish with items open: assert no `eval:` step is created).
- **Effort:** S to M.
- **Risk:** a legitimately finished run needs one more planner step to say finish after the checklist is ticked. That is cheap.

### A5-05. Sub-agents: low yield, no result contract, and no rescue below depth 0
- **Severity:** Medium-High.
- **Evidence:** research sub-agents (`agents.py` type `researcher`, 14 steps): WTI rep3 and rep4 and electricity a1 ended `not_met` on the same gap (no current primary-source level; "research briefing is incomplete"). Stuck analysts (retail a1, electricity a5, beijing a5) ended in `evalstall` after 6 to 13 calls with partial answers. `goal.py:1298-1305`: stall rescue only at `cfg.depth == 0`. `goal.py:1002-1027`: a sub-agent's report is returned as free text (`answer`, `reason`), not a typed contract. External: Anthropic multi-agent research system (`https://www.anthropic.com/engineering/multi-agent-research-system`) says the lead must give each subagent an objective, an output format and clear task boundaries, and that subagents may store work externally and return lightweight references.
- **Root cause:** the brief asks for "a short report"; the lead has no structured fields to check (figures, run ids, source ids, open items). Sub-agents hit their 14-step cap while still searching, and a stalled sub-agent cannot be rescued by the parent.
- **Proposed fix:** (a) a result schema per agent type (`{status, figures:[{name, value, source_id or run_id}], files, open_items}`) validated locally, like `EVAL_SCHEMA`; the lead gets the typed fields and the grounding check runs on them; (b) split research into one bounded sub-agent per angle, run in parallel (the `_delegate` thread pool already exists); (c) allow stall rescue at depth 1 by returning a structured `stalled` observation to the parent, which may hand the item to a new analyst.
- **Replay determinism:** the sub-agent result is a stored step output; the schema check is local and deterministic.
- **Files:** `runtime/agents.py`, `runtime/goal.py` (`_brief`, `_one_agent`, stall branch, `_delegate`), `runtime/grounding.py` (`unverified` input).
- **Regression test:** `tests/prophet_agent/test_subagent_contract.py::test_report_without_run_id_is_flagged` and `::test_stall_at_depth_one_returns_structured_observation`.
- **Effort:** M (2 to 3 days).
- **Risk:** the typed report is harder for a model to produce; fall back to a repair turn (as in `transport.MAX_REPAIRS`).

### A5-06. Deadlines are checked only between steps, so one model call can overrun
- **Severity:** Medium.
- **Evidence:** `runner.py:77-84` (`_guard_time` runs before each step and tool call only). `nim/transport.py:60` (`BUDGET_S = 420`), `:67` (`RATE_WAIT_LIMIT_S = 1800`); `transport.py:495` already accepts a `checkpoint` callback and `budget_s`, but `goal.py` does not pass the remaining segment time. Measured: electricity segment 1 ran 922 s against `max_seconds` 900; act steps of 162 s and 171 s (beijing `act:d0:3`, `act:d0:9`) were single tool calls. The lease (`store.py:325`, 300 s) is renewed only in `_guard_time`, so a long model wait can expire it (inferred; no takeover observed in these runs).
- **Root cause:** deadline and lease are not passed into the transport's waits.
- **Proposed fix:** pass `budget_s=min(BUDGET_S, remaining)` from the runtime into `brain.decide`; have the transport call `checkpoint` (which renews the lease and checks cancel) while it sleeps.
- **Replay determinism:** the timeout changes only whether a call fails, which is recorded as a stored step outcome.
- **Files:** `nim/transport.py` (`stream`, `_wait`), `runtime/goal.py` (`NimBrain.decide`), `runtime/runner.py`.
- **Regression test:** `tests/prophet_agent/test_deadline_model_call.py::test_call_stops_at_segment_deadline` with a fake client that sleeps.
- **Effort:** S.
- **Risk:** a shorter call may fail more often; the failover chain already handles that.

### A5-07. Control decisions are recomputed on each run, not read from the ledger
- **Severity:** Medium (inferred; needs a test to confirm).
- **Evidence:** `goal.py:1194` (`out_of_calls`), `goal.py:1195-1200` (`at_part_end`, the evaluator gate), `goal.py:1189` (`repair_granted` is a local that resets each run), `goal.py:1349-1350` (`cfg.tool_budget += REPAIR_CALLS`, a mutation of the config), `goal.py:1249-1259` (`_harness_plan` is computed before the `auto:` step and depends on the allowance). `runner.py:~55-63` checks only the step name on replay, not the branch that produced it.
- **Root cause:** the allowance and the repair grant are inputs to control flow, but are not stored. On a resumed part the allowance is recomputed (`service.py:315, 361`), so a branch taken in the original segment can differ on replay even when the stored step names still match.
- **Proposed fix:** store a `control:{tag}:{i}` step at each segment start holding `{tool_budget, repair_granted, contract_blocks, rejected_finishes}`; read it back on replay. Remove the in-place mutation of `cfg`.
- **Replay determinism:** this is the fix itself: the decision becomes a stored value.
- **Files:** `runtime/goal.py` (`_run` head and the repair branch), `api/service.py` (`_build_program`).
- **Regression test:** `tests/prophet_agent/test_replay_budget_boundary.py::test_resume_across_topup_takes_same_branch`: run a mission that blocks an answer at the budget edge, checkpoint, resume after a top-up, and assert the same stored steps are used with no `replay diverged` error.
- **Effort:** S.
- **Risk:** low. Existing ledgers without `control:` steps must fall back to the current behaviour.

### A5-08. Budget accounting differs across runtime, top-up and planner
- **Severity:** Medium.
- **Evidence:** `service.py:315, 361` (top-up is `store.count_tool_calls(mission_id)` with no exclude, so `analyst_skill` and `notes` reads count). `runner.py:85-90` (the runtime guard excludes `READ_ONLY`). `goal.py:974-983` (`_harness_allowance` raises the runtime limit by the sub-agent spend, so delegated calls are not charged to the lead at all). Electricity segment 2 `max_tool_calls` 107 against 91 total calls: the top-up includes reads and sub-agent calls.
- **Root cause:** three definitions of "a tool call".
- **Proposed fix:** one function in `store.py` (`budget_calls(mission_id, scope)`) with the excluded tools and the delegated flag defined once; use it in the guard, the top-up and the planner note.
- **Replay determinism:** stored tool rows are the input, so it is replay-stable.
- **Files:** `runtime/store.py`, `runtime/runner.py`, `api/service.py`, `runtime/goal.py`.
- **Regression test:** `tests/prophet_agent/test_budget_scope.py::test_reads_not_charged_in_topup`.
- **Effort:** S.
- **Risk:** low; changes the allowance number by a few calls.

### A5-09. Identical tool calls are run again, and stall detection only sees three in a row
- **Severity:** Medium.
- **Evidence:** measured identical (tool, args) repeats: electricity 21 of 91 (panel_backtest 6, sandbox_python 9, inspect_data 4, web_research 2); beijing 9 of 74; retail rep4 7 of 56. The 2026-11-22 panel note in `panel_tool.py:96-101` tells the planner that "the same call again returns the same result", which the planner then obeys. `goal.py:870-880` (`_stalled`) fires only on three consecutive identical signatures.
- **Root cause:** no pre-call check against the ledger; the only stop is the stall window.
- **Proposed fix:** before `ctx.call`, look up the ledger for the same tool and args in this mission. If a done result exists, return it as an observation with `"repeat_of": <step>` and no budget charge, and tell the planner the result is unchanged. Keep the note in `panel_tool.py` but point at `read_observation` (A5-02) instead of "call again".
- **Replay determinism:** the lookup reads stored rows; the result is the same stored value.
- **Files:** `runtime/runner.py` (`Ctx.call` or a wrapper in `goal._act`), `runtime/store.py`, `tools/panel_tool.py` (note text).
- **Regression test:** `tests/prophet_agent/test_repeat_call_dedupe.py::test_identical_call_returns_stored_result_uncharged`.
- **Effort:** S.
- **Risk:** a tool whose result is expected to change (web search, a live feed) must be excluded. Add a per-tool `dedupe` flag defaulting to False for `web_*` and `notes`, True for analysis tools.

### A5-10. `sandbox_python` is declared idempotent but ignores its idempotency key
- **Severity:** Medium.
- **Evidence:** `tools/sandbox_tool.py:96` (`idempotent=True`); `sandbox_tool.py` `_run(args, _key)` never uses `_key`. `runner.py:80` (`Ctx.call` docs: "receives the same key so the external system can deduplicate"). A crash after a sandbox run but before `tool_set` re-runs it into a new run folder and a second charge.
- **Root cause:** the flag claims a property the tool does not provide. Consequence: a re-run after a crash doubles the outputs and the cost; if the flag were honest, a resume would stop with `outcome_unknown` more often.
- **Proposed fix:** use the key as the run id seed (`run_id = key[:32]`); on re-run, if the directory exists and holds a manifest, return it; otherwise rerun into a fresh directory. Alternatively mark it non-idempotent and accept the stop.
- **Replay determinism:** the key is a function of mission, step, tool and args; deterministic.
- **Files:** `tools/sandbox_tool.py`, `sandbox/runner.py` (run id).
- **Regression test:** `tests/prophet_agent/test_sandbox_idempotent.py::test_rerun_with_same_key_returns_same_run`.
- **Effort:** S to M.
- **Risk:** keyed run folders must not collide across missions. The key includes the mission id, so they do not.

### A5-11. One planner decision per step, so independent tool calls run serially
- **Severity:** Low-Medium.
- **Evidence:** `goal.py:90-110` (`PLAN_SCHEMA` has one `tool` and one `args`). Measured: one tool call per planner call throughout (retail rep3 27 tools, 30 plans; electricity 91 and 97). External: Anthropic multi-agent research system reports the lead runs subagents in parallel and their tools in parallel, which cut time "up to 90%" for complex queries (`https://www.anthropic.com/engineering/multi-agent-research-system`); parallel tool calls are a standard harness capability.
- **Root cause:** the plan schema allows one action; delegation is the only parallel path.
- **Proposed fix:** allow `calls: [{tool, args}]` (max 4, all read-only or independent) in one plan. Each call becomes its own ledger row and step (`act:{tag}:{i}.{k}`), executed through a thread pool like `_delegate`. The planner sees all results in one observation.
- **Replay determinism:** the branch step numbers are reserved as in `BRANCH_BLOCK`, so replay asks for the same steps.
- **Files:** `runtime/goal.py` (`PLAN_SCHEMA`, act branch), `runtime/runner.py` (`branch`).
- **Regression test:** `tests/prophet_agent/test_parallel_calls.py::test_parallel_calls_replay_in_same_order`.
- **Effort:** M to L.
- **Risk:** a larger failure surface and budget accounting (A5-08 first).

### A5-12. Argument errors are charged and repeated
- **Severity:** Low.
- **Evidence:** measured `tool.failed`: `web_research` family enum (electricity, retail), `web_fetch` max_chars (WTI rep3 and rep4, the same error twice), `forecast_report` centre enum, `backtest_series` missing file (retail rep4), `calibrate_intervals` scope (beijing). Each counts as a tool call in `count_tool_calls` (`store.py:500`) because `tool_start` runs before validation (`runner.py:~110-120`).
- **Root cause:** argument validation happens inside the tool, after the ledger row is opened.
- **Proposed fix:** validate arguments against the tool's declared JSON schema before `tool_start`; return the error as an observation with no ledger row. Tool docs already name the allowed values (`_tool_lines`).
- **Replay determinism:** validation is a pure function of args.
- **Files:** `runtime/runner.py`, `tools/*` (schema per tool).
- **Regression test:** `tests/prophet_agent/test_arg_validation_uncharged.py::test_bad_enum_not_charged`.
- **Effort:** S.
- **Risk:** the schema must match what the tool accepts; a mismatch would block valid calls, so generate it from the same constants the tool reads.

## What the current design does well (keep)

- Replay by step number with a name check (`runner.py:Ctx.step`) and a tool ledger keyed by mission, step, tool and args (`store.py:158`). This is the durability core and matches the external guidance: "Recorded results are then retrieved from the persistence layer rather than re-executed" (external: `https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/concepts/durable_execution.md`).
- Non-idempotent calls left `started` stop as `outcome_unknown` (`runner.py:Ctx.call`), not retried.
- Deterministic contract checks on figures and horizon (`contracts.blocking`, `grounding.unverified`) run before an answer leaves, so a `met` verdict is not trusted by itself.
- Sub-agent isolation with a report-only return (`goal._view`, `_one_agent`). This matches the Claude Code model: a subagent "returns only its final report or summary to the parent conversation", with its own context window (external: `https://code.claude.com/docs/en/sub-agents`).
- Harness auto-delegation for the panel and research cases (`_harness_plan`), stored as `auto:` steps so the decision replays.

## Limits of this audit

- Snapshots are six runs; the sample is small. Rates (for example refusal 26 of 42) are measured on these runs, not the population.
- "Step time" is inferred from gaps between `step.done` events and includes lease and queue time.
- A5-07 and A5-10 are inferred from code and need a targeted test to confirm.
- Measured clipping shares use an approximate string match for stdout position (described in A5-02).
- External references were read with WebFetch on 2026-11-22: the Anthropic context-editing page, the Anthropic multi-agent research article, the LangGraph durable-execution concept page (pinned commit), and the Claude Code sub-agents page. Figures quoted are as the pages state them.

## Suggested order

1. A5-02 and A5-09 (view and dedupe): both reduce wasted calls with no behaviour change to answers.
2. A5-01 and A5-08 (one budget ledger): fixes the forced early finish.
3. A5-06 and A5-07 (deadline and control state): needed before any change to continuation.
4. A5-03 (deterministic context editing): largest time saving, after the view is fixed.
5. A5-04 and A5-05 (finish gate and sub-agent contract).
6. A5-10 to A5-12.
