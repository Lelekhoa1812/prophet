# Durable agent runtimes: research for Prophet

Status: research note. No code was changed. Only this file was written.
Date of research: 2026-11-21.

## 1. Question and method

**Question.** How do current systems handle long-running agent loops, context compaction, prompt caching, tool-result
clearing, planning, idempotent side effects, supervision, evaluate-repair loops and structured user questions? Which of
these should Prophet adopt now, later, or not at all, given its locked decisions in `docs/prophet/ARCHITECTURE.md`?

**Method.**
- Read `docs/prophet/ARCHITECTURE.md` first. Ran `graphify query` from the repo root (graphify-out exists), then read the
  cited Prophet files directly (`goal.py`, `runner.py`, `store.py`, `prompts/compactor.v1.md`) to check code claims.
- Searched and fetched primary sources: vendor docs (Anthropic, OpenAI, Google, LangGraph, DBOS, Temporal, MCP, SQLite,
  Redis), arXiv papers, and engineering posts. Every claim below cites a source ID from section 2.
- Reliability labels: `PRIMARY-DOC`, `PAPER`, `ENG-POST`, `THIRD-PARTY`.
- Confidence labels on every conclusion:
  - `HIGH`: documented by a primary source, or measured in a paper with a clear protocol.
  - `MEDIUM`: measured in a paper or case study, but on a different domain, or only partly checked.
  - `LOW`: single vendor example, or inferred from a design argument.
  - Each conclusion also says whether it is **documented**, **measured**, or **inferred**.
- "Date seen" is the date I accessed the page (2026-11-21). Published dates are given where the page showed them;
  for arXiv, the ID month is noted as "ID-implied".
- Limits of method: no Prophet test was run. Code claims are from reading the code, not from reproducing behaviour.
  Vendor numbers are as published and were not re-measured.

## 2. Source table

| ID | Title | URL | Type | Date seen (published if shown) |
|---|---|---|---|---|
| S1 | Anthropic: Prompt caching | https://platform.claude.com/docs/en/build-with-claude/prompt-caching | PRIMARY-DOC | 2026-11-21 (undated page) |
| S2 | Anthropic: Create a Message (cache_control ttl) | https://platform.claude.com/docs/en/api/messages/create | PRIMARY-DOC | 2026-11-21 |
| S3 | Anthropic: Tool use with prompt caching | https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching | PRIMARY-DOC | 2026-11-21 |
| S4 | Anthropic: Context editing | https://platform.claude.com/docs/en/build-with-claude/context-editing | PRIMARY-DOC | 2026-11-21 |
| S5 | Anthropic SDK: clear_tool_uses edit param types | https://github.com/anthropics/anthropic-sdk-python/blob/49d639a6/src/anthropic/types/beta/beta_clear_tool_uses_20250919_edit_param.py | PRIMARY-DOC | 2026-11-21 |
| S6 | Anthropic Cookbook: context engineering, compaction, tool clearing | https://platform.claude.com/cookbook/tool-use-context-engineering-context-engineering-tools | PRIMARY-DOC | 2026-11-21 |
| S7 | Anthropic: Code execution with MCP | https://www.anthropic.com/research/code-execution-with-mcp | ENG-POST | 2026-11-21 (published Nov 4, 2025) |
| S8 | MarkTechPost: summary of S7 | https://www.marktechpost.com/2025/11/08/anthropic-turns-mcp-agents-into-code-first-systems-with-code-execution-with-mcp-approach/ | THIRD-PARTY | 2026-11-21 (Nov 8, 2025) |
| S9 | OpenAI: Prompt caching guide | https://developers.openai.com/api/docs/guides/prompt-caching | PRIMARY-DOC | 2026-11-21 |
| S10 | OpenAI: Prompt Caching in the API (launch post) | https://openai.com/index/api-prompt-caching/ | PRIMARY-DOC | 2026-11-21 (search snippet only) |
| S11 | OpenAI: Compaction guide | https://developers.openai.com/api/docs/guides/compaction | PRIMARY-DOC | 2026-11-21 |
| S12 | OpenAI: Responses create reference (context_management) | https://developers.openai.com/api/reference/resources/responses/methods/create | PRIMARY-DOC | 2026-11-21 |
| S13 | OpenAI Agents SDK: Human-in-the-loop | https://openai.github.io/openai-agents-python/human_in_the_loop/ | PRIMARY-DOC | 2026-11-21 |
| S14 | OpenAI Agents SDK: Run state reference | https://openai.github.io/openai-agents-python/ref/run_state/ | PRIMARY-DOC | 2026-11-21 |
| S15 | LangGraph: Checkpointers and durability modes | https://docs.langchain.com/oss/python/langgraph/checkpointers | PRIMARY-DOC | 2026-11-21 |
| S16 | LangGraph: Durable execution concept | https://raw.githubusercontent.com/langchain-ai/langgraph/c4144bb48f97e6afeaa002de318580cc41c19893/docs/docs/concepts/durable_execution.md | PRIMARY-DOC | 2026-11-21 (pinned commit) |
| S17 | DBOS: Steps (Go tutorial) | https://docs.dbos.dev/golang/tutorials/step-tutorial | PRIMARY-DOC | 2026-11-21 |
| S18 | DBOS: Architecture | https://docs.dbos.dev/architecture | PRIMARY-DOC | 2026-11-21 |
| S19 | DBOS: Transactional steps (Java) | https://docs.dbos.dev/java/tutorials/step-factory-tutorial | PRIMARY-DOC | 2026-11-21 |
| S20 | DBOS: Workflows and guarantees (Go) | https://docs.dbos.dev/golang/tutorials/workflow-tutorial | PRIMARY-DOC | 2026-11-21 |
| S21 | Temporal: Activity definition | https://docs.temporal.io/activity-definition | PRIMARY-DOC | 2026-11-21 |
| S22 | Temporal: Standalone activity | https://docs.temporal.io/standalone-activity | PRIMARY-DOC | 2026-11-21 |
| S23 | Temporal: Detecting activity failures (heartbeats) | https://docs.temporal.io/encyclopedia/detecting-activity-failures | PRIMARY-DOC | 2026-11-21 |
| S24 | Temporal: Long-running activity heartbeat pattern | https://docs.temporal.io/design-patterns/long-running-activity | PRIMARY-DOC | 2026-11-21 |
| S25 | Temporal: Retry policies | https://docs.temporal.io/encyclopedia/retry-policies | PRIMARY-DOC | 2026-11-21 |
| S26 | Kleppmann: How to do distributed locking | https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html | ENG-POST | 2026-11-21 (published 2016-02-08) |
| S27 | Redis: Distributed locks pattern | https://redis.io/docs/latest/develop/clients/patterns/distributed-locks/ | PRIMARY-DOC | 2026-11-21 |
| S28 | SQLite: Write-Ahead Logging | https://www.sqlite.org/wal.html | PRIMARY-DOC | 2026-11-21 |
| S29 | MCP specification: Elicitation (2025-11-25) | https://modelcontextprotocol.io/specification/2025-11-25/client/elicitation | PRIMARY-DOC | 2026-11-21 |
| S30 | Google ADK: Events compaction | https://github.com/google/adk-python/blob/main/docs/guides/apps/events_compaction/index.md | PRIMARY-DOC | 2026-11-21 (also adk.dev/context/compaction) |
| S31 | Gemini API: Context caching | https://ai.google.dev/gemini-api/docs/caching?authuser=1&lang=python | PRIMARY-DOC | 2026-11-21 (page says updated 2026-10-13) |
| S32 | Firebase AI Logic: Context caching | https://firebase.google.com/docs/ai-logic/context-caching | PRIMARY-DOC | 2026-11-21 |
| S33 | Google Cloud: Context caching overview | https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/context-cache/context-cache-overview | PRIMARY-DOC | 2026-11-21 |
| S34 | vLLM: Automatic prefix caching design | https://docs.vllm.ai/en/stable/design/prefix%5Fcaching/ | PRIMARY-DOC | 2026-11-21 (search snippet only) |
| S35 | Manus: Context engineering lessons | https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus | ENG-POST | 2026-11-21 (published 2025-07-18) |
| S36 | Factory: Evaluating context compression | https://factory.com/news/evaluating-compression | ENG-POST | 2026-11-21 (published 2025-12-16) |
| S37 | The Complexity Trap (JetBrains), arXiv 2508.21433 | https://arxiv.org/abs/2508.21433 | PAPER | 2026-11-21 (first public 29 Aug 2025) |
| S38 | JetBrains Research blog on context management | https://blog.jetbrains.com/research/2025/12/efficient-context-management/ | ENG-POST | 2026-11-21 (Dec 2025 per URL) |
| S39 | Lost in Context Compaction (ConPACT), arXiv 2608.11242 | https://arxiv.org/html/2608.11242 | PAPER | 2026-11-21 (ID-implied Aug 2026) |
| S40 | Governance Decay, arXiv 2606.22528 | https://arxiv.org/abs/2606.22528 | PAPER | 2026-11-21 (ID-implied Jun 2026) |
| S41 | TrueFoundry: Governance Decay explainer | https://www.truefoundry.com/blog/governance-decay-context-compaction-enterprise | THIRD-PARTY | 2026-11-21 (updated 2026-10-15) |
| S42 | Magentic-One, arXiv 2411.04468 | https://arxiv.org/abs/2411.04468 | PAPER | 2026-11-21 (ID-implied Nov 2024) |
| S43 | AutoGen: Magentic-One orchestrator source | https://github.com/microsoft/autogen/blob/main/python/packages/autogen-agentchat/src/autogen_agentchat/teams/_group_chat/_magentic_one/_magentic_one_orchestrator.py | PRIMARY-DOC | 2026-11-21 |
| S44 | Huang et al., LLMs Cannot Self-Correct Reasoning Yet, arXiv 2310.01798 | https://arxiv.org/abs/2310.01798 | PAPER | 2026-11-21 (ICLR 2024) |
| S45 | CRITIC, arXiv 2305.11738 | https://arxiv.org/abs/2305.11738 | PAPER | 2026-11-21 (ICLR 2024) |
| S46 | Reflexion, arXiv 2303.11366 | https://arxiv.org/abs/2303.11366 | PAPER | 2026-11-21 (NeurIPS 2023) |
| S47 | ReAct, arXiv 2210.03629 | https://arxiv.org/abs/2210.03629 | PAPER | 2026-11-21 (ICLR 2023) |
| S48 | MemGPT, arXiv 2310.08560 | https://arxiv.org/abs/2310.08560 | PAPER | 2026-11-21 (published 2023-10-12) |
| S49 | Voyager, arXiv 2305.16291 | https://arxiv.org/abs/2305.16291 | PAPER | 2026-11-21 (v2 19 Oct 2023) |

Source counts by type: PRIMARY-DOC 32, PAPER 10, ENG-POST 5, THIRD-PARTY 2 (49 total). Sources S47-S49 were checked through
their abstracts or search snippets only. Where a claim rests on a snippet, the conclusion says so.

## 3. Topic 1: Long-running agent loops and checkpoint/resume

| Approach | Checkpoint unit | When it is written | Resume behaviour | Source |
|---|---|---|---|---|
| LangGraph, `durability` = sync | Each super-step | Before the next step starts | Replays from the last checkpoint; nodes re-run | S15, S16 |
| LangGraph, `durability` = async | Each super-step | While the next step runs | Small risk the last checkpoint is lost on crash | S15 |
| LangGraph, `durability` = exit | Run end only | At exit, including HITL interrupt | No mid-run crash recovery | S15 |
| DBOS | Each step output; workflow inputs | After the step body returns, in Postgres | Replays stored step outputs, resumes at first unrecorded step | S17, S18 |
| Temporal | Workflow event history | Activity results recorded on return | Workflow replays history; activity retries restart from top | S21, S22 |
| SQLite WAL (used by Prophet) | Page-level log | Commit appends to WAL | Readers do not block writer; one writer at a time; same host only | S28 |
| Prophet (`runner.py`, `goal.py`) | Each `step()` and `call()` | Stored after `fn` returns, before the run moves on | Done steps return stored value; decisions are not re-asked | repo code |

**Conclusion.**
- Step-level checkpoints with stored outputs and deterministic replay are the shared pattern across LangGraph, DBOS, Temporal
  and Prophet. **Confidence: HIGH (documented, S15-S18, S21).**
- The durability setting decides what survives a crash. `exit` loses everything mid-run; `sync` writes before the next step.
  **Confidence: HIGH (documented, S15).**
- Prophet's `step()` writes the result before the run continues, which matches `sync`. **Confidence: MEDIUM (inferred from
  code reading, not tested).**
- Prophet's SQLite WAL design is sound for one process on one host. It stops working for multi-host runners, where DBOS-style
  Postgres would be needed. **Confidence: HIGH (documented, S28).**

## 4. Topic 2: Context compaction, summarisation, and failure modes

| Approach | Mechanism | Retention evidence | Failure mode | Source |
|---|---|---|---|---|
| OpenAI server-side compaction | `context_management` with `compact_threshold` (min 1000 tokens); opaque compaction item | Not evaluated in the sources read | Output is opaque, so it cannot be checked; provider-specific | S11, S12 |
| Anthropic server-side compaction | Described as the primary strategy for long conversations | Not evaluated in the sources read | Provider-specific | S4 |
| Google ADK events compaction | Sliding window (interval + overlap) or token threshold + retention size; pairs must be set together; never splits a tool call from its response | Events are kept in session; only the prompt view changes | Summary quality depends on the chosen summarizer | S30 |
| Anchored iterative summary (Factory) | Structured sections (intent, files, decisions, next steps); only the newly dropped span is summarized and merged | Probe eval on 36,611 production messages: 3.70 vs 3.44 (Anthropic) vs 3.35 (OpenAI) | Vendor's own evaluation with an LLM judge | S36 |
| Observation masking (JetBrains) | Old tool observations replaced by placeholder; reasoning kept | Solve rate matched or exceeded LLM summary in most SWE-bench setups; about 50% lower cost than raw | Coding-agent domain; placeholder can hide facts the agent later needs | S37, S38 |
| Recent-N and LLMLingua-2 | Keep last N turns, or prompt compression | About 0% retention of injected side constraints (SCs) across three datasets | Drops constraints silently | S39 |
| LLM compactors (four open and proprietary) | LLM writes the summary | SC retention 17% on average across compactors; GPT-5.4-mini reached 98% on one dataset and 6.7% on another | Silent constraint loss | S39 |
| SC extractor (ConPACT) | Constraints kept apart from task summary, appended to compacted context | Over 90% retention in three scenarios | Tested only on the paper's suite | S39 |
| Constraint pinning (Governance Decay) | Policies in a compaction-exempt buffer, re-injected verbatim | 0% violations in tested attacks (control 0%; compaction 30% pooled, up to 59%) | Forged "operator update" messages still a gap | S40, S41 |
| Prophet (`goal.py`) | Older observations folded into memo by a durable `compact:*` step; newest 4 kept verbatim; answers kept verbatim outside memo | Not measured for Prophet | See failure modes below | repo code |

**Failure modes seen in the evidence:**
1. **Silent constraint loss.** In an inbox experiment, an agent deleted emails without approval in 30 of 100 sessions after compaction, with none before it. 28 of those 30 had the constraint dropped in the summary (S39).
2. **Compaction can be worse than no policy.** For the most affected model, compacted context led to more violations than having no policy at all, because the summary "normalizes the pending task while discarding the rule" (S40).
3. **Structure reduces loss, but not to zero.** Factory's sections act as a checklist. Freeform summaries lose file paths and decisions over repeated compressions (S36).
4. **Memo truncation drops the oldest text.** Prophet's memo cap keeps the last `MEMORY_CHARS` characters (`text[-memory_chars:]`), so the earliest memo content is removed first, with a marker (`goal.py`, near line 329).
5. **Re-asking answered questions.** Prophet's own code comment says a decision dropped by compaction "would be asked again". Its `_decisions` helper keeps answers verbatim outside the memo (`goal.py`, `_decisions`).

**Conclusion.**
- Compaction silently drops user constraints, measured across several compactors in two 2026 papers. **Confidence: HIGH (measured,
  S39, S40; the two papers agree).**
- Constraints and user decisions should live in a channel the summarizer cannot edit: verbatim, pinned, re-injected each
  planning turn. **Confidence: MEDIUM (measured in S39 and S40 on their own suites; not tested on Prophet).**
- Prophet already pins answers verbatim, but not user instructions that are not answers to a question. The memo cap is a
  separate defect. **Confidence: HIGH (inferred from code reading, which I checked directly).**

## 5. Topic 3: Prompt caching and KV-cache reuse

| Provider / engine | Minimum cacheable prefix | Lifetime and write/read price | What invalidates | Source |
|---|---|---|---|---|
| Anthropic | Model-dependent (512 on some models; 1024 and 4096 on others) | Default 5 min, refreshed on hit; 1 h at 2x base input price (S2). Earlier read of S1: 5-min write 1.25x, read 0.1x | Any change to tools, system or messages at or before the breakpoint; up to 4 breakpoints; 20-block lookback; changing thinking parameters | S1, S2, S3 |
| OpenAI (GPT-5.6 and later) | 1,024 visible input tokens | `prompt_cache_options.ttl` = "30m" (default, write 1.25x) or "12h" (write 2.5x); read 0.1x | Whole rendered prefix must match before the breakpoint | S9 |
| OpenAI (earlier models) | Model- and setting-dependent; implicit only; 128-token steps | Implicit; `prompt_cache_retention` up to 24 h | Same prefix rule | S9, S10 |
| Gemini (implicit caching, on by default) | 4,096 (Gemini 4 and 3.x Pro); 2,048 for 2.5 (per S31); 1,024 for Flash (per S32) | Expires no later than 24 h (S33); 90% discount (S33) | Prefix must match; large common content first | S31, S32, S33 |
| vLLM (self-hosted) | Full KV blocks only | Hash of parent hash + block tokens + extra keys (LoRA, MM, cache_salt) | Different tokens in any earlier block; default `sha256` not reproducible across versions | S34 |
| Manus (engineering practice) | Provider-dependent | Quoted 10x price gap for Claude Sonnet, cached vs uncached (a price ratio, not a measurement) | Timestamps at top of prompt; JSON key order; edits to past actions | S35 |

**What invalidates a cache (documented across sources):** any byte change before the last breakpoint (S1, S9, S34);
reordering JSON keys (S35); changing the tool list (S1, S3); changing thinking settings (S3); tool-result clearing, which
changes the prefix (S4); TTL expiry (S1, S9); switching to another model, since the cache is per model (S9, implicit from
the rule that the prefix must match).

**Source conflict to note.** The Gemini minimum for 2.5 Flash is 2,048 in S31 but 1,024 in S32, and S33 gives 6,144 for some
3.x Flash models. The pages disagree. I recorded each value with its source and did not pick one.

**Conclusion.**
- Provider caching is exact-prefix. A single changed byte before the last breakpoint loses everything after it.
  **Confidence: HIGH (documented, S1, S9, S34).**
- Minimums, TTLs and prices vary by provider and model, and the docs change often. Treat any number as a value to recheck.
  **Confidence: HIGH (documented; the Gemini conflict is evidence of this).**
- Prophet's stable head (instructions, tools, goal, memo, then volatile content last) is the right design. Whether it saves
  money on NIM is **not known**: it depends on whether NIM reports cache hits and keeps the prefix. **Confidence: MEDIUM
  (inferred from the design and the Prophet docs; not measured).**

## 6. Topic 4: Dynamic tool orchestration and tool-result clearing

| Approach | What is removed or deferred | Evidence | Where it runs | Source |
|---|---|---|---|---|
| Anthropic tool result clearing (`clear_tool_uses_20250919`) | Oldest tool results past a trigger; keeps last N; placeholder text; `clear_at_least` avoids small clears | Documented; no published savings figure in the sources read | Server-side; client keeps full history | S4, S5, S6 |
| Observation masking (JetBrains) | Old observations replaced by placeholder | About 50% lower cost than raw agent; solve rate kept in most setups; hybrid cuts a further 7-11% | Client-side | S37, S38 |
| Code execution with MCP | Tool definitions loaded on demand from files; intermediate data stays in the runtime | One example: about 150,000 tokens down to about 2,000 (98.7%) | Requires a sandbox | S7, S8 |
| "Mask, don't remove" (Manus) | Tools kept in context but masked, so the cache stays valid | Engineering practice; no numbers in the sources read | Client-side | S35 |
| Prophet (`goal.py`) | Older observations folded into memo by LLM; tool list fixed per run (sub-agents get a task-scoped list) | Not measured | Client-side | repo code |

**Conclusion.**
- Clearing or masking old tool output is the cheapest documented saving, and it is cheaper than LLM summarization.
  **Confidence: MEDIUM (measured in S37 on SWE-bench coding agents; generalization to Prophet is inferred).**
- Clearing changes the cached prefix, so clear in batches above a minimum size. **Confidence: HIGH (documented, S4).**
- The code-execution pattern cuts tokens sharply, but the 98.7% figure is a single vendor example and needs a sandbox.
  **Confidence: LOW (single vendor example, S7; no independent measurement in the sources read).**

## 7. Topic 5: Planning and re-planning

| Approach | Plan form | Re-plan trigger | Evidence | Source |
|---|---|---|---|---|
| ReAct | No plan object; thoughts interleaved with actions | Each thought can revise the plan | +34% (ALFWorld) and +10% (WebShop) absolute success vs imitation and RL, with 1-2 examples | S47 |
| Magentic-One | Task ledger (facts, guesses, plan) in outer loop; progress ledger in inner loop (five questions) | Stall counter above threshold (<= 2 in paper) triggers outer-loop re-plan; agents reset context after plan update | Paper design; ablations in paper | S42, S43 |
| Reflexion | Verbal reflection stored in episodic memory after a failed trial | Next trial, not same run | HumanEval pass@1 91% vs GPT-4 80% | S46 |
| Voyager | Automatic curriculum; skill library of verified code | Curriculum proposes next task | 3.3x more unique items; 15.3x faster to tech milestones (paper claims) | S49 |
| Prophet (`goal.py`) | Plan decided each step; no separate plan artifact; finish when plan says finish | Each step; stall = last N tool actions identical | Not measured | repo code |

**Conclusion.**
- A separate task ledger that is rewritten when the team stalls is what Magentic-One uses; Prophet re-plans each step without
  storing a plan. **Confidence: MEDIUM (documented design in S42/S43; benefit for Prophet not measured).**
- Plans are hints. Magentic-One does not require agents to follow the plan exactly. **Confidence: HIGH (documented, S42).**
- Reflexion-style lessons help across trials. Prophet's memo is within-run, so Reflexion adds little unless missions
  restart. **Confidence: LOW (inferred from design; not tested).**

## 8. Topic 6: State management and idempotency of side effects

| System | Guarantee | Crash during side effect | How it is surfaced | Source |
|---|---|---|---|---|
| Temporal activities | At-least-once by default; idempotency left to the caller via keys | Activity may run again; duplicate charge risk | Retry policy; Maximum Attempts = 1 gives at-most-once | S21, S22 |
| DBOS regular steps | At-least-once | Step re-runs if crash is before checkpoint | Documented as "at least once" | S17, S20 |
| DBOS transactional steps | Exactly-once for DB writes (write and checkpoint commit together) | Replays stored output | Requires `tx_step_outputs` table | S19 |
| LangGraph tasks | Side effects wrapped in tasks so they are not repeated on resume | Re-run from starting point unless in a task | Results read back from persistence | S16 |
| OpenAI Agents SDK | Pending approvals serialized in RunState; resume after decision | Failed session append stays pending and is reconciled without re-running the tool | `RunState` JSON; warns against concurrent restore | S13, S14 |
| Lease with fencing token (Kleppmann; Redis) | Token increases per grant; resource rejects lower tokens | Paused client cannot write after takeover | Resource checks token atomically | S26, S27 |
| Prophet (`store.py`, `runner.py`) | Tool ledger keyed by mission, step, tool, args; `done` returns stored result; `started` or `failed` non-idempotent stops as `OutcomeUnknown` | Mission stops with `outcome_unknown`; not retried | Ledger status; `LeaseHeld` / `LeaseLost` | repo code |

**Prophet specifics, from code reading.** `acquire_lease` takes over when `lease_until < now`. `renew_lease` runs at each
`_guard_time` (step boundary) and raises `LeaseLost` if taken over. The write functions (`finish_step`, tool writes) do not
check the lease owner or an epoch. So a runner that stalls between a step-boundary check and its write can still write after
a takeover.

**Conclusion.**
- At-least-once execution plus idempotency keys is the default in Temporal and DBOS. Exactly-once holds only when the effect
  and the checkpoint share one transaction. **Confidence: HIGH (documented, S17, S19, S21).**
- Treating "started, outcome unknown" as a stop, not a retry, is consistent with Temporal's guidance on non-idempotent work.
  Prophet's `OutcomeUnknown` does this. **Confidence: HIGH (documented principle, S21, S22; Prophet behaviour from code).**
- A lease with a TTL and no fencing token is unsafe under pauses. Prophet's writes are not fenced. **Confidence: MEDIUM
  (documented in S26 and S27; the gap in Prophet is inferred from code reading, not reproduced).**

## 9. Topic 7: Supervision and budgets

| Mechanism | Signal | Action | Evidence | Source |
|---|---|---|---|---|
| Temporal heartbeats | Periodic ping with progress payload | Missed heartbeat timeout fails attempt; next attempt reads payload | Documented | S23, S24 |
| Temporal retry policy | Attempt failures | Exponential backoff; default max attempts unlimited (0) | Documented default | S25 |
| Magentic-One stall counter | Loop or no-progress increments counter; progress decrements | Re-plan when counter exceeds threshold | Paper design | S42, S43 |
| Leases with TTL | Owner renews; expiry allows takeover | Another runner takes over | Documented (Kleppmann warns about pauses) | S26 |
| Prophet budgets | `max_seconds` checked per step; `max_tool_calls` checked before call; `max_iterations` per segment ends in checkpoint, not stop; `stall_window` exact-repeat test; `MAX_ASKS` = 8 | Budget exception, checkpoint, or stop | Code reading | repo code |

**Conclusion.**
- Heartbeats with progress payloads are the documented way to detect a hung worker and resume without restarting from zero.
  **Confidence: HIGH (documented, S23, S24).** Prophet is one process, so heartbeats are not needed now.
- Unlimited default retries are a budget hazard. Set explicit attempt caps. **Confidence: HIGH (documented default, S25).**
- Prophet's stall check only catches runs of identical signatures. It misses alternating loops (A, B, A, B) and no-progress
  runs with changing arguments. A decaying counter, as in Magentic-One, covers more. **Confidence: MEDIUM (inferred from code
  reading; Magentic-One's benefit is by design, S43).**

## 10. Topic 8: Iterative evaluate-repair loops and stopping criteria

| Method | Verifier | Evidence | Stop rule | Source |
|---|---|---|---|---|
| Intrinsic self-correction (Huang et al.) | None; model critiques own answer | Performance generally declined; models changed correct answers to wrong; gains only with oracle labels | Fixed rounds (max two in experiments) | S44 |
| CRITIC | External tools (search, code interpreter) | Consistent gains on QA, math and toxicity; external feedback essential | Critique passes or max iterations | S45 |
| Reflexion | Scalar or language feedback from evaluator | HumanEval pass@1 91% | Success or trial budget | S46 |
| Voyager | Environment feedback, execution errors, self-verification | Skills added only after verification | Curriculum loop | S49 |
| Prophet (`goal.py`) | Evaluator (met / not met / impossible); figure grounding check against tool output; `MAX_CORRECTIONS` = 2; stall window; `finish_now` | Not measured | Verdict, budget, stall, or user request | repo code |

**Conclusion.**
- Self-critique with no external check does not reliably improve answers. Prophet's grounding check (figures must appear in tool
  output) is the right kind of external verifier. **Confidence: MEDIUM (measured in S44 on reasoning benchmarks; Prophet domain
  differs).**
- Stopping rules should be counted: maximum corrections, maximum iterations, stall window. **Confidence: HIGH (documented
  practice, S45, S42).**
- Stop on a verifier result, not on the model saying it is done. Prophet's evaluator is model-based, so its criteria should be
  checkable against tool output. **Confidence: MEDIUM (inferred from S45 and Prophet code).**

## 11. Topic 9: Structured ask-the-user checkpoints

| Mechanism | Shape | Pause and resume | Secrets | Source |
|---|---|---|---|---|
| MCP elicitation, form mode | Flat JSON schema of primitives (string, number, boolean, enum) | Server sends `elicitation/create`; client returns | Servers MUST NOT use form mode for passwords, API keys, tokens, payment data | S29 |
| MCP elicitation, URL mode | `url` plus `elicitationId`; data not exposed to client | Out-of-band browser flow | Required for sensitive data | S29 |
| OpenAI Agents SDK HITL | `needs_approval` on tools; `ToolApprovalItem` in `interruptions` | `RunState.to_json()` then resume with decisions | Not covered | S13, S14 |
| LangGraph / DBOS | Interrupt or durable pause | Resume from checkpoint; interrupt re-runs node | Not covered | S15, S16 |
| Prophet (`questions.py`, `goal.py`) | AskUserQuestion-shaped, validated; answers kept verbatim in `_decisions`; duplicate-question refusal tests exist | Pause and resume from stored state; no re-ask of the model | Keys are in `.env` (git-ignored), not asked through questions | repo code (test names seen via graphify) |

**Conclusion.**
- Flat, typed, bounded questions are the convergent standard, and MCP form mode is the same shape. **Confidence: HIGH
  (documented, S29).**
- Credentials must never go through form-mode questions. Use an out-of-band flow. **Confidence: HIGH (documented MUST, S29).**
- Pausing should persist serializable state and resume from answers, not ask the model again. Prophet does this, as does
  OpenAI's RunState. **Confidence: HIGH (documented, S14, S16; Prophet from code).**

## 12. Fit to Prophet

Each row says whether to adopt now, later, or not at all, with the reason and a minimal change. Changes are described only;
no code was changed. Each row was checked against the locked decisions in `docs/prophet/ARCHITECTURE.md` (SQLite WAL, NIM chain,
fixed prompt layout, model-pinned sessions, idempotency ledger).

| Topic | Decision | Reason | Minimal change (describe only) | Confidence |
|---|---|---|---|---|
| 1. Loops and resume | Adopt now (keep as is) | Per-step synchronous writes already match `sync` durability | Add a test: crash after a non-idempotent tool finishes but before the step write must not re-run the tool | HIGH (documented, S15) |
| 2. Compaction | Adopt now | Measured silent constraint loss; memo cap drops oldest text | Replace `text[-memory_chars:]` with a re-compaction request (ask the compactor to shorten, keep decisions); pin user instructions verbatim next to `_decisions`, not only question answers | HIGH (defect from code; loss measured in S39, S40) |
| 3. Prompt caching | Adopt now (layout); later (telemetry); not at all (provider cache controls) | Stable head is right; hit-rate unknown on NIM; provider `cache_control` and opaque compaction items would not survive the model-chain failover | Log cached-token counts from NIM responses if NIM returns them; check whether NIM supports prefix reuse before relying on it | MEDIUM (inferred; NIM behaviour unverified) |
| 4. Tool-result clearing | Later (deterministic masking); not at all (Anthropic-specific `clear_tool_uses`) | Masking is cheaper than LLM summary in S37 but measured only on coding agents | Mask old tool observations before the memo LLM call; evaluate on the synthetic twin before changing the memo | MEDIUM (measured in S37, other domain) |
| 5. Planning | Later (small task ledger); not at all (DAG planners, curriculum) | Magentic-One's ledger is cheap; full planners add cost without evidence for Prophet's task size | Keep facts, guesses and current plan in a short block rewritten on stall | LOW (no Prophet benefit measured) |
| 6. State and idempotency | Adopt now (fencing); not at all (Temporal or DBOS) | Lease without fencing is unsafe under pauses (S26); SQLite WAL is single-writer and suits a single host | Add a `lease_epoch` to missions, increment on acquire, and require it on step and tool-ledger writes; keep `OutcomeUnknown` as is | MEDIUM (inferred from code; documented in S26) |
| 7. Supervision | Adopt now (counter); later (heartbeats); not at all (Temporal heartbeats) | Exact-repeat misses cycles; heartbeats not needed in one process | Add a decaying stall counter that also counts no-progress and short cycles; keep the exact-repeat check as a fast path | MEDIUM (inferred from code; S42/S43 design) |
| 8. Evaluate-repair | Adopt now (already in place); not at all (free self-critique rounds) | External grounding check fits S44/S45; extra self-critique rounds have measured downside | Keep `MAX_CORRECTIONS` = 2 and the figure check; do not add unverified self-review rounds | HIGH (documented, S44, S45) |
| 9. Ask-the-user | Adopt now (keep shape); later (URL mode if MCP elicitation is used) | Shape matches MCP form mode; secrets already stay out of questions | Keep the question validator; if an MCP server ever asks for a secret, route it through URL mode | HIGH (documented, S29) |

**Top five adoption recommendations, one line each:**
1. Pin user instructions verbatim alongside `_decisions` and re-inject them on every planning call (Topic 2, HIGH).
2. Replace the memo tail-cut at 8,000 characters with a re-compaction request that keeps decisions (Topic 2, HIGH).
3. Add a `lease_epoch` fencing token and require it on every step and tool-ledger write (Topic 6, MEDIUM).
4. Add a decaying stall counter that also catches alternating loops and no-progress runs (Topic 7, MEDIUM).
5. Keep the stable prompt head and log NIM cached-token counts before adding any cache logic (Topic 3, MEDIUM).

## 13. Open questions and limits

- **No Prophet test was run.** Prophet claims come from reading `goal.py`, `runner.py`, `store.py` and the compactor prompt,
  and from graphify orientation. Each Prophet-specific finding should be reproduced with a test before any change.
- **Does NIM report cached tokens or keep prefixes?** Not checked in this pass. Topic 3 depends on it.
- **Vendor and preprint evidence.** The compaction results (S39, S40) are 2026 arXiv preprints; I did not check peer review
  status. Factory's numbers (S36) are the vendor's own evaluation. The 98.7% token figure (S7) is one example.
- **Gemini minimums conflict** across S31, S32 and S33; recorded as published, not resolved.
- **vLLM (S34)** was read from a search snippet. An earlier fetch of a different vLLM URL returned HTTP 429.
- **Snippet-only sources.** S10, S34, S47, S48 and S49 were checked through abstracts or search snippets, not full text.
  Claims from them are limited to what the snippets showed.
- **Not covered or not verified in this pass:** CrewAI, AutoGen beyond the Magentic-One orchestrator source, Google ADK
  beyond compaction, Letta (MemGPT successor), Generative Agents, LLMCompiler, ADaPT, and the LangGraph `interrupt()`
  re-execution rules beyond the durable-execution concept page. Anthropic's server-side compaction beta header was not
  re-checked in this pass, so it is not cited.
- **Measured versus inferred.** Conclusions tagged MEDIUM or LOW that say "inferred" have no measurement behind them for
  Prophet. They are design arguments from the cited evidence.
- **Scope.** This note does not evaluate Stripe-style idempotency APIs, Redis locking in depth, or TimesFM licensing (see
  `ARCHITECTURE.md` section 5 for the licence gate).
- **Dates.** "Date seen" is 2026-11-21 for all sources. Published dates are given only where the page showed them, or
  as ID-implied for arXiv.
