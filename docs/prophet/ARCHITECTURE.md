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
quantiles from the 10th to the 90th percentile. Maximum context and horizon are not published. The repo's context cap
of 512 and the library default of 15,360 are our settings, not model limits. Covariates are supported (past-only and
past-future) but there is no independent evidence of covariate skill, and our own calendar covariate gave no gain on
the twin [R]. No training hook is in the 3.0 package; a LoRA example exists in the GitHub README with no stated version
[V-docs, unverified for 3.0].

**Routing.** Intermittent series go to TSB or Croston, not TimesFM [R]. Test covariates with and without, using a
rolling-origin backtest scored with MASE, CRPS or WQL, pinball loss and coverage [R].

## 6. Decisions needed from the user

These block nothing in the foundation work, but they block the moves listed.

1. **Legacy `src/lab` and the `timesfm-erp` package.** Keep, move under `legacy/`, or delete? Deleting or moving it
   changes the test suite (110 tests). I will not change it without a yes.
2. **Package name:** `prophet_agent` or a new name (§4).
3. **TimesFM 3 licence** (§5): who signs off, and until then, R&D only?
4. **`NIM.md` and `.env`.** `NIM.md` is untracked and not git-ignored; my scan found no `nvapi-` strings of 20 or more
   characters in it, but the legacy audit reported six key-shaped strings, so the discrepancy is unresolved. `.env` is
   mode 644; I recommend 600. Both change files outside the new work, so I need a yes.
5. **Commit and push.** Nothing is committed yet. Approve the foundation commit, and whether to push to `origin/main`.
