# Durable agent runtimes for Prophet: adopt or build

Snapshot date: 2026-11-20. Scope: durable-execution and agent-runtime options for a single-user, local-first, macOS (Apple silicon) FastAPI app that calls NVIDIA NIM (OpenAI-compatible chat completions, key pool, model fallback, streaming), runs multi-hour missions, survives crashes, pauses for human answers, calls tools (TimesFM, sandboxed Python), and streams progress to React.

Labels: [Verified: URL] = read in a fetched or search-returned primary source during this research. [Hypothesis] = my prior knowledge or inference, not checked in this session. Where a fetch returned only a landing page, I say so rather than fill the gap.

Coverage caveat: the research budget was short. Most vendor docs pages returned landing-page text only (Temporal, Restate, Pydantic AI, Letta, Inngest licence), so several "Gaps" below are real gaps, not omissions. Verify the licence and infra rows before committing.

## KQ1. Durability, resume, human-in-the-loop and interrupt primitives per candidate

### Takeaway
Every candidate offers some form of checkpointed resume, but the mechanism differs in kind. Server-backed engines (Temporal, Restate, Inngest) durably replay the whole run. Library-level options (DBOS, LangGraph checkpointers) checkpoint at step or superstep boundaries in your own database. Agent SDKs (OpenAI Agents, Claude Agent SDK, Microsoft Agent Framework) give you a pause-and-resume object, and persistence is largely your job or only partly provided.

### Cited Findings
- DBOS (Python): workflows are `@DBOS.workflow()`, side effects are `@DBOS.step()`. "If a workflow is interrupted, DBOS automatically recovers its execution from the last completed step." Completed steps are not re-executed; failed steps retry, so steps are at-least-once. Workflow functions must be deterministic; non-deterministic work (clock, random, API calls) belongs in steps; mismatches raise `DBOSStepNondeterminismError`. Workflow IDs double as idempotency keys; `workflow_id_reuse_policy="reject"` raises `DBOSWorkflowIDInUseError`. [Verified: https://docs.dbos.dev/python/tutorials/workflow-tutorial]
- DBOS storage: default is SQLite. The Python SDK uses an on-disk SQLite file when no `system_database_url` is set, and the LlamaIndex and Pydantic AI integration docs show `sqlite:///...`. Postgres is recommended for production or multi-process use. [Verified via search summaries: https://pydantic.dev/docs/ai/integrations/durable_execution/dbos/index.md and https://docs.resonatehq.io/evaluate/coming-from/dbos] The current DBOS docs page I fetched did not name the backends, so treat the SQLite claim as "verified by secondary sources", not by DBOS's own page. A 2025 Reddit post said Postgres was the only supported database. It is superseded but shows the claim changed in the past year.
- DBOS messaging: `DBOS.recv` lets a workflow wait for a message. I did not get the send/recv details, so the human-answer pattern is [Hypothesis]: a workflow blocks on `recv(topic)` and the API calls `send(workflow_id, ...)`.
- LangGraph: a checkpointer persists graph state per `thread_id`. Checkpointers named: `InMemorySaver`, `SqliteSaver`, `PostgresSaver` / `AsyncPostgresSaver`. In-memory checkpoints are lost on restart; the docs recommend a persistent checkpointer for production and `SqliteSaver` for local development. Checkpoints accumulate and need pruning. [Verified: https://docs.langchain.com/oss/python/langgraph/persistence]
- LangGraph interrupts: `interrupt(...)` pauses. Resume by invoking again with `Command(resume=value)`. On resume the node re-runs from its start, and `interrupt()` returns the resume value rather than pausing again, so code before `interrupt()` must be minimal and idempotent. Multiple interrupts in one node are matched by order to a list of resume values. Losing the `thread_id` loses the pointer to the saved state. [Verified (search summary of docs.langchain.com interrupts and HITL pages): https://docs.langchain.com/oss/python/langgraph/interrupts , https://docs.langchain.com/oss/python/langgraph/human-in-the-loop] Note: my fetch of the durable-execution page returned the persistence page, so durability modes (exit/async/sync) are unverified.
- OpenAI Agents SDK: tools declare `needs_approval` (bool or async callable; callables fail closed). A run pauses and `RunResult.interruptions` lists pending `ToolApprovalItem`s. Resume with `result.to_state()`, then `state.approve()` or `state.reject()`, then `Runner.run(agent, state)`. Persistence is developer-managed: `state.to_json()` / `RunState.from_json()` into your own store. The docs warn to deserialise only from trusted storage, and that approval does not guarantee exactly-once execution, so use idempotency keys for consequential side effects. [Verified: https://openai.github.io/openai-agents-python/human_in_the_loop/] Whether HITL works over a non-OpenAI chat-completions endpoint was not confirmed on that page.
- Temporal: the Python SDK runs workflows and activities on a Temporal service; signals and queries deliver human input. The OpenAI Agents integration (`temporalio-openai-agents`) runs the agent loop inside a Workflow and turns each model call into an Activity, and Temporal states that its durability does not extend to MCP servers, which run outside the Workflow. [Verified: https://temporal.io/blog/announcing-openai-agents-sdk-integration , https://docs.temporal.io/develop/python/openai-agents] The Temporal Python docs index I fetched did not describe determinism rules or server persistence options. [Verified only as a link index: https://docs.temporal.io/develop/python]
- Restate: the landing page describes steps stored as they complete, resume from the last completed step, durable timers, and workflows that wait for human approvals or webhooks. It names three hosting modes: Restate Cloud, BYOC, and self-hosted. Architecture (single binary, log-based journal, awakeables) and server licence were not in the fetched text. [Verified: https://docs.restate.dev/ for the landing claims only]
- Inngest: the CLI is a single binary (`inngest start`, port 8288). Default is SQLite for persistence and in-memory Redis for the queue and state. Postgres (`postgres-uri`) and Redis (`redis-uri`) are optional. SQLite "does not support scaling beyond a single node". [Verified: https://www.inngest.com/docs/self-hosting] Licence not stated on that page.
- Microsoft Agent Framework (workflows): executors call `ctx.request_info()` for human input and handle replies with `@response_handler`. Checkpoints are taken at superstep boundaries and store outstanding requests; restoring a checkpoint re-emits open requests so a host can answer them later. Changing the workflow's executors or edges invalidates existing checkpoints. [Verified (search summary of learn.microsoft.com checkpointing page and mirror): https://learn.microsoft.com/en-us/agent-framework/tutorials/workflows/checkpointing-and-resuming , https://www.mintlify.com/microsoft/agent-framework/workflows/human-in-the-loop]
- Claude Agent SDK: sessions persist the conversation, not the filesystem. `ClaudeAgentOptions.resume` takes a session ID; `fork_session` branches. [Verified: https://code.claude.com/docs/en/agent-sdk/sessions] It is a wrapper around Claude's agent loop; I found no documented way to point it at an arbitrary OpenAI-compatible base URL.
- Letta: positioned as a stateful-agent platform with a self-hostable App Server and persistent memory. Its docs page does not describe durable execution, crash resume, or approval pauses. [Verified as absence on https://docs.letta.com/; Gap]
- Pydantic AI: durable-execution integrations exist for Temporal and DBOS (and others). Its docs page redirected (ai.pydantic.dev to pydantic.dev) and I could not read the body. [Verified redirect: ai.pydantic.dev/durable_execution/overview/; Gap on content]
- Anthropic long-running harness pattern: an initializer session writes a feature list (JSON, all `passes: false`), an `init.sh`, a progress log, and an initial git commit. Coding sessions make one feature of progress, leave the tree clean, and update state. Agents may only flip `passes`; tests must not be removed or edited. Each session starts by checking the app still works. [Verified: https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents]

### Inferences
- Only DBOS, LangGraph (SQLite checkpointer) and a hand-built event log give a pure-library, no-daemon durable loop. Temporal, Restate and Inngest all need a separate service process.
- For HITL, the common shape across LangGraph, OpenAI Agents, DBOS and Microsoft is the same: persist a "pending request" record with a stable ID, return control, and resume by matching the answer to that ID. Your own tables should copy this shape regardless of framework.

### Gaps
- Pydantic AI DBOS and Temporal integration details (how tool calls map to steps) not read.
- Restate architecture and server licence not read.
- DBOS `recv`/`send` signatures not read.
- LangGraph durability modes (exit/async/sync) not read; the durable-execution page I fetched returned persistence content instead.
- smolagents, CrewAI and Letta durability not researched beyond the Letta landing page.

## KQ2. Licence, latest release, maintenance health, Python 3.12

### Takeaway
All actively released candidates support Python 3.12 on their declared floors, and the ones with PyPI packages released in the last two weeks are clearly maintained. AutoGen's PyPI package has been quiet since September 2025, which points to the Agent Framework successor. Licence claims below come from PyPI metadata where stated.

### Cited Findings
Latest versions from the PyPI JSON API as fetched on 2026-11-20 (`https://pypi.org/pypi/<name>/json`):

| Candidate (PyPI name) | Latest | Upload date | requires_python | Licence (PyPI) |
|---|---|---|---|---|
| dbos | 3.6.0 | 2026-11-17 (3.7.0a3 on 2026-11-19) | >=3.10 | MIT |
| temporalio | 1.36.1 | 2026-11-14 | >=3.11 | MIT (classifier) |
| pydantic-ai | 2.71.0 | 2026-11-19 | >=3.11 | MIT |
| langgraph | 1.2.24 | 2026-11-20 | >=3.10 | MIT |
| openai-agents | 0.25.0 | 2026-11-18 | >=3.10 | MIT |
| claude-agent-sdk | 0.2.186 | 2026-11-19 | >=3.10 | MIT |
| agent-framework (Microsoft) | 1.26.0 | 2026-11-12 | >=3.11 | not fetched |
| autogen-agentchat | 0.7.5 | 2025-09-30 | >=3.10 | not fetched |
| crewai | 1.15.44 | 2026-11-19 | >=3.10,<3.14 | not fetched |
| smolagents | 1.26.0 | 2026-05-29 | >=3.10 | not fetched |
| letta | 0.34.12 | 2026-10-12 | >=3.9 | not fetched |
| inngest (Python SDK) | 0.5.20 | 2026-11-02 | >=3.10 | not fetched |
| restate-sdk | 1.5.4 | 2026-11-17 | >=3.11 | not fetched |

Source for all rows: https://pypi.org/pypi/<name>/json (queried 2026-11-20). Python 3.12 is inside every declared range, so none are blocked on version. Note that requires_python is a floor check, not a tested-matrix claim.

Naming and status checks:
- DBOS: alive and releasing weekly on PyPI. Project name `dbos` (the "DBOS Transact" library). [Verified: PyPI]
- Temporal: alive, `temporalio` releasing monthly. Server is separate (Temporal Server / CLI dev server). [Verified: PyPI]
- LangGraph: alive, multiple releases in November 2026. [Verified: PyPI]
- OpenAI Agents SDK: alive, package `openai-agents`. [Verified: PyPI]
- Claude Agent SDK: alive, package `claude-agent-sdk`, Python SDK wraps the Claude Code CLI. [Verified: PyPI]
- Pydantic AI: alive, `pydantic-ai`. [Verified: PyPI]
- AutoGen: `autogen-agentchat` last release 2025-09-30. I did not verify Microsoft's maintenance status for AutoGen; Agent Framework is the newer Microsoft package, released 2026-11-12. [Verified: PyPI date; Hypothesis: AutoGen is superseded]
- Restate: `restate-sdk` (import name `restate_sdk`) is alive. The server is a separate binary. [Verified: PyPI]
- Inngest: Python SDK alive. [Verified: PyPI]
- Letta: alive, last release 2026-10-12. [Verified: PyPI]
- Licences of Temporal server, Restate server, Inngest server, Letta server, Microsoft Agent Framework, CrewAI and smolagents were not verified in this session. [Hypothesis, unverified: Temporal server MIT; Restate server BSL 1.1 (verify before use); Inngest server SSPL-style "source available" (verify); smolagents Apache-2.0; CrewAI MIT.]

### Inferences
- Cadence of releases (DBOS, LangGraph, Agents SDK, Claude Agent SDK releasing within days of each other) signals active churn. Pin exact versions and expect breaking changes in 0.x packages (openai-agents at 0.25, claude-agent-sdk at 0.2.x).

### Gaps
- Licence and server-side licence for most server-backed candidates.
- Temporal and Restate Python 3.12 test matrix (declared floor only).

## KQ3. Arbitrary OpenAI-compatible endpoint with a custom key-rotating transport; lock-in; footprint; does it fight a custom loop

### Takeaway
A thin custom transport is the hard requirement for NIM. Frameworks that accept a pre-built OpenAI-compatible client (OpenAI Agents SDK, LangGraph via LangChain's OpenAI chat model, Pydantic AI's OpenAI-compatible provider) can probably take it, but I did not confirm this in the fetched docs for any of them. The Claude Agent SDK is bound to Claude and is a poor fit for NIM. Server-backed engines add infrastructure that a single-user local app does not need.

### Cited Findings
- OpenAI Agents SDK: HITL state is plain JSON you serialise yourself, so there is no framework-owned store to fight. [Verified: https://openai.github.io/openai-agents-python/human_in_the_loop/]
- Temporal OpenAI integration: model calls become Activities, so the model transport sits behind an activity boundary and a custom client is possible in principle. [Verified structure: https://docs.temporal.io/develop/python/openai-agents; custom-client support: Hypothesis]
- Claude Agent SDK: sessions and tool loop belong to Claude Code's CLI. I found no `ClaudeAgentOptions` base-URL field in search results. [Verified absence in search results only: https://code.claude.com/docs/en/agent-sdk/sessions]
- Inngest and Temporal both need a server process (Inngest: single binary, SQLite default, optional Postgres and Redis; Temporal: separate service). [Verified for Inngest: https://www.inngest.com/docs/self-hosting]
- Lock-in: LangGraph state is in LangGraph's checkpoint schema (SQLite/Postgres tables you do not own); Temporal and Restate put history in their service; the OpenAI Agents SDK state blob is versioned by the SDK; DBOS stores workflow state in its own system tables inside your DB. [Verified for LangGraph checkpointers and DBOS system database; the schema-level claims are Hypothesis.]

### Inferences
- The NIM key pool, per-key rate-limit backoff, and model fallback chain belong in your transport layer, below any framework. Any framework can sit on top of it if it accepts a client object. Confirm that before choosing a framework: the decision hinges on whether the framework lets you inject a client and retry policy.
- A framework's retry layer will fight your key rotation if it retries 429s internally. Disable or wrap it.

### Gaps
- No verified example of injecting a custom base URL and key pool into OpenAI Agents SDK, LangGraph, or Pydantic AI in this session.
- Claude Agent SDK base-URL support unverified; likely not usable for NIM.

## KQ4. What production agent teams do for crash-resume

### Takeaway
Production teams converge on three things: an append-only record of what happened, a checkpoint or journal entry before and after each non-deterministic step, and idempotency keys on every external side effect. Framework choice changes where those records live, not the rules.

### Cited Findings
- Anthropic (long-running agents): state lives in the progress log and git history, not in the model's context. Each session starts by re-verifying the environment. One feature per session. Tests are not edited to pass. [Verified: https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents]
- Temporal: model calls as activities, deterministic workflow code, durability by event history replay. [Verified: https://temporal.io/blog/announcing-openai-agents-sdk-integration]
- DBOS: checkpoint per completed step, recover from the last completed step, steps at-least-once, deterministic workflows. [Verified: https://docs.dbos.dev/python/tutorials/workflow-tutorial]
- Microsoft Agent Framework: checkpoints at superstep boundaries including open requests; topology change invalidates checkpoints. [Verified: https://learn.microsoft.com/en-us/agent-framework/tutorials/workflows/checkpointing-and-resuming]
- OpenAI Agents SDK: explicit warning that approval does not guarantee exactly-once; use idempotency keys for consequential side effects. [Verified: https://openai.github.io/openai-agents-python/human_in_the_loop/]
- LangGraph: node re-runs from its start on resume, so pre-interrupt code must be idempotent. [Verified (search summary of docs.langchain.com): https://docs.langchain.com/oss/python/langgraph/interrupts]

Named public engineering write-ups on production agent teams (Vercel/Temporal durable agents blog, Temporal AI Engineer talk) were found in search but not read; treat as leads only.

### Inferences
- The common failure is re-running a model call or tool call after a crash between "performed" and "recorded". The fix is an intent record written before the call, an idempotency key passed to the tool, and a status that says "unknown" when the result is missing.
- For this app, a sandboxed Python run and a TimesFM forecast are idempotent if keyed by inputs, so they can be cached and replayed. An LLM call is not idempotent in cost or in output; replay should reuse the stored response, not re-ask.

### Gaps
- No first-hand post-mortem from a team running long agent missions on SQLite was found in this session.

## KQ5. Recommendation: adopt, thin library, or build

### Takeaway
Build the durable core yourself: an event log plus step and tool ledgers in SQLite, a single-writer lease, and a thin NIM transport you own. Do not adopt a server-backed engine (Temporal, Restate, Inngest) for a single-user local app. If you later want a library to avoid writing recovery code, DBOS Transact is the closest fit: MIT, library not server, SQLite default, step-level checkpointing. Use LangGraph only if you want its graph model; it would pull the loop shape out of your hands.

### Cited Findings
- Server-backed options need extra services: Temporal (separate server), Inngest (single binary plus optional Postgres and Redis, SQLite default not scaling past one node), Restate (separate server; hosting options listed). [Verified: sources in KQ1]
- DBOS runs in-process against SQLite or Postgres. [Verified: docs and secondary sources in KQ1]
- LangGraph checkpointers work with SQLite. [Verified: https://docs.langchain.com/oss/python/langgraph/persistence]
- Anthropic harness pattern: state outside the model, one unit per session, re-verify on start. [Verified: https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents]

### Inferences
- Build vs DBOS: DBOS saves you writing checkpoint and recovery code but dictates a decorator-based workflow shape and a SQLite system schema you do not own. For a lab app where replay rules matter (no future leakage, TimesFM reproducibility), owning the ledger is worth the few hundred lines. This is a judgement call, not a verified fact.
- Frameworks that give a pause-and-resume object (OpenAI Agents SDK `RunState`) are useful references for the serialised form of a pending approval, even if you do not import the SDK.

### Recommendation
Build own durable loop. Adopt nothing for the core. Optionally use `dbos` as a thin library only for mission scheduling if the custom ledger becomes a burden.

Minimum viable durable design (Hypothesis-level design, derived from the cited patterns above):

Tables (SQLite, WAL mode, one database file):
- `missions(id PK, goal, status, budget_json, lease_owner, lease_expires_at, created_at, updated_at)`. Status: queued, running, paused, awaiting_human, cancel_requested, cancelled, done, failed.
- `events(mission_id, seq, ts, type, payload_json, PRIMARY KEY(mission_id, seq))`. Append-only. Append with `BEGIN IMMEDIATE`, and the next seq is read in the same transaction. This table is the source for the UI stream (SSE with a seq cursor).
- `steps(step_id PK, mission_id, turn, kind, status, idempotency_key UNIQUE, request_hash, response_json, model_used, key_slot, attempt, started_at, completed_at)`. `kind` is llm, tool, or sandbox. `step_id` = hash(mission_id, turn, kind, index).
- `tool_calls(idempotency_key PK, mission_id, tool_name, args_hash, status: pending|done|unknown|failed, result_json, side_effect: none|idempotent|external)`.
- `interrupts(id PK, mission_id, question_json, options_json, status: open|answered|expired, answer_json, asked_at, answered_at)`.
- `nim_keys(slot PK, cooldown_until, last_error)`. Transport state only; never part of the replay.

Replay rules on resume:
1. Acquire the mission lease (single worker per mission). Renew by heartbeat; on expiry another worker may take over.
2. Rebuild state by folding `events` in seq order. Never rebuild from in-memory objects.
3. For each `steps` row in status `started` with no response: for llm, re-issue with the same idempotency key and record a new attempt (accept the cost; this is the only case where a model call repeats). For tool with `side_effect=none` or `idempotent`, re-run. For `external`, mark `unknown` and raise an interrupt asking the human whether it happened.
4. For `steps` with status `done`, return the stored response; do not call the model or tool again.
5. Sandboxed Python and TimesFM calls are keyed by hash of code or input series; cache hits short-circuit.
6. Model fallback chain: record `model_used` per step so replays are auditable and fallback changes are visible.

Idempotency keys: `sha256(mission_id || turn || kind || index || canonical_args)`. Pass the key to any tool that supports it; record it even when the tool does not.

HITL pause: write an `interrupts` row with status open, set mission to awaiting_human, stop scheduling. The answer endpoint writes `answer_json`, sets status answered, appends an event, and releases the lease. The worker resumes from the next step.

Cancel: cooperative. Set cancel_requested; the worker checks it between steps and before each tool or model call, kills any running sandbox subprocess, and writes a `cancelled` event. Cancel never interrupts a committed step.

Pause: set paused; the worker finishes the current step then stops. Resume: clear paused, acquire the lease, replay per rules above.

Crash between step start and completion: covered by rule 3. Crash after completion but before the event append: make the step row and event append one transaction so they commit together.

Streaming tokens: write deltas to the UI through SSE from memory, and only persist the final response to `steps`. A crash mid-stream loses the partial text, which is acceptable; the step is re-issued.

Testing: a crash-injection test that kills the worker after every step boundary and asserts the final event log matches an uninterrupted run, with the model replaced by a fixture. This is the single most valuable test for this design.

### Gaps
- Not verified: whether DBOS's SQLite backend is fully supported for production use on a single node. Treat as Hypothesis.
- Not verified: Pydantic AI's durable integrations and their step semantics.
- Not verified: NIM-specific behaviour on streamed tool calls and whether NIM returns idempotency or request IDs usable for dedupe.

## Source list (verified pages)
- https://pypi.org/pypi/<name>/json for dbos, temporalio, pydantic-ai, langgraph, openai-agents, claude-agent-sdk, agent-framework, autogen-agentchat, crewai, smolagents, letta, inngest, restate-sdk
- https://docs.dbos.dev/python/tutorials/workflow-tutorial
- https://pydantic.dev/docs/ai/integrations/durable_execution/dbos/index.md (search result)
- https://docs.resonatehq.io/evaluate/coming-from/dbos (search result)
- https://docs.langchain.com/oss/python/langgraph/persistence
- https://docs.langchain.com/oss/python/langgraph/interrupts (search result)
- https://docs.langchain.com/oss/python/langgraph/human-in-the-loop (search result)
- https://openai.github.io/openai-agents-python/human_in_the_loop/
- https://docs.temporal.io/develop/python (link index only)
- https://docs.temporal.io/develop/python/openai-agents (search result)
- https://temporal.io/blog/announcing-openai-agents-sdk-integration
- https://docs.restate.dev/ (landing only)
- https://www.inngest.com/docs/self-hosting
- https://docs.letta.com/ (landing only)
- https://code.claude.com/docs/en/agent-sdk/sessions (search result)
- https://learn.microsoft.com/en-us/agent-framework/tutorials/workflows/checkpointing-and-resuming (search result)
- https://www.mintlify.com/microsoft/agent-framework/workflows/human-in-the-loop (third-party mirror)
- https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
- https://ai.pydantic.dev/durable_execution/overview/ (redirects to https://pydantic.dev/docs/ai/integrations/durable_execution/overview/; body not read)
