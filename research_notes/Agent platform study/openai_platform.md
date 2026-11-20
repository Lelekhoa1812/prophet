# OpenAI Agent Products: Architecture Evidence for Prophet

Scope: ChatGPT agent mode, Deep Research, Operator/computer use, Codex, Code Interpreter, Responses API, Agents SDK, background mode, MCP. Current as of 2026-11-20. Public evidence only.

Evidence tags: [V-docs] documented publicly (URL given); [V-source] seen in open-source code (none inspected directly this round; docs pages generated from the repo were used instead); [V-observed] none (no first-hand product session); [H] hypothesis, UNVERIFIED.

Access note: several primary pages returned HTTP 403 to the fetch tool (the Deep Research launch post, the ChatGPT agent launch post, and the Deep Research FAQ). The Operator system card PDF returned binary data that could not be read. Claims that rest only on search snippets or third-party coverage are marked as such. Some OpenAI doc URLs now redirect to learn.chatgpt.com (noted below). Fetched doc summaries used a summariser, so exact wording should be re-checked before quoting externally.

## Key Question 1: Agent loop and stopping rules

### Takeaway
The Agents SDK defines the loop as: call the model, then either stop on final typed output with no tool calls, run a handoff and rerun, or run tool calls and rerun. Turn limits are explicit and raise an error when hit.

### Cited Findings
- Each pass calls the LLM for the current agent. Output is final when it "produces text output with the desired type, and there are no tool calls." Handoffs update the agent and input and rerun. Tool calls run and their results are appended, then the loop reruns. [V-docs: https://openai.github.io/openai-agents-python/running_agents/]
- The runner raises `MaxTurnsExceeded` when `max_turns` is exceeded; `max_turns=None` disables the limit. [V-docs: https://openai.github.io/openai-agents-python/running_agents/]
- `RunConfig` exposes `max_function_tool_concurrency` (via `tool_execution`), `tool_not_found_behavior` and `tool_error_formatter`. [V-docs: https://openai.github.io/openai-agents-python/running_agents/]
- The Responses API loop for Codex-style agents: Codex's own loop is not documented in the pages I could read (see Gaps).

### Inferences
- The SDK's stop rule is structural (no tool calls plus typed output), not a model-declared "done" flag. For Prophet, a deterministic completion condition plus a hard turn cap is the safer pattern than trusting the model to stop.

### Gaps
- The `tool_use_behavior` option is not covered on the running-agents page I fetched, so I cannot describe it.
- OpenAI's own consumer agent loop (ChatGPT agent) is not documented in any page I could read.

## Key Question 2: Planning and clarifying questions

### Takeaway
Deep Research in ChatGPT is documented (help centre, via search snippet) as asking clarifying questions and showing an editable research plan before it starts, with the ability to interrupt. The API version is documented as not having a clarification step. The consequential-action confirmation pattern is documented for Operator and ChatGPT agent.

### Cited Findings
- OpenAI's help centre says Deep Research "may ask clarifying questions to confirm your goals, alongside a research plan before it starts, and you can review and edit that plan." Newer help text describes answering clarifying questions, following progress, and interrupting to refine focus. [V-docs, search snippet only, full page 403: https://help.openai.com/en/articles/10500283]
- The developer documentation reportedly says Deep Research via the Responses API "does not include a clarification or prompt rewriting step." This appeared in a search snippet; the fetched developer page did not contain that sentence. [V-docs, snippet only: https://developers.openai.com/api/docs/guides/deep-research]
- The Operator system card (January 2025) describes confirmation prompts before critical actions, with examples including financial transactions, sending emails and deleting calendar events. [V-docs, search snippet of https://openai.com/index/operator-system-card ; PDF https://cdn.openai.com/operator_system_card.pdf was unreadable]
- Secondary coverage of ChatGPT agent (July 2025) says it asks for confirmation before "actions of consequence" such as sending an email, and that confirmation reduces prompt-injection risk. [Secondary: https://www.processexcellencenetwork.com/ai/news/openai-launches-chatgpt-agent-that-performs-complex-multistep-tasks ; primary ChatGPT agent system card 403]
- Secondary coverage (Decrypt) says the agent sometimes seeks approval before purchases or sensitive data access, and users can take over the browser. [Secondary: https://decrypt.co/330714/openais-chatgpt-agent-launches-expanded-powers-elevated-risk?amp=1]

### Inferences
- [H] The product pattern is "plan, confirm, run, interrupt" for research, and "confirm consequential action" for computer use. The two are separate mechanisms, not one.

### Gaps
- Could not read the Deep Research launch post or the ChatGPT agent system card (both 403). Exact clarification behaviour, plan editing UI and take-over mechanics rest on help-centre snippets and secondary coverage.

## Key Question 3: Tool use and tool discovery

### Takeaway
Tools are declared per request as typed items. Built-in hosted tools (web search, code interpreter, MCP) run server-side. Client functions run in your code. Handoffs are exposed to the model as tools. The SDK keeps the set of tools fixed per run and gates them with rules.

### Cited Findings
- Responses API MCP tool: a `"type": "mcp"` entry with `server_label`, `server_url`, `allowed_tools` and `require_approval`. The example used `require_approval: "never"` and says to keep approval on for tools that can modify data. [V-docs: https://developers.openai.com/api/docs/mcp]
- Custom MCP servers are described as "not developed or verified by OpenAI"; "Any MCP server may receive sensitive data as part of querying"; prompt injection risk applies even when the developer is trusted. [V-docs: https://developers.openai.com/api/docs/mcp]
- Handoffs "are represented as tools to the LLM"; an agent named `Refund Agent` becomes `transfer_to_refund_agent`. `handoff()` options include `input_filter`, `is_enabled`, `on_handoff`, `input_type`. Default: receiving agent sees full history. [V-docs: https://openai.github.io/openai-agents-python/handoffs/]
- `is_enabled` is evaluated before the model returns arguments, so it cannot check argument values; parsed-field checks belong at the start of `on_handoff`. [V-docs: https://openai.github.io/openai-agents-python/handoffs/]
- Hosted tools (`WebSearchTool`, `FileSearchTool`, `HostedMCPTool`, `CodeInterpreterTool`, `ImageGenerationTool`) and built-in execution tools (`ComputerTool`, `ShellTool`, `ApplyPatchTool`, `LocalShellTool`) do not use the SDK's tool-guardrail pipeline. [V-docs: https://openai.github.io/openai-agents-python/guardrails/]
- Responses API `web_search` emits a `web_search_call` item with action `search`, `open_page` or `find_in_page`, and `url_citation` annotations. [V-docs: https://developers.openai.com/api/docs/guides/deep-research]
- Codex `web_search` modes: `"cached"` (default), `"live"`, `"disabled"`, `"indexed"`. [V-docs: https://learn.chatgpt.com/docs/agent-approvals-security]

### Inferences
- Tool discovery in these products is mostly static declaration per run, not dynamic search. The "tool search" hints in the compaction guide suggest a deferred-tool mechanism exists, but I did not verify it.

### Gaps
- Tool discovery/deferral mechanics are not documented on any page I read.
- The Responses MCP approval item types (`mcp_approval_request` / `mcp_approval_response`) were not described on the MCP page I read. The SDK documents its own approval items (`ToolApprovalItem`), which is a different mechanism.

## Key Question 4: Long-running and background execution, resumption

### Takeaway
Background mode is a documented Responses API feature: create with `background: true`, poll or receive webhooks, cancel by ID, and resume a stream by sequence number. It temporarily stores request content even when `store` is false or ZDR applies.

### Cited Findings
- Start with `background: true`; retrieve by ID while status is `queued` or `in_progress`; cancel with POST `/v1/responses/{id}/cancel` (idempotent). [V-docs: https://developers.openai.com/api/docs/guides/background]
- Streaming is available only with the `soonest` completion window. For `2h` and `8h` windows, omit `stream` or set it to false. If a connection drops, resume with the last event's `sequence_number` as `starting_after`. [V-docs: https://developers.openai.com/api/docs/guides/background]
- Webhooks: subscribe to `response.completed` plus failed, incomplete or canceled events. Verify signatures, return 2xx promptly, deduplicate via the `webhook-id` header, retrieve using `event.data.id`. Polling is the documented alternative. [V-docs: https://developers.openai.com/api/docs/guides/background]
- Discounted completion windows: `2h` (30% below Standard) and `8h` (40% below Standard) via `background_options.completion_window`; these require `service_tier` `default`. [V-docs: https://developers.openai.com/api/docs/guides/background]
- Retention, quoted from the page: "Background mode temporarily stores response data even when you set `store` to `false` or use Zero Data Retention." Request content is encrypted at rest for all windows. [V-docs: https://developers.openai.com/api/docs/guides/background]
- Multi-step: `background_options.continuation_id` gives higher queue priority on follow-ups. If a prior response has expired, replay the original input and every saved output item in order, including encrypted reasoning items, then append `function_call_output`. [V-docs: https://developers.openai.com/api/docs/guides/background]
- Approval does not guarantee exactly-once execution; use idempotency keys for consequential tools and reconcile after a crash before retrying. [V-docs: https://openai.github.io/openai-agents-python/human_in_the_loop/]
- The Deep Research developer guide recommends background mode for "a request that might take several minutes." [V-docs: https://developers.openai.com/api/docs/guides/deep-research]
- Agents SDK run state is serialisable: `state.to_json()` / `RunState.from_json(...)`, and pending approvals resume with `Runner.run(agent, state)`. [V-docs: https://openai.github.io/openai-agents-python/human_in_the_loop/]

### Inferences
- The "ZDR-safe" assumption does not hold for background mode. Any Prophet workflow that moves client-derived data through background mode should be treated as temporarily stored at OpenAI, even with `store=false`.

### Gaps
- Webhook setup and event names are as summarised from the guide, not checked against an API reference.
- Maximum background job duration is not stated on the pages I read.

## Key Question 5: State and context management

### Takeaway
Three documented state modes exist: server-side chaining with `previous_response_id`, a durable Conversations object, and client-side manual input chaining. Responses are stored 30 days by default. Compaction is available server-side (threshold-triggered) or via a stateless endpoint, and produces an opaque encrypted item.

### Cited Findings
- `previous_response_id` chains responses; previous input tokens in the chain are still billed as input. [V-docs: https://developers.openai.com/api/docs/guides/conversation-state]
- Conversations: a durable object with its own ID, reusable across sessions, devices or jobs, storing messages, tool calls and tool outputs. [V-docs: https://developers.openai.com/api/docs/guides/conversation-state]
- "Response objects are saved for 30 days by default." Setting `store` to false disables this. "Conversation objects and items in them are not subject to the 30 day TTL." [V-docs: https://developers.openai.com/api/docs/guides/conversation-state]
- For reasoning models, preserve every item in the response's `output` array (reasoning items and `phase` values). `reasoning.context: "all_turns"` exists for models that support it. [V-docs: https://developers.openai.com/api/docs/guides/conversation-state]
- Server-side compaction: `context_management` with `compact_threshold` on `/responses`; "When the rendered token count crosses the configured threshold, the server runs server-side compaction." The stream includes an encrypted `compaction` item. [V-docs: https://developers.openai.com/api/docs/guides/compaction]
- Standalone `/responses/compact` is "fully stateless and ZDR-friendly." It returns a window that should be passed back "as-is"; it "generally contains more than just the compaction item." Input must fit the context window. [V-docs: https://developers.openai.com/api/docs/guides/compaction]
- Explicit trigger: append `{"type": "compaction_trigger"}` as the last input item. [V-docs: https://developers.openai.com/api/docs/guides/compaction]
- Compaction limits: `/v1/responses/compact` rejects `configuration_update` items; tools added with `additional_tools` are not restored by the standalone endpoint; hosted MCP servers need credentials supplied again. [V-docs: https://developers.openai.com/api/docs/guides/compaction]
- Codex instruction files count toward context: `project_doc_max_bytes` defaults to 32 KiB, and Codex "stops adding files once the combined size reaches the limit." [V-docs: https://learn.chatgpt.com/docs/agent-configuration/agents-md]
- SDK sessions cannot be combined with `conversation_id`, `previous_response_id` or `auto_previous_response_id` in one run. [V-docs: https://openai.github.io/openai-agents-python/running_agents/]

### Inferences
- [H] Compaction in Codex is not documented on the pages I read. The guide's `remote_compaction` capability hint (from an earlier search snippet) is unverified.

### Gaps
- Codex's own compaction and session resume: not documented on the pages I could read (the AGENTS.md page explicitly does not cover them).
- Compaction summary contents are opaque by design, so their quality cannot be checked from public evidence.

## Key Question 6: Code execution sandboxes

### Takeaway
Code Interpreter runs in a container with a memory tier and a 20-minute idle expiry. Expired containers are not recoverable. Generated files come back as citations and must be downloaded. Persistence is the caller's job.

### Cited Findings
- Container modes: auto (creates or reuses an active container) and explicit (created via `v1/containers`, then passed by ID). [V-docs: https://developers.openai.com/api/docs/guides/tools-code-interpreter]
- Memory tiers: `1g` (default), `4g`, `16g`, `64g`; "The selected `memory_limit` applies for the entire life of that container." Higher tiers bill at the built-in tools rate. [V-docs: https://developers.openai.com/api/docs/guides/tools-code-interpreter]
- "A container expires after 20 minutes without use. Once expired, it cannot be reactivated, and its data is discarded and unrecoverable." Any container operation refreshes the timer. [V-docs: https://developers.openai.com/api/docs/guides/tools-code-interpreter]
- Generated files are cited as `container_file_citation` annotations in the next assistant message and downloaded with the get-container-file-content method. [V-docs: https://developers.openai.com/api/docs/guides/tools-code-interpreter]
- Rate limit: 100 RPM per organisation. [V-docs: https://developers.openai.com/api/docs/guides/tools-code-interpreter]
- Codex sandboxing: `read-only`, `workspace-write` (default in the Auto preset), `danger-full-access`. Writes to `.git`, `.agents`, `.codex` and `.aws` stay read-only in workspace-write. Network is off by default in workspace-write. macOS uses Seatbelt (`sandbox-exec`), Linux uses `bwrap` and `seccomp`. [V-docs: https://learn.chatgpt.com/docs/agent-approvals-security]

### Inferences
- Code Interpreter is a stateful but ephemeral scratch area, not a store. Prophet should copy outputs into its own DuckDB or file store immediately.

### Gaps
- Available Python libraries and internet access inside the container are not stated on the page I read.
- The billing unit for Code Interpreter is not stated on the page (only "built-in tools rates").

## Key Question 7: Artefact and file delivery

### Takeaway
Outputs leave the sandbox as downloadable files referenced by citation annotations. There is no documented charts-as-first-class-artefact pipeline in the pages I could read; charts are generated as files by the code tool.

### Cited Findings
- Supported output formats listed for Code Interpreter include CSV, PDF, XLSX, DOCX, PPTX, JSON, common image types and several code file types. [V-docs: https://developers.openai.com/api/docs/guides/tools-code-interpreter]
- Deep Research citations: inline `url_citation` annotations with URL, title and position; `web_search_call.action.sources` returns every consulted URL. The developer page requires displayed citations to be "prominent, clickable links." [V-docs: https://developers.openai.com/api/docs/guides/deep-research]

### Inferences
- [H] Delivery is file-plus-citation, not a rendered artefact pane. Prophet's console/API serving model is compatible with this pattern.

### Gaps
- Canvas (ChatGPT) internals and its document-versioning mechanism are not covered by any page I could read. No canvas evidence was gathered.

## Key Question 8: Handoffs and multi-agent orchestration

### Takeaway
The SDK's multi-agent model is a handoff (control transfer, with optional history filtering) and Agent-as-tool (nested run). Approvals raised inside nested runs surface on the outer run.

### Cited Findings
- Handoff options: `agent`, `tool_name_override`, `tool_description_override`, `on_handoff`, `input_type`, `input_filter`, `is_enabled`, `nest_handoff_history` (beta, default off). [V-docs: https://openai.github.io/openai-agents-python/handoffs/]
- Server-managed conversations (`conversation_id`, `previous_response_id`, `auto_previous_response_id`) do not support handoff input filters. Nested history does not redact sensitive data, so tool arguments and outputs can remain in summaries. [V-docs: https://openai.github.io/openai-agents-python/handoffs/]
- Approvals arising after a handoff or inside `Agent.as_tool()` "still surface on the outer run," so they are resolved on the top-level run. [V-docs: https://openai.github.io/openai-agents-python/human_in_the_loop/]
- Input guardrails apply only to the first agent in the chain; output guardrails only to the agent producing the final output. Handoffs stay within a single run. [V-docs: https://openai.github.io/openai-agents-python/handoffs/]

### Inferences
- [H] Multi-agent orchestration is intentionally shallow: one run, one top-level agent, handoffs as tools. Prophet's agent set (shortfall, freshness, customer, margin) maps better to parallel pure functions plus a governor than to handoffs.

### Gaps
- The Agents SDK repository source was not inspected, so the handoff implementation details are from docs only.

## Key Question 9: Guardrails, human-in-the-loop, and tracing

### Takeaway
Guardrails are typed checks attached to agents or tools, with tripwire exceptions. Human approval is a per-tool `needs_approval` rule that pauses the run and resumes from serialised state. Tracing is built into runs and configurable per run.

### Cited Findings
- Three guardrail types: input (first agent only), output (final agent only), tool (wraps guarded function tools, input before execution, output after). Each raises a specific tripwire exception. [V-docs: https://openai.github.io/openai-agents-python/guardrails/]
- Input guardrails run in parallel by default; `run_in_parallel=False` runs them to completion first so the agent "never runs, so no tokens are spent and no tools execute." The page recommends blocking mode for cost control and to avoid side effects. [V-docs: https://openai.github.io/openai-agents-python/guardrails/]
- Rejected output is not persisted to the session; a rejected terminal tool output is replaced with "Output withheld by an output guardrail." (customisable via `RunConfig.output_guardrail_blocked_message`). [V-docs: https://openai.github.io/openai-agents-python/guardrails/]
- `needs_approval` on `function_tool`, `Agent.as_tool()`, `ShellTool`, `ApplyPatchTool` takes `True` or an async callable `(ctx, params, call_id)`. Callable rules "fail closed": malformed or missing arguments force manual approval. [V-docs: https://openai.github.io/openai-agents-python/human_in_the_loop/]
- Flow: the run pauses; `result.interruptions` returns `ToolApprovalItem`s; `result.to_state()`; `state.approve()` or `state.reject()`; resume with `Runner.run(agent, state)`. [V-docs: https://openai.github.io/openai-agents-python/human_in_the_loop/]
- `always_approve` decisions persist for the same tool. Function-tool decisions are scoped to the requesting agent; shell and apply_patch decisions are not agent-scoped. [V-docs: https://openai.github.io/openai-agents-python/human_in_the_loop/]
- Security cautions from the same page: deserialising a snapshot does not authenticate it; keep snapshots in server-controlled storage, authorise reviewers, validate decisions against server-held pending items; version pending tasks so resumes match the agent definitions they were paused under. [V-docs: https://openai.github.io/openai-agents-python/human_in_the_loop/]
- Codex: `approvals_reviewer` can be `"user"` (default) or `"auto_review"`, which routes eligible approval requests to a reviewer agent. [V-docs: https://learn.chatgpt.com/docs/agent-approvals-security]
- Tracing: `RunConfig` has `tracing_disabled`, `tracing` (TracingConfig), `trace_include_sensitive_data`, `workflow_name`, `trace_id`, `group_id`, `trace_metadata`. [V-docs: https://openai.github.io/openai-agents-python/running_agents/]
- Tool guardrails: `RunConfig.tool_execution` with `pre_approval_tool_input_guardrails=True` runs input checks before an approval interruption; calls are rechecked after approval. [V-docs: https://openai.github.io/openai-agents-python/guardrails/]

### Inferences
- [H] Approval is a pause-and-resume over serialised run state, not a blocking prompt. That suits Prophet's overnight or batch review pattern.

### Gaps
- The dedicated tracing page was not fetched; only RunConfig tracing options are confirmed.
- Whether trace data is retained or where it is exported beyond the default is not confirmed.

## Key Question 10: Codex long-horizon autonomy and instruction files

### Takeaway
Codex combines a sandbox (what it can touch) with an approval policy (when it must ask), reads AGENTS.md instructions from global and project scope, and can route approvals to a reviewer agent. Plans, resume and compaction are not documented on the pages I could read.

### Cited Findings
- Sandbox modes: `read-only`, `workspace-write`, `danger-full-access`. Approval policies: `on-request`, `never`, a `granular` table; `untrusted` is retired. [V-docs: https://learn.chatgpt.com/docs/agent-approvals-security]
- Combinations: Auto = `workspace-write` + `on-request`; read-only browsing = `read-only` + `on-request`; read-only CI = `read-only` + `never`. `codex exec --sandbox workspace-write` is recommended for non-interactive runs. [V-docs: https://learn.chatgpt.com/docs/agent-approvals-security]
- AGENTS.md discovery: global from `CODEX_HOME` (default `~/.codex`), `AGENTS.override.md` before `AGENTS.md`; project scope walks from project root (Git root) down to cwd; "at most one file per directory"; files concatenated root-first so closer files "override earlier guidance." [V-docs: https://learn.chatgpt.com/docs/agent-configuration/agents-md]
- Chain is built "once per run," so edits apply on restart. Loaded files can be audited via `codex-tui.log` or `session-*.jsonl` when session logging is on. [V-docs: https://learn.chatgpt.com/docs/agent-configuration/agents-md]
- Codex docs have moved: `developers.openai.com/codex/...` pages now 308-redirect to `learn.chatgpt.com/docs/...`. [V-docs: https://developers.openai.com/codex/agent-approvals-security.md redirects to https://learn.chatgpt.com/docs/agent-approvals-security.md]

### Inferences
- [H] The sandbox/approval split is a useful design: separate "what can execute" from "when a human is asked." Prophet's autonomy ladder (capped at "prepares") is a coarser version of the same idea.

### Gaps
- Codex plans, session resume, and compaction behaviour: not documented on the pages I read.
- The `openai/codex` repository was not inspected, so no [V-source] claims are made.
- Codex cloud/background execution: not researched in this round.

## Key Question 11: Anthropic comparison

### Takeaway
Not researched in this round. Another researcher covers Claude. The only comparison points here are where OpenAI's docs mention Anthropic-equivalent concepts; none were verified.

### Gaps
- No Anthropic evidence gathered by this researcher.

## Key Question 12: Table and Prophet design ideas

### Mechanism table

| Mechanism | Evidence tag | What it does | Prophet adaptation |
|---|---|---|---|
| Loop stop: no tool calls plus typed output; `max_turns` raises error | [V-docs: openai-agents-python running_agents] | Deterministic completion and hard cap | Set explicit turn/time caps on any agent-driven step; never rely on model "done" |
| Background mode with polling, webhooks and cancel | [V-docs: developers.openai.com background guide] | Long jobs run server-side; resume via `sequence_number` | Use for multi-minute forecasts or Deep Research-style jobs; keep the existing file-per-job pattern and add cancel |
| Background data retention | [V-docs: background guide] | Temporary storage even with `store=false` or ZDR | Do not assume ZDR covers background jobs; keep client-derived inputs out of background mode unless approved |
| Stored responses, 30-day default TTL; Conversations not subject to TTL | [V-docs: conversation-state guide] | Server-side state with retention clock | Prefer own DB as system of record; treat OpenAI-side state as a cache |
| Server compaction (`compact_threshold`) and standalone `/responses/compact` | [V-docs: compaction guide] | Opaque encrypted carry-forward state | Only if a long conversational agent is added; keep raw history in own store so compaction can be audited or replaced |
| Code Interpreter 20-minute idle expiry, unrecoverable | [V-docs: tools-code-interpreter] | Ephemeral container | Copy every artefact to DuckDB or file store immediately; never keep state only in a container |
| Container memory tiers 1g to 64g | [V-docs: tools-code-interpreter] | Fixed memory for container life | Size up front for large twin/replay work |
| Handoff as tool, `input_filter`, nested history beta | [V-docs: openai-agents-python handoffs] | Control transfer with history control | Prefer agent-as-pure-function plus governor (current design) over handoffs |
| Input guardrail blocking mode (`run_in_parallel=False`) | [V-docs: openai-agents-python guardrails] | Stops before tokens or tools spent | Run data-validity and no-future-leak checks before TimesFM is called |
| Tool guardrails do not cover hosted tools or handoffs | [V-docs: guardrails] | Coverage gap | Do not assume a guard covers every path; test each path |
| `needs_approval` with pause and `RunState` resume | [V-docs: openai-agents-python human_in_the_loop] | Serialised pause; resume after hours | Model proposals as pending items; resume from server-held state |
| Fail-closed approval callables | [V-docs: human_in_the_loop] | Malformed args force manual review | Default to human review when a proposal's evidence is incomplete |
| Always-approve scoped per agent (shell and patch not scoped) | [V-docs: human_in_the_loop] | Persistent approval rules, scope varies by tool | Scope any standing approval to one agent and one tool; never global |
| Idempotency and no exactly-once guarantee | [V-docs: human_in_the_loop] | Retries can double-act | Idempotency keys on any write path; reconcile after crash |
| Tracing via RunConfig (`workflow_name`, `trace_metadata`) | [V-docs: running_agents] | Per-run trace with sensitive-data toggle | Set `trace_include_sensitive_data=False` for client data; tag runs with `workflow_name` |
| Codex sandbox (`read-only` / `workspace-write`, network off by default) | [V-docs: learn.chatgpt.com agent-approvals-security] | Separates execution rights from approval | Mirror the split: read-only lab access to FoodFlow, no network from agent workers |
| Codex `approvals_reviewer: auto_review` | [V-docs: learn.chatgpt.com agent-approvals-security] | Reviewer agent handles routine approvals | Possibly useful for low-value alerts; keep humans on order-affecting items |
| AGENTS.md chain (global, root to cwd, one per dir, 32 KiB cap) | [V-docs: learn.chatgpt.com agents-md] | Nested, scoped instructions with size cap | Already aligned with CLAUDE.md; keep per-directory instructions and watch size |
| Deep Research: clarify, editable plan, interrupt | [V-docs: help.openai.com, snippet only] | Plan review before long run | Show the planned analysis scope before a long backtest; allow edit |
| Operator/ChatGPT agent: confirm consequential actions | [V-docs: Operator card, snippet only; secondary for ChatGPT agent] | Confirmation before side effects | Matches Prophet's "prepares only" rule; extend to any future write path |
| MCP: `allowed_tools`, `require_approval`, third-party servers unverified | [V-docs: developers.openai.com/api/docs/mcp] | Narrow tool exposure, approval for writes | Expose only read tools to agents; approval on any write tool; treat server output as untrusted |

### Design ideas worth stealing
1. Split execution rights (sandbox) from approval policy (when to ask). Use two independent config axes for agent workers.
2. Pause-and-resume approval over serialised run state, with server-held pending items, rather than in-process prompts. Matches Prophet's proposals/decisions store.
3. Input guardrails in blocking mode, so cheap validity checks run before expensive model calls.
4. Fail-closed approval rules: missing or malformed evidence forces human review.
5. Background jobs with explicit cancel, idempotent cancel, and webhook-or-poll completion.
6. Ephemeral-container discipline: treat any sandbox as scratch and copy outputs out at once.
7. Plan-then-run for long research: show scope and let the user edit before compute is spent.
8. Scoped standing approvals: per agent and per tool, never global.

### Ideas to avoid
1. Assuming ZDR or `store=false` covers background mode. The docs say it does not.
2. Relying on a container's state persisting. It expires after 20 minutes idle and cannot be recovered.
3. Treating compaction output as auditable. It is opaque by design; keep raw history separately.
4. Assuming tool guardrails cover handoffs, hosted tools or built-in execution tools.
5. Assuming exactly-once execution after approval. Use idempotency keys.
6. Trusting a deserialised approval snapshot without server-side authentication and validation.
7. Using `untrusted` approval policy (retired in Codex) or unbounded `max_turns=None` by default.
8. Treating third-party MCP servers as verified or safe because the developer is trusted.

### Takeaway
OpenAI's agent platform is built from a small set of primitives: a fixed-tool loop with explicit stop conditions, server-side background jobs with polling or webhooks, serialisable approval pauses, and sandboxed ephemeral execution. Most of the safety design is about separating execution rights from approval and bounding state and retention, which Prophet can borrow directly.

### Inferences
- [H] Prophet's design (pure agents, governor, prepares-only autonomy, file-per-job backgrounds) already matches most of the primitives. The main gaps to close are retention assumptions for background mode and explicit idempotency on any future write path.

### Gaps
- Deep Research and ChatGPT agent primary sources were not readable (403). Their behaviour rests on help-centre snippets and secondary coverage.
- Operator system card text was not readable; the confirmation claim rests on a search snippet.
- Codex agent loop, plans, resume, compaction, cloud execution: not documented on any page read.
- Open-source repos (`openai/codex`, `openai/openai-agents-python` source) were not inspected; all SDK claims are from docs pages.
- Canvas internals and Code Interpreter library and internet access: not covered.
- Tracing destinations and retention: not confirmed.
