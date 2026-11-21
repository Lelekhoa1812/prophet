# Agent platforms: how Claude and ChatGPT handle long-running agent work

Status: research note, not a design decision. Date of research: 2026-11-21.
Purpose: map public evidence on the Claude and ChatGPT agent platforms against Prophet's current design, and list concrete gaps with minimal fits. No implementation code is proposed.

## Scope and method

Scope:
- Claude: Claude.ai (public product pages only), Claude Code, Claude Agent SDK, Claude Managed Agents, Anthropic API tool use, extended and adaptive thinking, context editing, compaction, prompt caching, memory tool.
- ChatGPT: ChatGPT agent and Operator (launch posts), OpenAI Responses API, Agents SDK, Codex (CLI and cloud).
- Eight capabilities: (1) complex task reasoning, (2) dynamic planning and tool use, (3) multi-stage coordination and sub-agents, (4) long-running interactions, (5) context and session management, (6) confirmation UX and asking the user, (7) execution sandboxes and code tools, (8) recovery.

Method:
- Sources were located with WebSearch and read with WebFetch. Only public official docs, official blog posts and API reference pages were used.
- Private system prompts, hidden instructions and non-public internals were not sought and are not described.
- The three public GitHub repositories named in the brief (`anthropics/claude-code`, `openai/codex`, `openai/openai-agents-python`) were fetched, but the fetch returned only repository landing metadata. No claims are drawn from repository code, so they are not counted as sources.
- Prophet files were read through graphify queries first, then targeted Grep and Read. Prophet claims were checked against the code, not only against notes.
- No live product testing was done. Anything about behaviour that is not in a document is marked LOW.
- "Date seen" is 2026-11-21 for every source. Post dates are included only where the post itself states one. Web fetches return extracted text, so quoted wording is close paraphrase unless shown in code formatting.
- Figures from vendors (accuracy gains, token multipliers, sandbox prompt reductions) are self-reported and are labelled as such. They are not independent measurements.

Confidence labels:
- HIGH: an official document or post states the claim.
- MEDIUM: an official source implies the claim, or a claim follows from two official statements.
- LOW: inference from described behaviour or a product blog's UI description. No official mechanism is stated.

Counts: 36 distinct source URLs are cited in the evidence table. 62 findings are listed: 56 HIGH, 3 MEDIUM, 3 LOW.

## Evidence table

| ID | Claim | Source URL | Date seen | Confidence |
|---|---|---|---|---|
| E01 | Manual extended thinking sets `budget_tokens` (minimum 1,024) as a target; `max_tokens` is the hard ceiling. | https://platform.claude.com/docs/en/build-with-claude/extended-thinking | 2026-11-21 | HIGH |
| E02 | Manual thinking mode is deprecated on the newest models; adaptive thinking with an effort setting is the documented path there. | https://platform.claude.com/docs/en/build-with-claude/extended-thinking | 2026-11-21 | HIGH |
| E03 | Interleaved thinking lets the model reason between tool calls inside one assistant turn; availability depends on model and mode. | https://platform.claude.com/docs/en/build-with-claude/extended-thinking | 2026-11-21 | HIGH |
| E04 | Changing the thinking budget invalidates message-level prompt cache entries. | https://platform.claude.com/docs/en/build-with-claude/extended-thinking | 2026-11-21 | HIGH |
| E05 | Anthropic's multi-agent research system (lead plus parallel subagents) beat a single agent on its internal research eval. The 90.2% figure is self-reported. | https://www.anthropic.com/engineering/multi-agent-research-system (post dated 2025-06-13) | 2026-11-21 | HIGH |
| E06 | The same post reports agents use about 4x the tokens of chat, and multi-agent systems about 15x. Self-reported. | https://www.anthropic.com/engineering/multi-agent-research-system | 2026-11-21 | HIGH |
| E07 | Lead agent uses extended thinking to plan; subagents use interleaved thinking after tool results. | https://www.anthropic.com/engineering/multi-agent-research-system | 2026-11-21 | HIGH |
| E08 | Subagents write results to external storage and return references. The post lists synchronous batches (lead waits for each batch) as a limitation. | https://www.anthropic.com/engineering/multi-agent-research-system | 2026-11-21 | HIGH |
| E09 | The post says most coding tasks have fewer truly parallel parts than research, so multi-agent designs fit coding less well. The "less well" is the post's implication, not a measured result. | https://www.anthropic.com/engineering/multi-agent-research-system | 2026-11-21 | MEDIUM |
| E10 | An agent is a model using tools in a loop and gaining ground truth at each step. Context is a finite resource that must be curated. | https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents (post dated 2025-09-29) | 2026-11-21 | HIGH |
| E11 | Advanced tool use: Tool Search with deferred loading, programmatic tool calling, and tool-use examples. Accuracy gains are self-reported. | https://www.anthropic.com/engineering/advanced-tool-use | 2026-11-21 | HIGH |
| E12 | Agents SDK handoffs are exposed to the model as tools named `transfer_to_<agent>`. Nested handoff history (`nest_handoff_history`) is a beta option. | https://openai.github.io/openai-agents-python/handoffs/ | 2026-11-21 | HIGH |
| E13 | The Agents SDK runner loops model call, tool call and model call until a final output or `max_turns`, which raises `MaxTurnsExceeded`. | https://openai.github.io/openai-agents-python/running_agents/ | 2026-11-21 | HIGH |
| E14 | Codex subagents run in parallel, the parent waits for their results, and they use more tokens. The docs advise caution with parallel write-heavy work. | https://developers.openai.com/codex/subagents | 2026-11-21 | HIGH |
| E15 | ChatGPT agent has a virtual computer with a visual browser, text browser, terminal, API access and connectors. The user can interrupt, clarify and take over (post dated 2025-07-17). | https://openai.com/index/introducing-chatgpt-agent/ | 2026-11-21 | HIGH |
| E16 | Claude Code subagents have their own context window, system prompt, tool restrictions and permissions. Only a summary returns to the parent. | https://code.claude.com/docs/en/sub-agents | 2026-11-21 | HIGH |
| E17 | Claude Code subagents can pin a model (for example a cheaper one) in their definition. | https://code.claude.com/docs/en/sub-agents | 2026-11-21 | HIGH |
| E18 | The Claude Agent SDK loop has `max_turns` and `max_budget_usd` stop conditions and returns `error_max_turns` and `error_max_budget_usd`. The budget covers subagent cost. | https://code.claude.com/docs/en/agent-sdk/agent-loop | 2026-11-21 | HIGH |
| E19 | Claude Managed Agents: a session combines an agent (model, system prompt, tools, MCP servers, skills), an environment and an event stream. History and sandbox state persist; an idle sandbox is checkpointed and files are kept 30 days. | https://platform.claude.com/docs/en/managed-agents/sessions | 2026-11-21 | HIGH |
| E20 | A Managed Agents `user.define_outcome` event takes a rubric. Only one outcome is allowed per session. | https://platform.claude.com/docs/en/managed-agents/sessions | 2026-11-21 | HIGH |
| E21 | Managed Agents accept steering or interruption as events while a run is in progress. | https://platform.claude.com/docs/en/managed-agents/sessions | 2026-11-21 | HIGH |
| E22 | A Claude Code session starts with a fresh context window. `CLAUDE.md` and auto memory carry knowledge across sessions. | https://code.claude.com/docs/en/memory | 2026-11-21 | HIGH |
| E23 | Anthropic's long-running harness uses an initializer agent and a coding agent, a JSON feature list with pass flags, a progress file, git commits and an init script. One feature is worked per session. | https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents | 2026-11-21 | HIGH |
| E24 | The same post reports two failure modes: declaring the task finished too early, and trying to do everything in one attempt. | https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents | 2026-11-21 | HIGH |
| E25 | OpenAI background mode runs long responses asynchronously. Clients poll by response id or use webhooks, and can cancel. | https://developers.openai.com/api/docs/guides/background | 2026-11-21 | HIGH |
| E26 | Operator (post dated 2025-01-23) has a takeover mode for sensitive input, asks for confirmation before significant actions, and has a watch mode. | https://openai.com/index/introducing-operator/ | 2026-11-21 | HIGH |
| E27 | ChatGPT agent asks for confirmation before consequential actions. Takeover does not retain entered information. The user can stop the task. | https://openai.com/index/introducing-chatgpt-agent/ | 2026-11-21 | HIGH |
| E28 | Agent SDK permission order: hooks, then deny rules, ask rules, permission mode, allow rules, and finally `canUseTool`. `AskUserQuestion` always goes to `canUseTool`. The `dontAsk` mode denies it. | https://code.claude.com/docs/en/agent-sdk/permissions | 2026-11-21 | HIGH |
| E29 | `canUseTool` can stay pending indefinitely. A `PreToolUse` hook can defer the decision so the process exits and resumes later from the saved session. | https://code.claude.com/docs/en/agent-sdk/user-input | 2026-11-21 | HIGH |
| E30 | Managed Agents emit `requires_action` for a tool that needs confirmation. The client replies with `user.tool_confirmation` (allow or deny, with an optional message). | https://platform.claude.com/docs/en/managed-agents/sessions | 2026-11-21 | HIGH |
| E31 | Agents SDK human-in-the-loop: `needs_approval` per tool; a run returns interruptions; `RunState` can be serialised; the caller approves or rejects; the run resumes. | https://openai.github.io/openai-agents-python/human_in_the_loop/ | 2026-11-21 | HIGH |
| E32 | Approval permits a call but does not guarantee exactly-once execution. Tools with side effects need a stable idempotency key at the destination. Unknown outcomes should be reconciled and recorded, not replayed automatically. | https://openai.github.io/openai-agents-python/human_in_the_loop/ | 2026-11-21 | HIGH |
| E33 | Codex has sandbox modes (`read-only`, `workspace-write`, full access) and approval policies (`on-request`, `untrusted`, `never`). Network is off by default. Side-effecting MCP and connector tools need approval. | https://developers.openai.com/codex/agent-approvals-security | 2026-11-21 | HIGH |
| E34 | Codex cloud: setup runs with network, the agent phase runs offline by default, and secrets are removed before the agent phase. | https://developers.openai.com/codex/agent-approvals-security | 2026-11-21 | HIGH |
| E35 | Codex safety monitoring can pause a task asynchronously. The task should be resumed only after the pause has been checked. | https://developers.openai.com/codex/agent-approvals-security | 2026-11-21 | HIGH |
| E36 | Claude Code permission modes include `default`, `acceptEdits`, `plan`, `auto` (classifier), `dontAsk` and `bypassPermissions`. Hooks can block tool calls. | https://code.claude.com/docs/en/permissions | 2026-11-21 | HIGH |
| E37 | Claude Code sandboxing uses bubblewrap on Linux and Seatbelt on macOS, with filesystem isolation and a network proxy with a domain allow-list. Sandboxing cut permission prompts by 84% (internal figure). | https://www.anthropic.com/engineering/claude-code-sandboxing (post dated 2025-10-20) | 2026-11-21 | HIGH |
| E38 | Managed Agents give each session a cloud container. The isolation level is not described in the pages read. | https://platform.claude.com/docs/en/managed-agents/overview | 2026-11-21 | LOW |
| E39 | Agents SDK sandbox agents: a `Manifest`, snapshots, resume from saved sandbox state, and optional agent memory. The local Unix client has no OS-level isolation on Linux; Docker or a hosted sandbox is recommended for untrusted work. | https://openai.github.io/openai-agents-python/sandbox_agents/ | 2026-11-21 | HIGH |
| E40 | ChatGPT agent's terminal has limited network access. | https://openai.com/index/introducing-chatgpt-agent/ | 2026-11-21 | HIGH |
| E41 | Claude context editing: `clear_tool_uses_20250919` replaces the oldest tool results with a placeholder once a trigger is hit; `clear_thinking_20251015` manages old thinking blocks. Clearing invalidates the cache. | https://platform.claude.com/docs/en/build-with-claude/context-editing | 2026-11-21 | HIGH |
| E42 | Claude server-side compaction (beta, `compact-2026-09-04`) summarises older turns at a threshold, or on demand. The SDK can also compact client-side. | https://platform.claude.com/docs/en/build-with-claude/compaction | 2026-11-21 | HIGH |
| E43 | Agent SDK automatic compaction emits `compact_boundary`. A `PreCompact` hook runs first. `CLAUDE.md` is re-injected on every request. | https://code.claude.com/docs/en/agent-sdk/agent-loop | 2026-11-21 | HIGH |
| E44 | Claude Code checkpointing snapshots files before each edit, keeps the 100 most recent, and retains them for about 30 days. Bash-side changes and external changes are not tracked. | https://code.claude.com/docs/en/checkpointing | 2026-11-21 | HIGH |
| E45 | `CLAUDE.md` is loaded as context and is advisory, not enforced. Enforcement uses settings and hooks. | https://code.claude.com/docs/en/permissions | 2026-11-21 | HIGH |
| E46 | Claude memory tool: a client-side `/memories` directory. The application runs the commands and must validate paths. | https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool | 2026-11-21 | HIGH |
| E47 | Claude prompt caching: 5-minute default TTL, refreshed on hit; 1-hour TTL is an option; a 20-block lookback; breakpoints placed at stable prefixes; per-model minimum length. | https://platform.claude.com/docs/en/build-with-claude/prompt-caching | 2026-11-21 | HIGH |
| E48 | OpenAI prompt caching is automatic for prefixes at 1,024 tokens or more. Default retention is 30 minutes with a 12-hour option at extra cost. Breakpoints are implicit or explicit depending on the model. | https://developers.openai.com/api/docs/guides/prompt-caching | 2026-11-21 | HIGH |
| E49 | OpenAI server-side compaction is set by `context_management` with `compact_threshold`. A standalone `/responses/compact` endpoint also exists. The output is an opaque `compaction` item. | https://developers.openai.com/api/docs/guides/compaction | 2026-11-21 | HIGH |
| E50 | Because the compaction item is encrypted, the client cannot read the summary it carries. This follows from the stated encryption, not from a stated guarantee. | https://developers.openai.com/api/docs/guides/compaction | 2026-11-21 | MEDIUM |
| E51 | OpenAI conversation state has three options: manual input chaining, `previous_response_id`, and the Conversations API. Stored responses have a 30-day retention unless `store=false`. | https://developers.openai.com/api/docs/guides/conversation-state | 2026-11-21 | HIGH |
| E52 | Agents SDK `SQLiteSession` stores run items, and an interrupted run can resume in the same session. | https://openai.github.io/openai-agents-python/sessions/ | 2026-11-21 | HIGH |
| E53 | Claude Agent SDK sessions persist the conversation, not the filesystem. `fork` branches the conversation, not the files. | https://code.claude.com/docs/en/agent-sdk/sessions | 2026-11-21 | HIGH |
| E54 | Managed Agents retry transient errors automatically and expose session states including idle, running, rescheduling and terminated. | https://platform.claude.com/docs/en/managed-agents/sessions | 2026-11-21 | HIGH |
| E55 | Anthropic's multi-agent post describes regular checkpoints, resume from the point of error, deterministic retry safeguards, and telling agents when a tool is failing. | https://www.anthropic.com/engineering/multi-agent-research-system | 2026-11-21 | HIGH |
| E56 | Codex CLI creates git checkpoints before and after a task so changes can be reverted. `codex resume` reopens a prior session. | https://developers.openai.com/codex/cli/features | 2026-11-21 | HIGH |
| E57 | Codex discovers `AGENTS.md` from the global config and from the project root down to the working directory. The default size cap is 32 KiB. | https://developers.openai.com/codex/guides/agents-md | 2026-11-21 | HIGH |
| E58 | Agent SDK hooks include `Stop`, `SubagentStop`, `PreCompact`, `PreToolUse` and `PostToolUse`. A hook that times out counts as no decision. | https://code.claude.com/docs/en/agent-sdk/hooks | 2026-11-21 | HIGH |
| E59 | Codex subagents inherit the parent's model and reasoning effort unless their config overrides them. | https://developers.openai.com/codex/subagents | 2026-11-21 | HIGH |
| E60 | ChatGPT agent's watch mode monitors the page while the user is away. The post does not describe how the session resumes after a pause. | https://openai.com/index/introducing-chatgpt-agent/ | 2026-11-21 | LOW |
| E61 | ChatGPT agent takeover appears to keep the browser state across the handoff to the user. This is implied by the post's description, not stated. | https://openai.com/index/introducing-chatgpt-agent/ | 2026-11-21 | MEDIUM |
| E62 | ChatGPT agent's planning is shown to the user as progress summaries. The planner's internals are not documented. | https://openai.com/index/introducing-chatgpt-agent/ | 2026-11-21 | LOW |

## 1. Complex task reasoning

Claude:
- Reasoning is controlled by thinking settings. Manual mode uses `budget_tokens` as a target (E01). Newer models use adaptive thinking with an effort level, and manual mode is deprecated there (E02).
- Interleaved thinking lets the model reason between tool calls within one turn (E03). Changing the budget invalidates message cache entries (E04).
- The multi-agent post gives a concrete split: the lead plans with extended thinking, and subagents use interleaved thinking after each tool result (E07).
- Anthropic reports that a lead-plus-subagent setup beat a single agent on its internal eval (E05), at a large token cost (E06). These numbers are self-reported.

ChatGPT:
- The public Responses and Agents SDK docs describe the loop and its stop condition (E13). They do not describe a reasoning mechanism beyond what the model API exposes.
- Codex states that subagents use more tokens (E14), and inherit the parent's effort unless configured (E59).
- The ChatGPT agent planner is not documented. Its progress summaries are visible (E62, LOW).

Comparison:
- Both vendors publish budget controls: Claude through thinking budgets and max_budget_usd (E01, E18), OpenAI through max_turns (E13). Neither public source gives a reasoning-quality guarantee.
- Claude publishes more detail on the thinking mechanics; OpenAI publishes more on loop limits.

## 2. Dynamic planning and tool use

Claude:
- The core statement is that an agent is a model using tools in a loop, grounded at each step (E10).
- Advanced tool use adds Tool Search with deferred loading, programmatic tool calling and tool-use examples, with self-reported accuracy gains (E11).
- Subagent tool access is restricted per definition (E16).

ChatGPT:
- Handoffs are the planning primitive. They appear to the model as `transfer_to_<agent>` tools (E12).
- Hosted tools in ChatGPT agent cover a virtual computer, browser, terminal, APIs and connectors (E15).

Comparison:
- Claude exposes planning through tool metadata features (deferred loading, examples). OpenAI exposes it through handoff topology.
- Neither public source states how the planner chooses between tools. That is not verifiable from public evidence.

## 3. Multi-stage coordination and sub-agents

Claude:
- Claude Code subagents have isolated context, a system prompt, tool limits, and their own permissions. Only a summary returns to the parent (E16). Subagents can pin a cheaper model (E17).
- The multi-agent post says subagents should write to external storage and pass back references, not full text (E08). It says coordination is synchronous in batches and lists that as a limitation (E08). It says multi-agent designs fit research better than tightly coupled coding (E09).
- The Agent SDK budget covers subagent cost (E18).

ChatGPT:
- Agents SDK coordination uses handoffs, which can nest the history (E12).
- Codex subagents run in parallel and the parent waits for all of them (E14). Codex advises caution with parallel write-heavy work (E14).

Comparison:
- Both use isolated sub-contexts that return results to a parent. Claude's public guidance is more specific about what a subagent should return and how to store large outputs (E08). Codex is more specific about parallel fan-out and its risk (E14).

## 4. Long-running interactions

Claude:
- Managed Agents keep session history and sandbox state, checkpoint an idle sandbox, and retain files for 30 days (E19). Sessions accept steering and interruption events (E21). Transient errors are retried (E54).
- Anthropic's harness for long-running agents uses a feature list with pass flags, a progress file, git commits and an init script. One feature is worked per session (E23). The same post reports two failure modes: declaring the task done too early and trying to do too much in one attempt (E24).
- Claude Code starts each session fresh; continuity comes from `CLAUDE.md` and auto memory (E22).

ChatGPT:
- Background mode runs responses asynchronously. Clients poll by id or use a webhook, and can cancel (E25).
- Operator and ChatGPT agent describe a watch mode that monitors while the user is away (E26, E60). The resume mechanics are not described (E60, LOW).

Comparison:
- Claude publishes a concrete harness pattern (feature list, progress file, git) for long coding runs (E23). OpenAI publishes an API-level mechanism for async responses (E25). Claude's public harness is more directly reusable as a pattern.
- No public source gives a maximum session duration for Managed Agents. Not verifiable from public evidence.

## 5. Context and session management (compaction, memory, prompt caching)

Claude:
- Context editing clears old tool results and old thinking blocks, with a placeholder and a trigger. Clearing invalidates the cache (E41).
- Server-side compaction (beta) summarises older turns at a threshold or on demand (E42). The Agent SDK emits `compact_boundary`, offers a `PreCompact` hook, and re-injects `CLAUDE.md` on every request (E43).
- Memory: `CLAUDE.md` is advisory context (E45). The memory tool is client-side and the application must validate paths (E46).
- Prompt caching: 5-minute default TTL refreshed on hit, optional 1-hour TTL, a 20-block lookback and stable-prefix breakpoints (E47).
- Claude Code checkpoints files before edits (E44). Sessions persist conversation, not files (E53).

ChatGPT:
- Server-side compaction is set with `compact_threshold` and returns an encrypted compaction item (E49). That item is opaque to the client (E50, MEDIUM).
- Session state has three options, and stored responses have 30-day retention unless `store=false` (E51). SQLite sessions in the Agents SDK store run items (E52).
- Prompt caching is automatic from 1,024 tokens. Default retention is 30 minutes, with a 12-hour option (E48).
- Codex reads `AGENTS.md` from several levels, with a 32 KiB cap (E57).

Comparison:
- Claude offers a lighter first step (clear old tool results) before summarising (E41). OpenAI's documented path is compaction through a single threshold (E49).
- Claude's compaction and context-editing settings are documented more fully for the client. OpenAI's compaction output is opaque by design (E50).
- Caching: both are prefix-based. Claude's breakpoints and 20-block lookback (E47) make placement a design concern. OpenAI's automatic prefix caching (E48) reduces that work but gives less control.
- Memory: Claude has file-based memory and a memory tool (E22, E46). The ChatGPT side has no public memory tool in the pages read. Not verifiable from public evidence for the consumer product.

## 6. Confirmation UX and asking the user questions

Claude:
- Agent SDK permission order is documented (E28). `AskUserQuestion` always reaches `canUseTool` and is denied by `dontAsk` (E28).
- `canUseTool` can stay pending. A `PreToolUse` hook can defer, so the process exits and resumes later from the saved session (E29).
- Managed Agents use `requires_action` and `user.tool_confirmation` events (E30).
- Claude Code offers permission modes, including an `auto` classifier mode (E36).

ChatGPT:
- Operator asks for confirmation before significant actions, and has takeover for sensitive input (E26).
- ChatGPT agent asks before consequential actions and does not keep what the user typed in takeover (E27). Browser state appears to persist through takeover (E61, MEDIUM).
- Agents SDK HITL uses `needs_approval` per tool and a serialisable `RunState`, with approve and reject (E31).
- Codex splits sandbox mode from approval policy. Network is off by default and side-effecting connector tools need approval (E33).

Comparison:
- Both vendors separate "ask the user" from "approve this tool call". Claude routes both through one callback and a documented decision order (E28). OpenAI makes approval a per-tool flag with a saved run state (E31).
- Approval does not mean exactly-once execution (E32). Stable idempotency keys at the destination and reconciliation are the documented answer (E32). This is the most important point for Prophet's tool design.

## 7. Execution (sandboxes and code tools)

Claude:
- Claude Code sandboxing uses bubblewrap (Linux) and Seatbelt (macOS), with a filesystem boundary and a network proxy allow-list. The post reports 84% fewer prompts (internal) (E37).
- Managed Agents run each session in a cloud container. The isolation details are not described (E38, LOW).

ChatGPT:
- Agents SDK sandbox agents define a manifest, take snapshots and can resume from saved state (E39). The local Unix client has no OS-level isolation on Linux; Docker or a hosted sandbox is recommended for untrusted work (E39).
- ChatGPT agent's terminal has limited network access (E40).
- Codex cloud runs setup with network and the agent phase offline by default, with secrets removed before the agent phase (E34).

Comparison:
- Both publish a sandbox boundary. Claude documents its OS-level mechanisms (E37). OpenAI documents its sandbox client options and states the gap for the local client (E39).
- Both keep network access narrow by default (E33, E34, E40, E37).

## 8. Recovery (retries, resume, checkpoints)

Claude:
- Anthropic's multi-agent post describes deterministic retries, checkpoints, resume from the point of error, and telling agents about failing tools (E55).
- Agent SDK sessions resume and fork the conversation, but not the files (E53). Claude Code checkpoints files and can rewind, with Bash and external changes untracked (E44).
- Managed Agents retry transient errors (E54). `Stop` and `SubagentStop` hooks run at the end of a run (E58).

ChatGPT:
- Background responses can be polled and cancelled (E25). The Agents SDK resumes an interrupted run in the same SQLite session (E52). Interrupted approval runs resume from serialised state (E31), but E32 says a resumed snapshot may repeat a side effect.
- Codex creates git checkpoints and offers `codex resume` (E56). Codex safety monitoring can pause a task, which should be resumed only after a check (E35).

Comparison:
- Both documents describe resumption at the session level. Only Claude documents file-level rewind for an agent's edits (E44). Only OpenAI documents the exactly-once limit of approval-based resume in this form (E32).
- Anthropic's multi-agent guidance is the most concrete public statement on retries and checkpoints (E55).

## Gaps for Prophet

Prophet baseline, verified in code and notes:
- Goal loop: plan, act, evaluate. The plan action is one of `call_tool`, `ask_user`, `delegate`, `finish`. Delegation is a nested loop with depth 1 and a default of 12 steps (`src/prophet_agent/runtime/goal.py`).
- Questions: `ask()` records an open question and raises `Paused` (`src/prophet_agent/runtime/runner.py`). Identical questions are refused, and questions per mission are capped (`tests/prophet_agent/test_goal_decisions.py`).
- Compaction: `_compact()` folds older observations into a memo once the view exceeds `VIEW_CHARS = 16000`, keeping the last `KEEP_RECENT = 4` (`src/prophet_agent/runtime/goal.py`).
- Transport: NIM responses carry `cached_tokens` when reported (`src/prophet_agent/nim/transport.py`). The usage ledger records prompt and completion totals (`src/prophet_agent/usage.py`).
- Sandbox: child process with `-I`, scrubbed environment, rlimits, audit hook, hard timeout, sha256 manifest (`CLAUDE.md`).
- Idempotency: tools declare whether they are idempotent. A non-idempotent call left as started becomes `outcome_unknown` (`CLAUDE.md`).

Gaps, in order of importance:

G1. No per-tool approval gate.
- Evidence: E28, E30, E31, E32, E33.
- Prophet today: the only human gate is `ask_user`, a planner choice. A tool cannot be declared as needing approval before it runs.
- Minimal fit: add an approval flag to the tool declaration. When the planner picks a flagged tool, the act step records an approval question through the existing `ask` path and runs the tool only after a stored allow. A deny becomes an observation. Record the decision before the call, and keep the existing idempotency key so a resumed approval cannot replay a side effect (E32).

G2. No per-criterion completion checklist.
- Evidence: E20, E23, E24.
- Prophet today: the evaluator judges the whole goal against the evidence. Completion is not tied to separately stored criteria.
- Minimal fit: at mission start, store a short list of checkable criteria with a status and evidence ids. The evaluator may close an item only by citing an observation id. The finish action requires all items closed or explicitly waived by the user. This addresses the early-completion failure mode in E24.

G3. Compaction has no cheaper first step.
- Evidence: E41, E42, E47.
- Prophet today: once the view exceeds 16,000 characters, `_compact()` makes a model call and replaces older observations with a memo. The only lever is the summary, and the threshold is in characters rather than tokens.
- Minimal fit: first replace old large tool outputs with a stub that points to the stored ledger row, keeping the last `KEEP_RECENT` outputs in full. Use the summary only if that is not enough. Clear in batches above a minimum size so small edits do not break the cache prefix (E41, E47). The full output stays in the ledger, so grounding is unaffected.

G4. Cache health is not visible per step.
- Evidence: E04, E41, E47, E48.
- Prophet today: the transport captures `cached_tokens` when the provider reports it, but the usage ledger records only prompt and completion totals. I did not find a test that asserts prefix stability. The tests matching `cache` are in the transport and provider test files, which I did not read in full.
- Minimal fit: store cached input tokens per model call in the usage ledger, and add a test that the fixed prefix (system text, goal, tool schema, memo) is identical across consecutive steps. This is a measurement, not a design change. Whether NIM reports `cached_tokens` for the models Prophet uses is not verified (see below).

G5. The delegation contract is thin and synchronous.
- Evidence: E05, E08, E09, E14, E16, E17, E59.
- Prophet today: delegation is a nested loop with depth 1, a task-scoped tool list and a memo of its own. It is sequential and has no declared output format or per-delegate model setting.
- Minimal fit: extend the delegate action with an objective, an expected output shape, a tool allow-list and a step budget. The sub-agent writes its full result to the run folder and returns a reference plus a short summary (E08). Keep depth 1. Parallel fan-out can wait until read-only delegation is proven, given E09 and E14.

G6. No cross-mission memory.
- Evidence: E22, E45, E46, E57.
- Prophet today: in the code reviewed, memory lives inside one mission (the memo and ledger). I found no user-scoped notes file.
- Minimal fit: a small user-scoped notes file, capped in length, loaded into the brief at mission start. Writes go through one path-restricted tool, and changes happen between missions so they do not disturb the cache prefix within a mission. Treat it as advisory context, not enforcement (E45).

G7. No persisted workspace across goal parts.
- Evidence: E19, E43, E44, E53.
- Prophet today: each sandbox run is a child process with a run folder and a manifest. The reviewed files do not show one workspace reused across goal parts.
- Minimal fit: store the run-folder id on the mission and reuse it on resume, recording the manifest per step. Rewind is lower priority, because analytics outputs are write-once per run and E44 applies mainly to editing agents.

G8. No per-role model or effort settings.
- Evidence: E01, E02, E17, E59.
- Prophet today: the NIM transport has a model chain with demotion and a thinking mode. I did not confirm whether the planner, evaluator and summariser can each use a different model.
- Minimal fit: a per-role setting (planner, evaluator, summariser, sub-agent) for model and thinking mode. Put the summariser on the cheapest model that passes validation. Lower priority than G1 to G5.

Top five gaps, in order:
1. No per-tool approval gate (G1).
2. No per-criterion completion checklist (G2).
3. Compaction has no cheaper first step (G3).
4. The delegation contract is thin and synchronous (G5).
5. Cache health is not visible per step (G4).

Prophet claims checked against the evidence:
- ARCHITECTURE.md says approval does not guarantee exactly-once execution. This is supported by E32, so Prophet's `outcome_unknown` rule matches the documented guidance. This is a strength.
- ARCHITECTURE.md says the Claude Agent SDK has no base-URL option. Not confirmed. The Claude Code sub-agents page mentions routing through an LLM gateway set by an environment variable, but I did not find a documented base-URL option in the Agent SDK reference pages read. Re-check before relying on the claim.
- The ARCHITECTURE.md compaction threshold (summarise at about 60 to 70 percent of the window) has no public counterpart. The platforms use a configurable token threshold (E42, E49). Prophet's threshold is in characters. A token-based threshold would match the platforms more closely.

## Not verifiable from public evidence

- System prompts, planner prompts, tool descriptions and any hidden instructions in Claude.ai, ChatGPT or Codex. Not sought and not described.
- How the planner chooses between tools, and how it decides a task is finished.
- The internal implementation of Claude.ai and ChatGPT consumer memory, compaction and sandboxing.
- Maximum session duration for Managed Agents and ChatGPT agent. No figure was found in the pages read.
- Exact per-model minimum cacheable lengths for Claude (the table was not fully read).
- Whether NIM reports `cached_tokens` for the models in Prophet's chain.
- Whether Claude Code subagents can nest. Not checked.
- The isolation boundary of Managed Agents containers (E38, LOW).
- Independent benchmarks. The accuracy and token figures cited (E05, E06, E11, E37) are vendor-reported.
- Source code. The three GitHub repositories returned landing pages only, so no repository-level claims are made.
- Live behaviour of any product. No runs were made.
</contents>
