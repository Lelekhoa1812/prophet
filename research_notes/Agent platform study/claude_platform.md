# Claude agent platforms: public-evidence study for Prophet

## Method and evidence limits

- Sources are public Anthropic docs and engineering posts, read on 20 Nov 2026 through WebFetch. WebFetch returns a model-written summary of each page, not verbatim text. Figures and rules below are therefore as stated in those summaries and should be spot-checked against the live page before they go into a client or board document.
- Tags: [V-docs: URL] documented publicly; [V-observed] seen in a live Claude Code desktop session (the list supplied with the brief); [V-source] none used; [H] inference, unverified.
- Not covered: claude.ai web product internals, the Managed Agents internals beyond its public overview, MCP docs, the "Effective harnesses for long-running agents" post, "Advanced tool use", the compaction API page and the Claude Code model-config page (auto-compact threshold). No open-source code was read, and nothing here is decompiled or extracted from proprietary code.

## 1. Architecture rules from Anthropic's engineering guidance

**Workflows vs agents.** Start with the simplest design that works; add agent autonomy only when it measurably helps. [V-docs: https://www.anthropic.com/engineering/building-effective-agents, 19 Dec 2024] Workflow patterns named there: prompt chaining (with programmatic gates between steps), routing, parallelisation (sectioning and voting), orchestrator-workers, and evaluator-optimiser. Agents need stop conditions (for example a maximum iteration count) and checkpoints for human input. [V-docs: same URL]

**Tool design is as important as UI design.** Write tool descriptions "as a great docstring for a junior developer", with examples and edge cases. Test many inputs and iterate. Use poka-yoke parameters, such as requiring absolute paths. [V-docs: building-effective-agents URL]

**Multi-agent research system.** [V-docs: https://www.anthropic.com/engineering/multi-agent-research-system, 13 Jun 2025]
- Lead agent plans, spawns subagents and synthesises. Subagents each have their own context window, tools and trajectory, and return condensed findings.
- Delegation messages must state objective, output format, tool and source guidance, and task boundaries. Vague delegation caused duplicated searches and gaps.
- Effort is scaled in the prompt: simple lookups use 1 agent and 3–10 calls; comparisons use 2–4 subagents; complex research uses more. Early versions spawned 50 subagents for simple queries.
- Reported numbers (internal eval, as summarised): multi-agent beat single-agent by 90.2%. Token usage explained 80% of BrowseComp variance, and token usage, tool-call count and model choice together explained 95%. Agents use about 4x chat tokens and multi-agent systems about 15x. Parallel subagent and tool calls cut research time by up to 90% on complex queries.
- Long tasks: the lead saves its plan to external memory because context beyond 200k tokens is truncated. Agents resume from checkpoints rather than restarting. Subagents can write outputs to external storage and return references.
- Evaluation: start with about 20 representative queries. Use an LLM judge with a 0.0–1.0 score and a rubric (factual accuracy, citations, completeness, source quality, tool efficiency). Judge end-state outcomes, and keep human testing for source-quality issues.
- Stated limits: synchronous execution (lead cannot steer subagents mid-run), no subagent coordination, compounding errors, non-determinism, and poor fit for tasks with many inter-agent dependencies (most coding tasks).

**Context engineering.** [V-docs: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents, 29 Sep 2025]
- Aim for the smallest set of high-signal tokens. System prompts should sit at the "right altitude": specific enough to guide, flexible enough to allow judgement. Test a minimal prompt first and add rules only where tests fail.
- Keep tool sets minimal and non-overlapping. Prefer few, diverse canonical examples over long edge-case lists.
- Just-in-time retrieval: keep lightweight references (paths, queries) and load data when needed. Metadata (folder, names, size, timestamps) helps relevance. Hybrid (some up front, some on demand) is common.
- Compaction: summarise near the limit and keep architectural decisions, open bugs and implementation details; discard redundant tool output. Clearing old tool results is the safest form. Tune by maximising recall first, then precision.
- Structured note-taking: write progress notes to storage outside the window and reload them later.
- Sub-agents: each explores in a clean window and returns a condensed summary, often 1,000–2,000 tokens.
- "Context rot": recall degrades as the window fills, gradually rather than as a cliff.

**Tool writing.** [V-docs: https://www.anthropic.com/engineering/writing-tools-for-agents]
- Consolidate related steps into one tool. Avoid "list everything" tools; prefer filtered search. Give each tool a distinct purpose.
- Namespace tools by service and resource prefix.
- Return high-signal, human-readable fields. Offer a concise vs detailed response format. The post reports concise responses used about a third of the tokens in its Slack example.
- Paginate and truncate large outputs, and say how to narrow the next request. Errors should say what was wrong and show the correct format.
- Evaluate with realistic multi-call tasks, verifiable outcomes, held-out sets, and track accuracy, runtime, call counts, tokens and errors. The post reports an agent-rewritten MCP tool description cut task time by 40%.

## 2. Claude Code and platform mechanisms

| Mechanism | Evidence | What it does (as documented or observed) |
|---|---|---|
| Agent loop | [V-docs: https://code.claude.com/docs/en/agent-sdk/overview] | The SDK exposes "the same tools, agent loop, and context management" as Claude Code. Loop detail is on the linked agent-loop page, not read. |
| Deferred tool loading / tool search | [V-docs: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool]; [V-docs: https://code.claude.com/docs/en/costs] for MCP deferral in Claude Code; [V-observed] ToolSearch UI | Tools marked `defer_loading: true` are excluded from the prompt prefix. Claude searches by regex or BM25 over names, descriptions and argument text, and the API expands up to 5 matches (limit up to 10,000). Docs claim over 85% token reduction for a ~55k-token multi-server setup, and say selection accuracy degrades past 30–50 tools. Deferred tools cannot carry `cache_control`. Claude Code defers MCP tool definitions by default. |
| Subagents | [V-docs: https://code.claude.com/docs/en/sub-agents]; [V-observed] Agent tool, fork, SendMessage resume | Markdown + YAML definition with its own system prompt, tool allowlist or denylist, and model. Starts with a fresh context, receives its delegation message and CLAUDE.md (not for Explore/Plan agents). Foreground blocks; background runs concurrently and reports via notification. Nesting up to 3 layers by default. Only the final report returns; transcripts are saved and can be resumed. |
| Hooks | [V-docs: https://code.claude.com/docs/en/hooks]; [V-docs: https://code.claude.com/docs/en/costs] for `updatedInput` example | Command, HTTP or MCP hooks at lifecycle events: SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, SubagentStop, PreCompact, PostCompact and others. Input is JSON on stdin. Exit 2 blocks. JSON can return `permissionDecision` and `updatedInput`, which rewrites tool input. Command/HTTP/MCP hooks are deterministic; prompt and agent hooks are model-judged. Docs say hard allow/deny belongs in the permission system. |
| Skills | [V-docs: https://code.claude.com/docs/en/skills] | `SKILL.md` with `name` and `description` front-matter. Descriptions are always in context (truncated at 1,536 chars). The body loads on invocation or match. Supporting files load on demand; scripts run and are not read in. Keep SKILL.md under ~500 lines. After compaction, the latest invocation of each skill is re-attached (up to 5,000 tokens per skill, 25,000 combined). `context: fork` runs a skill in a subagent. |
| CLAUDE.md and auto memory | [V-docs: https://code.claude.com/docs/en/memory]; [V-docs: https://code.claude.com/docs/en/costs] | Both load at session start. CLAUDE.md is advisory context, not enforced config; the docs say to use a PreToolUse hook to block an action. Auto memory's first 200 lines or 25KB load each session. Docs advise keeping CLAUDE.md under ~200 lines and moving workflow-specific instructions into skills. [V-observed] the MEMORY.md index with one fact per file matches that loading behaviour, but the one-fact-per-file layout itself is not documented. |
| Compaction and context management | [V-docs: https://code.claude.com/docs/en/costs]; [V-docs: https://code.claude.com/docs/en/checkpointing]; [V-docs: https://platform.claude.com/docs/en/build-with-claude/context-editing] | `/compact [focus]` and auto-compaction summarise history. `/clear` starts fresh. `PreCompact` and `PostCompact` hooks exist. CLAUDE.md can hold compaction instructions. Checkpoint menu offers "Summarize from here" and "Summarize up to here" without changing files. API context editing clears old tool results (default trigger 100k input tokens, keeps 3 recent pairs) and old thinking blocks, and the client keeps the full history. [H] the auto-compact threshold is not verified here. |
| Compaction, API side | [V-docs: https://platform.claude.com/docs/en/managed-agents/overview] mentions "compaction" in the harness | Managed Agents says its harness supports prompt caching and compaction. Details not read. |
| Prompt caching | [V-docs: https://platform.claude.com/docs/en/build-with-claude/prompt-caching] | Cache order is tools → system → messages. A change at one level invalidates that level and all after it. Up to 4 explicit breakpoints, or one automatic. Default TTL 5 minutes (write 1.25x base, read 0.1x); 1-hour TTL writes at 2x. Minimum cacheable length 512–4,096 tokens depending on model. A cache lookback checks up to 20 blocks before a breakpoint. Put breakpoints on the last stable block, not on content that changes each request. |
| Checkpoints and rewind | [V-docs: https://code.claude.com/docs/en/checkpointing]; [V-observed] | A snapshot is taken before each turn; the last 100 are kept, and the retention sweep deletes snapshots after about 30 days. `/rewind` (or double Esc) restores code, conversation or both. Not tracked: edits made by Bash commands, subagent edits (except a foreground forked skill), external edits, symlinks. Docs say checkpoints are not a substitute for git. |
| Plan mode and permission modes | [V-docs: https://code.claude.com/docs/en/permission-modes]; [V-observed] plan mode | Modes: default, acceptEdits, plan (reads only until a plan is approved), auto (classifier reviews actions), dontAsk, bypassPermissions. Deny rules apply in every mode. Auto mode checks rules first, then safe local actions, then a classifier. Tool results are stripped from classifier input to limit prompt injection. Subagents are checked at start, during the run and on report. |
| Background tasks and scheduling | [V-docs: https://code.claude.com/docs/en/sub-agents] for background subagents; [V-observed] wakeup, cron and monitor tools | Background subagents keep running while the parent works. Scheduled and session-scoped tasks exist and cost tokens while idle (costs page). Detailed scheduling pages not read. |
| Sessions and resume | [V-docs: https://code.claude.com/docs/en/agent-sdk/overview] lists sessions as "resume or fork later"; [V-docs: https://code.claude.com/docs/en/checkpointing] | Sessions are saved with their checkpoints and can be resumed; `--continue --fork-session` branches a session. |
| Agent SDK | [V-docs: https://code.claude.com/docs/en/agent-sdk/overview] | Python and TypeScript library that runs the Claude Code binary. Provides built-in tools, hooks, subagents, MCP, permissions, sessions, skills, plugins. Also: run CLI with `-p --output-format json` from other languages. Anthropic does not allow third parties to offer claude.ai login or rate limits for SDK products. |
| Managed Agents | [V-docs: https://platform.claude.com/docs/en/managed-agents/overview] | Beta (header `managed-agents-2026-04-01`). Four concepts: Agent (model, prompt, tools, MCP, skills), Environment (cloud or self-hosted sandbox), Session (stateful running instance), Events (SSE stream). Stateful: transcripts and sandbox state persist until deleted. Not ZDR or HIPAA eligible, except self-hosted. Steer or interrupt by sending events. |
| Memory tool (API) | [V-docs: https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool] | Client-side `/memories` directory with view, create, str_replace, insert, delete, rename. The app stores the files and must block path traversal. The API adds an instruction to view memory first and to assume the context may be reset at any time. |
| Tool-use loop, streaming | [V-docs: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool] | Tool search results come back as server blocks, which must be passed back unchanged. Errors use `is_error` on tool results. |
| AskUserQuestion, update_plan, wrap_up, Artifact | [V-observed] only | Structured question cards (1–4 questions, 2–4 options, "Other" added automatically), a pinned task list, a closing recap box, and artifacts published to claude.ai. None of these appeared in the public docs read here. Behaviour is described from observation only. |
| Tool results as data | [V-docs: permission-modes page]; [V-observed] | Tool output is treated as data, and only user chat messages are instructions. The classifier design shows this in practice. |

## 3. Which mechanisms matter for Prophet

Prophet is a deterministic forecasting and alerting pipeline (TimesFM forecasts, then pure-function agents, then a policy governor), read-only against FoodFlow. Most Claude Code mechanisms serve interactive coding, so only some carry over.

| Mechanism | Essential for Prophet? | Prophet adaptation |
|---|---|---|
| Simplest-workflow rule | Essential | Keep the forecast → agents → policy path as a fixed workflow (prompt chaining with gates). Use LLM-driven agency only where a task is open-ended, such as research triage. The agents in `agents/` are already pure functions; keep them that way. |
| Stop conditions and checkpoints | Essential | Add iteration and budget limits to any agentic loop. Persist job state per run, as `api/jobs.py` already does; resume from the last completed stage. |
| Orchestrator-workers / subagents | Useful, scoped | Use for parallel read-only work such as experiment audits or literature sweeps, with a delegation template (objective, output format, sources, boundaries). Do not use for tightly coupled pipeline changes, where the docs themselves say multi-agent is a poor fit. |
| Context isolation with condensed reports | Essential for long experiments | Each experiment writes a short summary back to `docs/GOAL_CHANGELOG.md` or a run record, rather than the raw logs entering the conversation. |
| External state and notes (memory, progress file) | Essential | Prophet already has `CONTINUOUS_GOAL.md`, `GOAL_BACKLOG.md` and the changelog. Treat them as the memory tool does: check them first and update them before a session ends. |
| Hooks for hard rules | Essential | CLAUDE.md invariants (no FoodFlow writes, no future leakage, labelling synthetic vs real vs measured) are advisory today. Enforce the critical ones with tests (already partly done in `tests/test_no_future_leak*.py`) and, for agent sessions, PreToolUse hooks that deny writes outside `data/` or connections to non-lab hosts. |
| Least privilege at the data boundary | Essential | Anthropic's model is deny rules that hold in every mode. Prophet's SELECT-only role on staging is the right place for this. Do not rely on prompt wording. |
| Skills (progressive disclosure) | Useful | Move the long runbook (RUNBOOK §4), the experiment recipes and the chart/report style into skills. Keep CLAUDE.md short. |
| Prompt caching | Useful only if Prophet calls an LLM | Relevant to the optional Bedrock narration. Put stable instructions before variable data; put breakpoints on the last stable block. Tool definitions sit at the start of the prefix, so changing them invalidates the cache. |
| Tool design (concise vs detailed, pagination) | Essential if Prophet exposes tools to agents | Keep the API and any agent tools high-signal and paginated. Return IDs in detailed mode only, and use actionable error messages. |
| Tool search / deferred loading | Not essential | Prophet has few tools. Only relevant if the tool count passes 10–30. |
| Checkpoints and rewind | Not essential | Use git and run-versioned DuckDB files instead. Checkpoints do not track Bash side effects, so Prophet's own versioning is the safer design. |
| Plan mode | Optional | Useful for design changes to the policy or governor. |
| Permission classifier (auto mode) | Not essential | Prophet is read-only by design. The classifier is a claude.ai-style control for coding sessions. |
| AskUserQuestion, update_plan, wrap_up, Artifacts | Optional UX | Claude-Code-specific. Worth copying only as a UX pattern for the console's decision prompts, not as an architecture. |
| Sessions and resume | Essential | Same role as the checkpoint and resume pattern above. |
| Evaluation with an LLM judge | Useful | Anthropic's rubric (accuracy, citation, completeness, source quality, efficiency) maps onto Prophet's scorer. Keep the score and its outcome separate, and use it only where there is no deterministic scorer. |
| Managed Agents | Not recommended now | Beta, stateful, not ZDR-eligible. Conflicts with the "no client data" and read-only boundary in CLAUDE.md until cleared. |

## 4. Caveats for Prophet

- Synthetic-twin figures in the repo are labelled as method demonstrations. The Anthropic figures above (90.2%, 15x, 85%, 40%) come from vendor posts and internal evals, and they should be labelled the same way if quoted.
- Anthropic's posts are dated (Dec 2024, Jun 2025, Sep 2025), and the 2026 docs show newer models, so verify specific numbers before reuse.
- Several observed Claude Code mechanisms (AskUserQuestion, update_plan, wrap_up, Artifacts, the internal shape of ToolSearch in the app) have no public description here. Treat them as [V-observed] UX, not as documented architecture.

## 5. Gaps (not verified)

- MCP documentation, the "Advanced tool use" post, "Effective harnesses for long-running agents", and the Managed Agents session and event reference.
- The auto-compact threshold and compaction prompt contents.
- The full hooks page past 100k characters, and the agent-loop page.
- claude.ai product mechanisms (Projects, memory feature, Artifacts internals) beyond what was observed.
