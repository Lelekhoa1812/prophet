# Goal-driven agent loops: planning, verification, repair, tools, questioning, supervision

Scope: evidence for building Prophet's goal-pursuing loop on open-weight models via NVIDIA NIM. Research date: 20 Nov 2026. Labels: [Verified: URL] = read or confirmed in a fetched/search source this session; [Hypothesis] = my inference or recollection not checked against a source this session. Search-snippet-only claims are flagged as such. Fetch failures are listed under Gaps.

## Key Question 1: Which loop patterns exist, and what makes "done" verifiable?

### Takeaway
The durable pattern is plan, act, verify, with completion judged by a party other than the actor and against an observable end state. Anthropic advises the simplest workflow that works and reserves open-ended agent loops for problems where the step count cannot be predicted. Claude Code's `/goal` is a concrete reference design: a separate evaluator model checks a stated condition after every turn, and the loop stops on met, impossible, stall, or error.

### Cited Findings
- Anthropic names five workflow patterns (prompt chaining, routing, parallelisation, orchestrator-workers, evaluator-optimizer) and separately defines agents as LLMs that direct their own tool use in a loop with environmental feedback. Advice: "find the simplest solution possible, and only increasing complexity when needed"; agents bring "higher costs, and the potential for compounding errors." [Verified: https://www.anthropic.com/engineering/building-effective-agents]
- Evaluator-optimizer works "best with clear evaluation criteria and when feedback measurably improves output." [Verified: https://www.anthropic.com/engineering/building-effective-agents]
- Claude Code `/goal`: after each turn a small fast model receives the condition and the conversation so far and returns not met, met, or impossible. The evaluator "doesn't run commands or read files independently, so write the condition as something Claude's own output can demonstrate." [Verified: https://code.claude.com/docs/en/goal]
- `/goal` conditions should have one measurable end state, a stated check (for example "`npm test` exits 0"), and constraints that must not change. Bounds are set by a turn or time clause in the condition, such as "or stop after 20 turns". Conditions can be up to 4,000 characters. [Verified: https://code.claude.com/docs/en/goal]
- `/goal` stops with the goal still set when Claude keeps answering the evaluator without tool use for several turns in a row. It skips evaluation while a subagent or background shell is still running. [Verified: https://code.claude.com/docs/en/goal]
- `/goal` is a wrapper around a session-scoped prompt-based Stop hook. Auto mode approves tool calls within a turn but does not start a new turn. `/goal` adds a separate evaluator, so "completion is decided by a fresh model rather than the one doing the work." [Verified: https://code.claude.com/docs/en/goal]
- Retry and pause policy for `/goal`: after transient failures it retries automatically, and after three automatic retries it pauses. Rate limits and usage limits pause. Credit exhaustion, context overflow that compaction cannot clear, and auth failure clear the goal. [Verified: https://code.claude.com/docs/en/goal]
- Anthropic's multi-agent research post reports about 4x tokens for agents versus chat and about 15x for multi-agent systems. It says token usage explained about 80% of variance on BrowseComp. An Opus 4 lead with Sonnet 4 subagents beat single-agent Opus 4 by 90.2% on an internal research eval. Coding, which shares context heavily, is a poorer fit for multi-agent designs. [Verified: https://www.anthropic.com/engineering/multi-agent-research-system]
- The same post says the lead agent decides when to stop with no explicit stopping criteria. Effort-scaling rules set budgets, for example 3–10 tool calls for simple fact-finding and more than 10 subagents only for complex research. Early versions spawned 50 subagents for simple queries and over-continued after results were sufficient. Premature completion is not discussed. For state-changing agents the post recommends end-state evaluation rather than step checks. [Verified: https://www.anthropic.com/engineering/multi-agent-research-system]
- The LLM judge in that post used a single prompt returning a 0.0–1.0 score plus pass/fail, scored against a rubric covering factual accuracy, citation accuracy, completeness, source quality and tool efficiency. Evaluation started with about 20 queries. [Verified: https://www.anthropic.com/engineering/multi-agent-research-system]
- Reflexion (Shinn et al.) improves agents with verbal reflections stored in an episodic memory buffer, without weight updates. Reported results: 91% pass@1 on HumanEval against 80% for GPT-4 at the time, and gains on AlfWorld. [Verified: arXiv search summary, https://arxiv.org/pdf/2303.11366. The HumanEval and AlfWorld figures come from a search summary, not the full paper.]
- Multi-agent failure taxonomy (MAST, Cemri et al.): 14 failure modes across three categories, derived from 1,642 annotated traces over seven frameworks and more than 200 tasks. Category shares: specification 41.77%, inter-agent misalignment 36.94%, task verification 21.30%. Largest single modes: step repetition 17.14%, reasoning-action mismatch 13.98%, failure to ask for clarification 11.65%, unaware of termination conditions 9.82%, premature termination 7.82%, no or incomplete verification 6.82%, incorrect verification 6.66%. [Verified: https://arxiv.org/html/2503.13657v2]
- MAST intervention results: a better role specification raised ChatDev's success by 9.4%. Adding a high-level verification step raised it by 15.6% absolute. The authors conclude that "simple fixes are insufficient". The paper is internally inconsistent on ChatDev's baseline (33.33% correctness in one place, 75% failure in another). [Verified: https://arxiv.org/html/2503.13657v2]

### Inferences
- [Hypothesis] Prophet should treat "done" as a claim the loop must justify, not a state the model asserts. Completion should be checked by an evaluator that sees tool outputs and checks named criteria. Ideally the check is deterministic (for example, a forecast-accuracy threshold computed by code) and the model judges only what code cannot check.
- [Hypothesis] The MAST shares suggest supervision should target step repetition and termination awareness first. A repeated-action detector and an explicit termination predicate would address the two largest single modes.
- [Hypothesis] Goal conditions in Prophet should be written as measurable criteria with a required check and a budget clause, copying `/goal`'s structure. Example: "backtest WAPE on holdout is at or below X, using the `run_backtest` tool output, or stop after N turns."
- [Hypothesis] ReAct (Yao et al., 2022) and plan-and-execute are the usual baseline shapes. Prophet's core loop can be ReAct with a plan object held in state, plus a separate verifier. This is design reasoning, not a cited finding.

### Gaps
- ReAct, plan-and-execute, Codex's goal-like features, and the "Ralph Wiggum" loop were not fetched or verified this session. Treat them as unresearched.
- Anthropic's Building Effective Agents post is dated December 2024 and its page points to newer guidance. I did not locate that newer guidance.
- The Reflexion numbers are from a search summary, not the paper body.
- No source found on the false-"done" rate of LLM evaluators. Evaluator reliability is unquantified here.

## Key Question 2: How should Prophet repair tool failures and bad model output?

### Takeaway
Intrinsic self-correction (asking the model to re-check its own answer without external signal) does not reliably help on reasoning tasks. Repair should rely on external signals: tool errors, schema validation, deterministic checks and tests. Retry, re-plan and ask-user are different responses and should be chosen by error class.

### Cited Findings
- Huang et al. (Google DeepMind and UIUC) find that intrinsic self-correction "consistently leads to performance degradation" on reasoning tasks across the models and benchmarks tested. Example: GPT-3.5 on GSM8K fell from 75.9% to 74.7% after two rounds. Models change correct answers to wrong ones about as often as they fix wrong ones. Earlier gains often relied on oracle labels. The authors recommend external feedback. [Verified: https://arxiv.org/pdf/2310.01798 and https://arxiv.org/html/2310.01798v2 (abstract and summary sources)]
- MAST attributes many multi-agent failures to design and misalignment, not only model capability, and recommends structural redesign. [Verified: https://arxiv.org/html/2503.13657v2]
- Anthropic's tool-writing post recommends making error messages "actionable rather than opaque codes or tracebacks," and suggests pagination, filtering and truncation with sensible defaults. Claude Code caps tool responses at 25,000 tokens by default. [Verified: https://www.anthropic.com/engineering/writing-tools-for-agents]
- Anthropic's building-agents post says to "change the arguments so that it is harder to make mistakes" (poka-yoke). In SWE-bench work, requiring absolute file paths fixed errors caused by relative paths after directory changes. [Verified: https://www.anthropic.com/engineering/building-effective-agents]
- Anthropic's multi-agent post says the team resumes work from checkpoints instead of restarting, uses retry logic, and lets agents adapt when tools fail. [Verified: https://www.anthropic.com/engineering/multi-agent-research-system]

### Inferences
- [Hypothesis] Error taxonomy for Prophet, to map to actions:
  - Transport or rate-limit or 5xx: retry with backoff, bounded.
  - Invalid JSON or schema violation from model output: one constrained retry with the validator message, then fall back to a grammar-constrained or structured-output call if available.
  - Wrong tool arguments (validator error): return the error as a tool result with the expected schema, one retry.
  - Hallucinated tool name: reject, list the valid tools in the error, do not execute.
  - Empty or non-progressing reply: count as a stall, not as completion.
  - Tool ran but output contradicts the criterion: re-plan (change the approach), not retry the same call.
  - Missing user input or a consequential choice: ask the user through a structured question.
- [Hypothesis] Self-correction should be gated on an external signal. A "review your answer" pass with no new evidence should not count as verification.

### Gaps
- No primary source found this session on the frequency of each error type for open models on NIM. Measure this on Prophet's own traces.
- Repair-policy numbers (retry caps, backoff) are design choices, not sourced findings.

## Key Question 3: Tool surface, dynamic discovery, and open-model tool-calling reliability

### Takeaway
Defer tool schemas when the tool set is large, because loading everything costs context and degrades selection. Write tool descriptions as briefs with unambiguous parameter names, and namespace tools. Open-model tool calling on NIM depends on server flags, thinking mode and model size. Small models (3B and under) are reported as unreliable for tools, so the model choice needs testing before design.

### Cited Findings
- Anthropic's advanced tool use post: a Tool Search tool with `defer_loading: true` keeps tools out of context until searched. Loading about 50+ MCP tools upfront costs about 77K tokens against about 8.7K with deferral. MCP evaluation accuracy went from 49% to 74% (Opus 4) and from 79.5% to 88.1% (Opus 4.5). Built-in search is regex or BM25, and custom search such as embeddings is supported. [Verified: https://www.anthropic.com/engineering/advanced-tool-use]
- The same post recommends deferring when definitions exceed about 10K tokens, there are 10+ tools, or several MCP servers are running. It keeps the 3–5 most used tools loaded. Tool Use Examples (`input_examples`) raised complex-parameter accuracy from 72% to 90%. The features are beta behind header `advanced-tool-use-2025-11-20`. Note: the post's own context-savings figures are inconsistent (191,300 vs 122,800 tokens), so check the API docs before quoting them. [Verified: https://www.anthropic.com/engineering/advanced-tool-use]
- Tool writing guidance: write descriptions "as you would brief a new team member"; use unambiguous names such as `user_id` rather than `user`; define strict input and output models; namespace by service and resource; offer a `concise` or `detailed` response-format enum; and evaluate with dozens of realistic agent-generated tasks with verifiable outcomes, tracking accuracy, runtime, tool calls, tokens and tool errors. [Verified: https://www.anthropic.com/engineering/writing-tools-for-agents]
- The same post says Claude appended "2025" to web search queries, biasing results, and that improving the tool description fixed it. [Verified: https://www.anthropic.com/engineering/writing-tools-for-agents]
- NIM function calling (search-snippet level, not fetched): controlled by `tools` and `tool_choice` (`none`, `auto`, or a named tool). Tool calling requires environment variable `NIM_ENABLE_AUTO_TOOL_CHOICE=1` in the v1.12 and v1.15 docs. [Verified as search snippet: https://docs.nvidia.com/nim/large-language-models/1.12.0/function-calling.html and https://docs.nvidia.com/nim/large-language-models/1.15.0/function-calling.html]
- NIM v1.10 docs: Llama Nemotron Nano, Super and Ultra support tool calling only with detailed thinking turned off. The v1.15 docs list Nemotron 3 Nano and say tool calling is enabled automatically for the listed models. [Verified as search snippet: https://docs.nvidia.com/nim/large-language-models/1.10.0/function-calling.html]
- NIM v1.12 release notes (search snippet): for models of 3B parameters or fewer, tool calling is "highly inaccurate" and should not be used for tool use. The same notes say `tool_choice` is always `auto`, which conflicts with the general docs. [Verified as search snippet: https://docs.nvidia.com/nim/large-language-models/1.12.0/release-notes.html]
- BFCL V4 (Berkeley) is described by its official leaderboard as measuring accurate function calling, last updated 16 Dec 2025. I could not retrieve open-model rankings. An aggregator (BenchLM) shows figures for a snapshot dated 13 Nov 2026, but its leader is from a lab I could not verify and its scores do not appear on the official page, so I do not use them. [Verified: https://gorilla.cs.berkeley.edu/leaderboard (metadata only); aggregator not relied on: https://benchlm.ai/benchmarks/bfcl-v4]

### Inferences
- [Hypothesis] With Prophet's current tool count, deferral is probably not yet needed. Use a tool registry with a small always-loaded core (for example, data load, forecast, evaluate, ask-user) and a search tool for the rest once the count passes about 10.
- [Hypothesis] Mitigations for open-model tool-calling on NIM: pin the NIM version and read its tool-calling docs; disable thinking for tool turns if the model requires it; validate every argument against the JSON Schema before execution; return validator errors as tool results; add 1–3 few-shot examples per tool for ambiguous parameters; prefer named `tool_choice` for forced steps such as the ask-user tool.
- [Hypothesis] Shortlist candidate models by running Prophet's own tool-call eval (accuracy, invalid-argument rate, hallucinated-tool rate) rather than relying on leaderboards.

### Gaps
- The NIM function-calling page itself did not fetch (empty content). The flags and model lists above are from search snippets and need checking against the NIM version in use.
- No verified current list of open models that reliably support `tool_choice` on NIM.
- Constrained decoding (grammar or JSON-schema decoding) support in NIM was not checked.
- BFCL open-model standings for November 2026 could not be verified.

## Key Question 4: Structured user questioning

### Takeaway
Claude Code's AskUserQuestion is the clearest published schema for this: 1–4 questions per call, 2–4 options each, a short `header`, a `multiSelect` flag, and answers keyed by question text. The MCP elicitation spec offers a second model: flat, primitive-only JSON-schema forms with explicit accept, decline and cancel outcomes. Both pause the agent until the user answers, and both need a pending-question state that survives restart.

### Cited Findings
- Claude Code AskUserQuestion fields (official SDK docs): `question`, `header` (max 12 characters), `options` (2–4 items with `label` and `description`), and `multiSelect`. Limits: "each AskUserQuestion call supports 1-4 questions with 2-4 options each." Not available in subagents spawned via the Agent tool. [Verified: https://code.claude.com/docs/en/agent-sdk/user-input]
- Answers return as `answers`, an object keyed by the question text, with the option label as the value (or an array or joined string for multi-select). An optional `response` field carries a free-form reply when the user dismisses the card. [Verified: https://code.claude.com/docs/en/agent-sdk/user-input]
- Free text: the docs say the app must add an "Other" choice itself and put the user's typed text in the answer. Third-party sites say "Other" is added automatically. The official docs contradict that, so the app builds it. [Verified: https://code.claude.com/docs/en/agent-sdk/user-input (official); third-party claim not relied on]
- The callback can stay pending indefinitely. For users who may take longer than the process can run, the docs recommend a PreToolUse hook returning a `defer` decision, so the process can exit and resume from the persisted session. [Verified: https://code.claude.com/docs/en/agent-sdk/user-input]
- Option previews: with `previewFormat` set, options can carry Markdown or HTML previews. Claude includes previews only where a visual comparison helps. [Verified: https://code.claude.com/docs/en/agent-sdk/user-input]
- Claude Code's AskUserQuestion is a tool call that goes through `canUseTool`, and the SDK docs recommend a custom tool for forms or approval workflows that exceed multiple choice. [Verified: https://code.claude.com/docs/en/agent-sdk/user-input]
- Community reports (search snippet, not verified against Anthropic's own issue tracker): a null-output bug where the tool returned without prompting, a request for select-all controls on multi-select, and a request for radio controls on single-select. [Verified as search snippet: https://claudeissues.com/issue/20855-askuserquestion-tool-returns-null-output-without-prompting-user-for-input and https://claudeissues.com/issue/83902-single-select-askuserquestion-options-need-a-radio-control-matching-the-checkbox]
- MCP elicitation (spec 2025-06-18): `elicitation/create` takes a `message` and a `requestedSchema`, which is a flat object of primitives only (string with formats email, uri, date, date-time; number or integer with bounds; boolean; enum with `enumNames`). Nested objects and arrays of objects are excluded. The result `action` is `accept` (with `content`), `decline`, or `cancel`. Servers "MUST NOT" request sensitive information. Clients "SHOULD" allow users to decline at any time and rate-limit. The spec notes that elicitation "may evolve in future protocol versions." [Verified: https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation]
- MCP distinguishes decline (explicit no) from cancel (dismissed without choice), and servers should handle each. [Verified: https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation]

### Inferences
- [Hypothesis] Prophet should adopt the AskUserQuestion shape for its own `ask_user` tool, since it maps cleanly to a chat UI and to the MCP enum schema. Constraints to keep: 1–4 questions, 2–4 options, a required recommended option marked by position or a `recommended` flag (a design choice, not a sourced rule), explicit `multiSelect`, and a free-text "Other" that the app renders.
- [Hypothesis] Confirmation gates before consequential actions should be a single-question call with two options (for example "Run the forecast refit" / "Cancel"), not free text.
- [Hypothesis] State machine: the run enters `awaiting_user` with a persisted pending-question record (id, questions, schema, created time, timeout). Independent work continues in other branches while waiting. On answer, the run resumes with the answers appended as a tool result. On cancel or timeout, the run records a decision and does not proceed with the consequential action.
- [Hypothesis] Decline and cancel must be distinct states in Prophet, following MCP, so a dismissal does not count as a "no".

### Gaps
- No source found on how often open models pick good questions or ask too many. Measure this.
- The recommended-option and Other-option UX choices above come from design reasoning, not sourced findings.
- OpenAI Agents SDK and Cursor equivalents were not fetched this session.

## Key Question 5: Supervision, tracing and observability

### Takeaway
Supervision needs four things: hard budgets (turns, tool calls, tokens, time), repeated-action and no-progress detection, a persisted trace of each step with its evidence, and a human interrupt that cancels cleanly. The OpenTelemetry GenAI conventions are the likely standard for traces, but I could not verify their current attribute names this session.

### Cited Findings
- Claude Code `/goal` shows the condition, elapsed time, turn count, token spend and the evaluator's latest reason. It supports Ctrl+C interrupt and `/goal clear`, and it resumes an active goal on session resume with turn count and token baseline reset. [Verified: https://code.claude.com/docs/en/goal]
- `/goal` check-ins: background work that keeps the goal waiting for 30 minutes triggers a check-in. Later check-ins back off, up to 4x the first interval. At most three idle check-ins per goal between prompts. [Verified: https://code.claude.com/docs/en/goal]
- The Anthropic multi-agent post recommends full production tracing and monitoring of decision patterns rather than conversation content, because non-deterministic runs are hard to debug. [Verified: https://www.anthropic.com/engineering/multi-agent-research-system]
- Anthropic's post says that for long-running agents minor failures can derail the whole run, so checkpoint-and-resume is needed. [Verified: https://www.anthropic.com/engineering/multi-agent-research-system]
- OpenTelemetry GenAI semantic conventions have moved to https://github.com/open-telemetry/semantic-conventions-genai. The old page is no longer maintained. I could not read the current attribute list. [Verified: fetch notice only, https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/]

### Inferences
- [Hypothesis] Budgets to enforce in code, not in the prompt: max turns, max tool calls per goal, max tokens and cost (the ledger sums provider usage per call), wall-clock limit, and max consecutive turns without a new tool result. Trip any budget to pause, not silently stop.
- [Hypothesis] Stuck-loop detection: hash each (tool name, normalised args) and flag repeats; track whether the set of observed facts, or the criterion's metric, changed in the last N turns. MAST's top mode (step repetition, 17.14%) makes this a priority.
- [Hypothesis] Trace schema: one span per model call and per tool call, with a run id, the goal and criteria version, the verdict and its reason, the budget state, and the pending-question id. Store full content only under an opt-in flag, consistent with the OTel GenAI pattern of opt-in content capture (not verified this session).
- [Hypothesis] Interrupt: cancel should propagate to in-flight tool calls, leave the run in a `cancelled` state with a resumable checkpoint, and never record a success.

### Gaps
- Exact OTel GenAI attribute names (for example `gen_ai.operation.name`, `gen_ai.usage.input_tokens`) and their stability status were not verified. Check the new repository before coding.
- Langfuse and Phoenix trace models were not researched this session.
- No source located on the false-"done" rate of evaluator models, or on cost data for NIM-hosted evaluators.

## Overall gaps and caveats for the report writer
- Not researched or not verified this session: ReAct, plan-and-execute, Codex's goal features, the "Ralph Wiggum" loop, OpenAI Agents SDK and Cursor equivalents, Langfuse and Phoenix, and the current OTel GenAI attributes.
- The NIM function-calling page did not fetch, so the NIM flags and model lists are from search snippets and conflict between NIM versions.
- Anthropic's "Building effective agents" post is dated December 2024 and says much tooling has changed. Treat its specifics as dated.
- Several Anthropic figures (tool search accuracy, the multi-agent 90.2% gain) come from vendor engineering posts on internal evals. They are not independent benchmarks.
- MAST percentages are from one taxonomy paper with an internal inconsistency noted above.
- All measurements above are design evidence. Prophet should measure its own tool-call reliability, question quality and loop termination on its synthetic and staging data before relying on any figure.
