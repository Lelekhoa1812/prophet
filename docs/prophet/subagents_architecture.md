# Sub-agent and context architecture: what Claude Code and Cursor do

Status: research note. Written from static string evidence in shipped binaries, not from runtime traces.
It feeds the `delegate` action and the auto-compact design in Prophet. Where a point is inferred rather than
read from a string, it is marked as inferred.

## 1. Method and limits

- Target 1, Claude Code 2.1.304: `/opt/homebrew/Caskroom/claude-code/2.1.304/claude`, a 258 MB arm64 Mach-O Bun
  single-file executable. Its JavaScript is embedded as readable text in some regions and as UTF-16 in others.
  Strings were pulled with a byte-window extractor, not by running the binary.
- Target 2, Cursor client: `/Applications/Cursor.app/Contents/Resources/app/out/vs/workbench/*.js` and
  `extensions/cursor-agent-host`. The agent loop itself runs server-side; the client only carries protobuf schemas
  and rendering code.
- The REA MCP server was not connected in this session (its namespace was in an error state), so the REA
  structured tools (`open_binary`, `analyze_javascript_application`) were not used. This study is therefore a
  string-level reading, lower confidence than a REA Evidence record. Behaviour below is labelled `observed` when a
  string or schema states it, and `inferred` when it follows from several strings.

## 2. Sub-agent delegation

### 2.1 The Task tool contract (Claude Code)

Observed fields and descriptions in the Task tool schema:

| Field | Observed meaning |
| --- | --- |
| `description` | "A short (3-5 word) description of the task" (observed) |
| `prompt` | "The task for the agent to perform" (observed). The prompt is the only task text the child receives. |
| `subagent_type` | Names a registered agent definition. Errors: `subagent_type_not_found`, `subagent_type_missing`, `subagent_type_ambiguous`, `subagent_type_denied` (permission rule), `subagent_type_tools_denied`. `"fork"` is a special type that inherits the parent context (observed). |
| `run_in_background` | "Agents run in the background by default; you will be notified when one completes. Set to false only when your very next action depends on this..." (observed). |
| `name` | Makes the agent addressable by `SendMessage`-style mailbox while it runs (observed). In-process teammates cannot spawn background agents (`subagent_teammate_background_denied`). |
| `model`, `effort` | Per-agent model and reasoning effort. Effort is ignored for `fork` and remote (observed). |
| `isolation` | `worktree` / `remote` variants, with a tag `worktreeBranch` / `worktreePath` on the result (observed). |

Result handling (observed):
- A background launch returns an `agent_id` and the message "The agent is now running and will receive
  instructions via mailbox." The ID is internal metadata and is not to be quoted to the user.
- If a child ends without delivering a report, the parent is told: "The subagent ended without delivering a
  report ... so no report was delivered", and it is offered a continuation by task ID.
- A `SubagentHandback` frame exists, so the child hands its result back as one message, not as a transcript.

### 2.2 Context isolation (observed fields, inferred mechanism)

The agent-loop options (observed names) show what a child gets and what it does not:

- `forkContextMessages`: a fork is seeded with the parent's message prefix. A normal subagent is not.
- `appendSubagentSystemPrompt` / `appendSubagentSystemPromptFile`: the child's system prompt is the base plus an
  appended block, not a copy of the parent prompt.
- `excludeDynamicSystemPromptSections`: the child drops per-session dynamic sections so its prefix stays
  byte-stable for prompt caching (inferred from the name and the cache-oriented code around it).
- `systemPromptSnapshot`: the system prompt can be frozen and replayed.
- `maxTurns`, `maxBudgetUsd`, `taskBudget.total`: each child has its own turn and cost ceiling (observed).
- `forwardSubagentText`, `forwardNestedSubagentToolCalls`: the parent may choose to see the child's text and
  nested tool calls. By default it sees only the report (inferred from the default and the flag names).
- Each child writes its own transcript marked as a sidechain: "Failed to record sidechain transcript", and the
  session list shows `(sidechain)` (observed). A parent transcript therefore does not contain child tool traffic.
- `subagent_no_directory_in_cwd_scope`: a subagent cannot start outside the project scope; the model is told to do
  the task without one (observed).
- Hooks may deny a spawn: `subagent_spawn_denied_by_hook` (observed).

### 2.3 Cursor's Task tool (client schema)

Observed in `agent.v1.TaskArgs` and `aiserver.v1.TaskV2Params`:

- `description`, `prompt`, `subagent_type` (a typed union: `generalPurpose`/unspecified, `explore`, `bash`,
  `shell`, `browser_use`, `computer_use`, `media_review`, `debug`, `vmSetupHelper`, or `custom:<name>`).
- `model`, `resume` and `agent_id`: a subagent can be resumed by its ID.
- `mode` and `responding_to_message_ids`: the child can be asked to respond to specific messages.
- `environment` and `machine`: the child runs in a chosen environment (local, cloud, or another attached
  machine).
- Persistent state, `agent.v1.SubagentPersistedState`: `conversation_state`, `created_timestamp_ms`,
  `last_used_timestamp_ms`, `subagent_type`, `model_id`, `environment`, `cloud_subagent`. The child's
  conversation is stored separately and reloaded on resume (observed).

### 2.4 Common pattern (both products)

1. The parent emits a bounded task: a short description, a self-contained prompt, a type, and optional model.
2. The child runs on its own message list with its own system prompt, turn and cost budget, and tool scope.
3. The child's full transcript stays in a sidechain store. Only a final report returns to the parent.
4. A child can be resumed by ID. An unfinished child is reported as "stopped", not as failed.
5. Spawning can be denied by policy and is bounded by scope.

## 3. Context-window management

### 3.1 Auto-compact trigger (Claude Code, observed)

- Log line: `autocompact: tokens=<H> level=<compact|...> effectiveWindow=<W>`, plus `serverCleared=<n>`.
- `effectiveWindow` = `contextWindow - min(maxOutputTokens, reserved)`. The reserve for output is subtracted before
  any threshold (observed, from `WY` / `Mbr`).
- Threshold: a percentage of the effective window, minus a buffer, can be overridden with
  `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`. Warnings show "% until auto-compact" and "Context low (N% remaining)".
- The token count adjusts for tokens the server reports as cleared: the log carries `serverCleared`, read from
  `context_management.applied_edits[*].cleared_input_tokens`. The exact arithmetic was not fully traced (inferred).
- In remote sessions a fork ignores the percentage override. If it is under the threshold without the override,
  it does not compact early (`tengu_fork_early_compact`, log text "not compacting the fork").

### 3.2 Compaction summary (Claude Code, observed)

The compaction prompt asks for a summary that "will be placed at the start of a continuing session; newer messages
that build on this context will follow". It is structured as numbered sections: primary request and intent,
key technical concepts, files and code, errors and fixes, problem solving, all user messages, pending tasks,
current work, and an optional next step with a direct quote. The summary is written for recent messages only,
after a retained earlier-context block. After compaction the boundary is recorded with `tokensBefore`,
`tokensAfter`, `compact_boundary`, and `microcompact_boundary` frames, and the message "Conversation summarized"
is shown.

### 3.3 Tool-result clearing (Claude Code, observed)

Two mechanisms run before a full summary is needed:

- Server-side: context-management edits `clear_tool_uses_20250919` (clears old tool results) and
  `clear_thinking_20251015` (clears old thinking blocks). Their cleared counts feed the token estimate above.
- Client-side microcompact: `[KEEP-RECENT MC] context_hint trigger, cleared N tool results (~X tokens), kept last K`.
  `compact_micro_keep_recent` sets K. Recent results are kept whole; older ones are replaced by a placeholder
  (`content omitted`).

After compaction the runtime re-reads files it had read (`extractReadFilesFromMessages`,
`pre_compact_artifact_read_versions`), so a file's current content is restored into the new context.

### 3.4 Prompt-cache stability (inferred)

The design keeps the prefix fixed: system prompt, tool schemas and the compacted summary come first and do not change
between turns. Per-turn variability goes at the end. Subagents get a separate prefix so that they do not invalidate
the parent's cache.

## 4. Streaming and continuation

- A streamed response that is cut by a 529 or a stream error is retried as a stream first
  (`tengu_streaming_529_retry`) and only then falls back to a non-streaming call. The fallback is logged with
  `cli_nonstreaming_fallback_started` (observed).
- A non-streaming fallback is not a silent failover. It is a separate, logged path.

## 5. Applying this to Prophet

| Claude Code / Cursor mechanism | Prophet design |
| --- | --- |
| Child gets its own message list and budget | `delegate` action runs a child goal with its own history, step budget and tool scope. Depth is 1; children cannot delegate. |
| Only the report returns | The child returns `{report, evidence_ids, unknowns}`. Its transcript is stored as a sidechain under the mission, not in the parent's history. |
| Child is resumable | The child's mission/step ledger is durable like a parent mission; a stopped child is reported as checkpointed. |
| Auto-compact on a threshold | `context.py` estimates tokens and compacts the oldest evidence into a memory block once the view passes `view_chars`. Recent observations and the plan prefix are kept whole. |
| Tool-result clearing | Older tool results are replaced by a short stub that keeps the id, tool name and digest. The full result stays in the ledger and can be re-read by id. |
| Stable prefix | The system prompt, tool schema and memory block lead the prompt; per-step content trails it. |
| Stream first, fall back second | The NIM transport streams; a cut stream is continued on the same model before any failover (already implemented as `MAX_CONTINUATIONS`). |

Invariants kept from `CLAUDE.md`: a completed tool call is never re-run; compaction writes a durable step so a
replay sees the same memory; synthetic and measured numbers stay labelled.
