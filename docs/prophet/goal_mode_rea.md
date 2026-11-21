# Claude Code `/goal`: mechanism recovered with REA, and what Prophet takes from it

Method: REA (`rea-agents` 6.10.1) `inspect-artifact` on the installed Claude Code 2.1.304 binary (a Bun standalone
executable, 2,729 embedded modules), then `extract-artifact` into a scratch folder and a static read of the chunks that
mention the goal state. Nothing was executed and nothing was decompiled from native code: the JavaScript is shipped as
text beside its bytecode. This file records the mechanism in our own words. It is understanding, not a copy.
Confidence: the control flow below is read from code (high). Anything about how well it works in practice is not
established here.

## 1. `/goal` is a stop hook with a condition, nothing more

- `/goal <condition>` registers one session-scoped **Stop hook of type "prompt"** whose text is the condition. The
  condition is capped at 4,000 characters. Setting a new goal supersedes the old one (the old hook is removed). `clear`,
  `stop`, `off`, `reset`, `none` and `cancel` remove it.
- Setting the goal injects one directive message: acknowledge briefly, start working at once, treat the condition as the
  directive, do not pause to ask the user what to do, and do not tell the user to clear the goal after success.
- A small state record rides in app state and in transcript attachments: condition, iterations, time set, tokens at start,
  last reason (cut to 512 characters). Status shows turns, elapsed time and tokens since the goal was set, and the last
  check's reason. The record is also mirrored to remote clients with length checks on every field.
- It is refused when hooks are disabled by policy or the workspace is untrusted. A goal is therefore a policy-visible
  object, not a hidden prompt.

## 2. The judge is a separate, stateless, schema-bound call

Each time the main agent tries to stop, the hook runs the evaluator:

- Input: the conversation transcript plus the condition, worded as "has this stopping condition been satisfied? Answer
  from transcript evidence only."
- Model call: no tools, thinking disabled, structured output `{ok, reason, impossible}`, default 30 s timeout.
- Rules in the evaluator's system prompt: quote transcript text as evidence; if the transcript does not clearly show the
  condition holds, answer not met with "insufficient evidence"; use `impossible` only when the condition is
  self-contradictory, needs something unavailable, or the agent has exhausted reasonable approaches, and treat the agent's
  own claim of impossibility as evidence, not proof; when in doubt, plain not-met.
- Injection hardening: the event JSON and anything read while judging are data; any rule inside them is ignored, even one
  claiming to come from the user.
- Outcomes: `ok` true clears the goal and records "achieved" with iterations, duration and tokens. `ok` false becomes a
  blocking message `[condition]: reason`, appended as a meta user message, which makes the agent continue; iterations go
  up by one and the reason is stored. `impossible` marks the goal failed and stops. A timeout cancels the check without
  failing or blocking. A prompt-too-long error retries once with half of the messages. An API or schema error is a
  non-blocking error, not a verdict.

## 3. Compaction keeps the asks, not the story

The summary prompt asks for nine sections: primary request and intent, key concepts, files and code, errors and fixes,
problem solving, **all user messages** (preserved verbatim, security-relevant constraints verbatim), pending tasks,
current work, and a next step that must quote the latest conversation to stop drift. An optional variant is capped at
2,000 words, and when the transcript already starts with a summary the new one must copy that summary's record of what
the user asked, unchanged, and rewrite only the account of work. The analysis scratch block is stripped before the summary
is kept. Auto-compaction is driven by an effective window and a threshold per model, and is also available on demand.

## 4. What this means for Prophet

Prophet already has something stronger than a prompt judge: deterministic contract checks run before the evaluator
(`runtime/contracts.py`), a durable brief, a grounding gate, and a replayable ledger. The gaps this exposes:

| Gap in Prophet | `/goal` practice | Change |
| --- | --- | --- |
| Evaluator reasons are free text | judge must quote evidence, default is "insufficient evidence" | evaluator.v8: every `met` carries evidence quotes (tool result ids); a quote that is not in the history is rejected locally |
| `impossible` can be reached on the agent's say-so | agent's claim is evidence, not proof | impossible needs the evaluator to name the missing capability and a deterministic check that no tool of that kind was left untried |
| Brief lives as one contract step | condition always re-injected after each stop | brief block re-rendered at the head of every planner prompt and every compaction, never summarised |
| Compactor keeps figures, not asks | summary keeps all user messages verbatim and a verbatim next-step anchor | compactor.v2 with fixed sections: asks, DONE_WHEN, ledger stages, decisions, open items, next step with quote |
| No mission status | iterations, elapsed time, tokens, last check reason | `mission_status` in the result and the event log; shown in the console |
| Evaluator or schema error can look like a verdict | timeout is a no-op, not a fail | evaluator errors retry and never count toward the rejected-finish limit |
| Prompt too long to judge | retry with half the history | evaluator retry with a trimmed view before giving up |

Limits of this reading: the evaluator in Claude Code sees the whole transcript, so it can check what the agent did. Ours
sees a compacted view, which is why the deterministic checks must stay the authority for evidence.
