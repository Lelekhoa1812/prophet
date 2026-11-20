# Prophet: mission and completion criteria

Status as of 2026-11-20. Every claim below is labelled with its evidence:
**[LIVE]** a real NIM or TimesFM call made in this repo, **[TEST]** an offline test, **[DOC]** a document or research
note, **[TODO]** not started or not verified.

## Mission

Turn the legacy `timesfm-erp` lab into **Prophet**: a domain-independent forecasting agent that plans, calls tools
(including TimesFM 3), asks the user structured questions, produces charts and documents, and keeps its state durable
across crashes. Read-only against any ERP. No client data goes to TimesFM 3 weights until the licence question is
settled (see ARCHITECTURE.md §5).

## Completion criteria

Each criterion is done only when its check passes and its evidence is recorded in this file.

| # | Criterion | Check | Status |
|---|---|---|---|
| G1 | NIM transport implements NIM.md §4.2 row by row | `PYTHONPATH=src .venv/bin/python -m pytest tests/prophet_agent -q` (one test per row) | **[TEST]** 46 transport tests pass; [TODO] wire into scripts/check.sh |
| G2 | Every configured key answers a 1-token chat call | `python -m prophet_agent.nim.probe keys` | **[LIVE]** 6 of 6 slots 200 on 2026-11-20 |
| G3 | Each chain model answers a schema-bound claim prompt, thinking off | `python -m prophet_agent.nim.probe models` | **[LIVE]** 5 of 5 valid JSON; GLM first call 100 s |
| G4 | Same, thinking on | `python -m prophet_agent.nim.probe models --thinking` | **[LIVE]** running; result in ARCHITECTURE.md §3 when it lands |
| G5 | Key failover works against the live API | invalid key first; it must be retired and a real key must serve | **[LIVE]** 403 retired, real key served (2026-11-20) |
| G6 | Durable runtime: an event log in SQLite (WAL); a crash mid-step resumes from the last completed step with no duplicate side effects | crash-injection test using idempotency keys | **[TODO]** |
| G7 | Goal loop: planner, tool use, evaluator returns met / not met / impossible; hard budgets; stall detection | scripted-NIM tests for each outcome and each budget | **[TODO]** |
| G8 | Structured questions (AskUserQuestion-style) validate the schema (1–4 questions, 2–4 options, header ≤ 12 chars), pause the run, and resume from the answer | schema tests and a pause/resume test | **[TODO]** |
| G9 | TimesFM 3 tool: real forecast on a real series, with quantiles and a recorded config | `pytest -m timesfm` on Apple silicon with weights cached | **[TODO]** (weights not yet loaded here) |
| G10 | Sandboxed Python for analytics: subprocess with `-I`, scrubbed env, timeouts, POSIX rlimits, output manifest with sha256 | tests for timeout, memory cap, out-of-dir write refused | **[TODO]** (labelled "guardrail, not isolation") |
| G11 | Charts and documents (PNG/SVG, XLSX, DOCX, PDF) produced from the sandbox and listed in a manifest | generate, reopen, check hashes | **[TODO]** |
| G12 | Chat UI with artefact panel, Dune-meets-space identity (impeccable, shadcn/ui, React Bits), works at 375 px with no horizontal scroll | browser check with screenshots and console clean | **[TODO]** |
| G13 | TimesFM evaluation on the synthetic twin vs baselines, reported honestly against the 15 % bar | backtest report with origins labelled (synthetic replay / real staging / measured) | **[TODO]** |
| G14 | No key or secret reaches logs, errors, transcripts or git | redaction tests plus a grep of the tree and the git diff before each commit | **[TEST]** for the transport; [TODO] whole-tree scan |

## Working rules for this mission

- Use Haiku sub-agents for research workers and bounded reviews; keep design and integration in the main thread.
- No access to the user's SharePoint (standing instruction, 2026-11-20).
- Do not delete or restructure `src/lab` without the user's explicit yes (open question, ARCHITECTURE.md §6).
- Commit locally as each criterion passes. Push to `origin/main` only after the user approves ARCHITECTURE.md §6,
  because a push is outward-facing and cannot be undone once others fetch it.
- Label every figure as synthetic replay, real staging, or measured. Never present expected dollars as observed savings.
