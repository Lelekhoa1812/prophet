# Prophet: mission and completion criteria

Status as of 2026-11-20. Every claim below is labelled with its evidence:
**[LIVE]** a real NIM or TimesFM call or a live browser or API check made in this repo, **[TEST]** an offline test in
the gate (`scripts/check.sh`), **[DOC]** a document or research note, **[TODO]** not started or not verified.

Gate at this status: `scripts/check.sh` 155 passed, 1 skipped (the real-weights test); `scripts/check.sh --real` adds
2 passed (TimesFM 3 with cached weights).

## Mission

Turn the legacy `timesfm-erp` lab into **Prophet**: a domain-independent forecasting agent that plans, calls tools
(including TimesFM 3), asks the user structured questions, produces charts and documents, and keeps its state durable
across crashes. Read-only against any ERP. The legacy lab was deleted in commit 9cf4c01 on the user's instruction of
2026-11-20. The TimesFM 3 licence position is in ARCHITECTURE.md §5.

## Completion criteria

Each criterion is done only when its check passes and its evidence is recorded in this file.

| # | Criterion | Check | Status |
|---|---|---|---|
| G1 | NIM transport implements NIM.md §4.2 row by row | `scripts/check.sh` (transport tests in `tests/prophet_agent/test_nim_transport.py`) | **[TEST]** in the gate; one test per row |
| G2 | Every configured key answers a 1-token chat call | `PYTHONPATH=src .venv/bin/python -m prophet_agent.nim.probe keys` | **[LIVE]** 6 of 6 slots 200 on 2026-11-20 |
| G3 | Each chain model answers a schema-bound claim prompt, thinking off | `... -m prophet_agent.nim.probe models` | **[LIVE]** 5 of 5 valid JSON in two runs; GLM 91–100 s |
| G4 | Same, thinking on | `... -m prophet_agent.nim.probe models --thinking` | **[LIVE]** 5 of 5 valid JSON on a trivial prompt; does not test heavy reasoning (ARCHITECTURE.md §3.1) |
| G5 | Key failover works against the live API | invalid key first; it must be retired and a real key must serve | **[LIVE]** 403 retired, real key served (2026-11-20) |
| G6 | Durable runtime: SQLite WAL event log; a crash mid-step resumes from the last completed step with no duplicate side effects | `tests/prophet_agent/test_runtime_durability.py` and `test_crash_mid_loop_resumes_without_repeating_decisions_or_tools` | **[TEST]** |
| G7 | Goal loop: planner, tool use, evaluator returns met / not met / impossible; hard budgets; stall detection | scripted-brain tests for each outcome and budget, including the stored mission status | **[TEST]** for all outcomes. **[LIVE]** met in one iteration on 2026-11-20 (mission `1a1639de…`). Two live defects fixed after the run (below) |
| G8 | Structured questions (AskUserQuestion-style) validate the schema, pause the run, and resume from the answer | validator and pause/resume tests; UI round-trip | **[TEST]** validator and pause/resume. UI round-trip verified with a scripted brain. [TODO] live NIM `ask_user` not exercised in the final pass |
| G9 | TimesFM 3 tool: forecast with quantiles and a recorded config | `scripts/check.sh --real` and a live planner mission | **[LIVE]** `google/timesfm-3.0-pytorch` on MLX; config recorded (global context 15360, context used 24, nine quantiles). Input was a synthetic series, so "real series" is not yet met |
| G10 | Sandboxed Python for analytics: subprocess with `-I`, scrubbed env, timeouts, rlimits, output manifest with sha256 | `tests/prophet_agent/sandbox/test_sandbox.py` | **[TEST]**. Labelled "guardrail, not isolation". The memory cap is not enforced on macOS |
| G11 | Charts and documents (PNG/SVG, XLSX, DOCX, PDF) produced from the sandbox and listed in a manifest | live forecast-pack mission; downloads re-hashed | **[LIVE]** 5 files from mission `1a1639de…`; each download's sha256 matched the manifest (2026-11-20) |
| G12 | Chat UI with artefact panel, Dune-meets-space identity, works at 375 px with no horizontal scroll | browser check with screenshots and console | **[LIVE]** 375 px: no horizontal overflow (scroll width 375), artefacts card renders. 1280 px: three columns, fan chart and file list render. Console not re-read after the last fix |
| G13 | TimesFM evaluation on the synthetic twin vs baselines, reported honestly against the 15 % bar | `scripts/backtest_twin.py` (written to `data/backtests/`, git-ignored) | **[LIVE]** SYNTHETIC REPLAY. The 15 % bar is **not met**: see the result below |
| G14 | No key or secret reaches logs, errors, transcripts or git | redaction tests plus a whole-tree scan before each commit | **[TEST]** redaction; **[LIVE]** whole-tree scan on 2026-11-20 (below) |

### Evidence for G13 (synthetic replay, 2026-11-20)

- Data: the synthetic generator, seed 20261120, 12 entities, 137 weekly series, three years of history, a 4-week
  horizon. History is the recorded sales (censored by stockouts, as a planner would see them). The future four weeks
  are the generator's true demand.
- Method: walk-forward, 26 origins (131 to 156). Head set = series carrying 80 % of volume (54 series).
- Head-set WAPE: TimesFM 3 **0.908**; best baseline `moving_average` 0.976; `seasonal_naive` 1.342; `seasonal_year` 1.616.
- Improvement over the best baseline: **+6.9 %**. Improvement over the per-series oracle: **−10.0 %**.
- The 15 % bar is **not met**. TimesFM 3 under-forecasts on this set (bias −0.25 across all series). The 80 % band
  covers 81.7 % of head-set actuals, close to nominal.
- Output: `data/backtests/twin_backtest_20261120-205503.json`. Not committed (`data/` is git-ignored).

### Evidence for the live planner defects fixed on 2026-11-20

- Stored mission status followed the goal outcome, not "completed" for every finished program. Covered by a test that
  asserts the mission row reads `stalled`.
- The goal and the tool list were lost once the evidence grew past the observation budget, because the JSON was key-sorted
  and clipped as one string. The goal now sits in its own section, and each observation is clipped on its own. Covered
  by a regression test.
- The planner now sees each tool's argument names. Before, it passed the forecast output as `forecast` to the report
  tool.

### Evidence for G14 (whole-tree scan, 2026-11-20)

- No value from the git-ignored `.env` appears in any file to be committed (count 0).
- `NIM.md` (kept out of the commit) contains no key-shaped string.
- Four test files contain key-shaped placeholders such as `nvapi-LEAKED…` and `nvapi-AAAA…`, used to prove redaction.
  They are not live keys.

## Working rules for this mission

- Use Haiku sub-agents for research workers and bounded reviews; keep design and integration in the main thread.
- No access to the user's SharePoint (standing instruction, 2026-11-20).
- Keep the flow going until everything is shipped and pushed end to end. Use the question tool only for decisions the
  user must make (standing instruction, 2026-11-20).
- Push to `origin/main` after each green gate. The user's 2026-11-20 instruction to ship and push end to end is the
  authorisation.
- Label every figure as synthetic replay, real staging, or measured. Never present synthetic numbers as observed results.
