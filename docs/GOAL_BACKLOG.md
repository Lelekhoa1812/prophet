# GOAL backlog (ranked by impact × confidence ÷ effort)

| # | Hypothesis | Acceptance test | Status |
|---|---|---|---|
| B1 | A same-origin guard (Host allow-list, Origin check on writes, token for `/api/reveal`) stops cross-site writes and name reveal without breaking the console | TestClient: foreign Origin POST → 403; foreign Host → 400; console calls still pass | Done loop 2 (login still absent) |
| B2 | Stockout-censored demand is under-counted in the twin and in staging history (ordered qty is used, so censoring is partial) | Measure share of weeks where short_qty > 0 per series and bias of ordered vs true demand on the twin | Not started |
| B3 | Leakage in daily paths (weekday profile, stock_at, pipeline_at, unit_cost_at fallback uses today's cost) | Metamorphic tests for each: perturb data at/after as_of, outputs unchanged | Not started |
| B4 | Repeat decisions on one proposal double-count in learning and create duplicate lessons | Test: second decision on same proposal rejected or idempotent | Not started |
| B5 | Dollars-protected what-if (Liam's open decision) with honest labels: expected, not observed | Replay with a stated action-acceptance rate; output marked Synthetic replay | Awaiting decision |
| B6 | Durable jobs (persist to lab DB, mark interrupted on restart) | Kill-and-restart test shows status `interrupted` | Not started |
| B7 | Dismissal lessons with per-lesson regression test and owner approval for wide scope | Lesson create/undo/replay test | Not started |
| B8 | Calendar uplift | no gain on twin earlier; needs real multi-year history | Blocked on data |

## Resume instruction
Start at B3 (leakage metamorphic tests), then B4. Run `.venv/bin/python -m pytest -q` (54 pass after loop 2). Write the failing test first.
