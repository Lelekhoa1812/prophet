# GOAL backlog (ranked by impact × confidence ÷ effort)

| # | Hypothesis | Acceptance test | Status |
|---|---|---|---|
| B1 | A same-origin guard (Host allow-list, Origin check on writes, token for `/api/reveal`) stops cross-site writes and name reveal without breaking the console | TestClient: foreign Origin POST → 403; foreign Host → 400; console calls still pass | Done loop 2 (login still absent) |
| B2 | Censoring | Measured on twin: no bias, but generator has no order suppression. Next: add a censoring option to the twin generator (orders drop when a product was out of stock the day before) and measure forecast bias and shortfall recall | Measured; generator change open |
| B3 | Leakage in daily paths | End-to-end truncation test across agents | Done loop 3: no leak found; open caveats: same-day order entry time, staging cost fallback |
| B4 | Repeat decisions / repeated learning | tests | Done loop 4 |
| B5 | Dollars-protected what-if (Liam's open decision) with honest labels: expected, not observed | Replay with a stated action-acceptance rate; output marked Synthetic replay | Awaiting decision |
| B6 | Durable jobs | restart test | Done loop 5 (file per job) |
| B9 | No CI exists: tests that need the twin never run off this laptop; add a CI workflow for the fast suite and a tiny seeded twin fixture | Workflow runs the fast suite green on a clean checkout | Workflow added loop 6; remote run result recorded in changelog; twin-dependent tests still skip in CI (seeded mini-twin fixture open) |
| B7 | Dismissal lessons with per-lesson regression test and owner approval for wide scope | Lesson create/undo/replay test | Not started |
| B8 | Calendar uplift | no gain on twin earlier; needs real multi-year history | Blocked on data |

## Resume instruction
Start at B9 or the B2 generator option. Run `.venv/bin/python -m pytest -q` (66 pass after loop 5, 17 s). Write the failing test first.

| B10 | Shortfall is net negative under the default economics. Raise its score floor or lower its alert volume until expected net is positive without losing more than 5 points of dollar recall | Policy simulator sweep with whatif() net as the objective, chronological split (tune first 60% of window, test on last 40%) | Next |
| B11 | Decision audit identity (free-text decided_by) | Named operator header, validated; decisions record it | Next |
| B11 done loop 9 | validated, labelled `unverified:` identity | 3 tests | Real login remains open (needs a decision on where authentication lives) |
| B10 done loop 10 | Per-agent ranking false-alarm cost, shortfall $1,500 (user chose per-agent over global $1,000) | held-out +17% expected net | Open: raising it for freshness too gave a further ~+11 pts on the twin; ask before applying. Shortfall remains net negative: its alerts are right only about half the time |
