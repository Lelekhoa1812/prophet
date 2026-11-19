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
| B7 | Dismissal lessons with per-lesson regression test and owner approval for wide scope | Lesson create/undo/replay test | Done loop 19 (API + audit); console picker/approve button open |
| B8 | Calendar uplift | no gain on twin earlier; needs real multi-year history | Blocked on data |

## Resume instruction
Start at B9 or the B2 generator option. Run `.venv/bin/python -m pytest -q` (66 pass after loop 5, 17 s). Write the failing test first.

| B10 | Shortfall is net negative under the default economics. Raise its score floor or lower its alert volume until expected net is positive without losing more than 5 points of dollar recall | Policy simulator sweep with whatif() net as the objective, chronological split (tune first 60% of window, test on last 40%) | Next |
| B11 | Decision audit identity (free-text decided_by) | Named operator header, validated; decisions record it | Next |
| B11 done loop 9 | validated, labelled `unverified:` identity | 3 tests | Real login remains open (needs a decision on where authentication lives) |
| B10 done loop 10 | Per-agent ranking false-alarm cost, shortfall $1,500 (user chose per-agent over global $1,000) | held-out +17% expected net | Open: raising it for freshness too gave a further ~+11 pts on the twin; ask before applying. Shortfall remains net negative: its alerts are right only about half the time |
| B2 follow-up | Sweep censoring p in {0.2,0.4,0.6,0.8} and lag {1,2,4}; measure recall loss | `build censored` variants + replay | Done (5 variants, p 0.2-0.8, lag 2-4): recall loss is graceful, at most about 4 points at p0.8 or lag 4. Lag 1 and combined extremes not run. Real-data check still open |
| B12 | Dedicated read-only staging login (member of `axis_semantic_reader` only) and an `axis_semantic.products` view (or grant) | `role_audit` reports can_write false; a write attempt fails at the database; extract still returns the same row counts | Needs Liam's approval and an upstream grant |
| B12 decision | Liam, 19 Nov: accept the risk for now (current login, console warning, stricter guard) | revisit before any shared or client use | Accepted risk |
| B13 | Dollar recalibration | held-out net | Rejected loop 13 (-0.1%); duplicate-alert collapse (one problem, one claim) is the better lever for executive totals | Open: report problem-level dollars (distinct stockout events alerted) instead of alert-level sums |
| B14 | Customer and margin have no dollar truth on the twin, so their dollar outcomes are unmeasured and excluded from what-if | Define a dollar truth per agent from truth_events or daily_outcomes, then report customer and margin dollar recall with the same replay | Next (user chose 'mark acts unmeasurable, move on', loop 23). Not started |

## Resume instruction (loop 23)
Next: B14. Read the twin's truth_events and truth_expected schema, define dollar truth for margin (price drift and below-floor keying) and customer (lapse and slowdown), then replay with the existing `exp_*` pattern. Acceptance: a dollar recall per agent with a labelled method, not an expected value.
Open, not runnable here: API login, staging reader role (B12, accepted risk), screen-reader pass, live shadow run (not decided), TimesFM licence.
