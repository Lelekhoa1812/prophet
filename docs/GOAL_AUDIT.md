# GOAL audit: strategy promise → code → tests → evidence → gap

Reviewed 19 Nov 2026 at commit 5a11bc9 (+ loop 1). Evidence types: Code-derived (C), Test-derived (T), Synthetic replay (S), Real staging retrospective (R), Proposal/assumption (P).
The strategy page `docs/ai-native-value.html` (read in full: prose, 8 capabilities, 3 moves, 7 rules titles in the JS, 5 decisions, demo caveats; its own header says figures are illustrative, not an AXION position) is a vision, not a spec. It is kept out of Git because it is marked Commercial in Confidence and this repo sits on a personal GitHub account.
Hidden in the HTML and not reviewed as functionality: the interactive autonomy dial (Part 1f) and "AXIS across products" (Part 2a), both switched off by `SHOW_HIDDEN`.

## 1. Eight capabilities vs this repo

| Capability (HTML) | Status | Code | Tests | Evidence | Gap / where it belongs |
|---|---|---|---|---|---|
| Adapts per business and role | Partial/prototype | `policy/rollout.py` tenant JSON, `engine/profile.py` percentile thresholds | `test_rollout.py` | S only | No role- or tenant-composed views; console is one fixed layout. Views belong upstream (AXIS). |
| Learns from mistakes | Partial/prototype | `policy/learning.py` SegmentLearner, `/api/tenant/learn`, dismissal lessons in `api/main.py` `decide` | `test_learning.py` | S: +3.2 pts shortfall precision from a simulated, noise-free operator | No real operator verdicts exist. Lessons have scope but no approval step or regression test per lesson. Repeat decisions on one proposal are not rejected. |
| Ask and search anything | Vision-only here | none (narration of one proposal only) | – | – | Belongs in AXIS. Not built here on purpose. |
| Acts before being asked | Implemented, inadequately verified | `agents/*`, `sim/replay.py` | agent logic via replay only | S | Daily replay on a synthetic twin; never run live or in shadow on real data. |
| Does the work end to end | Not applicable | – | – | – | Lab is read-only and recommends. Writes need FoodFlow governed actions. |
| Earns autonomy | Partial/prototype | `policy/governor.py` LADDER, `ladder_status` | none found | C | Ladder is advisory display. `PUT /api/agents/{agent}` accepts any level with no effect on behaviour, and no auth. |
| Thinks in food | Partial/prototype | per-sample FEFO `agents/freshness.py`, shelf-life classes, pack sizes `agents/shortfall.py` | `test_distribution.py` | S | Catch-weight, substitutes, contract prices, delivery runs not modelled. |
| Trustworthy by design | Partial/prototype | pseudonymisation `source/pseudo.py`, read-only session `source/staging.py`, number check on narration `narration/bedrock.py` | narration number check only | C | See section 3: no API auth, free-text `decided_by`, read-only not proven at DB role level. |

## 2. Three moves and seven rules (challenge criteria)
- Living model of the business: typed local tables only; no event stream; `proposals` are the only machine-readable object. Partial.
- Agents that do and learn: proposals + decisions + lessons exist; corrections become suppressions, not lessons with evidence. Partial.
- Autonomy under control: three-layer limits, demotion and undo are not implemented; only a ladder display. Vision-only.
- Rules (contract-first actions, events, per-action policy, replayable identity/audit, scoped reversible lessons, composed views, evaluation gates): the repo has evaluation gates (readiness, drift, scorer) and reversible lessons (`/api/lessons/{id}/undo`); it has no action contracts, no agent identity and no per-action policy.

## 3. Security and operations findings
| ID | Finding | Evidence | Severity | Status |
|---|---|---|---|---|
| S1 | Forecast cache keyed by date only: a model or route change, or a corrected history, served the old forecast | T: `tests/test_forecast_cache.py` failed 2 of 3 before the fix | High for any model comparison | Fixed in loop 1 |
| S2 | API has no authentication; `GET /api/reveal` returns real names; every route is open to any local page (CSRF on POST, DNS rebinding on GET) | C: `api/main.py` has no middleware, no Origin or Host check | High before any shared deployment, low on a single-user laptop | Open (backlog B1) |
| S3 | `decided_by` is client-supplied text, so decision audit identity is spoofable | C: `decide()` | Medium | Open |
| S4 | Read-only staging relies on a session option plus hard-coded SELECT; `assert` is stripped under `python -O`; no proof the DB role itself lacks write rights | C: `source/staging.py` | Medium | Open (needs DB role check; staging access is Liam's decision) |
| S5 | Jobs live in process memory: a restart loses progress and results; no cancel | C: `api/jobs.py` | Medium (lab convenience) | Open |
| S6 | Pickle caches under `data/` are loaded without integrity checks | C | Low (local, git-ignored); high if the directory is ever shared | Open |
| S7 | Unknown dataset names return 500 on several routes instead of 404 | C | Low | Open |

## 4. Falsified hypotheses
- "Weekly forecast sees data from as_of onward": falsified. Metamorphic test sets every value from as_of on to 999; history matrix and quantiles are identical (`test_future_data_does_not_change_the_forecast`). Scope: weekly forecast path only. Daily weekday profile, stock and pipeline were read by eye (all strictly before as_of), not yet by a test.

## 5. Evidence limits that stay in force
Twin results are method demos. Staging has about eight weeks. The ship bar (15% better than the best baseline) was not met earlier (about 17% only against one baseline on the twin, none on real data). TimesFM weights are non-commercial. No dollar here is an observed saving.
