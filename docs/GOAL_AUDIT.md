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
| S2 | API has no authentication; `GET /api/reveal` returns real names; every route is open to any local page (CSRF on POST, DNS rebinding on GET) | C: `api/main.py` has no middleware, no Origin or Host check | High before any shared deployment, low on a single-user laptop | Fixed in loop 2 for cross-site and rebound-host attacks; still no login, so not for a shared deployment |
| S3 | `decided_by` is client-supplied text, so decision audit identity is spoofable | C: `decide()` | Medium | Open |
| S4 | Read-only staging relies on a session option plus hard-coded SELECT; `assert` is stripped under `python -O`; no proof the DB role itself lacks write rights | C: `source/staging.py` | Medium | Open (needs DB role check; staging access is Liam's decision) |
| S5 | Jobs lived in process memory: a restart lost progress and results | C | Medium | Fixed in loop 5 (file per job, interrupted marking); still no cancel, no resume |
| S6 | Pickle caches under `data/` are loaded without integrity checks | C | Low (local, git-ignored); high if the directory is ever shared | Open |
| S8 | Learning re-applied the full decision history on every call (weights ratcheted to bound with no new evidence); a changed decision left a stale suppression; stats double-counted | T: 2 of 5 learning tests failed before; 3 integrity tests | High for the 'learns from corrections' claim | Fixed in loop 4 |
| S7 | Unknown dataset names return 500 on several routes instead of 404 | C | Low | Open |

## 3b. Haiku adversarial review of loops 1-3 (reconciled, loop 5)
Fixed: non-atomic cache write and corrupt-file crash (now temp file + rename, bad file recomputed); guard treated every local port as same origin (now Origin must equal the request's own host:port, and same-site fetches are refused); env host list ignored ports and schemes (normalised); `testserver` removed from the allow-list; weak guard tests tightened.
Accepted, recorded: fingerprint ignores model parameters (names, routing, licences only; a tuned model needs a new name); customer cache does not compare customer ids; `GET /api/forecast` writes cache files so a cross-site image could trigger work; the leakage test skips where `data/` is absent and the repo has no CI at all (B9); a concurrent `set_router` could mislabel one entry (no caller exists).

## 4. Falsified hypotheses and validity limits
- Twin demand censoring (loop 4, haiku sub-agent, reconciled with the code): history = order qty = true demand on the twin (5,465,171 both; fulfilled is 7.3% lower); top-stockout products ratio 0.998 of truth. This is because the generator never suppresses orders during stockouts, so the twin CANNOT test censoring. Whether real customers stop ordering when out of stock is unknown; staging (about 8 weeks) cannot tell either.
- "Weekly forecast sees data from as_of onward": falsified. Metamorphic test sets every value from as_of on to 999; history matrix and quantiles are identical (`test_future_data_does_not_change_the_forecast`). Scope: weekly forecast path. Loop 3 extended this end to end: all four agents raise identical proposals on a twin world with everything after the as_of day removed (8 Mondays; shortfall 45, freshness 341, customer 42, margin 1 proposals seen; negative control cutting 10 days early changed shortfall 3→7 and freshness 38→44). Caveats: margin has almost no power in the test; the same-day price and keying checks read orders delivered on the as_of day itself, and the data has no order-entry time, so replay assumes those orders were already entered at day start; the cost fallback `products.unit_cost` on staging is today's last cost, not the cost then.

## 5. Evidence limits that stay in force
Twin results are method demos. Staging has about eight weeks. The ship bar (15% better than the best baseline) was not met earlier (about 17% only against one baseline on the twin, none on real data). TimesFM weights are non-commercial. No dollar here is an observed saving.
