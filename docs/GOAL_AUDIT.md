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
| Acts before being asked | Implemented, inadequately verified | `agents/*`, `sim/replay.py` | agent logic via replay only | S | Daily replay on a synthetic twin; never run live or in shadow on real data. Loop 23: staging has 31 delivery days (14 Sep to 6 Nov 2026), too few for a replay to support any precision or recall claim. Only a live shadow run or longer history can test it. |
| Does the work end to end | Not applicable | – | – | – | Lab is read-only and recommends. Writes need FoodFlow governed actions. |
| Earns autonomy | Partial/prototype | `policy/governor.py` LADDER, `ladder_status`; ceiling `MAX_LAB_LEVEL` in `api/main.py` | `test_lab_cannot_be_set_above_prepares` | C | Loop 21: the level is a label only. Grep finds no code path that reads `autonomy_level` to change behaviour; the lab only proposes. The API now refuses `acts`/`runs` and the console disables them, so the label cannot overstate what the lab does. Promotion has no effect and no auth. |
| Thinks in food | Partial/prototype | per-sample FEFO `agents/freshness.py`, shelf-life classes, pack sizes `agents/shortfall.py` | `test_distribution.py` | S | Catch-weight, substitutes, contract prices, delivery runs not modelled. |
| Trustworthy by design | Partial/prototype | pseudonymisation `source/pseudo.py`, read-only session `source/staging.py`, number check on narration `narration/bedrock.py` | narration number check only | C | See section 3: no API auth, free-text `decided_by`, read-only not proven at DB role level. |

## 2. Seven design rules vs this repo (rule text read from the HTML's own `PR` data)

| # | Rule (HTML) | What the repo has | Status | Missing, and where it belongs |
|---|---|---|---|---|
| 1 | Design the action before the screen: each action defined once with inputs, limits, evidence, undo | `Proposal.action` dict (`type`, lots, etc.) with evidence; `/api/proposals/{id}/decision`; lessons undo | Partial | No action schema, limits or undo for the action itself (lab never acts). Contracts belong in AXIS governed actions; the lab can only emit proposals shaped like them |
| 2 | Describe the business so a machine can read it | `store.py` canonical tables (shelf life, catch-weight flag, lots, use-by), `engine/profile.py` tenant profile | Partial | No contract floors, units conversion, delivery runs. Upstream FoodFlow model |
| 3 | Autonomy is policy, not code (3 layers) | `agent_config.autonomy_level` stored per agent; `ladder_status` promotion/demotion criteria | Partial | Ladder is display only; no hard limit/business/person layers; nothing is enforced because the lab cannot act |
| 4 | A trace that can be replayed | `runs`, `proposals.evidence`, `decisions`, deterministic `as_of` replay; loop 1 made cached forecasts keyed by model and history; leakage tests | Partial, strengthened | No event stream; decision identity is `unverified:`; model weights not versioned in the fingerprint |
| 5 | Learning is a product feature (scope, owner, approval, test) | scoped suppression lessons with undo; bounded, logged weight learning with watermark (loop 4) | Partial, strengthened (loop 19) | Wide lessons need a different approver and a clean audit (API only, no console control); entity lessons still need none; verdicts only from simulated operators |
| 6 | The screen is a view (composed per business and role) | One console with Executive / Operator / Developer / Client demo lenses over the same API | Partial | No composition from approved parts; belongs in AXIS |
| 7 | Prove it continuously (evaluations gate every change) | readiness backtest, drift check, scorer, CI for fast tests (loop 6), evaluation gate (loop 14) | Partial, strengthened | CI gates correctness but not decision quality: no evaluation regression gate (loop 14 target) |

## 2b. Eight questions (HTML chapter 3) answered for the lab
1 Built for people or AI first: lab outputs are machine-readable proposals; the console is a view (Partial). 2 What happens when nobody asks: agents run on a daily replay clock, never live (Prototype). 3 How AI acts: it does not act, it proposes (by design). 4 Knowledge of the business: tenant profile and canonical tables (Partial). 5 Correction: becomes a suppression lesson or a bounded weight (Partial). 6 Limits: ladder display only (Prototype). 7 Fit per business: tenant economics and percentile thresholds, not composed screens (Partial). 8 Gets smarter: monthly drift/re-tune planned, not run live (Prototype).
The HTML says to claim AI Native only per capability once it passes the eight questions; nothing in this lab passes all eight, so no lab output may be described as AI Native.

## 2c. Three moves
Living model of the business: tables only, no events (Partial). Agents that do and learn: propose and learn from simulated verdicts (Partial). Autonomy under control: Vision-only here.

## 3. Security and operations findings
| ID | Finding | Evidence | Severity | Status |
|---|---|---|---|---|
| S1 | Forecast cache keyed by date only: a model or route change, or a corrected history, served the old forecast | T: `tests/test_forecast_cache.py` failed 2 of 3 before the fix | High for any model comparison | Fixed in loop 1 |
| S2 | API has no authentication; `GET /api/reveal` returns real names; every route is open to any local page (CSRF on POST, DNS rebinding on GET) | C: `api/main.py` has no middleware, no Origin or Host check | High before any shared deployment, low on a single-user laptop | Fixed in loop 2 for cross-site and rebound-host attacks; still no login, so not for a shared deployment |
| S3 | `decided_by` is client-supplied text, so decision audit identity is spoofable | C: `decide()` | Medium | Open |
| S4 | Read-only staging relied on a client session option plus a first-word check inside an `assert` | C + live catalog audit 19 Nov: the role is a member of `neon_superuser` (plus `foodflow_reporting`, `foodflow_ask_data`, `axis_semantic_reader`), has createdb/createrole/bypassrls/replication, owns 1,463 relations and can INSERT/UPDATE/DELETE on 209 of 209 visible tables | High: protection by convention only; the lab itself ran no writes | Partly fixed loop 12: guard is a real check (single statement, write words and INTO refused, not an assert), transactions start READ ONLY, role audit stored with the dataset and shown as a console warning. Root cause open: needs a reader login |
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

## 6. Loop 8 finding: claimed dollars overstate (executive risk)
Summing `dollars_at_stake` over alerts overstates the real problem 2.5x (shortfall $1.25M claimed vs $510k real stockout dollars in the window) and 2.3x for freshness ($9.7M vs $4.2M). One problem can be alerted on several days. Loop 13 refinement: per alert the claims are close (realised/claimed 0.89 freshness, 0.76 shortfall on shown alerts, and 1.27 across all candidates, so the shown-alert shortfall gap is mostly selection), and recalibrating shortfall dollars gave no held-out gain. The aggregate overstatement is mainly duplication, so the warning stands: never sum claimed dollars. The what-if card shows both figures side by side. Under the default assumptions the shortfall agent is net negative at 50% acceptance, so its false-alarm cost, not its recall, is what to tune next (backlog B10).

Loop 9 (S3): `decided_by` is now validated (1 to 64 plain characters), an `X-Operator` header beats the body, and the stored value reads `unverified:<name>`. This labels the identity, it does not prove it: without a login anyone can claim any name. Real identity needs authentication in front of the lab or in FoodFlow.

Loop 11, censoring (B2 closed for the assumed strength): the forecast under-reads latent demand more (up to 4.8 points in the worst-stockout fifth), but alerts caught only about 1 point fewer stockouts. Not evidence for real clients; a sweep of p and lag is the next test once a client history shows whether orders drop after stockouts.

## 7. Staging role facts (loop 12, catalog reads only, no connection detail recorded)
- `axis_semantic_reader` exists: not superuser, no bypassrls, zero write privileges on all 209 relations, SELECT on all 8 curated views the lab reads.
- It has no SELECT on `public.products`, the one raw table the lab reads (shelf life, catch-weight flag, cost, price). Using it needs either a grant or an `axis_semantic` products view (an upstream FoodFlow change).
- `SET ROLE axis_semantic_reader` is refused for the lab's current login, so privileges cannot be dropped in-session.
- Therefore the fix is a dedicated login that is only a member of the reader role, plus the products view. That is an access change for Liam to approve; nothing was written to staging.

## 8. Accessibility and UX audit (loop 15, automated checks on all 7 console routes, browser pane)
Passed: every button and link has a name, every input has a label, one h1 per page, `main` and `nav` landmarks, `lang="en"`, no images without alt, no table without header cells, no horizontal overflow at 375 px on the pages checked.
Fixed: 60 "Open" links on Proposals were 57x20 px (WCAG 2.2 target size minimum is 24x24): now at least 24 px tall. Four agent enable switches were 36x20: now 44x24. Heading levels skipped from h1 to h3 on every page (card titles were h3, proposal titles h3, Rollout sub-headings h4): card and proposal titles are now h2 and Rollout sub-headings h3.
Code-checked: a global `:focus-visible` outline exists (`index.css`), `prefers-reduced-motion` is handled, `outline-none` appears only on the dialog container. Loop 16 contrast: muted text, status red/green/amber and the dark-mode primary button failed AA; fixed with text-safe tokens (`--muted-foreground`, `--status-*`, dark overrides, dark `--primary-foreground`); 0 failing text elements on 7 routes x 2 themes. Loop 16 keyboard: focus order and outlines fine on the Briefing; added a skip-to-content link and descriptive names for repeated Open links. Not checked: a screen reader on the briefing, other pages' keyboard flows, the Dismiss menu with arrow keys, non-text contrast (charts, borders), the Explorer horizon slider (native control, 20 px). These need a manual pass.
