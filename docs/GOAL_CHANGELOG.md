# GOAL changelog

## Loop 1, 19 Nov 2026
- Gap: forecast cache keyed by date only (S1).
- Root cause: `ForecastService.weekly`, `customer_fc`, `cost_fc` wrote and read `data/cache/*.pkl` by name without model or history.
- Change: entries carry the forecaster fingerprint (`Router.fingerprint`) and the history matrix; a hit needs both to match (`state.cached`); in-memory memo per service. Old entries count as the default TimesFM route.
- Test: 3 new tests (2 failed before). Full suite 47 passed (was 44).
- Before vs after: stale forecast served on route change → recomputed. Cost: a route change now recomputes that week's forecast (intended).
- Residual: fingerprint covers names and routing, not weights.
- Next: B1.

## Loop 2, 19 Nov 2026
- Gap: S2, any web page could POST to the local API; a rebound hostname could read `/api/reveal`.
- Change: middleware refusing non-loopback Host (400) and cross-site writes (403); `TIMESFM_ERP_ALLOWED_HOSTS` for deliberate hosts.
- Test: `tests/test_api_guard.py` (7), suite 54 passed; live curl and browser checks.
- Side effect to disclose: before the guard existed, the first test run reset the real twin replay cursor (`/api/time/reset`). Tests now use a dummy dataset. The cursor is demo state and can be stepped again.
- Residual: no authentication, `decided_by` still free text (S3).
- Next: B3.

## Loop 3, 19 Nov 2026
- Hypothesis: agents see data after the as-of day. Result: falsified (no code change). Added `tests/test_no_future_leak.py` as a guard; suite 55 passed.
- Open caveats recorded in the audit (same-day order entry time; staging cost fallback).
- Next: B4 (repeat decisions on one proposal).

## Loop 4, 19 Nov 2026
- Gap: learning re-applied the same decisions each call; a changed decision left a stale lesson; stats double-counted.
- Change: `rollout.learn` records consumed decision ids and steps only on 20 new agent decisions (8 per segment); API sends latest decision per proposal with ids; `decide` retires lessons from earlier decisions on that proposal and always resets the governor.
- Test: suite 60 passed (was 55). Old test that looped on identical rows and called hitting the bound correct was rewritten.
- Residual: the console's Learn button now often returns no changes; the UI text does not yet say why (candidate for a copy fix). Reviewer verdicts are still free-form `decided_by`.
- Next: B2 generator option, B6 durable jobs.

## Loop 5, 19 Nov 2026
- Gaps: S5 jobs lost on restart; haiku adversarial review findings (non-atomic cache write, guard accepted other local ports, env host parsing).
- Change: `api/jobs.py` persists jobs to `data/jobs`, marks running ones interrupted on start, writes the final state to disk before exposing it; cache writes atomic and corrupt files recomputed; guard tightened.
- Test: suite 66 passed (was 60). Live: browser same-origin POST 200.
- Note: the Learn endpoint has no console button (loop 4 residual about UI copy withdrawn).
- Next: B9 CI, or B2 generator censoring option.

## Loop 6, 19 Nov 2026
- Gap: no CI; tests silently depended on local `data/`.
- Change: `.github/workflows/tests.yml` (installs everything except `timesfm[mlx]`, package with `--no-deps`); `httpx` in dev extras; one test no longer needs the twin.
- Test: clean venv with an empty `data/`: 65 passed, 1 skipped (the twin leakage test). Full local: 66 passed.
- Residual: CI cannot exercise agents or the leakage guard until a small seeded twin fixture exists.
- Remote: GitHub Actions run 44517852886 on 408956e passed (fast job, Ubuntu, no model, no data).

## Loop 7, 19 Nov 2026 (user chose: separate censored twin, then CI mini-twin, what-if, audit identity)
- Gap (B2): the twin cannot test stockout censoring (orders = latent demand exactly).
- Change: `twin/censor.py`, `build.py` option, dataset `twin_censored` (python -m lab.twin.build censored). Base twin untouched.
- Result: censoring at an assumed p=0.4 lowers recorded orders 2.7% and the median forecast a further 4.8 points in the worst-stockout fifth; replay on `twin_censored` pending (see below).
- Gap (B9 part 2): CI could not run agents. Change: `tests/test_no_future_leak_mini.py` builds a 60-product seeded twin in the test with a cheap router (no TimesFM), plus a planted-leak negative control. Clean checkout with empty `data/`: 71 passed, 1 skipped. Full local 72 passed.

## Loop 8, 19 Nov 2026
- Gap (B5): no honest answer to 'what is it worth'.
- Change: `policy/whatif.py`, `GET /api/policy/whatif`, console card on the Developer page (phone checked, no overflow). Customer and margin listed as unmeasured, not netted.
- Test: 7 new tests; measured at defaults: see evidence. Finding: claimed dollars overstate real problem dollars 2.3-2.5x; shortfall is net negative at 50% acceptance (B10).

## Loop 9, 19 Nov 2026
- Gap (S3, B11): audit identity was free text. Change: validated name, `X-Operator` header, `unverified:` prefix. Tests: 3 new, suite 82 passed.
- Residual: identity is asserted not authenticated.

## Loop 10, 19 Nov 2026
- Gap (B10): shortfall alerts net negative under default economics.
- Decision (user, AskUserQuestion): per-agent cost, shortfall only.
- Change: `Economics.ranking_false_alarm_cost` (ranking and gating only; the what-if still values with the stated $200); default {"shortfall": 1500}; old tenant files inherit it. Tests: 2 new, 2 existing made independent of the override; suite 84 passed.
- Result: held-out expected net +17%, shortfall alerts -41%, shortfall dollar recall -7.3 pts (see evidence).
- Residual: stored scorecard stale until a full twin replay is rerun; twin-tuned default must be re-checked per client in readiness.

## Loop 11, 19 Nov 2026
- User approved raising the freshness ranking cost. Default is now shortfall $1,500 and freshness $1,500 (ranking only; the what-if still values at $200). Held-out expected net +28% vs the old default. Suite run below.
- Full twin replay under the new defaults persisted; stored scorecard now matches the policy. Shortfall now catches fewer stockouts (41% of events, 59.9% of dollars) in exchange for fewer wrong alerts; freshness improves on every measure.

## Loop 12, 19 Nov 2026
- Gap (S4): read-only staging was a client promise. Audit found the login can write everywhere.
- Change: `check_select` (raises, single statement, write words refused), READ ONLY transactions, `role_audit` stored in dataset meta, exposed in `/api/meta`, console warning. Tests: 9 new, suite passes.
- Residual: the role itself is unchanged (needs a reader login, B12). Existing safeguards limit the lab, not a mistaken or malicious caller with these credentials.

## Loop 13, 19 Nov 2026
- Hypothesis: alert dollar claims are miscalibrated and recalibrating helps. Result: per-alert claims are close; shown-alert shortfall gap is selection; recalibration gives -0.1% held-out net. Rejected, no code change. Next lever: problem-level (de-duplicated) dollars for executive totals.
- Loop 12 decision recorded: Liam accepted the staging role risk for now (B12).
- Loop 13b: what-if card now shows one claim per problem beside the alert-level sum (`problem_level`, 1 new test; suite passes; browser check, no overflow). Executives should quote the per-problem figure.

## Loop 14, 19 Nov 2026
- Gap: rule 7 (prove it continuously): CI checked correctness, not decision quality; a change that halved what the agents catch would pass.
- Change: shared `mini` fixture in `tests/conftest.py`; `tests/test_eval_gate.py` replays 90 days on the mini twin and fails below recorded precision, recall and volume floors, with a control that silences shortfall.
- Rule: if a deliberate change moves a number, re-measure, change the floor in the same commit and say why here.
- Remote: GitHub Actions run 44523550916 on 9f1f6e3 concluded as above (fast suite plus evaluation gate, no model, no data).

## Loop 15, 19 Nov 2026
- Gap (lens H): no accessibility audit. Automated route scan found target sizes below 24 px and heading skips. Fixed in the console (links, switches, h2/h3 levels). Re-scan: no skips except none remaining on 7 routes, small targets only the native slider.
- Residual: manual keyboard, screen-reader and dark-mode contrast checks not done.

## Loop 16, 19 Nov 2026
- Gap: contrast never measured. Measured, found light and dark failures, fixed with colour tokens, re-measured clean (text only). Methodological note: measure with transitions disabled in the background browser pane.
- Loop 16b: keyboard walk-through found no skip link and unnamed repeated links; both fixed (WCAG 2.4.1, 2.4.4).

## Loop 17, 20 Nov 2026
- B2 follow-up: censoring sweep finished. Shortfall recall degrades gradually (41.0% base to 37.2% at the harshest tested), no cliff; precision flat. Synthetic only.
- Loop 18: audit of the value paper's 'learns' capability found lessons matched days before they were created and used the wall clock. Fixed with two tests (98 pass). Still not built from the paper: lesson scope choice (this order / customer / everyone) and owner approval for wide lessons (B7).
- Loop 19: design rule 5. Agent-wide lessons are pending until another person approves and `audit_lesson` is clean; capped at the dismissed alert's dollars. 100 tests pass; mutation check fails the tests as intended.
- Loop 20: console support for rule 5: Dismiss menu has an 'apply to the whole agent' checkbox; Lessons panel shows waiting lessons with a name field and Approve button (API error shown). Type-check and build clean; Dismiss menu checked in the browser pane. The Approve path was not clicked through in the browser (that would write to the real lab database); the API flow is covered by tests.
- Loop 21: earns-autonomy audit. The level changes no behaviour in the lab, so the API now refuses levels above 'prepares' and the console disables them. Test added; 101 pass. The label is still not enforcement.
- Loop 22: TIMESFM_ERP_LAB_DB env override (default unchanged) lets the console run on a throwaway lab database. Approve flow tested in the browser: same-person refused, different person approved, audit 3 checked / 2 would suppress. Real lab DB untouched (verified 0 rows). Full suite 101 passed.
- Loop 23: staging feasibility for 'acts before asked'. 31 delivery days, 14 Sep to 6 Nov 2026. A replay can run as a smoke test but cannot support a performance claim. Recorded; no code change.
- Loop 23 decision (user): acts-before-asked marked unmeasurable; next is B14, customer and margin dollar truth on the twin. Not started.
- Loop 24 (B14): customer and margin dollars from order lines (estimate basis). Unvalued events excluded from dollar totals. Customer dollar recall 84% of $593k estimate; margin 100% of $198k, concentrated in three cost spikes. Estimates kept out of the what-if headline. 109 tests pass.
- Loop 25: CI for 0948e9a and 2631976 failed with zero steps run; a clean clone of origin/main passes the same commands (108 pass, 1 skipped), so the failure is at the runner. Re-run requested. Scorer now reports dollar_recall in every replay result. Earlier claim of 'missing replay dollar stats' was my script reading the wrong key; corrected.
- Loop 25 (CI): 0948e9a and 2631976 never started. The runner annotation says 'recent account payments have failed or your spending limit needs to be increased'. Code is fine: clean clone passes 108 and 1 skipped. Needs the account owner to fix billing; not something the loop can change.
- Loop 25 (B14 follow-up): shown-alert dollar recall under the default policy recorded in GOAL_EVIDENCE.md.
- Loop 26: read the value paper's script and checked its 16 cited URLs (13 resolve, 3 blocked). Claims in the page are not verified against the sources. Recorded in GOAL_AUDIT section 10.
- Loop 27: consolidated the strategic-promise matrix into GOAL_AUDIT section 11; every referenced code path confirmed present. No code change.
- Loop 28: full visible prose of the value paper read. Two claims with no behaviour recorded (automatic demotion; per-action hard limits). No code change.
- Loop 29: checked the 16 cited articles for their key figures. All 16 carry them. Two quotations (Anil Thawani, OnlineMedEd) remain unsourced. Recorded in GOAL_EVIDENCE and GOAL_AUDIT section 13.
- Loop 30: searched the two unsourced quotes. Both are Rillet vendor material; no independent source. Recorded.
- Loop 31: orientation reads completed (goal document in full; HTML rule, capability and move data checked against the audit). No code change.
- Loop 32: value paper interactions clicked in a browser (about 128 of 213 controls): no JavaScript errors. Partial coverage. Recorded in GOAL_AUDIT section 15.
- Loop 33: re-checked CI. Still blocked: the runner says account payments failed or the spending limit must be raised (run 23:03 UTC on c76a80c, 0 steps). Re-ran the CI steps on a fresh clone of c76a80c: 108 passed, 1 skipped. Account billing cannot be changed from the repo.
- Loop 34 (20 Nov 2026): GitHub Actions workflow removed (billing blocked, Liam decision). Replaced by `scripts/check.sh` (pytest plus console typecheck and build): 109 passed locally, console build ok. TimesFM smoke test: weights load in 1.4 s, 8-step forecast returns point and 9 quantiles. CLAUDE.md added. B9 now means local gate, not remote CI.
- Loop 35 (20 Nov 2026): TimesFM audit run, results in GOAL_EVIDENCE. Twin gain 17.4% vs 7.3% recorded in PLAN.md (unreconciled); censored twin 13.5%. Staging too short. No code change.
- Loop 36 (20 Nov 2026): freshness precision investigation. Found a scorer window-edge defect: expiry truth was cut at the window end although the twin runs to 6 Nov, so late alerts were marked false. Fixed (`sim/scorer.py`; new test `tests/test_scorer_maturity.py`; 110 pass). Remaining September and October dip (about 82% against 89 to 90%) traced to forecast under-forecasting of rising demand: actual sell-through 16 to 17% above forecast since September. Not yet fixed; decision pending (B16).
