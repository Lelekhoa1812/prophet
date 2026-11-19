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
