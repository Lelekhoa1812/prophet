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
