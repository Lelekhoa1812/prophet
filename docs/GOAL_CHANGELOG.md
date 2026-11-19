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
