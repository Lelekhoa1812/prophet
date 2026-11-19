# GOAL changelog

## Loop 1, 19 Nov 2026
- Gap: forecast cache keyed by date only (S1).
- Root cause: `ForecastService.weekly`, `customer_fc`, `cost_fc` wrote and read `data/cache/*.pkl` by name without model or history.
- Change: entries carry the forecaster fingerprint (`Router.fingerprint`) and the history matrix; a hit needs both to match (`state.cached`); in-memory memo per service. Old entries count as the default TimesFM route.
- Test: 3 new tests (2 failed before). Full suite 47 passed (was 44).
- Before vs after: stale forecast served on route change → recomputed. Cost: a route change now recomputes that week's forecast (intended).
- Residual: fingerprint covers names and routing, not weights.
- Next: B1.
