# GOAL evidence ledger

| Item | Type | Method | Result | Limits |
|---|---|---|---|---|
| Forecast cache stale on route change | Test-derived | `tests/test_forecast_cache.py::test_route_change_is_not_served_from_cache`, in-memory 2-product world, two constant models | Before: failed (served 10 not 30). After: passes | Synthetic fixture, not the twin |
| Forecast cache stale on revised history | Test-derived | `::test_changed_history_is_not_served_from_cache` | Before: failed. After: passes | same |
| Future data cannot reach the weekly forecast | Test-derived | `::test_future_data_does_not_change_the_forecast`, values from as_of on set to 999 | Passes before and after | Weekly path only |
| Existing twin cache stays usable | Synthetic replay | Sampled every 10th of 281 weekly files in `data/cache`; checked fingerprint and stored history against recomputed history | 10 of 10 accepted by the new rule | Sample, not all files; fingerprint of old files is assumed to be the default TimesFM route (no earlier code installed a router: grep of `set_router`) |
| Earlier headline numbers affected? | Code-derived | No code path called `set_router`; routing was evaluated only in `fit_routing` backtests that call models directly | Not affected | The first tenant routing install would have been |

Datasets: twin = 4-year synthetic (seeded generator); staging = about 8 weeks real, pseudonymised, not in Git.

| Cross-site write and rebound host | Test-derived + live probe | `tests/test_api_guard.py` (3 of 7 failed before), curl against restarted API | Evil Origin POST 403, Host attacker.example 400, console origins and no-Origin pass, browser same-origin POST 200 | Local checks only; no login added; a non-browser attacker on the machine is out of scope |
| No look-ahead in agents | Synthetic replay + test | `tests/test_no_future_leak.py`: 4 agents × 8 Mondays on the twin, world truncated at the as_of day vs full | Identical proposals; negative control (cut 10 days early) differs | Twin only; margin 1 proposal; skipped where `data/` is absent (so not in a fresh CI) |
| Twin censoring | Synthetic replay | Haiku sub-agent read-only SQL on data/twin.duckdb, 63,000 product-weeks; numbers re-derived from code paths (`engine/data.py`, `state.py` use order_lines.qty) | qty = demand exactly; stockout product-weeks 22%; fulfilled 7.3% below qty; top-stockout ratio to truth 0.998 | Says nothing about real customers; sub-agent figures not independently re-run by me |
| Learning ratchet | Test-derived | `tests/test_rollout.py` (3 new/rewritten), `tests/test_decision_integrity.py` (3) | before: repeated call kept moving the weight; after: no change without 20 new decisions (8 per segment) | Operators are simulated; real verdict noise untested |
