# GOAL evidence ledger

| Item | Type | Method | Result | Limits |
|---|---|---|---|---|
| Forecast cache stale on route change | Test-derived | `tests/test_forecast_cache.py::test_route_change_is_not_served_from_cache`, in-memory 2-product world, two constant models | Before: failed (served 10 not 30). After: passes | Synthetic fixture, not the twin |
| Forecast cache stale on revised history | Test-derived | `::test_changed_history_is_not_served_from_cache` | Before: failed. After: passes | same |
| Future data cannot reach the weekly forecast | Test-derived | `::test_future_data_does_not_change_the_forecast`, values from as_of on set to 999 | Passes before and after | Weekly path only |
| Existing twin cache stays usable | Synthetic replay | Sampled every 10th of 281 weekly files in `data/cache`; checked fingerprint and stored history against recomputed history | 10 of 10 accepted by the new rule | Sample, not all files; fingerprint of old files is assumed to be the default TimesFM route (no earlier code installed a router: grep of `set_router`) |
| Earlier headline numbers affected? | Code-derived | No code path called `set_router`; routing was evaluated only in `fit_routing` backtests that call models directly | Not affected | The first tenant routing install would have been |

Datasets: twin = 4-year synthetic (seeded generator); staging = about 8 weeks real, pseudonymised, not in Git.
