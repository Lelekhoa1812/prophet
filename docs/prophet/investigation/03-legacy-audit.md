# 03 Legacy Audit: `prophet` repo (package `lab`)

Scope: read-only audit of `/Users/liamle/Downloads/prophet` ahead of restructuring into Prophet, a domain-independent autonomous forecasting platform. No existing file was edited. Secret values were not printed; only names, paths and counts are recorded.

Evidence labels: **Verified** (read in code or confirmed by command), **Doc** (stated in a repo doc, not yet checked against code), **Inferred** (reasoned from code, not tested).

Totals: `src/lab` is 5,166 lines of Python across 39 modules (excluding `__init__.py`). `tests/` has 20 test files plus `conftest.py`, collecting 110 tests (`pytest --co`). CLAUDE.md says 109.

---

## 1. Module table

Verdicts: KEEP (take as-is into Prophet core), ADAPT (keep the pattern, rewrite the coupling), REMOVE (ERP or Victorian specific, replace with an injectable port).

| Path (under `src/lab/`) | Lines | Purpose (verified from code) | ERP-specific? | Verdict | Dependents (imports, verified by reading) |
|---|---|---|---|---|---|
| `config.py` | 40 | Paths, dataset registry, lab DB path, Bedrock region and model, $30 hard cap, env-file reader. | Partly (staging env path, Bedrock model) | ADAPT | Nearly all modules |
| `store.py` | 110 | DuckDB schemas for datasets (12 tables) and the lab DB (9 tables); `open_dataset`, `open_lab`. | Yes (order_lines, lots, purchase_lines, credits) | ADAPT | engine, sim, api, source |
| `calendar_vic.py` | 74 | Hard-coded Victorian public holidays 2022–2026, Easter 2022–2027, school holidays; `flags(days)`, `calendar_channel`. | Yes (Victoria specific) | REMOVE (replace with injectable calendar port) | backtest, digest tests |
| `engine/forecast.py` | 68 | Process-wide TimesFM 3 (MLX) singleton behind a lock; `predict`, `predict_cov`; `Forecast(point, q)` with 9 quantiles. | No (model wrapper) | KEEP (change checkpoint and import config) | models.py, api/main.py, backtest |
| `engine/models.py` | 236 | Model-agnostic adapter: `ForecastModel` protocol, `TimesFM`, `Baseline`, `TSB`, `Ensemble`, `series_class`, `Router`, `fit_routing`, `default_registry`. | No | KEEP | state, backtest, api, tests |
| `engine/distribution.py` | 93 | One-factor Gaussian copula (`Draws.make`), `week_samples`, `daily_samples`, `expected_shortfall`, `expected_leftover`, `prob_exceeds`. | No (zero coupling, verified) | KEEP | agents, state |
| `engine/baselines.py` | 76 | `moving_average`, `seasonal_naive`, `seasonal_year`, `run`; z-table bands. | No | KEEP | backtest, models |
| `engine/metrics.py` | 35 | `wape`, `bias`, `pinball` (normalised), `coverage`, `mase`. | No | KEEP | backtest, tests |
| `engine/data.py` | 100 | `Demand` dataclass; `load_demand` (SQL on order_lines and products.category), `load_promo_covariate` (truth_events 'PROMO'), `load_expected`. | Yes (SQL is ERP and twin specific) | ADAPT (generic Demand matrix plus adapter) | backtest |
| `engine/state.py` | 326 | `WorldData` (ERP tables in memory), `get_world`, `weekly_matrix`, `stock_at`, `pipeline_at`, `unit_cost_at`, `cached` (fingerprinted disk cache), `ForecastService` (weekly, weekday_profile, daily_path, demand_samples), customer and cost forecasts. | Yes (mixes generic forecast service with ERP world model) | ADAPT (split in two) | agents, api, sim |
| `engine/backtest.py` | 153 | Walk-forward `run_backtest`, head/tail split at 80% revenue, oracle baseline, ship verdict (15% bar, coverage 0.70–0.90), writes `backtests` table. | Partly (calls ERP loader, calendar) | ADAPT (injectable covariates) | api |
| `engine/profile.py` | 60 | `tenant_profile` (spend and qty percentiles, margins from order lines); `margin_for`, `spend_at_percentile`, `qty_at_percentile`. | Yes (field names) | ADAPT (generic "entity profile") | agents/customer, agents/margin |
| `agents/base.py` | 80 | `AgentContext`, `Proposal` (with `finalise()` producing `PR-` ids), `severity_for`, `interp_cdf`, `inv_cdf`, `series_evidence`, `dstr`. | No (contract) | KEEP | all agents, policy |
| `agents/__init__.py` | 21 | `REGISTRY` of four agents and `META` (titles, owners, cadence). | Yes (agent names and owners) | ADAPT (keep registry pattern) | api, sim |
| `agents/shortfall.py` | 108 | Stockout risk over review gap and lead time, using stock, pipeline, pack sizes. | Yes (stock, pipeline, pack) | ADAPT ("risk of shortfall in window") | registry |
| `agents/freshness.py` | 106 | FEFO over sampled demand paths against lots and use-by dates; targets recent buyers. | Yes (lots, use-by) | ADAPT ("expiry risk") | registry |
| `agents/customer.py` | 164 | `_keying` (quantity hold), `_slowdown`, `_lapse` (Mondays). | Yes (order entry, customer) | ADAPT ("activity anomaly") | registry |
| `agents/margin.py` | 123 | `_price_floor`, `_cost_drift` (Mondays). | Yes (price, cost) | ADAPT ("unit economics drift") | registry |
| `policy/alerts.py` | 133 | `Economics`, `net_value`, `value_of`, `select` (one score, one budget per owner, digest, false-alarm cost), `EXEMPT_ACTIONS` = `hold_order_line`. | No (except exempt action name) | KEEP (rename exempt action) | sim, api, governor |
| `policy/governor.py` | 139 | `LADDER` (explains, suggests, prepares, acts, runs), `Lesson`, `Governor.filter/commit/ladder_status`, `audit_lesson` (independent oracle). | No | KEEP | sim, api |
| `policy/learning.py` | 43 | `SegmentLearner` with bounded weights (prior 10). | No | KEEP | policy_sim, rollout |
| `policy/calibration.py` | 52 | `fit` (pool-adjacent-violators, MIN_POINTS 300), `apply`. | No | KEEP | rollout |
| `policy/whatif.py` | 101 | `measured` (excludes `dollar_basis='estimate'`), `whatif` (expected net, break-even), `problem_level` (collapses repeat alerts). | No | KEEP | api |
| `policy/rollout.py` | 262 | Tenant JSON file per dataset, `phase` (demo, setup, shadow 14d, live), `learn` (MAX_STEP 0.10), `readiness`, `drift_report`. | Partly (tenant file path, dataset coupling) | ADAPT (storage path) | api |
| `sim/replay.py` | 113 | `scored_window`, `replay` day-by-day loop; writes scorecard and digest JSON and a `runs` row. Has an unused `learn` flag (see section 5). | Partly | ADAPT | api, tests |
| `sim/scorer.py` | 203 | `score`: per-agent precision, recall, dollar recall; freshness maturity exclusion; `dollar_basis` per agent. | Yes (event definitions query truth_events, daily_outcomes, stock_movements, order_lines) | ADAPT (keep framework, swap event defs) | replay, policy_sim, whatif |
| `sim/policy_sim.py` | 77 | `load_candidates` (pickle), `flag_candidates`, `simulate` (offline re-scoring with cooldown, budget, optional learner). | No (except `cand_twin.pkl` default name) | KEEP | api |
| `sim/event_dollars.py` | 82 | Pre-event baseline dollars for keying, price-floor, cost-rise, customer loss and slowdown; HORIZON 56, PRE 56, COST_PRE 28, SLOWDOWN 0.35. | Yes (order-line semantics) | ADAPT or REMOVE (event dollars are estimates; port only if needed) | scorer, whatif |
| `source/staging.py` | 214 | `extract` (psycopg, read-only session, SELECT-only `check_select`), `role_audit`, ERP `QUERIES`, writes staging.duckdb and reveal_map. | Yes | REMOVE from core; keep patterns in an ADAPT connector | api (CLI scripts) |
| `source/pseudo.py` | 32 | `stable_hash` (salted SHA-256), `CodeBook` (C-0001 style codes); salt in `data/.salt`. | No | KEEP | staging |
| `twin/generator.py` | 319 | Synthetic world: CATEGORIES, SEGMENTS, CAL_EFFECTS (labelled SYNTHETIC), promotions, shock injection (nine shock types). | Yes (food domain) | ADAPT (keep shock harness, replace domain) | build, tests |
| `twin/supply.py` | 193 | `simulate` (FEFO stock simulation), `bau_policy` (safety stock, shelf-life cover, Mon/Wed/Fri), `SimResult`. | Yes | ADAPT (generic inventory simulator) | build |
| `twin/build.py` | 87 | `order_lines_frame`, `write_twin`, `main`. | Yes | ADAPT | tests (via fixture) |
| `twin/censor.py` | 31 | `censoring_mask` (drop and lag of outcomes, seed 101). | No | KEEP | build |
| `narration/templated.py` | 34 | Deterministic text per action (`raise_purchase_order`, `sell_down_offer`, `customer_check_in`, `hold_order_line`, `review_price`). | Yes (action vocabulary) | ADAPT | api |
| `narration/bedrock.py` | 85 | `spent`, `allowed_numbers`, `validate` (rejects invented numbers), `narrate` (Haiku via boto3, $30 cap, ledger). System prompt says "food wholesaler". | Partly (prompt) | KEEP validator and ledger; ADAPT prompt | api |
| `api/jobs.py` | 95 | Background runner (ThreadPoolExecutor, 2 workers), atomic JSON per job, marks interrupted jobs on load. | No | KEEP | api |
| `api/main.py` | 898 | FastAPI app: same-origin guard, decisions and lessons, backtest and replay jobs, forecast, briefing, tenant, policy, readiness, drift, time-step and reset, `/api/reveal`, SPA fallback. | Mostly | ADAPT (split into generic routers and ERP routers) | none (entry point) |

Not tabled: `__init__.py` files other than `agents/__init__.py`.

---

## 2. Dependency table

| Dependency (pyproject) | Where used (verified) | Verdict |
|---|---|---|
| `timesfm[mlx]==3.0.2` | `engine/forecast.py` (lazy import of `timesfm3.mlx.TimesFM3Forecaster`, checkpoint `google/timesfm-3.0-pytorch`) | KEEP as an optional provider extra, pinned. Weights are non-commercial; not for client production until licence is cleared (Doc: PLAN.md). Move behind an extra so core installs without MLX. |
| `numpy` | distribution, baselines, models, metrics, agents | KEEP (core) |
| `pandas` | replay, scorer, policy_sim, state, tests | KEEP, but limit to adapters. Core maths can be numpy-only. |
| `duckdb` | `store.py` and every data read; lab DB; datasets | KEEP for the reference store. Prophet core should take a storage port so Postgres or other ERP stores can be used. |
| `fastapi` | `api/main.py`, `api/jobs.py` | KEEP (API layer only, not in core) |
| `uvicorn` | Not imported in `src`. Used only to run the app (CLAUDE.md command). | KEEP as a run-time or dev extra, not a library dependency. |
| `psycopg[binary]` | `source/staging.py` only | REMOVE from core. Move to the ERP connector package. Binary wheel is heavy. |
| `boto3` | `narration/bedrock.py` only (lazy) | Optional extra (`narration-bedrock`). Keep validator and ledger in core without it. |
| `pydantic` (not declared) | Transitive import via FastAPI | Declare it explicitly if any module imports it. Otherwise document as transitive. |
| `pytest`, `httpx` (dev extra) | Tests; `httpx` used by FastAPI `TestClient` | KEEP (dev). Note: TestClient requires `httpx`. |
| Console (`console/`, React, Vite, Tailwind) | Not audited (out of scope) | Separate package. Keep out of Python core. |

Redundant stacks: DuckDB and pandas are both in the core path, and `numpy` maths is duplicated between `distribution.py` and `baselines.py` (both build z-tables or quantile helpers). Consolidate to one quantile helper.

---

## 3. Test table

20 test files plus `conftest.py`. Counts are from `pytest --co` on this machine.

| File (tests) | What it guards | ERP-only dependency | Verdict |
|---|---|---|---|
| `conftest.py` (fixture) | `mini` fixture: 60-product seeded twin, Router of ma8, ma4, tsb. | Uses twin builder | REWRITE (build a generic synthetic dataset fixture) |
| `test_core.py` (13) | Codebook, governor, metrics, narration validator, read-only guard, twin determinism. | Twin and narration parts | KEEP the generic parts; REWRITE twin parts |
| `test_models.py` (6) | Model classes, TSB, router, licence fallback, `fit_routing`. | None | KEEP |
| `test_distribution.py` (5) | Copula quantile reproduction, skew, correlation, daily shares, shortfall and leftover. | None | KEEP |
| `test_forecast_cache.py` (4) | Route, history and future-leak in forecast cache. | Mild (weekly ERP series) | KEEP (rewrite fixtures) |
| `test_daily_shape.py` (4) | Weekday profile and daily path. | Mild | KEEP |
| `test_no_future_leak.py` (1) | Future-leak guard on twin; skipped without `data/twin.duckdb`. | Twin | REWRITE (generic leak test on synthetic port) |
| `test_no_future_leak_mini.py` (2) | Same, on mini fixture; imports `key` and `truncated` from the twin test. | Mini fixture | KEEP (move helpers to a shared module) |
| `test_alert_policy.py` (9) | Floor, budget, digest, false-alarm cost. | Hold-order-line name only | KEEP |
| `test_api_guard.py` (10) | Origin, Sec-Fetch-Site, Host allow-list, lab cannot exceed `prepares`. | None | KEEP |
| `test_decision_integrity.py` (9) | Lesson retirement, operator identity, agent-wide approval. | Agent names | KEEP |
| `test_digest_calendar.py` (5) | Digest and calendar flags; Christmas and Easter. | Victorian calendar | REWRITE (calendar port) or DELETE with calendar_vic |
| `test_eval_gate.py` (2) | Mini replay floors (shortfall 0.45/0.10/10, freshness 0.78/0.70/180, customer 0.40/0.30/10, margin 0.60/0.50/2). | ERP agents | REWRITE (floors become per-domain config) |
| `test_event_dollars.py` (7) | Event dollar estimates. | ERP order-line semantics | REWRITE or DELETE with event_dollars |
| `test_jobs.py` (3) | Job persistence and interrupted-state recovery. | None | KEEP |
| `test_learning.py` (2) | `SegmentLearner` bounds. | None | KEEP |
| `test_rollout.py` (5) | Phase, bounded learning, no double-learning. `test_operator_context_dismissals` has a weak `or` assertion (Verified). | Tenant file path | KEEP; strengthen the weak assertion |
| `test_scorer_maturity.py` (1) | Freshness alerts whose horizon runs past data end are excluded from precision, not counted as false (verified, read in full). | Freshness event | KEEP the principle; REWRITE on generic expiry event |
| `test_staging_guard.py` (9) | `role_audit`, `check_select` refusals, guard raises PermissionError not assert. | Staging ERP | KEEP the guard tests as a connector test; move out of core |
| `test_twin_censor.py` (4) | Censoring mask behaviour. | None | KEEP |
| `test_whatif.py` (9) | Whatif arithmetic, break-even, endpoint uses stored scorecard, unmeasured agents not netted, claimed vs matched dollars, `problem_level` collapse, estimates excluded from headline (verified, read in full). | Endpoint uses `main` | KEEP (highest-value tests in the repo) |

Totals per file (verified): 9, 10, 13, 4, 9, 5, 5, 2, 7, 4, 3, 2, 6, 1, 2, 5, 1, 9, 4, 9 = 110.

---

## 4. Reusable generic building blocks

Domain-independent, keep with exact names:

**Forecasting core (`engine/`)**
- `forecast.Forecast`, `forecast.TimesFMEngine.get().predict`, `predict_cov`, `QUANTILES` (9 levels).
- `models.ForecastModel` (protocol), `models.TimesFM`, `models.Baseline`, `models.TSB`, `models.Ensemble`, `models.series_class`, `models.Router`, `models.fit_routing`, `models.default_registry`, `models.get_forecaster`, `models.set_router`.
- `baselines.moving_average`, `baselines.seasonal_naive`, `baselines.seasonal_year`, `baselines.run`.
- `distribution.Draws.make`, `week_samples`, `daily_samples`, `expected_shortfall`, `expected_leftover`, `prob_exceeds`.
- `metrics.wape`, `bias`, `pinball`, `coverage`, `mase`.
- `backtest.run_backtest` (generic once covariates and loader are injected).

**Agent contract (`agents/base.py`)**
- `AgentContext(world, as_of, fs, params, lessons)`, `Proposal` and `Proposal.finalise()`, `severity_for`, `interp_cdf`, `inv_cdf`, `series_evidence`.

**Policy and governance (`policy/`)**
- `alerts.Economics`, `alerts.net_value`, `alerts.value_of`, `alerts.select` (budget, digest, false-alarm cost).
- `governor.Governor.filter`, `Governor.commit`, `Governor.ladder_status`, `governor.Lesson`, `governor.audit_lesson`, `governor.LADDER`.
- `learning.SegmentLearner`; `calibration.fit`, `calibration.apply`.
- `whatif.whatif`, `whatif.measured`, `whatif.problem_level`.
- `rollout.phase`, `rollout.learn`, `rollout.readiness`, `rollout.drift_report`.

**Simulation and evaluation (`sim/`)**
- `replay.scored_window`, `replay.replay`; `policy_sim.load_candidates`, `flag_candidates`, `simulate`; `scorer.score` (framework only).

**Utilities**
- `source/pseudo.stable_hash`, `CodeBook`.
- `twin/censor.censoring_mask`.
- `narration/bedrock.validate` (rejects numbers not in the allowed set), `narration/bedrock.spent` and the ledger pattern.
- `api/jobs.submit`, `get`, `list_jobs` (atomic JSON jobs).
- `store.open_dataset(name, read_only)`, `open_lab()`.

**ERP data-layer needs (what Prophet's data port must supply)**

The engine needs only four things from any source system. These should sit behind a port (Python `Protocol`), not in SQL:
1. Demand history: entity by period quantity and value, with an `as_of` cut (no future rows). Today: `state.weekly_matrix`, `data.load_demand`.
2. Stock and expiry: on-hand by lot with expiry date, and inbound pipeline with lead time. Today: `state.stock_at`, `state.pipeline_at`, `lots`, `purchase_lines`.
3. Economics: unit cost and price per entity over time. Today: `state.unit_cost_at`, `profile.margin_for`.
4. Entity attributes: category, segment, owner. Today: `products.category`, `customers`.

Optional: calendar (injectable, not hard-coded), event truth for scoring (twin only).

---

## 5. Dead code, redundant stacks and architectural constraints

**Dead or unused (Verified by grep on `src/lab`)**
- `forecast_cache` table is in the lab schema (`store.py`) but has zero references in `src/lab` outside its CREATE statement. Forecast caching is done on disk by `state.cached`. Drop the table or wire it in.
- `sim/replay.py` `replay(..., learn: bool = False)`: the simulated-operator branch (line 59) is never called with `learn=True` in `src` or `tests`. Learning is exercised only through `policy_sim.simulate(learn_delay=...)`. Remove the flag or add a test that uses it.
- `event_dollars.py` and `calendar_vic.py` are used only by scoring and backtest respectively; both are candidates for removal once the port exists.

**Redundant or bypassing paths**
- `api/main.py` forecast endpoints (lines around 501, 520, 532) call `TimesFMEngine.get().predict` directly, bypassing the `Router` and `ForecastModel` adapter. The model-agnostic design is therefore not enforced at the API layer. Route all endpoints through `get_forecaster()`.
- Two quantile helpers (`distribution.py` and `baselines.py`) overlap. Consolidate.
- Lab DB and dataset DBs both use DuckDB; two schema modules in one file (`store.py`). Acceptable for lab, but Prophet should separate operational state from data sources.

**Licence and defaults (Verified)**
- `engine/models.py` line 131: `Router.allow_non_commercial: bool = True`. The default lets TimesFM (non-commercial weights) be used. For Prophet, default to `False` and require explicit opt-in per deployment.
- `api/main.py` line 257: `decided_by: str = "liam"`. A personal name is hard-coded as a default operator. Remove the default and require the caller to supply an identity.

**Constraints that must carry into Prophet**
- No future leakage: every agent and forecast at `as_of` sees data only up to that date. Guarded by `test_no_future_leak*.py` and `test_decision_integrity.py`.
- Labels: "flagged", "useful", "accepted", "acted" and "loss prevented" are distinct. Expected dollars must never be presented as observed.
- Autonomy capped at `prepares` (lab cannot claim acts or runs). Guarded by `test_api_guard.py`.
- Estimated dollars (customer and margin) stay out of the headline net (`whatif.measured`).
- 15% improvement-over-baseline ship bar is documented and **not met** (7–10% on the twin, Doc: CLAUDE.md and PLAN.md). Do not move the bar.
- Calendar covariate gave no gain on the twin (Doc: PLAN.md). Keep it optional.
- TimesFM weights are non-commercial. Licence unresolved (Doc: PLAN.md).

---

## 6. Secrets and data hygiene

Recorded by path and count only. No values were printed.

| Item | Status (Verified) | Action |
|---|---|---|
| `data/` | Git-ignored. Holds `twin.duckdb`, `staging.duckdb`, `lab.duckdb`, `reveal.duckdb`, experiment pickles and logs. | Never commit. Keep ignored. |
| `data/.salt` | Exists, mode 600, 32 bytes. Git-ignored through `data/`. | Never commit. Rotate if ever exposed. Document in the connector. |
| `reveal.duckdb` | Exists, 798,720 bytes, mode 644 (group and world readable). Holds the code-to-real-name map. | Tighten to mode 600. Never commit. |
| `.env` | Git-ignored. Mode 644. | Tighten to mode 600. Do not read in tests. |
| `NIM.md` (repo root) | **Untracked and not git-ignored.** Contains six key-shaped strings (`nvapi-` prefix) by count. No code in `src`, `tests` or `scripts` reads these values. | Do not commit. Add to `.gitignore` or move out of the repo. Consider rotating the keys, since they now sit in a plain file on disk. |
| `reveal` API (`/api/reveal`) | **No token check** (Verified: `@app.get("/api/reveal") def reveal(codes: str)`). CLAUDE.md says a token is required; that is false. | Add authentication before any use beyond local. Correct CLAUDE.md. |
| Staging connection | `source/staging.py` `role_audit` reports superuser-like privileges on the staging role (Doc: PLAN.md says owner role plus read-only session). Read-only is enforced only by session settings and a SELECT guard. | Use a dedicated read-only role for any real connection. Treat current access as higher than needed. |
| Staging credentials | Read from an env file outside the repo, via `config.read_env_value`. CLAUDE.md gives the path as `../foodflow/axis/env3.local`; `config.py` default is `~/Downloads/Axion/foodflow/axis/env3.local`. | Reconcile the path. Never print values. |
| SPA fallback (`/{path:path}` in `api/main.py`) | Builds `_DIST / path` and serves it if `is_file()`, with no containment check (Verified by reading). Path traversal is **likely** but **not tested**. | Add a test that requests a path outside `console/dist` and asserts 404. Do not exploit against a live host. Fix with `resolve()` and a parent check. |
| Git | No sensitive files tracked (checked by file-name pattern). | Keep the `.gitignore` list for `data/`, `.env`, `*.duckdb`, `.salt`, `NIM.md`. |

---

## 7. Recommended target layout

Prophet Python backend. Core is domain-free. ERP connectors and the reference twin are separate packages.

```
src/prophet/
├── __init__.py
├── core/                      # domain-free contracts and maths (KEEP from lab)
│   ├── forecast.py            # Forecast, QUANTILES, ForecastModel protocol   <- engine/forecast.py, models.py (protocol)
│   ├── models.py              # TSB, Baseline, Ensemble, Router, fit_routing  <- engine/models.py
│   ├── baselines.py           # moving_average, seasonal_naive, seasonal_year <- engine/baselines.py
│   ├── distribution.py        # Draws, samples, expected_shortfall            <- engine/distribution.py
│   ├── metrics.py             # wape, bias, pinball, coverage, mase           <- engine/metrics.py
│   ├── backtest.py            # run_backtest with injected loader/covariates  <- engine/backtest.py (ADAPT)
│   └── profile.py             # entity profile (generic)                       <- engine/profile.py (ADAPT)
├── agents/
│   ├── base.py                # AgentContext, Proposal, finalise               <- agents/base.py
│   ├── registry.py            # REGISTRY pattern                               <- agents/__init__.py
│   ├── shortfall.py           # generic shortfall risk                         <- agents/shortfall.py (ADAPT)
│   ├── expiry.py              # generic expiry risk                            <- agents/freshness.py (ADAPT)
│   ├── anomaly.py             # activity anomaly                               <- agents/customer.py (ADAPT)
│   └── drift.py               # unit-economics drift                           <- agents/margin.py (ADAPT)
├── policy/
│   ├── alerts.py              # Economics, select, net_value                   <- policy/alerts.py
│   ├── governor.py            # LADDER, Governor, Lesson, audit_lesson         <- policy/governor.py
│   ├── learning.py            # SegmentLearner                                 <- policy/learning.py
│   ├── calibration.py         # PAV calibration                                <- policy/calibration.py
│   ├── whatif.py              # whatif, measured, problem_level                <- policy/whatif.py
│   └── rollout.py             # phase, readiness, drift_report, learn           <- policy/rollout.py (ADAPT storage)
├── sim/
│   ├── replay.py              # day-by-day replay                              <- sim/replay.py (drop unused learn flag)
│   ├── scorer.py              # scoring framework; event defs injected         <- sim/scorer.py (ADAPT)
│   └── policy_sim.py          # offline candidate re-scoring                   <- sim/policy_sim.py
├── ports/                     # NEW: Protocols the data layer must satisfy
│   ├── demand.py              # weekly demand by entity, as_of cut
│   ├── inventory.py           # on-hand by lot, expiry, pipeline, lead time
│   ├── economics.py           # unit cost and price over time
│   ├── entities.py            # category, segment, owner
│   └── calendar.py            # injectable calendar (replaces calendar_vic)
├── storage/
│   ├── duckdb_store.py        # reference store                                <- store.py (split)
│   ├── lab_state.py           # runs, proposals, decisions, lessons            <- store.py LAB_SCHEMA
│   └── cache.py               # fingerprinted disk cache                        <- engine/state.py cached()
├── narration/
│   ├── templates.py           # generic templated text                         <- narration/templated.py (ADAPT)
│   ├── validator.py           # allowed-number check                           <- narration/bedrock.py validate
│   ├── ledger.py              # spend cap and ledger                           <- narration/bedrock.py spent
│   └── bedrock.py             # optional extra (boto3)                         <- narration/bedrock.py
├── util/
│   ├── pseudo.py              # stable_hash, CodeBook                          <- source/pseudo.py
│   └── censor.py              # censoring_mask                                 <- twin/censor.py
├── jobs/
│   └── runner.py              # ThreadPool job runner                          <- api/jobs.py
└── api/                       # FastAPI, optional extra
    ├── app.py                 # app factory, guard middleware                  <- api/main.py (split)
    ├── guard.py               # Host, Origin, Sec-Fetch-Site checks           <- api/main.py
    ├── decisions.py           # decisions and lessons routes                   <- api/main.py
    ├── forecast.py            # forecast routes via get_forecaster()           <- api/main.py (fix bypass)
    └── static.py              # SPA serving with containment check             <- api/main.py (fix traversal risk)

connectors/                    # NOT core; one package per source system
├── foodflow_axis/             # ERP read-only connector                        <- source/staging.py, twin-free
└── reference_twin/            # synthetic benchmark (labelled SYNTHETIC)       <- twin/generator.py, supply.py, build.py

tests/
├── conftest.py                # generic synthetic fixture (replaces twin mini)
├── core/                      # models, distribution, metrics, baselines, daily shape, forecast cache
├── policy/                    # alert policy, decision integrity, whatif, rollout, learning, jobs
├── api/                       # guard, reveal auth (new), SPA containment (new)
├── sim/                       # scorer maturity, eval gate (per-domain floors), no-future-leak
└── connectors/                # staging guard, role audit (move from core)
```

Migration order, lowest risk first: `core/` and `policy/` (pure, tested), then `ports/`, then `api/` split with the two security fixes (reveal auth, SPA containment), then connectors.

---

## Summary of verified discrepancies

1. `/api/reveal` has no token check. CLAUDE.md says it does.
2. CLAUDE.md says 109 tests; `pytest --co` collects 110.
3. CLAUDE.md staging env path differs from `config.py` default.
4. `forecast_cache` table is unused.
5. `replay(learn=...)` flag is never set to True.
6. API forecast endpoints bypass the Router.
7. `Router.allow_non_commercial` defaults to True.
8. `decided_by` defaults to a personal name.
9. `NIM.md` (key-shaped strings) is untracked and not git-ignored.
10. SPA fallback has no containment check (traversal unverified, must be tested, not exploited).
