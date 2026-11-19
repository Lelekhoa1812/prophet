# TimesFM ERP: plan (v0.1, 19 Nov 2026)

R&D sandbox to prove where TimesFM 3 adds value to FoodFlow and Axis, before anything ships to a client.
Separate from the FoodFlow repo. Read-only against FoodFlow data. Never writes to FoodFlow.

## Locked decisions

| Area | Decision | Source |
|---|---|---|
| Model | TimesFM 3.0 only. Licence (non-commercial weights) resolved later; no client production step until then | Liam, 19 Nov |
| First use cases | Shortfall and buy list; freshness sell-down; customer order anomalies; margin and price drift | Liam |
| Triggers | Hybrid: nightly re-forecast per tenant, event re-checks, on-demand through Axis. Each agent has an owner and quiet hours | Liam |
| Surfaces (client release 1) | 5 am morning briefing; Axis proposals inbox; email/Teams digest; in-context hints on PO, product, customer and order screens | Liam |
| Autonomy (client release 1) | Prepares for your OK. Agents draft with evidence and quantile bands; a named person approves | Liam |
| Production runtime | Separate Python forecast service on AWS ECS, ap-southeast-2, behind the Axis model gateway, batch-first | Liam |
| Success bar | Beats seasonal-naive and moving-average baselines; calibrated 10-90% bands; actionable precision (aim 70%+ useful); shadow-mode dollar outcome; every proposal explainable | Liam |
| Data source | FoodFlow staging tenant. Connection to be confirmed (see open question O1) | Liam |
| Data handling | Local pseudonymised snapshot in DuckDB, git-ignored, never uploaded. Names replaced by codes in the console unless reveal is on | Liam |
| Simulation | Walk-forward backtests; time-travel replay of history as events; injected shocks; live shadow mode | Liam |
| Console | Agent and trigger control; forecast explorer; proposals review queue; scorecard and dollars | Liam |
| Test login | Liam authorised a universal staging test account for any agent to use in the FoodFlow staging app. Credentials are not stored in this repo | Liam, 19 Nov |

## Round 2 decisions and results (19 Nov 2026)

Decided with Liam (AskUserQuestion):
- **TimesFM value bar:** not met yet. Stay on the 15% improvement bar; try to close the gap first.
- **Alert volume:** daily budget plus digest. Each owner sees up to 5 items a morning, ranked by dollars. The rest roll into one digest line per owner and agent. Order-entry stops (`hold_order_line`) are never digested.
- **Anomaly scope:** order-entry checks and nightly business checks in scope now. Weekly pattern checks and supplier delivery checks later, because the staging data is only partly there.
- **Holidays:** Victorian calendar tested first, weather later.

Results:
- Calendar covariate (`timesfm_cal`, `src/lab/calendar_vic.py`) does not help on the twin. Over 52 weekly origins, WAPE is 21.5% vs 21.0% for plain TimesFM. On the 28 origins whose horizon includes a public holiday or run-up, it is 22.2% vs 21.6%. Plain TimesFM still beats moving averages on those weeks (21.6% vs 23.1% for the 8-week average). The noise floor is 18.5%.
- Why it likely fails: the twin's calendar effects are real but segment-specific, and the weekly covariate is diluted (one or two events a year). A single covariate channel cannot learn per-segment magnitudes from two years of history. Next candidate: a calendar uplift layer per segment, estimated on the history and applied to the forecast. Not built yet.
- Scope caveat: the scored window only covers forecasts whose horizon includes Christmas 2025 when `n_origins` is 52. The 26-origin default starts in April 2026, so it misses Christmas entirely.
- Ship-bar margin vs best baseline: 7.3% on 52 origins, 9.5% on 26 origins. Both are below the 15% bar.
- The twin's calendar effects are generated from the same calendar module as the covariate. Any covariate gain on the twin is therefore an upper bound, and the twin cannot tell us real effect sizes.

## Round 3 decisions and audit (19 Nov, later)

Decided with Liam (AskUserQuestion):
- **Default history window:** all history on every page (briefing and Explorer agree). The roller still shortens it, and the panel says what the model saw.
- **Expected sales for expiry maths:** keep the median. It under-forecasts weekly units by 6.7% on the twin, but it is the most accurate basis (WAPE 27.5%). The 60th percentile over-forecasts by 2.2% (WAPE 28.1%). Corrected after I first mislabelled the quantiles.
- **Explorer Daily:** derived from the weekly forecast and the weekday pattern, the same as the briefing. Direct daily TimesFM is not used: for P-0079 it gives 0.6 to 1.2 units a day, against 3.4 from the weekly view.
- **Partly known week:** weekly forecasts start at the next full week. The partly known days are shown as a note.
- **Shortfall look-ahead:** review gap 4 days (was 2). On the four-year twin: 49% count recall and 67% dollar recall, 5.5 alerts a day, 46% precision.

Audit findings:
- Same product, same date: the Explorer (52-week default) and the briefing (all history) differed by about 10% a week. Fixed by the shared default.
- The Explorer's "how the forecast compares" bar summed signed errors, so over- and under-forecasts cancelled. Now absolute error.
- Known-events list showed the twin's planted shocks as if known in advance. Now labelled as checking-only.
- Shortfall stockouts: a repeat cooldown was not the cause. Removing it (shortfall cooldown 0) added 20% more alerts for 2 extra catches, so it was reverted.
- Shortfall misses are mostly routine stockouts: planted shocks explain less than half, and the catch rate is 26% without a nearby shock. Margin is not the driver (10 to 30% margins catch at similar rates).
- Residual: the weekly charts on shortfall and freshness cards still start at the partly known week. Needs the same fix as the Explorer.

## Open questions

- **O1 (resolved 19 Nov). Staging database** is STAGING_TENANT_DATABASE_URL in foodflow/axis/env3.local. Data depth: about 8 weeks of orders (delivery 14 Sep to 6 Nov 2026), so seasonality cannot be tested yet. Connection uses the owner role with a read-only session. Liam chose not to create a dedicated read-only role. Original note: The `.env` files hold `REPORTING_DATABASE_URL` and `TENANT_DATABASE_URL`. Their hosts are not labelled staging or production, so the lab will not read them until confirmed.
- **O2. Snapshot grain.** Proposed: daily per product and per customer, plus weekly rollups for the buy list.
- **O3. Baselines.** Proposed: seasonal-naive (weekly), 4-week moving average, and TimesFM 3 median. Compared on WAPE, pinball loss and band coverage.

## Build order

1. Scaffold the lab (done in this step): folders, config, data-access guard, scoring code.
2. Snapshot extractor: read-only role, aggregates only, pseudonymised codes. Blocked on O1.
3. Walk-forward backtest runner: TimesFM 3 (MLX) vs baselines, per use case.
4. Scenario injector: known shocks, to test anomaly and freshness agents.
5. Agents as pure functions over a snapshot date: each returns proposals with evidence.
6. Scheduler: replays history as events, and runs shadow mode on current data.
7. Console: FastAPI backend and a FoodFlow-branded React UI.

## Guardrails

- The lab never writes to FoodFlow. Connections are read-only roles only.
- Customer and product names stay pseudonymised in committed fixtures. Snapshot files stay in `data/`, which is git-ignored.
- Nothing is sent to an external service except the TimesFM model weights from Hugging Face.
- Every run records its code version, data cut-off date and model checkpoint, so results can be replayed.
