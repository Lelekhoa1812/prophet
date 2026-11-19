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
