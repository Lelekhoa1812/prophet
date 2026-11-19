# Runbook: standing up the forecasting and alert layer for a client

For engineers. Status: R&D lab. Nothing here runs against a client's production system, and TimesFM 3 weights are non-commercial until the licence is cleared. Numbers in `PLAN.md` come from the synthetic twin and show method, not client results.

## 1. Run the lab
```bash
cd ~/Downloads/Axion/timesfm-erp
source .venv/bin/activate
PYTHONPATH=src uvicorn lab.api.main:app --host 127.0.0.1 --port 8000     # console and API
cd console && npm run build                                               # rebuild the console after UI changes
.venv/bin/python -m pytest -q tests                                       # 40+ focused tests, under a second
```
The first forecast loads the model (about 80 seconds). Forecasts are cached under `data/cache`, so replays reuse them.

## 2. Onboard a client (in this order)
1. **Data.** Connect read-only and build a dataset the same shape as the twin: orders and lines, products, customers, stock lots with use-by dates, purchase orders, movements. Pseudonymise before it lands in `data/`, which is git-ignored.
2. **Profile.** `engine/profile.py` measures the client's customer size spread, order-line size spread and actual margins from the trailing 12 weeks. Customer and margin rules read these instead of fixed dollar figures. Check them first: `tenant_profile(world, date)`.
3. **Model routing.** `engine/models.py`: run `fit_routing(matrix, models, origins)` on the client's weekly demand. It keeps TimesFM unless a challenger wins by at least 2% pinball loss in a series class with enough history. For a production tenant while the licence is open, set `allow_non_commercial=False` on the `Router`; the free fallback (TSB) is about 14% worse on steady series on the twin.
4. **Client economics.** Developer page, "Client set-up and rollout" (or `POST /api/tenant`): cost to review one alert, share of the loss a reviewed alert saves, cost of acting on a false alarm, alerts per owner per morning. Defaults are $12.50, 50%, $200, 8.
5. **Readiness.** For each agent, `GET /api/readiness?agent=` replays the last 13 weeks under the client's policy. Verdicts: ready, start in shadow, not ready (under 12 weeks of history). Under 52 weeks the system makes no seasonal claims. A window with fewer than 10 recorded problems says recall is not reliable.
6. **Shadow.** Set a go-live date. For 14 days alerts are recorded and scored but not shown. The briefing says so.
7. **Go live.** Owners see the best N alerts a day; the rest are in a digest. Order-entry stops are always shown.

## 3. Keep it healthy
- **Drift check** (`GET /api/drift`, monthly): per volume class, share of actuals inside the 10-90 range (aim 80%) and below the median (aim 50%). It tests calibration, not unit bias, because the median of lumpy demand always sits below the mean.
- **Learning** (`POST /api/tenant/learn`): turns approvals and dismissals into bounded priorities per agent and per segment (product category). Steps of at most 10%, weights 0.5 to 2, every change logged and visible. 'One-off' and 'known promotion' dismissals do not count against an agent.
- **Calibration** (`POST /api/tenant/calibrate`, needs recorded candidates): fits how often alerts at each stated chance proved useful. Used for display ("alerts scored like this were useful 49% of the time"), not for ranking, because on the twin using it to price false alarms only made the system stricter.
- **Trade-off table** (`GET /api/policy/frontier`): what each alert budget buys. Re-scores recorded candidates under the client's economics in seconds.

## 4. Experiments without waiting for a replay
```bash
# record every candidate once (about 12 minutes), then re-score any setting in seconds
PYTHONPATH=src python data/exp_candidates.py        # writes data/cand_twin.pkl
PYTHONPATH=src python data/exp_sweep2.py            # budget x false-alarm cost sweep
```
`sim/policy_sim.simulate(candidates, economics, dataset, window, flags=..., learn_delay=...)` applies cooldown, the score-and-budget policy and optional learning, then scores against what happened.

## 5. What each file owns
| File | Job |
|---|---|
| `engine/distribution.py` | Sampled demand over a window from forecast quantiles (weeks lightly correlated, weekday pattern) |
| `engine/models.py` | Model adapter, router, `fit_routing`, licence fallback |
| `engine/profile.py` | Client-relative thresholds and margins |
| `agents/*.py` | Pure functions from business state at a date to proposals with evidence |
| `policy/alerts.py` | Net value, budget, digest |
| `policy/learning.py`, `policy/calibration.py` | Segment priorities and chance calibration |
| `policy/rollout.py` | Tenant file, phase, readiness, drift, learning log |
| `policy/governor.py` | Dedupe, cooldown, dismissal lessons, autonomy ladder |
| `sim/replay.py`, `sim/policy_sim.py`, `sim/scorer.py` | Replay, offline re-scoring, scoring against recorded problems |

## 6. Honest limits
- A useful alert means it matched a stockout, expiry or planted event in the data. Flagged is not prevented.
- Customer precision is understated on the twin: only planted slowdowns and losses count, natural ones count as false.
- The twin's learning gain assumes operators judge by the true outcome. Real operators are noisier.
