# Design review: a forecasting and anomaly platform that can be deployed per client (19 Nov 2026)

Scope: the lab as built, reviewed for what has to change before it serves any Axion client. All numbers below come from the four-year synthetic twin unless stated. They show method, not client results. Real staging has about eight weeks of data.

## 1. What the evidence says

### Forecast layer
- TimesFM quantiles are well calibrated in aggregate. Share of actuals at or below each forecast level (weekly origins June 2025 to September 2026, 4-week horizon): q10 17.6%, q50 49.6%, q90 89.9%. Only the lower tail runs high.
- Calibration differs by volume, not by category. For big sellers only 42% of actuals fall at or below the median forecast (under-forecast). For small sellers it is 57% (over-forecast). Categories sit within a few points of 50%.
- Correcting the point forecast does not pay. A trailing multiplicative correction removes the bias (−7.8% to +0.6%) but makes accuracy worse. By volume class: WAPE 29.9% to 30.8%, pinball 12.47 to 12.86. By category: WAPE 31.0%, pinball 12.94. The median is the right statistic for typical-case accuracy. The bias is median-below-mean on lumpy demand, not a model defect.
- Conclusion: keep the raw quantile distribution. Decisions that add up demand (expiry, shortfall) should use the whole distribution and an expectation, not a plug-in median.
- Calendar covariate (Christmas, Easter, school holidays) adds error on the twin. The effect is segment-specific and the history holds only one or two of each event.

### Decision layer
- Today's alert gates (probability of shortfall at least 55%, expected dollars at least $150, look-ahead of lead time plus 4 days) are not on the efficient frontier. Shortfall, four-year twin, about 650 days:

| Setting | Alerts | Per day | Count recall | Dollar recall | Precision |
|---|---|---|---|---|---|
| $150 floor, 55% (current) | 3,591 | 5.5 | 49.2% | 67.4% | 45.5% |
| $50 floor, 85% | 2,661 | 4.1 | 52.6% | 66.9% | 56.9% |
| $50 floor, 70% | 4,849 | 7.5 | 67.9% | 77.4% | 51.8% |
| $100 floor, 55% | 5,051 | 7.8 | 59.8% | 72.8% | 47.1% |
| $50 floor, 55% | 7,726 | 11.9 | 73.8% | 80.6% | 48.0% |
| $25 floor, 55% | 10,067 | 15.5 | 82.3% | 85.2% | 49.2% |
| $50 floor, 40% | 11,612 | 17.9 | 79.4% | 83.9% | 44.6% |

- The current setting is dominated: $50 and 85% gives fewer alerts, higher recall and higher precision. The two gates interact, so hand-tuning one at a time cannot find the frontier.
- Look-ahead window matters most: lead time plus 2 days saw only 30% of stockouts (51% of dollars). A 4-day gap lifted that to 49% (67%).
- A shortfall repeat cooldown was not the cause of misses (removing it added 20% more alerts for 2 more catches).

### Rigidity inventory (constants that assume this one business)
- Shortfall: probability 0.55, floor $150, lead default 3 days, cover percentile 0.80, pack size 6 above 21-day shelf life, cover days 3/5/10 by shelf life, margin fallback 15% of price.
- Freshness: horizon 21 days, floor $200, share at risk 25%, discount tiers 30/20/10%, median as the demand basis.
- Customer: minimum weekly spend $400, slowdown at 50% for two weeks, lapse at 21 days or 4x the usual gap, keying at 8x and 25 units, one fixed 18% gross margin for every customer.
- Margin: cost rise 6%, price floors 85% and 75%, 12% target margin.
- Cadence: cost and customer-pattern checks run on Mondays only.
- Severity cut-offs and confidence formulas are fixed per agent and are not calibrated against outcomes.
- Active products need sales in the last 26 weeks; accounts and products with under six weeks of history are skipped.

## 2. Target design

Principle: policy as data, decisions on distributions, every dial tied to a business quantity a client understands.

1. **Data contracts and profiling.** A tenant profile (history depth, order cadence, intermittency, shelf-life classes, margin distribution, currency, units) computed on connect. It decides which agents can switch on.
2. **Forecast service behind an adapter.** A model-agnostic interface that returns a quantile distribution per series and horizon. TimesFM 3, simple rules and an ensemble are interchangeable. Routing per series class (steady, intermittent, short history) is decided by backtest, not by assumption. This also removes the licence risk of depending on one model.
3. **Calibration and monitoring.** Per tenant and per volume class: coverage, bias and pinball tracked weekly, with alerts when they drift. New models and settings run in shadow before they drive alerts.
4. **Decision engines, one per decision type.** Each computes an expected value from the distribution (for example expected lost margin = E[max(demand − position, 0)] × margin) and returns it with its drivers. No fixed dollar or probability gates inside the engine.
5. **Alert policy.** A ranked list per owner under an alert budget, with a digest for the rest. A single score replaces separate probability and dollar gates. Dials are tenant economics: cost of reviewing an alert, value of margin at risk, tolerance for missed events.
6. **Learning loop.** Approvals and dismissals estimate precision per agent, owner and segment, and adjust the score threshold within guardrails. Dismissals still suppress, visibly and reversibly.
7. **Tenant onboarding.** A readiness backtest per agent on the tenant's own history before it switches on, with a plain report: what it would have caught, how many alerts a day, how many weeks of data it needed.
8. **Autonomy ladder** stays as built: explains, suggests, prepares, acts, runs. Promotion needs measured precision on the tenant's data.

## 3. Open decisions to settle with Liam
- Policy architecture (single score and budget, versus presets, versus autotuning from the tenant's backtest).
- Statistic for expiry and shortfall sums (expectation from the distribution, versus the median).
- Model strategy (adapter and routing, versus TimesFM only).
- Onboarding gate (automatic readiness backtest, versus manual set-up).
- Whether the learning loop may change thresholds on its own.

## 4. What is not yet known
- Whether any of this holds on real client data. Staging holds about eight weeks.
- Whether the twin's lumpiness and stockout rate resemble a real wholesaler. The twin was tuned to staging's category mix, price spread and basket size, not to its stockout behaviour.
- TimesFM 3.0 weights carry a non-commercial licence. No client production use until that is resolved or another model passes the same tests.
