# Analyst playbook: how Prophet works a dataset

Why this exists: the first live runs showed a planner that forecast quickly and inspected shallowly. One Gemini run
reported "no missing periods" on an hourly load file whose clock changes had removed or duplicated six hours; a NIM run
found them. The design below makes the careful route the default, and makes the careful route checkable.

Source: `research_notes/Principal analyst agent harness design/` and the report in `reports/`. Labels in those notes
(verified, summary-fetch, snippet, memory, proposed threshold) carry over: every number marked "proposed default" in the
skills is a starting value to calibrate on synthetic replay, not a finding.

## The funnel (one cell per stage, each recorded in a ledger)

1. Provenance and units. 2. Schema. 3. Keys and duplicates. 4. Time index. 5. Coverage. 6. Zeros and censoring.
7. Level shifts. 8. Seasonality. 9. Volatility. 10. Outliers. 11. Model gate (backtest against baselines on the same
origins). 12. Handover with a stated trust level.

The order matters: a missing timestamp looks like a seasonal dip, and a level shift looks like a run of outliers.

## Where each part lives

| Part | File | Role |
|---|---|---|
| Playbooks (11 skills) | `src/prophet_agent/skills/*.md`, read on demand with the `analyst_skill` tool | decision rules, short code, anti-patterns |
| Statistics toolkit (`ps`) | `src/prophet_agent/sandbox/prophet_stats.py` | causal outlier flags, gap-aware imputation, change points, censoring, stationarity, leakage-safe treatment backtest, ledger |
| Data helpers (`pl`) | `src/prophet_agent/sandbox/prophet_lib.py` | load, profile, prepare, chart, query |
| Planner rules 50 to 56 | `src/prophet_agent/prompts/planner.v11.md` | funnel, atomic breakdown, cause-led treatment, no leakage, method reconciliation, honest reporting |
| Deterministic checks | `src/prophet_agent/runtime/contracts.py`, called at the finish in `runtime/goal.py` | data inspected, baseline compared, health repaired, deliverables present |

## Principles

- A cleaning step earns its place by a leakage-safe backtest (`ps.backtest_treatment`). Whether smoothing, outlier
  replacement or imputation helps TimesFM is unmeasured in the sources reviewed; it is a hypothesis for the synthetic twin.
- Keep the raw series, treat a copy, write a reason code, count before and after every fill.
- Model input uses only information from before each origin: trailing windows, one-sided imputers.
- "Done" is a claim until a check on the evidence passes. The checks cost no model call.
- Label every figure as synthetic replay, real staging or measured. TimesFM 3 outputs are research only.
