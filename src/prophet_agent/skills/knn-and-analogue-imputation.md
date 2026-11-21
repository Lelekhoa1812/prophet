name: knn-and-analogue-imputation
summary: Fill short gaps from similar earlier days, causally, and keep the fill only if a backtest says it helps.

# KNN and analogue imputation

Fill a gap only from earlier rows that look like it. The fill reads nothing at or after the gap, so it is safe inside a backtest.

1. Apply when a short gap inside a seasonal series is a capture failure, not a real zero. Screen first with `ps.censoring_check(x)`: runs it lists as probable stockouts are never filled.
2. Gate: the gap is no longer than two seasonal cycles, and at least 30 earlier rows have observed targets (proposed default, so the donors are more than a handful of days).
3. Call it on a table with a date index, not on a bare array:

```python
filled, rep = ps.knn_impute(df, cols=["demand"], k=5, seasonal_lags=(1, 7))
print(rep["count"], rep["unfilled"], rep["donor_shortfalls"])
```

4. Read the report. Fills are listed by column and position. A donor shortfall means fewer than k donors were available, so that fill is thin. Unfilled gaps stay NaN; do not fill them another way without a recorded reason.
5. What it can destroy: a real weekday effect when the nearest donors come from another regime, a spike copied into the gap from its neighbours, and any stockout turned into demand. The method gives no uncertainty of its own.
6. Mandatory check: run `ps.causal_canary(fn, series)` with fn applying the fill. It must report causal True. Then run a walk-forward comparison. At each origin, forecast from history with the fill and from history without it, score both on held-out actuals with MASE, and keep the fill only if it wins at more than half the origins.

Record every fill in the Ledger with its method, k and the donor count, so a reviewer can see which values were inferred. `ps.fuse_report` merges this report with the other cleaning reports into one dict.

Do not fill trailing gaps, which are missing future values rather than holes. Do not fit the imputer on the full table. Do not report filled values as observed.
