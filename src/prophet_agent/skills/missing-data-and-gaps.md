name: missing-data-and-gaps
summary: Treat missing stamps and values: trim, interpolate short gaps, truncate after long ones.

# Missing data and gaps

Goal: never let a missing feed look like demand. Fill only what is provably a short gap, and log every filled point.

1. Put the series on a regular grid first. Missing stamps become NaN, never zero.
2. Trim leading zeros and leading NaN that precede first sale or launch. Log the trim date. This is not demand.
3. Treat a flat or empty tail with suspicion. Before calling it end of life, check whether other series in the group stopped on the same date.
4. For interior gaps, decide by length:
   - 1 to 3 consecutive steps: time-aware interpolation with a cap, flagged (proposed default).
   - Between 4 steps and about two seasonal cycles: seasonal fill is an option, to be tested by backtest.
   - Longer than about two seasonal cycles: do not bridge. Truncate training to the last complete run, or mark the series not forecastable (proposed default).
5. If gaps line up with a calendar pattern (closures, public holidays), model them with a calendar indicator. Do not interpolate them.
6. Report the imputed share per series, for example "2.1 percent of rows interpolated, 0 dropped".

```python
import prophet_lib as pl
import prophet_stats as ps
s, rep = pl.prepare_series(df, value="units", time="date", freq="D")
filled, rep2 = ps.impute(s, method="auto", max_gap=3, causal=True)
print(rep2)                                  # method, count filled, gaps left open
```

Decision rules:
- If the gap count is high in one series only, check that series' feed before choosing a fill.
- If the gap sits across a level shift, do not interpolate it. Treat the shift first.
- If the model's NaN handling is undocumented (the timesfm tool does not document it), fill or exclude gaps in the harness before any call.

Record each treatment in the ledger with its reason code, count and method.

Do not fill gaps with zero. Do not bridge a long gap by interpolation. Do not interpolate across a level shift. Do not back-fill history used by a backtest. Do not drop series quietly to remove gaps.
