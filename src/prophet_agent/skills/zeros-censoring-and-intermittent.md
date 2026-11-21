name: zeros-censoring-and-intermittent
summary: Separate real zeros from censored ones; route intermittent demand to a TSB or Croston benchmark.

# Zeros, censoring and intermittent demand

A zero can mean no demand, no stock, no feed or no opening. Classify it before you treat it.

1. Classify each zero run:
   - Coincides with a stock-out, outage, closure or sensor fault: censored. Set it to missing and exclude it from fitting. Never forecast zero for it.
   - Recurs at the same calendar position: real seasonality. Keep it.
   - Isolated zero in steady demand with no cause: keep it, flag it, and test sensitivity.
2. Run `ps.censoring_check(x)`. Confirm against a second signal (inventory, opening hours) where one exists.
3. Measure intermittence as the average inter-demand interval: total periods divided by periods with sales. Above 1.32 (repo constant) the demand is intermittent. Sales in fewer than about three in four periods is also intermittent (proposed default).
4. For intermittent series, do not let a continuous model be the only candidate. Include a TSB or Croston-type benchmark, and report expected demand, not the median, as the point. The timesfm tool already returns expected demand for these series; still state the route you chose.
5. Report bias beside any scaled error. Scaled errors can reward near-zero forecasts on lumpy demand, so the mean error must be visible (proposed practice).
6. Run `ps.seasonal_strength` only after the zeros are classified. A seasonal dip caused by censoring is not seasonality.

```python
import numpy as np
import prophet_stats as ps
x = s.to_numpy(dtype=float)
sold = int(np.count_nonzero(x > 0))
adi = len(x) / max(sold, 1)                      # average inter-demand interval
print(ps.censoring_check(x))
print("intermittent" if adi > 1.32 else "continuous", round(adi, 2))
```

Decision rules:
- If zeros sit next to an inventory drop, treat them as censored until shown otherwise.
- If the series is intermittent and the continuous model wins on point error only, check its bias before accepting it.
- If there are too few sales to estimate an interval, report the series as not forecastable rather than inventing a band.

Do not forecast zero for censored periods. Do not treat stock-out zeros as true demand. Do not use the median as the point for lumpy demand. Do not report a single scaled-error figure alone. Do not run a continuous model on intermittent demand with no benchmark beside it.
