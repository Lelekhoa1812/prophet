name: seasonality-and-calendar
summary: Test the seasonal period against noise and build calendar covariates over the full horizon.

# Seasonality and calendar

Seasonality is a claim to test, not a default. Find candidates, measure strength, then decide whether a seasonal model or a calendar covariate earns its place.

1. Find candidate periods with `pl.find_season(values, candidates=(7, 365))`. Then measure strength with `ps.seasonal_strength(x, period)`.
2. Decide by strength:
   - Strength at or above about 0.64 (proposed default) and the period matches the data frequency (weekly on daily data, yearly on monthly data): treat the seasonality as real.
   - Weak strength, or two candidate periods of similar strength: use the simpler model and test the seasonal option in a backtest.
   - Several periods (weekly and yearly): model each one, or use calendar covariates, and say which.
3. Trust a seasonal model only with about four seasonal cycles of history (proposed default). With about two cycles, classical methods fail most often.
4. Build calendar terms on the full index, history plus horizon. Never build them from the training window alone.
5. Drop or flag partial periods (the first and last incomplete week or month) before any seasonal estimate.
6. For public holidays use `pl.holiday_covariate` with country and state. Public calendars miss shop closures and school breaks, so test an own event list with `pl.event_covariate`.
7. Keep a calendar covariate only if the backtest gains more than the noise. A repo test on the synthetic twin found no gain from a calendar covariate, so test it each time.

```python
import prophet_lib as pl
import prophet_stats as ps
x = s.to_numpy(dtype=float)
print(pl.find_season(x, candidates=(7, 365)))
print(ps.seasonal_strength(x, period=7))
hol = pl.holiday_covariate(s.index, "AU", horizon=28, freq="D")
assert len(hol) == len(s) + 28
```

Decision rules:
- If the period is unknown, say so and do not guess a weekly cycle.
- If the calendar effect is a known closure, model it as a closure indicator rather than as seasonality.
- If the seasonal dip sits on censored zeros, fix censoring first.

Do not assume weekly seasonality without a test. Do not build calendar terms from the training window only. Do not count a holiday closure as demand. Do not read a seasonal dip caused by censoring as seasonality. Do not add a covariate because it seems right.
