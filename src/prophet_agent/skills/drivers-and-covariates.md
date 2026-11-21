name: drivers-and-covariates
summary: Add promotions and holidays only when known at the origin and proven better in backtest.

# Drivers and covariates

A driver helps only if it is known at the forecast origin for every future period and it improves the backtest.

1. Name each driver and its source: promotions, prices, stock-outs, public holidays. Do not invent one. Use the build_drivers tool for named events and holiday calendars, or `pl.event_covariate` and `pl.holiday_covariate` in analysis code.
2. Check knowledge time. A promotion plan already agreed is known. A month-end total, or the weather that actually happened, is not.
3. Use past-only covariates for values known up to the origin, and past-future covariates for values known over the horizon. Mixing these up is leakage.
4. Check length: history plus horizon, with no blanks. A covariate with partial history is not usable.
5. Compare with and without the driver on the same origins, using the same arguments. Keep it only if the gain is larger than the backtest noise band (proposed default: a paired bootstrap interval above zero).
6. Test one driver at a time, so any gain can be attributed.
7. An effect visible only in-sample is not evidence. Test rather than assume: a repo test found calendar covariates gave no gain on the synthetic twin.
8. In the report, state each driver: what it is, its source, its horizon coverage and its backtest result.

```python
import prophet_lib as pl
idx = s.index                                    # the dates saved with the series
promo = pl.event_covariate(idx, [("2026-11-27", "2026-12-02")], horizon=28, freq="D")
hol = pl.holiday_covariate(idx, "AU", horizon=28, freq="D")
assert len(promo) == len(s) + 28 and len(hol) == len(s) + 28
```

Decision rules:
- If the driver is a plan rather than an actual, confirm the plan is fixed before the origin.
- If the driver helps one origin only, treat it as no effect.
- If a driver is unknown for part of the horizon, fill the horizon with a scenario and label it as one.

Do not forecast with a promotion plan that is not agreed. Do not use weather or sales that only become known after the period. Do not add several drivers at once and credit the best one. Do not assume a public holiday means a shop closure. Do not let a covariate length differ from history plus horizon.
