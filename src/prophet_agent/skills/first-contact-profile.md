name: first-contact-profile
summary: Profile a new dataset before modelling: schema, keys, time index, coverage and units.

# First contact: profile before you model

Use this at the start of every analysis, before cleaning, forecasting or charting. Its job is to decide whether the data can be forecast at all.

1. Record provenance: what one row means, the unit, the currency, the timezone, and whether values are period totals or cumulative. If any of these is unknown for a value column, stop and ask.
2. Load with `pl.load`. Read `df.attrs["report"]` for converted columns, failed cells and ambiguous dates. Never ignore the report.
3. Check keys: the series identifier plus the timestamp must be unique. Identical duplicate rows: drop and log the count. Same key with conflicting values: stop that series.
4. Check the time index: parse, sort, find the regular step, count missing stamps, and detect a partial last period.
5. Measure coverage per series: first and last date with a valid value, count of active periods, and leading and trailing zeros. Fewer than about two seasonal cycles of active history means not forecastable at this grain (proposed default).
6. Find the period with `pl.dominant_period` and confirm it with `pl.find_season`. Treat a weak or multiple period as a question, not a fact.
7. Compare levels over time. A median jump of 10 times or more with no documented event is a unit change: stop and escalate (proposed default).

```python
import prophet_lib as pl
df = pl.load("in/sales.csv")
print(df.attrs["report"])                              # converted columns, failed cells
s, rep = pl.prepare_series(df, value="units", time="date", freq="D")
print(rep)                                             # every choice made, including trimmed tails
period, strength = pl.dominant_period(s.to_numpy())
print(period, strength)
```

Decision rules:
- If a column that should be numeric holds free text in more than a small share of rows, fix the parse before anything else.
- If a whole group of series stops on the same date, suspect a feed outage, not demand.
- If identifiers have lost leading zeros, reload them as text.
- If the profile contradicts the brief (wrong grain, wrong unit, wrong date range), report that before modelling.

Record the profile as a short table (rows, keys, date range, coverage, flags) and add it to the ledger. A clean-looking plot is a hypothesis, not a finding.

Do not model before the profile is recorded. Do not drop series silently. Do not treat a complete-looking table as complete. Do not ignore the loader's report. Do not assume a unit or a timezone that nobody stated.
