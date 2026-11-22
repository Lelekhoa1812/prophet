name: deliverables-and-handover
summary: Hand over results with trust status, baselines beside forecasts, a decision ledger and sourced charts.

# Deliverables and handover

A result is only usable when a reader can see its status, its baseline, what was changed and what was not tested.

1. Lead with the answer and a trust status per series: fit-for-use, use-with-caveat, or not forecastable. Put the reason beside the status.
2. Show naive and seasonal naive beside every forecast. If a baseline wins on the primary metric, say so in the first paragraph.
3. Report the share of data treated, not just the count of issues. For example: 2.1 percent of rows interpolated, 0 rows dropped.
4. State what the backtest did not cover: horizons not tested, regimes absent, series excluded and why.
5. Keep the decision log: each treatment and model choice, its evidence, the option rejected and why, and the run id. Save it with `ps.Ledger().save(path="out/ledger.json")`.
6. Make charts with `pl.chart`. Give a title, a subtitle stating the data status, a unit, and a source line. Save under out/.
7. End with the assumptions and open questions a reader must check before acting.

```python
import prophet_lib as pl
import prophet_stats as ps
led = ps.Ledger()
led.add("impute", "2.1 percent of rows interpolated, 0 dropped", "linear, gaps up to 3", evidence="out/impute.json")
led.save(path="out/ledger.json")
pl.chart("line", data=fc, title="Weekly demand forecast holds near its recent level", subtitle="Backtest measured", unit="units", path="out/forecast.png")
```

Decision rules:
- If a gate failed, report the failure in the same section as the result.
- If a figure comes from synthetic replay, label it synthetic in the same sentence.
- If a series was excluded, give the count and the reason in the summary.

Do not present synthetic numbers as observed. Do not hide a failed gate. Do not omit the baselines. Do not call a chart clean without its source line. Do not drop excluded series from the reported count.
