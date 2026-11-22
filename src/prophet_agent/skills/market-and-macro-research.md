name: market-and-macro-research
summary: What to research before forecasting FX, commodities, rates or equities, and how to turn it into scenarios.

# Market and macro research

A price series is driven by information the history cannot show. Research it before the forecast, and keep it separate from the data.

1. Identify the driver family. FX: interest-rate differentials and the policy path of both central banks, inflation and growth surprises, terms of trade, risk sentiment, intervention. Crude and energy: OPEC+ decisions and compliance, inventories, demand growth, sanctions and conflict, refining margins, the dollar. Metals and agriculture: weather, stocks-to-use, China demand, export policy. Rates: the policy path, inflation prints, issuance. Equities: earnings dates, guidance, index events, sector news, rates.
2. Build the event calendar inside the horizon: central-bank meetings, CPI and jobs releases, OPEC meetings, earnings, elections, auctions, option expiries. Dates only, from a primary source: the calendar pages that `web_research` reads in round 0 (its `family` registry), not a news story. An event with no dated passage is listed as unconfirmed. These are candidates for `build_drivers` events and for risk windows in the interval statement.
3. State the current regime in numbers with dates: the latest level, the 1-month and 12-month change, the realised volatility from the data, and what the market-implied view is where a primary source gives it. Compare the data's last date with the news date: if the file ends before the news, say the forecast starts from stale data.
4. Write 2 to 3 scenarios (base, upside, downside) each with a trigger, a direction and a rough size taken from history (for example the largest 20-day move in the data), never from a headline.
5. Use the scenarios in this order of strength: (a) widen or shift the interval statement and name the risk windows, (b) add an event driver only if backtest shows it helps, (c) never move the point forecast by hand. Evidence on judgemental adjustment is mixed to negative: adjustments helped only about half of items in published studies.
6. Always say that a random-walk-like series is not predictable day to day: the honest forecast is a level with a wide, growing interval. The research explains the risk, it does not remove it.
7. Keep a source list: source id, URL, date and date_source, tier, the quote used. Every number taken from the web needs a source id and must appear in that source's passage; a number without one is not reported.

```python
import pandas as pd
move20 = s.pct_change(20).abs().quantile(0.95)       # size a scenario from the data's own 20-period moves, not from a headline
print(f"95th percentile 20-period move: {move20:.1%}")
```

Decision rules:
- If the news is newer than the data, name the gap and its likely effect.
- If scenarios cannot be tied to a dated primary source, label them speculation.
- If a source is a forecast by a bank or agency, report it as that institution's view, with its date.

Do not give investment advice or tell the person to trade. Do not adjust a point forecast from a headline. Do not treat consensus as truth. Do not mix web numbers into the series. Do not use information dated after the forecast origin when testing a method in a backtest.
