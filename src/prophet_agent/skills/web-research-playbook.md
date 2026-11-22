name: web-research-playbook
summary: Plan angles, fan out searches, read passages, cross-check, and report sourced, dated findings.

# Web research playbook

Outside evidence is context with a source and a date, not data. Research in rounds, and stop when a round adds nothing.

1. Write the question as a sentence the report must answer, then split it into 4 to 6 angles that would each change the forecast: current level and regime, scheduled events inside the horizon, policy and central-bank stance, supply or demand shocks, consensus forecasts, and what could go wrong.
2. Round 0, no search: call `web_research` with `family` set, or let it detect the family from the queries. It reads the primary registry pages for that series first (statistics offices, central banks, EIA, OPEC, IEA). Note each source's id, tier, date and date_source.
3. Round 1, in parallel: one `web_research` call with one query per angle. For each angle send two wordings: a `site:` query for the primary domain, and a news query (`kind` news, `recency` week or month) for anything that moves. Pass `exclude_urls` with every URL already read.
4. Round 2, gaps only: list the conflicts and the missing dated facts from rounds 0 and 1, and search only for those, at most two fetches per gap. Use `web_fetch` with `queries` to read one page closely.
5. Stop after round 2, which is three rounds at most, or earlier when a round adds no new primary domain and no new dated claim. Read `metrics` in each result: `tier1_sources`, `dated_sources`, `newest_date`, `new_sources`.
6. Prefer tier 1 sources (central banks, statistics offices, IMF, EIA, FRED, OECD, regulators) for facts, and tier 2 outlets for events. Treat a tier 3 or 4 page as a lead to confirm, never as the answer. A claim with one weak source is reported as unconfirmed.
7. Check dates. A source with date_source "none" is background, not dated evidence. Say "as of DATE" for every level, rate or price. A figure from a search snippet is not a fact until a fetched passage shows it.
8. Cross-check: a number or event that matters needs two independent sources, not two copies of one wire story. Where sources disagree, report both and the gap.
9. Report as a list of findings, each with: the claim in one line, source id and URL, date, tier, and a short quote from that source's passage. Every number you report from the web needs a source id and must appear in that source's passage; drop or mark unconfirmed any that do not. Then give open questions, conflicts, and what you could not find. Put the scenario implication last, marked as your inference.
10. Never put a number in the report that a tool did not print. Never write a forecast; your job is evidence.

```python
# round 1 for one angle: a primary-site query and a news query (call the web_research tool with these arguments)
{"queries": ["site:eia.gov weekly crude inventories", "WTI crude inventories this week"],
 "kind": "news", "recency": "week", "fetch_top": 4, "exclude_urls": ["https://www.eia.gov/outlook"]}
```

Decision rules:
- If every search backend fails or returns nothing, say so; do not answer from memory.
- If the only evidence is opinion pieces, report that no primary evidence was found.
- If the question is about a future event, report the scheduled date and the stated position, not a prediction of the outcome.

Do not treat a search snippet as evidence. Do not cite a page you did not fetch. Do not repeat a query that returned nothing new. Do not present web claims as data from the person's file.
