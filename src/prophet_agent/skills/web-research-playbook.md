name: web-research-playbook
summary: Plan angles, fan out searches, read passages, cross-check, and report sourced, dated findings.

# Web research playbook

Outside evidence is context with a source and a date, not data. Research in rounds, and stop when a round adds nothing.

1. Write the question as a sentence the report must answer, then split it into 4 to 6 angles that would each change the forecast: current level and regime, scheduled events inside the horizon, policy and central-bank stance, supply or demand shocks, consensus forecasts, and what could go wrong. One search per angle, worded differently, never five paraphrases of one query.
2. Round 1: call `web_research` with those queries together (`kind` news for recent events, text for background; `recency` week or month for anything that moves). Read the passages it returns. Note each source's id, tier and date.
3. Prefer tier 1 sources (central banks, statistics offices, IMF, EIA, FRED, OECD, regulators) for facts, tier 2 outlets for events. Treat a tier 3 or 4 page as a lead to confirm, never as the answer. A claim with one weak source is reported as unconfirmed.
4. Round 2: list what is missing or conflicting, and search only for that. Use `web_fetch` with `queries` to pull the best passages from a page worth reading closely. Stop after round 3 or when a round brings no new claim.
5. Check dates. A page with no date, or older than the question's regime, is background. Say "as of DATE" for every level, rate or price. A figure from a search snippet is not a fact until a fetched passage shows it.
6. Cross-check: a number or event that matters needs two independent sources (not two copies of one wire story). Where sources disagree, report both and the gap.
7. Report as a list of findings, each with: the claim in one line, source id and URL, date, tier, and a short quote that appears in the fetched text. Then: open questions, conflicts, and what you could not find. Put the scenario implication last (what each finding does to level, trend or risk of the series), marked as your inference.
8. Never put a number in the report that a tool did not print. Never write a forecast; your job is evidence.

```python
# one round: several different angles at once (call the web_research tool with these arguments)
{"queries": ["WTI crude inventories EIA weekly report", "OPEC+ production decision November", "oil demand outlook IEA monthly report"],
 "kind": "news", "recency": "week", "fetch_top": 5}
```

Decision rules:
- If every search backend fails or returns nothing, say so; do not answer from memory.
- If the only evidence is opinion pieces, report that no primary evidence was found.
- If the question is about a future event, report the scheduled date and the stated position, not a prediction of the outcome.

Do not treat a search snippet as evidence. Do not cite a page you did not fetch. Do not repeat a query that returned nothing new. Do not present web claims as data from the person's file.
