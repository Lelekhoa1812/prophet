"""Self-hosted web research: search, safe page fetching, source ranking and quotable passages.

Motivation vs Logic
Motivation: the forecasting agent needs outside evidence (prices, policy, published outlooks) with citations it can check.
Logic: each module does one job and never raises into the caller. netguard guards every outbound request, search finds
candidate pages, fetch reads them, rank orders them by source tier, and passages selects the quotable text. The tools
that the goal loop calls live in prophet_agent.tools.research_tools.
"""
