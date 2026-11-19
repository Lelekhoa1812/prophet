from . import customer, freshness, margin, shortfall

REGISTRY = {
    "shortfall": shortfall,
    "freshness": freshness,
    "customer": customer,
    "margin": margin,
}

META = {
    "shortfall": {"title": "Shortfall & buy list", "owner": "buyer", "cadence": "daily, before the 5 am briefing",
                  "summary": "Warns when stock plus inbound will not cover forecast demand, and drafts the purchase order."},
    "freshness": {"title": "Freshness sell-down", "owner": "sales", "cadence": "daily, before the 5 am briefing",
                  "summary": "Finds lots that will not sell before use-by and drafts a sell-down offer to the people who usually buy them."},
    "customer": {"title": "Customer order anomalies", "owner": "account_manager / order_desk",
                 "cadence": "Mondays for patterns, daily for today's lines",
                 "summary": "Spots customers ordering well below pattern or gone quiet, and quantities that look mis-keyed."},
    "margin": {"title": "Margin & price drift", "owner": "pricing / order_desk",
               "cadence": "Mondays for cost, daily for today's lines",
               "summary": "Catches supplier cost moves the sell price has not followed, and prices keyed far below normal."},
}
