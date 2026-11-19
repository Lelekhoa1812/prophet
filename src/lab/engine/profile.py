"""Tenant profile: the shape of this client's business, measured from its own history.

Motivation vs Logic
Motivation: rules such as 'customers under $400 a week are too small to judge', 'a line of 25 units is large' and 'every
customer earns 18%' describe one business. Another client may sell by the pallet or earn 6%.
Logic: measure each of them from the trailing 12 weeks before the decision date, as percentiles and actual margins, so a
rule says 'smaller than 90% of this client's customers' instead of a dollar figure. Cached per week.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .state import WorldData, unit_cost_at, week_start

_CACHE: dict[tuple, dict] = {}
MARGIN_BOUNDS = (0.03, 0.60)


def tenant_profile(w: WorldData, as_of: pd.Timestamp, weeks: int = 12) -> dict:
    key = (w.name, week_start(as_of), weeks)
    if key in _CACHE:
        return _CACHE[key]
    lo = as_of - pd.Timedelta(days=7 * weeks)
    L = w.lines[(w.lines.delivery_date < as_of) & (w.lines.delivery_date >= lo)]
    prof = {"weeks": weeks, "customers": 0, "spend_pcts": None, "qty_pcts": None, "tenant_margin": None, "customer_margin": {}, "product_margin_p25": None}
    if len(L) == 0:
        _CACHE[key] = prof
        return prof
    v = L.qty * L.unit_price.fillna(0)
    cost = unit_cost_at(w, as_of)
    c = L.qty * L.product_id.map(cost).fillna(0)
    spend = v.groupby(L.customer_id).sum() / weeks
    prof["customers"] = int(len(spend))
    prof["spend_pcts"] = np.percentile(spend.values, np.arange(0, 101, 5)).tolist()          # index = percentile / 5
    prof["qty_pcts"] = np.percentile(w.lines[w.lines.delivery_date < as_of].qty.values, np.arange(0, 101, 1)).tolist()   # index = percentile
    cust_v, cust_c = v.groupby(L.customer_id).sum(), c.groupby(L.customer_id).sum()
    tm = 1 - c.sum() / v.sum() if v.sum() > 0 else None
    prof["tenant_margin"] = None if tm is None else float(np.clip(tm, *MARGIN_BOUNDS))
    ok = cust_v > 0
    prof["customer_margin"] = {k: float(np.clip(1 - cust_c[k] / cust_v[k], *MARGIN_BOUNDS)) for k in cust_v.index[ok]}
    pv, pc = v.groupby(L.product_id).sum(), c.groupby(L.product_id).sum()
    pm = (1 - pc / pv)[pv > 0].dropna()
    prof["product_margin_p25"] = float(np.clip(np.percentile(pm.values, 25), *MARGIN_BOUNDS)) if len(pm) else None
    _CACHE[key] = prof
    return prof


def spend_at_percentile(prof: dict, pct: float, fallback: float) -> float:
    s = prof.get("spend_pcts")
    return float(np.interp(pct, np.arange(0, 101, 5), s)) if s else fallback


def qty_at_percentile(prof: dict, pct: float, fallback: float) -> float:
    q = prof.get("qty_pcts")
    return float(np.interp(pct, np.arange(0, 101, 1), q)) if q else fallback


def margin_for(prof: dict, customer_id: str, fallback: float = 0.18) -> float:
    return prof.get("customer_margin", {}).get(customer_id, prof.get("tenant_margin") or fallback)
