"""Freshness sell-down: which lots will not sell before use-by, and who normally buys them?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.state import stock_at, unit_cost_at, weekly_matrix
from .base import AgentContext, Proposal, dstr, series_evidence, severity_for

NAME = "freshness"
DEFAULTS = {"horizon_days": 21, "min_dollars": 80.0, "min_share_at_risk": 0.25, "demand_percentile": "q50", "top_buyers": 5}


def run(ctx: AgentContext) -> list[Proposal]:
    P = {**DEFAULTS, **ctx.params}
    w, as_of = ctx.world, ctx.as_of
    stock = stock_at(w, as_of)
    stock = stock[stock.use_by_date.notna()].copy()
    stock["dte"] = (stock.use_by_date - as_of).dt.days
    stock = stock[(stock.dte >= 0) & (stock.dte <= P["horizon_days"])]
    if stock.empty:
        return []
    cost = unit_cost_at(w, as_of)
    prod = w.products.set_index("product_id")
    mat, weeks = weekly_matrix(w, as_of)
    res = ctx.fs.weekly(as_of, 4)
    pos_in_fc = {int(pi): k for k, pi in enumerate(res["active"])}
    out: list[Proposal] = []
    recent = w.lines[(w.lines.delivery_date >= as_of - pd.Timedelta(days=56)) & (w.lines.delivery_date < as_of)]
    for pid, lots in stock.groupby("product_id"):
        pi = w.pidx[pid]
        if pi not in pos_in_fc:
            demand_per_day = None
        lots = lots.sort_values("use_by_date")
        horizon = int(lots.dte.max()) + 1
        d = ctx.fs.daily_demand_quantiles(as_of, pi, horizon)
        if not d:
            continue
        per_day = d[P["demand_percentile"]] / max(horizon, 1)
        remaining_demand_pool = 0.0
        cum_before = 0.0
        for lot in lots.itertuples():
            consumable = per_day * (lot.dte + 1) - cum_before          # demand available to this lot before expiry
            used = float(np.clip(consumable, 0, lot.qty))
            cum_before += lot.qty
            at_risk = lot.qty - used
            share = at_risk / lot.qty if lot.qty else 0
            c = float(lot.unit_cost if pd.notna(lot.unit_cost) else cost.get(pid, 0) or 0)
            dollars = at_risk * c
            if dollars < P["min_dollars"] or share < P["min_share_at_risk"]:
                continue
            buyers = (recent[recent.product_id == pid].groupby("customer_id")
                      .agg(qty=("qty", "sum"), last=("delivery_date", "max"), orders=("order_id", "nunique"))
                      .sort_values("qty", ascending=False).head(P["top_buyers"]))
            tgt = [{"customer_id": cid, "name": w.customers.set_index("customer_id").display_name.get(cid, cid),
                    "qty_8w": round(r.qty, 1), "days_since_last": int((as_of - r.last).days), "orders": int(r.orders)}
                   for cid, r in buyers.iterrows()]
            discount = 30 if lot.dte <= 2 else 20 if lot.dte <= 4 else 10
            conf = float(np.clip(0.55 + 0.4 * share - 0.05 * max(0, lot.dte - 5) / 5, 0.3, 0.95))
            hist_x = [dstr(x) for x in weeks[-12:]]
            fx = [dstr(res["future_start"] + pd.Timedelta(days=7 * i)) for i in range(4)]
            k = pos_in_fc.get(pi)
            fc = res["fc"]
            ev = {
                "series": series_evidence(hist_x, mat[pi, -12:], fx, fc.q[k, :, 0], fc.q[k, :, 4], fc.q[k, :, 8], "units / week",
                                          "Weekly demand and forecast band") if k is not None else None,
                "facts": [
                    {"label": "Lot", "value": lot.lot_id, "unit": ""},
                    {"label": "Quantity on hand", "value": round(lot.qty, 1), "unit": "units"},
                    {"label": "Use-by", "value": dstr(lot.use_by_date), "unit": f"({lot.dte} days)"},
                    {"label": f"Forecast sells before use-by", "value": round(used, 1), "unit": "units"},
                    {"label": "Expected to expire unsold", "value": round(at_risk, 1), "unit": "units"},
                    {"label": "Cost at risk", "value": round(dollars, 2), "unit": "$"},
                ],
                "why": f"{at_risk:.0f} of {lot.qty:.0f} units are expected to still be on the shelf at use-by, after first-expiry-first-out against the forecast.",
                "method": "Weekly TimesFM 3 median demand spread evenly over the window; lots consumed first-expiry-first-out.",
                "targets": tgt,
            }
            out.append(Proposal(NAME, dstr(as_of), "lot", lot.lot_id,
                                f"{prod.display_name[pid]}: {at_risk:.0f} units expire in {lot.dte} days",
                                severity_for(dollars, conf, hi=1200, mid=300), round(dollars, 2), round(conf, 2), ev,
                                {"type": "sell_down_offer", "product_id": pid, "lot_id": lot.lot_id, "qty": round(at_risk, 1),
                                 "suggested_discount_pct": discount, "valid_until": dstr(lot.use_by_date - pd.Timedelta(days=1)),
                                 "target_customers": tgt},
                                f"freshness|{lot.lot_id}", owner="sales").finalise())
    out.sort(key=lambda p: -p.dollars_at_stake)
    return out
