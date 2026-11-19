"""Freshness sell-down: which lots will not sell before use-by, and who normally buys them?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.state import stock_at, unit_cost_at
from .base import AgentContext, Proposal, dstr, series_evidence, severity_for

NAME = "freshness"
DEFAULTS = {"horizon_days": 21, "min_dollars": 200.0, "min_share_at_risk": 0.25, "top_buyers": 5}


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
    out: list[Proposal] = []
    recent = w.lines[(w.lines.delivery_date >= as_of - pd.Timedelta(days=56)) & (w.lines.delivery_date < as_of)]
    for pid, lots in stock.groupby("product_id"):
        pi = w.pidx[pid]
        lots = lots.sort_values("use_by_date")
        horizon = int(lots.dte.max()) + 1
        # Motivation vs Logic
        # Motivation: first-expiry-first-out against the median demand ignores that demand is uncertain and skewed,
        # so it under-counts stock that is only likely (not certain) to expire.
        # Logic: sample cumulative daily demand from the forecast range, run first-expiry-first-out on every sample,
        # and take the expected unsold quantity per lot (plus a 10th-90th percentile range for the evidence).
        S = ctx.fs.demand_samples(as_of, pi, horizon)
        path = ctx.fs.daily_path(as_of, pi, horizon)
        if S is None or not path:
            continue
        cum = np.cumsum(S, axis=1)                                      # (n, days) units sold through each day
        cum_before = 0.0
        at_risk_lots = []
        range_samples = np.zeros(S.shape[0])
        for lot in lots.itertuples():
            through = cum[:, min(int(lot.dte), cum.shape[1] - 1)]
            used = np.clip(through - cum_before, 0, lot.qty)            # demand available to this lot before expiry
            cum_before += lot.qty
            left = lot.qty - used
            at_risk = float(left.mean())
            c = float(lot.unit_cost if pd.notna(lot.unit_cost) else cost.get(pid, 0) or 0)
            if at_risk > 0 and at_risk / lot.qty >= P["min_share_at_risk"]:
                range_samples += left
                at_risk_lots.append({"lot_id": lot.lot_id, "qty": round(lot.qty, 1), "use_by": dstr(lot.use_by_date), "days_left": int(lot.dte),
                                     "sells_before_use_by": round(lot.qty - at_risk, 1), "at_risk_qty": round(at_risk, 1),
                                     "at_risk_p10": round(float(np.quantile(left, 0.1)), 1), "at_risk_p90": round(float(np.quantile(left, 0.9)), 1),
                                     "at_risk_value": round(at_risk * c, 2)})
        dollars = sum(l["at_risk_value"] for l in at_risk_lots)
        if dollars < P["min_dollars"]:
            continue
        at_risk_total = sum(l["at_risk_qty"] for l in at_risk_lots)
        on_hand_total = float(lots.qty.sum())
        share = at_risk_total / on_hand_total if on_hand_total else 0
        soonest = min(l["days_left"] for l in at_risk_lots)
        buyers = (recent[recent.product_id == pid].groupby("customer_id")
                  .agg(qty=("qty", "sum"), last=("delivery_date", "max"), orders=("order_id", "nunique"))
                  .sort_values("qty", ascending=False).head(P["top_buyers"]))
        cmap = w.customers.set_index("customer_id").display_name
        tgt = [{"customer_id": cid, "name": cmap.get(cid, cid), "qty_8w": round(r.qty, 1),
                "days_since_last": int((as_of - r.last).days), "orders": int(r.orders)} for cid, r in buyers.iterrows()]
        discount = 30 if soonest <= 2 else 20 if soonest <= 4 else 10
        conf = float(np.clip(0.55 + 0.4 * share - 0.05 * max(0, soonest - 5) / 5, 0.3, 0.95))
        # Daily chart: the last 28 days of real orders, then the daily path. Weekly totals are kept in the method note.
        dem_pi = w.demand[pi]
        end_i = int(w.days.searchsorted(as_of))
        hist_days = w.days[max(0, end_i - 28):end_i]
        hist_x = [dstr(x) for x in hist_days]
        hist_y = list(dem_pi[max(0, end_i - 28):end_i])
        # Chart a fixed two weeks so every card looks alike; the expiry maths above uses only the lot horizon.
        chart = ctx.fs.daily_path(as_of, pi, max(horizon, 14)) or path
        fx = [dstr(x) for x in chart["dates"]]
        ev = {
            "series": series_evidence(hist_x, hist_y, fx, chart["q10"], chart["q50"], chart["q90"], "units / day",
                                      "Daily demand and forecast band"),
            "facts": [
                {"label": "On hand", "value": round(on_hand_total, 1), "unit": "units"},
                {"label": "Lots at risk", "value": len(at_risk_lots), "unit": f"of {len(lots)}"},
                {"label": "Soonest use-by", "value": f"{soonest} days", "unit": ""},
                {"label": "Expected to expire unsold", "value": round(at_risk_total, 1), "unit": "units"},
                {"label": "Plausible range (10th-90th)", "value": f"{np.quantile(range_samples, 0.1):.0f}-{np.quantile(range_samples, 0.9):.0f}", "unit": "units"},
                {"label": "Cost at risk", "value": round(dollars, 2), "unit": "$"},
            ],
            "why": f"{at_risk_total:.0f} of {on_hand_total:.0f} units are expected to still be on the shelf at use-by, averaged over the forecast range with first-expiry-first-out.",
            "method": "TimesFM 3 forecast range for each week, split by this product's weekday pattern (last 12 weeks, shrunk toward flat); lots consumed first-expiry-first-out on each of 1,000 sampled demand paths, then averaged.",
            "targets": tgt, "lots": at_risk_lots,
        }
        out.append(Proposal(NAME, dstr(as_of), "product", pid,
                            f"{prod.display_name[pid]}: {at_risk_total:.0f} units expire within {soonest + 1} days",
                            severity_for(dollars, conf, hi=1500, mid=400), round(dollars, 2), round(conf, 2), ev,
                            {"type": "sell_down_offer", "product_id": pid, "qty": round(at_risk_total, 1),
                             "suggested_discount_pct": discount, "valid_until": dstr(as_of + pd.Timedelta(days=max(soonest, 1))),
                             "lots": at_risk_lots, "target_customers": tgt},
                            f"freshness|{pid}", owner="sales").finalise())
    out.sort(key=lambda p: -p.dollars_at_stake)
    return out
