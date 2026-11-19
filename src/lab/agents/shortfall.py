"""Shortfall and buy list: will stock plus inbound cover forecast demand until the next delivery could arrive?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.state import pipeline_at, stock_at, unit_cost_at, weekly_matrix
from .base import AgentContext, Proposal, dstr, inv_cdf, interp_cdf, series_evidence, severity_for

NAME = "shortfall"
DEFAULTS = {"min_history_weeks": 6, "risk_threshold": 0.55, "min_dollars": 150.0, "review_gap_days": 4,
            "default_lead": 3, "cover_percentile": 0.80}


def _lead_days(w, product_id: str, as_of: pd.Timestamp, default: int) -> int:
    p = w.pos[(w.pos.product_id == product_id) & (w.pos.created_on < as_of)].tail(10)
    if len(p) < 3:
        return default
    return int(max(1, round((p.expected_on - p.created_on).dt.days.median())))


def _last_supplier(w, product_id: str, as_of: pd.Timestamp):
    p = w.pos[(w.pos.product_id == product_id) & (w.pos.created_on < as_of)].tail(1)
    return None if p.empty else p.supplier_id.iloc[0]


def run(ctx: AgentContext) -> list[Proposal]:
    P = {**DEFAULTS, **ctx.params}
    w, as_of = ctx.world, ctx.as_of
    mat, weeks = weekly_matrix(w, as_of)
    if mat.shape[1] < P["min_history_weeks"]:
        return []
    stock = stock_at(w, as_of)
    on_hand = stock.groupby("product_id").qty.sum()
    pipe = pipeline_at(w, as_of)
    cost = unit_cost_at(w, as_of)
    prod = w.products.set_index("product_id")
    res = ctx.fs.weekly(as_of, 4)
    out: list[Proposal] = []
    for k, pi in enumerate(res["active"]):
        pid = w.products.product_id.iloc[pi]
        lead = _lead_days(w, pid, as_of, P["default_lead"])
        window = lead + P["review_gap_days"]
        d = ctx.fs.daily_demand_quantiles(as_of, int(pi), window)
        if not d or d["q50"] <= 0:
            continue
        oh = float(on_hand.get(pid, 0.0))
        inbound = float(pipe[(pipe.product_id == pid) & (pipe.expected_on <= as_of + pd.Timedelta(days=window))].qty_ordered.sum())
        position = oh + inbound
        p_short = 1.0 - interp_cdf(position, d["q10"], d["q50"], d["q90"])
        if p_short < P["risk_threshold"]:
            continue
        price = float(prod.unit_price.get(pid, 0) or 0); c = float(cost.get(pid, 0) or 0)
        margin = max(price - c, 0.0)
        exp_short = (max(d["q50"] - position, 0.0) + max(d["q90"] - position, 0.0)) / 2
        dollars = exp_short * (margin if margin > 0 else price * 0.15)
        if dollars < P["min_dollars"]:
            continue
        shelf = int(prod.shelf_life_days.get(pid, 0) or 0)
        cover_days = 3 if shelf <= 7 else 5 if shelf <= 21 else 10
        cover_days = min(cover_days, max(1, int(shelf * 0.35))) if shelf else cover_days
        dt = ctx.fs.daily_demand_quantiles(as_of, int(pi), window + cover_days)
        target = inv_cdf(P["cover_percentile"], dt["q10"], dt["q50"], dt["q90"]) if dt else d["q90"]
        pack = 6.0 if shelf > 21 else 1.0
        qty = float(np.ceil(max(target - position, 0.0) / pack) * pack)
        fc = res["fc"]
        conf = float(np.clip(1.0 - (fc.q[k, 0, 8] - fc.q[k, 0, 0]) / (2 * max(fc.q[k, 0, 4], 1.0)), 0.25, 0.95))
        title = f"{prod.display_name[pid]}: likely to run short within {window} days"
        # Motivation vs Logic
        # Motivation: the weekly card started at the week containing as_of, which is only partly known, so its first point
        # mixed known and unknown days. Logic: chart the same derived daily path the sell-down and freshness cards use.
        path = ctx.fs.daily_path(as_of, int(pi), 14)
        dem_end = int(w.days.searchsorted(as_of))
        day_hist = list(w.demand[pi, max(0, dem_end - 28):dem_end])
        day_x = [dstr(x) for x in w.days[max(0, dem_end - 28):dem_end]]
        ev = {
            "series": series_evidence(day_x, day_hist, [dstr(x) for x in path["dates"]], path["q10"], path["q50"], path["q90"],
                                      "units / day", "Daily demand and forecast band") if path else None,
            "facts": [
                {"label": "On hand", "value": round(oh, 1), "unit": "units"},
                {"label": "Inbound before window ends", "value": round(inbound, 1), "unit": "units"},
                {"label": f"Forecast demand next {window}d (median / high)", "value": f"{d['q50']:.0f} / {d['q90']:.0f}", "unit": "units"},
                {"label": "Chance of running short", "value": f"{p_short * 100:.0f}%", "unit": ""},
                {"label": "Supplier lead time (recent POs)", "value": lead, "unit": "days"},
                {"label": "Margin per unit", "value": round(margin, 2), "unit": "$"},
            ],
            "why": f"Stock plus inbound ({position:.0f} units) sits below the forecast median demand over the next {window} days ({d['q50']:.0f} units).",
            "method": "TimesFM 3 median for each week, split by this product's weekday pattern; position = on hand + inbound by window end.",
        }
        out.append(Proposal(NAME, dstr(as_of), "product", pid, title, severity_for(dollars, conf), round(dollars, 2), round(conf, 2),
                            ev, {"type": "raise_purchase_order", "product_id": pid, "qty": qty, "supplier": _last_supplier(w, pid, as_of),
                                 "needed_by": dstr(as_of + pd.Timedelta(days=window)), "est_cost": round(qty * c, 2)},
                            f"shortfall|{pid}", owner="buyer").finalise())
    out.sort(key=lambda p: -p.dollars_at_stake)
    return out
