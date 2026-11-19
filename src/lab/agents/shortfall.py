"""Shortfall and buy list: will stock plus inbound cover forecast demand until the next delivery could arrive?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.state import pipeline_at, stock_at, unit_cost_at, weekly_matrix
from ..engine import distribution as dist
from .base import AgentContext, Proposal, dstr, series_evidence, severity_for

NAME = "shortfall"
DEFAULTS = {"min_history_weeks": 6, "risk_threshold": 0.55, "min_dollars": 150.0, "review_gap_days": 4,
            "default_lead": 3, "cover_percentile": 0.80, "rho": 0.1}


def _lead_days(w, product_id: str, as_of: pd.Timestamp, default: int) -> int:
    p = w.pos[(w.pos.product_id == product_id) & (w.pos.created_on < as_of)].tail(10)
    if len(p) < 3:
        return default
    return int(max(1, round((p.expected_on - p.created_on).dt.days.median())))


def _last_supplier(w, product_id: str, as_of: pd.Timestamp):
    p = w.pos[(w.pos.product_id == product_id) & (w.pos.created_on < as_of)].tail(1)
    return None if p.empty else p.supplier_id.iloc[0]


def run(ctx: AgentContext) -> list[Proposal]:
    """Will stock plus inbound cover demand until a new order could arrive?

    Motivation vs Logic
    Motivation: a plug-in median hides skew, and scaling weekly quantiles by days/7 overstates the spread of a window.
    Logic: sample daily demand from the forecast distribution (weeks lightly correlated, weekday pattern applied),
    sum over the window, then compute the chance of running out and the expected lost margin directly:
    dollars = E[max(demand - position, 0)] x margin per unit. The order quantity covers `cover_percentile` of demand
    over the window plus a shelf-life-dependent cover.
    """
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
        shelf = int(prod.shelf_life_days.get(pid, 0) or 0)
        cover_days = 3 if shelf <= 7 else 5 if shelf <= 21 else 10
        cover_days = min(cover_days, max(1, int(shelf * 0.35))) if shelf else cover_days
        S = ctx.fs.demand_samples(as_of, int(pi), window + cover_days, rho=P["rho"])
        if S is None:
            continue
        total = S[:, :window].sum(axis=1)
        if total.mean() <= 0:
            continue
        oh = float(on_hand.get(pid, 0.0))
        inbound = float(pipe[(pipe.product_id == pid) & (pipe.expected_on <= as_of + pd.Timedelta(days=window))].qty_ordered.sum())
        position = oh + inbound
        p_short = dist.prob_exceeds(total, position)
        if p_short < P["risk_threshold"]:
            continue
        price = float(prod.unit_price.get(pid, 0) or 0); c = float(cost.get(pid, 0) or 0)
        margin = max(price - c, 0.0)
        exp_short = dist.expected_shortfall(total, position)
        dollars = exp_short * (margin if margin > 0 else price * 0.15)
        if dollars < P["min_dollars"]:
            continue
        cover_total = S.sum(axis=1)
        target = float(np.quantile(cover_total, P["cover_percentile"]))
        pack = 6.0 if shelf > 21 else 1.0
        qty = float(np.ceil(max(target - position, 0.0) / pack) * pack)
        dem_end = int(w.days.searchsorted(as_of))
        day_hist = list(w.demand[pi, max(0, dem_end - 28):dem_end])
        day_x = [dstr(x) for x in w.days[max(0, dem_end - 28):dem_end]]
        path = ctx.fs.daily_path(as_of, int(pi), 14)
        fc = res["fc"]
        conf = float(np.clip(1.0 - (fc.q[k, 0, 8] - fc.q[k, 0, 0]) / (2 * max(fc.q[k, 0, 4], 1.0)), 0.25, 0.95))
        p50, p90 = np.quantile(total, [0.5, 0.9])
        title = f"{prod.display_name[pid]}: likely to run short within {window} days"
        ev = {
            "series": series_evidence(day_x, day_hist, [dstr(x) for x in path["dates"]], path["q10"], path["q50"], path["q90"],
                                      "units / day", "Daily demand and forecast band") if path else None,
            "facts": [
                {"label": "On hand", "value": round(oh, 1), "unit": "units"},
                {"label": "Inbound before window ends", "value": round(inbound, 1), "unit": "units"},
                {"label": f"Demand next {window}d (typical / high)", "value": f"{p50:.0f} / {p90:.0f}", "unit": "units"},
                {"label": "Chance of running short", "value": f"{p_short * 100:.0f}%", "unit": ""},
                {"label": "Expected units short", "value": round(exp_short, 1), "unit": "units"},
                {"label": "Supplier lead time (recent POs)", "value": lead, "unit": "days"},
                {"label": "Margin per unit", "value": round(margin, 2), "unit": "$"},
            ],
            "why": f"Stock plus inbound ({position:.0f} units) is below demand in {p_short * 100:.0f}% of the forecast range over the next {window} days; expected shortfall {exp_short:.0f} units.",
            "method": "TimesFM 3 weekly forecast range, split by this product's weekday pattern and summed over the window; shortfall and chance computed over the whole range.",
        }
        out.append(Proposal(NAME, dstr(as_of), "product", pid, title, severity_for(dollars, conf), round(dollars, 2), round(conf, 2),
                            ev, {"type": "raise_purchase_order", "product_id": pid, "qty": qty, "supplier": _last_supplier(w, pid, as_of),
                                 "needed_by": dstr(as_of + pd.Timedelta(days=window)), "est_cost": round(qty * c, 2)},
                            f"shortfall|{pid}", owner="buyer", p_event=round(p_short, 4)).finalise())
    out.sort(key=lambda p: -p.dollars_at_stake)
    return out
