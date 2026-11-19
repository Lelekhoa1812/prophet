"""Margin and price drift: supplier cost moves outside the forecast band, and sell prices keyed far below normal."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.profile import tenant_profile
from ..engine.state import cost_fc, week_start
from .base import AgentContext, Proposal, dstr, series_evidence, severity_for

NAME = "margin"
DEFAULTS = {"min_cost_rise": 0.06, "margin_drop_pts": 3.0, "margin_floor_pct": 8.0, "target_margin_pct": None, "price_floor_ratio": 0.85, "product_floor_ratio": 0.75, "min_dollars": 120.0, "price_min_dollars": 15.0}


def _cost_drift(ctx: AgentContext, P: dict) -> list[Proposal]:
    w, as_of = ctx.world, ctx.as_of
    prev = week_start(as_of) - pd.Timedelta(days=7)
    cur = cost_fc(ctx.fs, prev, 2); now = cost_fc(ctx.fs, as_of, 2)
    if cur["fc"] is None or now["fc"] is None:
        return []
    pos_cur = {int(i): k for k, i in enumerate(cur["active"])}
    recent_lines = w.lines[(w.lines.delivery_date < as_of) & (w.lines.delivery_date >= as_of - pd.Timedelta(days=28))]
    price28 = recent_lines.groupby("product_id").unit_price.median()
    wk_units = recent_lines.groupby("product_id").qty.sum() / 4
    prod = w.products.set_index("product_id")
    out = []
    for k, pi in enumerate(now["active"]):
        if int(pi) not in pos_cur:
            continue
        pid = w.products.product_id.iloc[pi]
        series = now["mat"][pi]
        last = series[-1]; base = float(np.nanmedian(series[-10:-2])) if len(series) > 10 else np.nan
        if not np.isfinite(last) or not np.isfinite(base) or base <= 0:
            continue
        q90 = float(cur["fc"].q[pos_cur[int(pi)], 0, 8])
        rise = last / base - 1
        price = float(price28.get(pid, np.nan))
        if rise < P["min_cost_rise"] or last <= q90 or not np.isfinite(price) or price <= 0:
            continue
        old_m = (price - base) / price * 100; new_m = (price - last) / price * 100
        if old_m - new_m < P["margin_drop_pts"] and new_m > P["margin_floor_pct"]:
            continue
        units = float(wk_units.get(pid, 0))
        dollars = units * (last - base) * 4          # four weeks of exposure at current volumes
        if dollars < P["min_dollars"]:
            continue
        target_m = max(old_m, P["target_margin_pct"])   # never recommend restoring a negative or tiny margin
        restore = last / (1 - target_m / 100)
        conf = float(np.clip(0.55 + min(rise, 0.3), 0.55, 0.95))
        fx = [dstr(now["future_start"] + pd.Timedelta(days=7 * i)) for i in range(2)]
        ev = {
            "series": series_evidence([dstr(x) for x in now["weeks"][-12:]], np.nan_to_num(series[-12:]), fx, now["fc"].q[k, :, 0],
                                      now["fc"].q[k, :, 4], now["fc"].q[k, :, 8], "$ / unit", "Weekly purchase cost and forecast band"),
            "facts": [{"label": "Latest purchase cost", "value": round(last, 2), "unit": "$"},
                      {"label": "Usual cost (prior 8 weeks)", "value": round(base, 2), "unit": "$"},
                      {"label": "Change", "value": f"+{rise * 100:.0f}%", "unit": ""},
                      {"label": "Margin now vs before", "value": f"{new_m:.1f}% vs {old_m:.1f}%", "unit": ""},
                      {"label": "Sell price (28d median)", "value": round(price, 2), "unit": "$"},
                      {"label": "Four-week exposure", "value": round(dollars), "unit": "$"}],
            "why": f"Cost rose {rise * 100:.0f}% above its usual level and above the forecast band, while the sell price has not moved.",
            "method": "Weekly TimesFM 3 forecast of purchase cost, made before the week; flagged when actual cost exceeds the 90% band.",
        }
        out.append(Proposal(NAME, dstr(as_of), "product", pid, f"{prod.display_name[pid]}: cost up {rise * 100:.0f}%, margin shrinking",
                            severity_for(dollars, conf, 2500, 600), round(dollars, 2), round(conf, 2), ev,
                            {"type": "review_price", "product_id": pid, "current_price": round(price, 2),
                             "suggested_price": round(restore, 2), "target_margin_pct": round(target_m, 1), "previous_margin_pct": round(old_m, 1)},
                            f"margin-cost|{pid}", owner="pricing").finalise())
    return out


def _price_floor(ctx: AgentContext, P: dict) -> list[Proposal]:
    """Sell price far below this customer's usual, or (no pair history) far below the product's usual across customers."""
    w, as_of = ctx.world, ctx.as_of
    today = w.lines[w.lines.delivery_date == as_of]
    if today.empty:
        return []
    hist = w.lines[(w.lines.delivery_date < as_of) & (w.lines.delivery_date >= as_of - pd.Timedelta(days=180))]
    pair = hist.groupby(["customer_id", "product_id"]).unit_price.agg(["median", "count"])
    prodp = hist[hist.delivery_date >= as_of - pd.Timedelta(days=90)].groupby("product_id").unit_price.agg(["median", "count"])
    names = w.customers.set_index("customer_id").display_name
    prod = w.products.set_index("product_id")
    out = []
    for r in today.itertuples():
        if not r.unit_price:
            continue
        key = (r.customer_id, r.product_id)
        if key in pair.index and pair.loc[key, "count"] >= 3:
            typical = float(pair.loc[key, "median"]); floor = P["price_floor_ratio"]; basis = "this customer's usual price"
        elif r.product_id in prodp.index and prodp.loc[r.product_id, "count"] >= 5:
            typical = float(prodp.loc[r.product_id, "median"]); floor = P["product_floor_ratio"]; basis = "the product's usual price across customers"
        else:
            continue
        if r.unit_price >= floor * typical:
            continue
        dollars = (typical - r.unit_price) * r.qty
        if dollars < P["price_min_dollars"]:
            continue
        conf = float(np.clip(0.55 + (1 - r.unit_price / typical), 0.55, 0.95))
        ev = {"facts": [{"label": "Price on order", "value": round(r.unit_price, 2), "unit": "$"},
                        {"label": "Usual price (median)", "value": round(typical, 2), "unit": "$"},
                        {"label": "Below usual by", "value": f"{(1 - r.unit_price / typical) * 100:.0f}%", "unit": ""},
                        {"label": "Margin given away on this line", "value": round(dollars), "unit": "$"},
                        {"label": "Order", "value": r.order_id, "unit": ""}],
              "why": f"The price is {(1 - r.unit_price / typical) * 100:.0f}% below {basis}.",
              "method": "Compared with price history; flagged below 85% of the customer's usual, or 75% of the product's usual when the pair is new."}
        out.append(Proposal(NAME, dstr(as_of), "pair", f"{r.customer_id}|{r.product_id}",
                            f"{names.get(r.customer_id, r.customer_id)}: {prod.display_name[r.product_id]} priced far below usual",
                            severity_for(dollars, conf, 1200, 300), round(dollars, 2), round(conf, 2), ev,
                            {"type": "hold_order_line", "order_id": r.order_id, "line_no": int(r.line_no), "usual_price": typical, "ordered_price": float(r.unit_price)},
                            f"price|{r.customer_id}|{r.product_id}", owner="order_desk").finalise())
    return out


def run(ctx: AgentContext) -> list[Proposal]:
    P = {**DEFAULTS, **ctx.params}
    if P["target_margin_pct"] is None:       # the margin worth restoring: this client's own 25th-percentile product margin
        m = tenant_profile(ctx.world, ctx.as_of).get("product_margin_p25")
        P["target_margin_pct"] = 100 * m if m else 12.0
    out = _price_floor(ctx, P)
    if ctx.as_of.dayofweek == 0:
        out += _cost_drift(ctx, P)
    out.sort(key=lambda p: -p.dollars_at_stake)
    return out
