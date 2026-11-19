"""Customer order anomalies: slowdowns, lapses, and keying errors on today's lines."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.state import customer_fc, week_start
from .base import AgentContext, Proposal, dstr, series_evidence, severity_for

NAME = "customer"
DEFAULTS = {"min_weekly_spend": 400.0, "slowdown_ratio": 0.5, "min_active_weeks": 9, "lapse_min_days": 21, "lapse_gap_multiple": 4.0,
            "lapse_min_orders": 12, "keying_multiple": 8.0, "keying_min_qty": 25.0, "gross_margin": 0.18}


def _slowdown(ctx: AgentContext, P: dict) -> list[Proposal]:
    """Two weeks in a row below the forecast's lowest band, for customers who order steadily enough to judge."""
    w, as_of = ctx.world, ctx.as_of
    ws = week_start(as_of)
    prev1, prev2 = ws - pd.Timedelta(days=7), ws - pd.Timedelta(days=14)
    f1, f2, now = customer_fc(ctx.fs, prev1, 2), customer_fc(ctx.fs, prev2, 2), customer_fc(ctx.fs, as_of, 2)
    if f1["fc"] is None or f2["fc"] is None or now["fc"] is None:
        return []
    out = []
    names = w.customers.set_index("customer_id").display_name
    p1 = {int(i): k for k, i in enumerate(f1["active"])}
    p2 = {int(i): k for k, i in enumerate(f2["active"])}
    for k, ci in enumerate(now["active"]):
        ci = int(ci)
        if ci not in p1 or ci not in p2:
            continue
        series = now["mat"][ci]
        if (series[-12:] > 0).sum() < P["min_active_weeks"]:
            continue
        last, before = series[-1], series[-2]            # last completed week, and the one before
        band_last, band_before = f1["fc"].q[p1[ci], 0], f2["fc"].q[p2[ci], 0]
        q50, q10 = float(band_last[4]), float(band_last[0])
        if q50 < P["min_weekly_spend"]:
            continue
        ratio = last / q50 if q50 else 1.0
        if not (last < q10 and before < float(band_before[0]) and ratio <= P["slowdown_ratio"]):
            continue
        cid = now["cids"][ci]
        gap = max(q50 - last, 0) * 2
        dollars = gap * P["gross_margin"]
        conf = float(np.clip(1 - ratio, 0.4, 0.95))
        fx = [dstr(now["future_start"] + pd.Timedelta(days=7 * i)) for i in range(2)]
        ev = {
            "series": series_evidence([dstr(x) for x in now["weeks"][-12:]], series[-12:], fx, now["fc"].q[k, :, 0],
                                      now["fc"].q[k, :, 4], now["fc"].q[k, :, 8], "$ / week", "Weekly spend and forecast band"),
            "facts": [{"label": "Last week's spend", "value": round(last), "unit": "$"},
                      {"label": "Expected (median / low)", "value": f"{q50:.0f} / {q10:.0f}", "unit": "$"},
                      {"label": "Vs expectation", "value": f"{ratio * 100:.0f}%", "unit": ""},
                      {"label": "Revenue gap over two weeks", "value": round(gap), "unit": "$"}],
            "why": f"Spend fell to {ratio * 100:.0f}% of what was expected, below the forecast's lowest band, for a second week running.",
            "method": "Weekly TimesFM 3 forecast of each customer's spend, made before each week began, compared with what they ordered.",
        }
        out.append(Proposal(NAME, dstr(as_of), "customer", cid, f"{names.get(cid, cid)}: ordering well below pattern",
                            severity_for(gap, conf, 3000, 800), round(gap, 2), round(conf, 2), ev,
                            {"type": "customer_check_in", "customer_id": cid, "suggested": "Account manager call; check stock-outs and competitor offers"},
                            f"customer-slowdown|{cid}", owner="account_manager").finalise())
    return out


def _lapse(ctx: AgentContext, P: dict) -> list[Proposal]:
    w, as_of = ctx.world, ctx.as_of
    L = w.lines[w.lines.delivery_date < as_of]
    g = L.groupby("customer_id").delivery_date.apply(lambda s: np.sort(s.dt.normalize().unique()))
    names = w.customers.set_index("customer_id").display_name
    rev = L[L.delivery_date >= as_of - pd.Timedelta(days=90)].assign(v=lambda d: d.qty * d.unit_price.fillna(0)).groupby("customer_id").v.sum()
    out = []
    for cid, dates in g.items():
        if len(dates) < P["lapse_min_orders"]:
            continue
        gaps = np.diff(dates).astype("timedelta64[D]").astype(int)
        med = float(np.median(gaps[-12:]))
        since = (as_of - pd.Timestamp(dates[-1])).days
        threshold = max(P["lapse_min_days"], P["lapse_gap_multiple"] * max(med, 1))
        weekly = float(rev.get(cid, 0)) / 13
        if since < threshold or since > 120 or weekly < P["min_weekly_spend"] * 0.5:
            continue
        dollars = weekly * (since / 7) * P["gross_margin"] * 0.5 + weekly * 4 * P["gross_margin"]
        conf = float(np.clip(0.5 + 0.1 * (since / max(threshold, 1) - 1), 0.45, 0.92))
        ev = {"facts": [{"label": "Days since last order", "value": since, "unit": "days"},
                        {"label": "Usual gap between orders", "value": med, "unit": "days"},
                        {"label": "Typical weekly spend (13w)", "value": round(weekly), "unit": "$"},
                        {"label": "Last order", "value": dstr(dates[-1]), "unit": ""}],
              "why": f"{since} days without an order, against a usual gap of {med:.0f} {'day' if med == 1 else 'days'} between orders.",
              "method": "Order cadence from the customer's own history; flagged beyond 3x the usual gap."}
        out.append(Proposal(NAME, dstr(as_of), "customer", cid, f"{names.get(cid, cid)}: may have stopped ordering",
                            severity_for(dollars, conf, 3000, 800), round(dollars, 2), round(conf, 2), ev,
                            {"type": "customer_check_in", "customer_id": cid, "suggested": "Call before the account is lost"},
                            f"customer-lapse|{cid}", owner="account_manager").finalise())
    return out


def _keying(ctx: AgentContext, P: dict) -> list[Proposal]:
    """Quantity far above this customer's usual, or (with no pair history) far above anyone's usual for the product."""
    w, as_of = ctx.world, ctx.as_of
    today = w.lines[w.lines.delivery_date == as_of]
    if today.empty:
        return []
    hist = w.lines[(w.lines.delivery_date < as_of) & (w.lines.delivery_date >= as_of - pd.Timedelta(days=365))]
    pair = hist.groupby(["customer_id", "product_id"]).qty.agg(["median", "count"])
    prodq = hist.groupby("product_id").qty.agg(med="median", p99=lambda s: s.quantile(0.99), n="count")
    names = w.customers.set_index("customer_id").display_name
    prod = w.products.set_index("product_id")
    out = []
    for r in today.itertuples():
        key = (r.customer_id, r.product_id)
        basis = None
        if key in pair.index and pair.loc[key, "count"] >= 3:
            m = float(pair.loc[key, "median"]); basis = "this customer's own orders of the product"
            hit = r.qty >= P["keying_multiple"] * m and r.qty >= P["keying_min_qty"]
        elif r.product_id in prodq.index and prodq.loc[r.product_id, "n"] >= 20:
            m = float(prodq.loc[r.product_id, "med"]); p99 = float(prodq.loc[r.product_id, "p99"]); basis = "all customers' orders of the product (no history for this pair)"
            hit = r.qty >= max(P["keying_multiple"] * m, 3 * p99) and r.qty >= P["keying_min_qty"] * 1.6
        else:
            continue
        if not hit:
            continue
        price = float(r.unit_price or 0)
        dollars = (r.qty - m) * price
        conf = float(np.clip(0.6 + 0.04 * min(r.qty / max(m, 1), 12), 0.6, 0.97))
        ev = {"facts": [{"label": "Quantity on order", "value": r.qty, "unit": "units"},
                        {"label": "Usual quantity (median)", "value": m, "unit": "units"},
                        {"label": "Multiple of usual", "value": f"{r.qty / max(m, 1):.0f}x", "unit": ""},
                        {"label": "Order", "value": r.order_id, "unit": ""}],
              "why": f"{r.qty:.0f} units is {r.qty / max(m, 1):.0f} times the usual quantity, judged against {basis}.",
              "method": "Robust comparison with history; flagged beyond 8x the usual quantity."}
        out.append(Proposal(NAME, dstr(as_of), "pair", f"{r.customer_id}|{r.product_id}",
                            f"{names.get(r.customer_id, r.customer_id)}: {prod.display_name[r.product_id]} quantity looks wrong",
                            severity_for(dollars, conf, 1500, 400), round(dollars, 2), round(conf, 2), ev,
                            {"type": "hold_order_line", "order_id": r.order_id, "line_no": int(r.line_no), "usual_qty": m, "ordered_qty": r.qty},
                            f"keying|{r.order_id}|{r.line_no}", owner="order_desk").finalise())
    return out


def run(ctx: AgentContext) -> list[Proposal]:
    P = {**DEFAULTS, **ctx.params}
    out = _keying(ctx, P)
    if ctx.as_of.dayofweek == 0:                      # cadence and volume checks run on Mondays
        out += _slowdown(ctx, P) + _lapse(ctx, P)
    out.sort(key=lambda p: -p.dollars_at_stake)
    return out
