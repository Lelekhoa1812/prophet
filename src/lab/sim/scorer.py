"""Score proposals against what really happened in the twin (precision, recall, detection delay, dollars)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..store import open_dataset
from . import event_dollars as ed


def _clusters(days: list[pd.Timestamp]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not days:
        return []
    days = sorted(days)
    out, s, e = [], days[0], days[0]
    for d in days[1:]:
        if (d - e).days <= 2:
            e = d
        else:
            out.append((s, e)); s = e = d
    out.append((s, e))
    return out


def score(dataset: str, proposals: list[dict], window: tuple[pd.Timestamp, pd.Timestamp]) -> dict:
    """proposals: dicts with agent, entity_type, entity_id, as_of (str), dollars_at_stake, severity, dedupe_key, action."""
    con = open_dataset(dataset, read_only=True)
    truth = con.execute("select * from truth_events").df()
    if truth.empty:
        con.close()
        return {"available": False}
    truth["start_date"] = pd.to_datetime(truth.start_date); truth["end_date"] = pd.to_datetime(truth.end_date)
    outc = con.execute("""select o.day, o.product_id, o.demand, o.fulfilled, o.expired_value, p.unit_price, p.unit_cost
                          from daily_outcomes o join products p using(product_id)""").df()
    outc["day"] = pd.to_datetime(outc.day)
    exp = con.execute("""select m.lot_id, m.product_id, m.moved_on, -m.qty_change as qty, l.unit_cost
                         from stock_movements m join lots l using(lot_id) where m.movement_type='Expired write-off'""").df()
    exp["moved_on"] = pd.to_datetime(exp.moved_on)
    # Motivation vs Logic: customer and margin dollars need the full line history (run-rates, pre-event prices and costs).
    # Bug: this query loaded only keying and price pairs, so customer and margin dollars could not be valued.
    # Fix: load every line, then value each event from its own pre-event baseline (sim/event_dollars.py).
    ol = con.execute("""select l.customer_id, l.product_id, l.delivery_date, l.qty, l.unit_price, l.unit_cost from order_lines l
                        where not coalesce(l.cancelled, false)""").df()
    ol["delivery_date"] = pd.to_datetime(ol.delivery_date)
    data_end = ol.delivery_date.max() if len(ol) else pd.Timestamp(hi)
    con.close()
    lo, hi = window
    P = pd.DataFrame(proposals)
    if P.empty:
        return {"available": True, "agents": {}, "events": [], "window": [str(lo.date()), str(hi.date())]}
    P["as_of"] = pd.to_datetime(P.as_of)
    res: dict = {"available": True, "agents": {}, "events": [], "window": [str(lo.date()), str(hi.date())]}

    def finish(agent: str, flags: list[bool], events: list[dict], extra: dict | None = None):
        sub = P[P.agent == agent]
        tp = int(sum(flags))
        sev_false = int(sum(1 for f, s in zip(flags, sub.severity) if (not f) and s == "high"))
        sev_total = int((sub.severity == "high").sum())
        caught = [e for e in events if e["caught"]]
        res.setdefault("flags", {})[agent] = dict(zip(sub.proposal_id.tolist(), [bool(f) for f in flags])) if "proposal_id" in sub else {}
        res["agents"][agent] = {
            "proposals": len(sub), "useful": tp, "precision": (tp / len(sub)) if len(sub) else None,
            "severe_false_rate": (sev_false / sev_total) if sev_total else 0.0,
            "recent_precision": (sum(flags[-20:]) / len(flags[-20:])) if flags else None,
            "events": len(events), "caught": len(caught), "recall": (len(caught) / len(events)) if events else None,
            "median_delay_days": float(np.median([e["delay_days"] for e in caught])) if caught else None,
            "dollars_at_stake_total": float(sub.dollars_at_stake.sum()),
            # Motivation vs Logic: an event that could not be valued (dollars None) is counted in recall but not in dollars.
            # Bug: unvalued events were summed as 0, which made missed events look worth nothing.
            # Fix: sum only valued events, and report how many were left out.
            "dollars_truth_caught": float(sum(e["dollars"] for e in caught if e["dollars"] is not None)),
            "dollars_truth_total": float(sum(e["dollars"] for e in events if e["dollars"] is not None)),
            "dollars_unvalued": int(sum(1 for e in events if e["dollars"] is None)),
            # Customer and margin dollars are estimates from a pre-event baseline, not recorded losses (see sim/event_dollars.py).
            "dollar_basis": "estimate" if agent in ("customer", "margin") else "recorded",
            **(extra or {}),
        }
        res["events"] += events

    # ---------- shortfall: stockout clusters
    sub = P[P.agent == "shortfall"]
    if len(sub):
        flags = []
        so = outc[(outc.demand > outc.fulfilled + 1e-6) & (outc.day >= lo) & (outc.day <= hi)]
        for r in sub.itertuples():
            win = so[(so.product_id == r.entity_id) & (so.day >= r.as_of) & (so.day <= r.as_of + pd.Timedelta(days=9))]
            flags.append(len(win) > 0)
        events = []
        for pid, g in so.groupby("product_id"):
            for a, b in _clusters(list(g.day)):
                seg = g[(g.day >= a) & (g.day <= b)]
                dollars = float(((seg.demand - seg.fulfilled) * (seg.unit_price - seg.unit_cost)).sum())
                if dollars < 150:
                    continue
                m = sub[(sub.entity_id == pid) & (sub.as_of >= a - pd.Timedelta(days=10)) & (sub.as_of <= b)]
                events.append({"agent": "shortfall", "kind": "STOCKOUT", "entity_id": pid, "start": str(a.date()), "end": str(b.date()),
                               "dollars": dollars, "caught": len(m) > 0,
                               "delay_days": float((a - m.as_of.min()).days) if len(m) else None, "event_id": f"SO-{pid}-{a:%m%d}"})
        finish("shortfall", flags, events)

    # ---------- freshness: lots that really expired (proposals are per product, listing the lots at risk)
    sub = P[P.agent == "freshness"]
    if len(sub):
        ex = exp.groupby("lot_id").agg(qty=("qty", "sum"), on=("moved_on", "max"), cost=("unit_cost", "first"), product_id=("product_id", "first")).reset_index()
        ex = ex[(ex.on >= lo) & (ex.on <= hi)]
        exmap = ex.set_index("lot_id")
        flags = []
        for r in sub.itertuples():
            lots = r.action.get("lots", [])
            hit = any(l["lot_id"] in exmap.index and exmap.loc[l["lot_id"], "qty"] >= 0.25 * l["at_risk_qty"] for l in lots)
            flags.append(hit)
        first_flag = {}
        for r in sub.itertuples():
            for l in r.action.get("lots", []):
                first_flag.setdefault(l["lot_id"], r.as_of)
                first_flag[l["lot_id"]] = min(first_flag[l["lot_id"]], r.as_of)
        events = []
        for r in ex.itertuples():
            dollars = float(r.qty * (r.cost or 0))
            if dollars < 80:
                continue
            ff = first_flag.get(r.lot_id)
            caught = ff is not None and ff < r.on
            events.append({"agent": "freshness", "kind": "EXPIRY", "entity_id": r.lot_id, "start": str(r.on.date()), "end": str(r.on.date()),
                           "dollars": dollars, "caught": bool(caught), "delay_days": float((r.on - ff).days) if caught else None,
                           "event_id": f"EX-{r.lot_id}"})
        finish("freshness", flags, events)

    # ---------- customer: slowdown / lapse / keying against truth events
    sub = P[P.agent == "customer"]
    if len(sub):
        flags = []
        t_cust = truth[truth.kind.isin(["CUSTOMER_LOST", "CUSTOMER_SLOWDOWN"])]
        t_key = truth[truth.kind == "KEYING_ERROR"]
        for r in sub.itertuples():
            if r.dedupe_key.startswith("keying"):
                flags.append(bool(((t_key.entity_id == r.entity_id) & (t_key.start_date == r.as_of)).any()))
            else:
                flags.append(bool(((t_cust.entity_id == r.entity_id) & (t_cust.start_date <= r.as_of) &
                                   (t_cust.start_date >= r.as_of - pd.Timedelta(days=56))).any()))
        events = []
        for e in truth[truth.kind.isin(["CUSTOMER_LOST", "CUSTOMER_SLOWDOWN", "KEYING_ERROR"]) & (truth.start_date >= lo) & (truth.start_date <= hi)].itertuples():
            if e.kind == "KEYING_ERROR":
                c_, p_ = e.entity_id.split("|")
                ln = ol[(ol.customer_id == c_) & (ol.product_id == p_) & (ol.delivery_date == e.start_date)]
                dollars = ed.keying_dollars(ln, e.magnitude)
                if ln.empty or dollars < 100:
                    continue            # unobservable or immaterial: nobody could or should flag it
                m = sub[(sub.entity_id == e.entity_id) & (sub.as_of == e.start_date) & sub.dedupe_key.str.startswith("keying")]
            else:
                m = sub[(sub.entity_id == e.entity_id) & (sub.as_of >= e.start_date) & (sub.as_of <= e.end_date + pd.Timedelta(days=7))]
                cl = ol[ol.customer_id == e.entity_id]
                v = ed.customer_lost_dollars(cl, e.start_date) if e.kind == "CUSTOMER_LOST" else ed.customer_slowdown_dollars(cl, e.start_date, e.end_date)
                dollars = None if v is None else float(v)   # estimate from the pre-event run-rate; None (unvalued) without history
            events.append({"agent": "customer", "kind": e.kind, "entity_id": e.entity_id, "start": str(e.start_date.date()),
                           "end": str(e.end_date.date()), "dollars": dollars, "caught": len(m) > 0,
                           "delay_days": float((m.as_of.min() - e.start_date).days) if len(m) else None, "event_id": e.event_id, "note": e.note})
        finish("customer", flags, events)

    # ---------- margin
    sub = P[P.agent == "margin"]
    if len(sub):
        flags = []
        t_cost = truth[truth.kind == "COST_SPIKE"]; t_price = truth[truth.kind == "PRICE_BELOW_FLOOR"]
        for r in sub.itertuples():
            if r.dedupe_key.startswith("price"):
                flags.append(bool(((t_price.entity_id == r.entity_id) & (t_price.start_date <= r.as_of) & (t_price.end_date >= r.as_of)).any()))
            else:
                flags.append(bool(((t_cost.entity_id == r.entity_id) & (t_cost.start_date <= r.as_of)).any()))
        events = []
        for e in truth[truth.kind.isin(["COST_SPIKE", "PRICE_BELOW_FLOOR"]) & (truth.start_date >= lo) & (truth.start_date <= hi)].itertuples():
            if e.kind == "PRICE_BELOW_FLOOR":
                c_, p_ = e.entity_id.split("|")
                pl = ol[(ol.customer_id == c_) & (ol.product_id == p_)]
                ln = pl[(pl.delivery_date >= e.start_date) & (pl.delivery_date <= e.end_date)]
                dollars = ed.price_below_floor_dollars(pl, e.start_date, e.end_date)
                if ln.empty or dollars is None or dollars < 15:
                    continue            # no order placed during the event, no pre-event price, or too small to matter
                m = sub[(sub.entity_id == e.entity_id) & (sub.as_of >= e.start_date) & (sub.as_of <= e.end_date) & sub.dedupe_key.str.startswith("price")]
            else:
                m = sub[(sub.entity_id == e.entity_id) & (sub.as_of >= e.start_date) & sub.dedupe_key.str.startswith("margin-cost")]
                v = ed.cost_rise_dollars(ol[ol.product_id == e.entity_id], e.start_date, data_end)
                dollars = None if v is None else float(v)   # eight-week margin lost, from recorded costs; None without a pre-event cost
            events.append({"agent": "margin", "kind": e.kind, "entity_id": e.entity_id, "start": str(e.start_date.date()),
                           "end": str(e.end_date.date()), "dollars": dollars, "caught": len(m) > 0,
                           "delay_days": float((m.as_of.min() - e.start_date).days) if len(m) else None, "event_id": e.event_id, "note": e.note})
        finish("margin", flags, events)
    return res
