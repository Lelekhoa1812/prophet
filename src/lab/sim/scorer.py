"""Score proposals against what really happened in the twin (precision, recall, detection delay, dollars)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..store import open_dataset


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
        res["agents"][agent] = {
            "proposals": len(sub), "useful": tp, "precision": (tp / len(sub)) if len(sub) else None,
            "severe_false_rate": (sev_false / sev_total) if sev_total else 0.0,
            "recent_precision": (sum(flags[-20:]) / len(flags[-20:])) if flags else None,
            "events": len(events), "caught": len(caught), "recall": (len(caught) / len(events)) if events else None,
            "median_delay_days": float(np.median([e["delay_days"] for e in caught])) if caught else None,
            "dollars_at_stake_total": float(sub.dollars_at_stake.sum()),
            "dollars_truth_caught": float(sum(e["dollars"] for e in caught)),
            "dollars_truth_total": float(sum(e["dollars"] for e in events)),
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

    # ---------- freshness: lots that really expired
    sub = P[P.agent == "freshness"]
    if len(sub):
        ex = exp.groupby("lot_id").agg(qty=("qty", "sum"), on=("moved_on", "max"), cost=("unit_cost", "first"), product_id=("product_id", "first")).reset_index()
        ex = ex[(ex.on >= lo) & (ex.on <= hi)]
        exmap = ex.set_index("lot_id")
        flags = []
        for r in sub.itertuples():
            at_risk = r.action.get("qty", 0)
            flags.append(r.entity_id in exmap.index and exmap.loc[r.entity_id, "qty"] >= 0.25 * at_risk)
        events = []
        for r in ex.itertuples():
            dollars = float(r.qty * (r.cost or 0))
            if dollars < 80:
                continue
            m = sub[(sub.entity_id == r.lot_id) & (sub.as_of < r.on)]
            events.append({"agent": "freshness", "kind": "EXPIRY", "entity_id": r.lot_id, "start": str(r.on.date()), "end": str(r.on.date()),
                           "dollars": dollars, "caught": len(m) > 0, "delay_days": float((r.on - m.as_of.min()).days) if len(m) else None,
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
                m = sub[(sub.entity_id == e.entity_id) & (sub.as_of == e.start_date) & sub.dedupe_key.str.startswith("keying")]
            else:
                m = sub[(sub.entity_id == e.entity_id) & (sub.as_of >= e.start_date) & (sub.as_of <= e.end_date + pd.Timedelta(days=7))]
            events.append({"agent": "customer", "kind": e.kind, "entity_id": e.entity_id, "start": str(e.start_date.date()),
                           "end": str(e.end_date.date()), "dollars": 0.0, "caught": len(m) > 0,
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
                m = sub[(sub.entity_id == e.entity_id) & (sub.as_of >= e.start_date) & (sub.as_of <= e.end_date) & sub.dedupe_key.str.startswith("price")]
            else:
                m = sub[(sub.entity_id == e.entity_id) & (sub.as_of >= e.start_date) & sub.dedupe_key.str.startswith("margin-cost")]
            events.append({"agent": "margin", "kind": e.kind, "entity_id": e.entity_id, "start": str(e.start_date.date()),
                           "end": str(e.end_date.date()), "dollars": 0.0, "caught": len(m) > 0,
                           "delay_days": float((m.as_of.min() - e.start_date).days) if len(m) else None, "event_id": e.event_id, "note": e.note})
        finish("margin", flags, events)
    return res
