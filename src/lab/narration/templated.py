"""Deterministic plain-English explanations built only from a proposal's own evidence."""
from __future__ import annotations


def _money(x) -> str:
    return f"${x:,.2f}" if abs(x) < 100 else f"${x:,.0f}"


def narrate(p: dict) -> str:
    a = p["action"]; ev = p["evidence"]; why = ev.get("why", "")
    subject = p["title"].split(":")[0]            # standalone text (briefing, email) needs its own subject
    if why and not why.startswith(subject):
        why = f"{subject}: {why[0].lower() + why[1:]}"
    t = a.get("type")
    stake = _money(p["dollars_at_stake"])
    if t == "raise_purchase_order":
        what = f"Order about {a['qty']:.0f} units, needed by {a['needed_by']}, estimated cost {_money(a['est_cost'])}."
        return f"{why} {what} Without it, about {stake} of margin is at risk."
    if t == "sell_down_offer":
        lots = a.get("lots", [])
        who = ", ".join(c["name"] for c in a.get("target_customers", [])[:3])
        what = (f"Offer about {a['qty']:.0f} units at roughly {a['suggested_discount_pct']}% off, valid to {a['valid_until']}"
                + (f", starting with {who}." if who else "."))
        return f"{why} {what} That is about {stake} of stock cost at risk of being written off."
    if t == "customer_check_in":
        return f"{why} {a.get('suggested', 'Check in with the customer.')} Roughly {stake} of gross margin is in play."
    if t == "hold_order_line" and "usual_qty" in a:
        return f"{why} Hold this line and confirm the quantity with the customer before it is picked. About {stake} would be over-delivered."
    if t == "hold_order_line":
        return f"{why} Hold this line and confirm the price before it is invoiced. About {stake} of margin is being given away."
    if t == "review_price":
        return (f"{why} Raising the sell price from {_money(a['current_price'])} to about {_money(a['suggested_price'])} would bring the margin back "
                f"to {a['target_margin_pct']}%. Around {stake} of margin is exposed over four weeks.")
    return why
