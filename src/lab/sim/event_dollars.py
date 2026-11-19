"""Dollar truth for customer and margin events, measured from order lines (not from generator labels).

Motivation vs Logic
Motivation: the scorer set customer and margin event dollars to 0.0, so their dollar recall was never measured and they
were excluded from what-if. The twin's labels hold only nominal magnitudes (price keyed to 0.7 is a label; the real factor
is drawn per event and not stored), so dollars must come from the order lines the business actually recorded.
Logic: each event is valued from the recorded prices and costs against the same pair or product before the event
(a pre-event run-rate). Customer and cost events are valued over a fixed horizon, so a lost customer is not valued
for the whole four-year history. Every figure is a counterfactual against a pre-event baseline: an estimate, labelled
as such, never an observed saving.
"""
from __future__ import annotations

import pandas as pd

HORIZON_DAYS = 56          # value at risk for lost customers and cost rises: eight weeks forward (a stated assumption)
PRE_DAYS = 56              # pre-event window for run-rates and prices
COST_PRE_DAYS = 28         # pre-event window for cost baselines
SLOWDOWN_FACTOR = 0.35     # the generator's nominal slowdown level (the magnitude field of CUSTOMER_SLOWDOWN)


def _window(lines: pd.DataFrame, lo, hi) -> pd.DataFrame:
    d = lines.delivery_date
    return lines[(d >= lo) & (d <= hi)]


def keying_dollars(lines: pd.DataFrame, factor: float) -> float:
    """Value of quantity over-keyed by `factor` on the event day: qty x (1 - 1/factor) x price."""
    if lines.empty or factor <= 1:
        return 0.0
    return float((lines.qty * (1 - 1 / factor) * lines.unit_price.fillna(0)).sum())


def price_below_floor_dollars(lines: pd.DataFrame, start, end) -> float | None:
    """Margin given away: qty x (pre-event price - keyed price) on the event's lines, if the pair had a pre-event price.

    Returns None when there is no pre-event price to compare with (the event cannot be valued, so it is not scored).
    """
    pre = lines[(lines.delivery_date >= start - pd.Timedelta(days=PRE_DAYS)) & (lines.delivery_date < start)]
    if pre.empty:
        return None
    ref = float(pre.unit_price.median())
    ev = _window(lines, start, end)
    if ev.empty:
        return 0.0
    return float((ev.qty * (ref - ev.unit_price).clip(lower=0)).sum())


def cost_rise_dollars(lines: pd.DataFrame, start, end_data) -> float | None:
    """Margin lost to a supplier cost rise: qty x (cost now - pre-event cost), over the next HORIZON_DAYS."""
    pre = lines[(lines.delivery_date >= start - pd.Timedelta(days=COST_PRE_DAYS)) & (lines.delivery_date < start)]
    if pre.empty:
        return None
    ref = float(pre.unit_cost.median())
    hi = min(pd.Timestamp(end_data), start + pd.Timedelta(days=HORIZON_DAYS))
    ev = _window(lines, start, hi)
    if ev.empty:
        return 0.0
    return float((ev.qty * (ev.unit_cost - ref).clip(lower=0)).sum())


def _run_rate(lines: pd.DataFrame, start) -> float | None:
    """Pre-event revenue per day for one customer, over PRE_DAYS. None when the customer had no history before the event."""
    pre = lines[(lines.delivery_date >= start - pd.Timedelta(days=PRE_DAYS)) & (lines.delivery_date < start)]
    if pre.empty:
        return None
    return float((pre.qty * pre.unit_price.fillna(0)).sum()) / PRE_DAYS


def customer_lost_dollars(lines: pd.DataFrame, start) -> float | None:
    """Revenue lost if the customer stops: pre-event run-rate x HORIZON_DAYS (estimate)."""
    rate = _run_rate(lines, start)
    return None if rate is None else rate * HORIZON_DAYS


def customer_slowdown_dollars(lines: pd.DataFrame, start, end) -> float | None:
    """Revenue lost in a slowdown: run-rate x (1 - SLOWDOWN_FACTOR) x days in the event (estimate)."""
    rate = _run_rate(lines, start)
    if rate is None:
        return None
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    return rate * (1 - SLOWDOWN_FACTOR) * days
