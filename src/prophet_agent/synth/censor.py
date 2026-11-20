"""Stock-out censoring: recorded sales are capped at the units available, so the record under-states demand.

Motivation vs Logic
Motivation: a forecaster sees sales, not demand. When an item runs out, the sales record stops at the stock on hand,
so a forecast trained on it under-states demand. The synthetic data must reproduce that, or the lab cannot test
whether a forecast or an agent copes with it.
Logic: one stock run per item, on the same period grid as the data (days or weeks). Each period, receipts arrive,
sales are the smaller of demand and units on hand, and a simple order-up-to planner reorders from the sales it can
see, so stock-outs also lower its own forecast, as they do in practice. In a short period, each entity's observed
sales are its share of the units sold, rounded down, so observed sales never exceed available supply. A series is
flagged censored in a period when its observed sales fall below its latent demand.
Assumption: the planner, opening cover, margin and lead times are synthetic choices, not measured from any client.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StockRun:
    """Item-level stock outcome per period. All arrays are (items, periods)."""

    demand: np.ndarray       # total latent demand for the item
    available: np.ndarray    # units on hand after receipts, before sales (the cap)
    sold: np.ndarray         # units sold = min(demand, available)
    short: np.ndarray        # bool, True where demand exceeded available units (censoring period)


def simulate_stock(demand: np.ndarray, lead: np.ndarray, *, window: int = 4, opening_cover: float = 3.0,
                   margin: float = 1.25, safety: float = 2.0) -> StockRun:
    """Run a daily-review order-up-to planner per item over `demand` (items, periods).

    `lead` is the replenishment lead time in whole periods (>= 1). The planner targets
    mean(sales over the last `window` periods) * (lead + 1) * margin + safety * sd, and orders the shortfall
    against on-hand plus on-order stock. It only ever reads sales that have already happened.
    """
    demand = np.asarray(demand, dtype=float)
    lead = np.asarray(lead, dtype=int)
    if demand.ndim != 2:
        raise ValueError("demand must be a 2-D array of (items, periods)")
    n_items, n_periods = demand.shape
    if lead.shape != (n_items,) or (lead < 1).any():
        raise ValueError("lead must hold one whole number of periods >= 1 per item")
    if window < 1:
        raise ValueError("window must be at least 1 period")

    arrivals = np.zeros((n_items, n_periods + int(lead.max()) + 1))
    rows = np.arange(n_items)
    on_hand = np.floor(demand[:, : min(window, n_periods)].mean(axis=1) * opening_cover)
    on_order = np.zeros(n_items)
    available = np.zeros_like(demand)
    sold = np.zeros_like(demand)
    short = np.zeros(demand.shape, dtype=bool)

    for t in range(n_periods):
        receipts = arrivals[:, t]
        on_hand = on_hand + receipts
        on_order = on_order - receipts
        available[:, t] = on_hand

        d = demand[:, t]
        s = np.minimum(d, on_hand)
        sold[:, t] = s
        short[:, t] = d > on_hand
        on_hand = on_hand - s

        seen = sold[:, max(0, t - window + 1): t + 1]
        target = seen.mean(axis=1) * (lead + 1) * margin + safety * seen.std(axis=1)
        order = np.ceil(np.maximum(0.0, target - (on_hand + on_order)))
        on_order = on_order + order
        arrivals[rows, t + lead] += order

    return StockRun(demand=demand, available=available, sold=sold, short=short)


def censor_series(latent: np.ndarray, item_index: np.ndarray, run: StockRun) -> tuple[np.ndarray, np.ndarray]:
    """Return (observed, censored) for each series.

    `latent` is (series, periods) with whole-number demand; `item_index[s]` is the item row of series s.
    `observed` is int64 and never above `latent`. `censored[s, t]` is True where observed < latent, i.e. the
    record under-states demand. Sum of observed over an item's series never exceeds `run.available`.
    """
    latent = np.asarray(latent, dtype=float)
    item_index = np.asarray(item_index, dtype=int)
    if latent.ndim != 2 or latent.shape[1] != run.demand.shape[1]:
        raise ValueError("latent must be (series, periods) with the same periods as the stock run")
    if item_index.shape != (latent.shape[0],):
        raise ValueError("item_index must hold one item row per series")

    totals = np.zeros_like(run.demand)
    np.add.at(totals, item_index, latent)
    if not np.allclose(totals, run.demand):
        raise ValueError("latent demand does not match the demand used in the stock run")

    share = np.divide(run.sold, totals, out=np.zeros_like(totals), where=totals > 0)
    capped = np.floor(latent * share[item_index])
    observed = np.where(run.short[item_index], capped, latent).astype(np.int64)
    censored = observed < latent
    return observed, censored
