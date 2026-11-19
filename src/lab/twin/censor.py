"""Demand censoring for the twin: customers order less of a product that was out of stock in the last days.

Motivation vs Logic
Motivation: in the base twin every order is placed whatever the stock, so recorded orders equal true demand (measured:
5,465,171 both) and the twin cannot show whether stockouts bias a forecast or hide demand. Real customers who were told
'not available' often stop asking for a while.
Logic: after the supply simulation, each order line of a product that had a stockout on any of the previous `lag_days`
days is dropped with probability `p_drop` (seeded). Latent demand stays in daily_outcomes and truth_expected, so the gap
between recorded orders and truth is exactly what a forecaster must cope with. The strength is an assumption, not a
measurement from any client; sweep it rather than trust one value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def censoring_mask(outcomes: pd.DataFrame, product_ids: list, days: pd.DatetimeIndex, pi: np.ndarray, c: np.ndarray,
                   p_drop: float = 0.4, lag_days: int = 2, seed: int = 101) -> np.ndarray:
    """keep[i] for order line i (product index pi[i], day index c[i]); False means the order was never placed."""
    P, D = len(product_ids), len(days)
    idx = {p: i for i, p in enumerate(product_ids)}
    out = np.zeros((P, D), bool)
    di = {d: j for j, d in enumerate(pd.DatetimeIndex(days).normalize())}
    for r in outcomes[outcomes.stockout.astype(bool)].itertuples():
        out[idx[r.product_id], di[pd.Timestamp(r.day).normalize()]] = True
    recent = np.zeros_like(out)
    for k in range(1, lag_days + 1):                     # a stockout on day d affects orders on d+1 .. d+lag_days
        recent[:, k:] |= out[:, :-k]
    drop = recent[pi, c] & (np.random.default_rng(seed).random(len(pi)) < p_drop)
    return ~drop
