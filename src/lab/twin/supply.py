"""Stock and ordering simulator for the twin.

Runs day by day: receive POs as lots, expire old lots (waste), fulfil demand first-expiry-first-out,
record stockouts, then let an ordering policy place new POs. Because demand is fixed, the same world
can be replayed under different policies, which is how the lab measures dollars protected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .generator import TwinWorld


@dataclass
class PolicyCtx:
    day: int                      # index into world.days
    product: int
    hist: np.ndarray              # daily demand history for this product, up to day-1
    on_hand: float
    on_order: float
    lead: int                     # expected lead time in days
    shelf_life: int
    unit_cost: float
    next_review_gap: int          # days until the next review for this product
    lots: list                    # [(use_by_idx, qty)] FEFO order


Policy = Callable[[PolicyCtx], float]


def bau_policy(ctx: PolicyCtx) -> float:
    """A competent human buyer using standard heuristics, so the lab does not flatter the forecast policy.

    Trailing 4-week mean plus safety stock from demand variability; cover is capped by shelf life so a
    perishable line is never stocked beyond what can sell; perishables are bought in singles, dry goods in cases of 6.
    Reviews Mon/Wed/Fri.
    """
    h = ctx.hist[-28:]
    if len(h) < 7:
        return 0.0
    rate = float(h.mean())
    if rate <= 0.0:
        return 0.0
    sd = float(h.std())
    base_cover = 3 if ctx.shelf_life <= 7 else 5 if ctx.shelf_life <= 21 else 10
    cover = min(base_cover, max(1.0, ctx.shelf_life * 0.35))
    horizon = ctx.lead + ctx.next_review_gap
    safety = (0.8 if ctx.shelf_life > 14 else 0.35) * sd * np.sqrt(horizon)
    position = ctx.on_hand + ctx.on_order
    reorder_point = rate * horizon + safety
    if position >= reorder_point:
        return 0.0
    target = rate * (horizon + cover) + safety
    target = min(target, rate * max(horizon + 1.0, ctx.shelf_life * 0.7) + safety)  # never stock more than can sell in shelf life
    need = max(0.0, target - position)
    pack = 6.0 if ctx.shelf_life > 21 else 1.0
    return float(np.ceil(need / pack) * pack)


@dataclass
class SimResult:
    lots: pd.DataFrame
    movements: pd.DataFrame
    purchases: pd.DataFrame
    outcomes: pd.DataFrame
    fill: np.ndarray              # (products, days) fulfilled / demand
    waste_value: float
    stockout_units: float
    stockout_value: float
    lost_margin: float
    holding_value: float
    meta: dict = field(default_factory=dict)


REVIEW_DOWS = {0, 2, 4}  # Mon, Wed, Fri


def simulate(world: TwinWorld, policy: Policy = bau_policy, policy_b: Optional[Policy] = None,
             switch_idx: Optional[int] = None, seed: int = 11) -> SimResult:
    rng = np.random.default_rng(seed)
    nd, npr = len(world.days), len(world.products)
    pidx = world.pairs.pi.values
    # total daily demand per product (units)
    dem = np.zeros((npr, nd))
    np.add.at(dem, pidx, world.demand)

    sup_index = {s: i for i, s in enumerate(world.suppliers.supplier_id)}
    prod_sup = np.array([sup_index[s] for s in world.products.supplier_id])
    base_lead = np.array([world.supplier_lead[s] for s in world.suppliers.supplier_id])
    shelf = world.products.shelf_life_days.values.astype(int)
    cost0 = world.products.unit_cost.values
    jitter = rng.integers(0, 2, size=(npr, nd))

    forced = {}
    for f in world.forced_orders:
        forced.setdefault((f["day"], f["product_idx"]), f["mult"])

    lots: list[list] = [[] for _ in range(npr)]            # per product [use_by_idx, qty, lot_no, cost, recv_idx]
    pipeline: list[list] = [[] for _ in range(npr)]        # [arrive_idx, qty, po_no, cost]
    lot_rows, move_rows, po_rows, out_rows = [], [], [], []
    fill = np.ones((npr, nd))
    waste_value = stockout_units = stockout_value = lost_margin = holding_value = 0.0
    lot_no = po_no = 0

    # warm-up stock: each product starts with ~10 days of cover so day 0 is not all stockouts
    for p in range(npr):
        r = dem[p, :28].mean() if nd >= 28 else 0.0
        if r > 0:
            lot_no += 1
            lots[p].append([min(nd + 400, int(max(5, shelf[p] * 0.6))), r * 10, lot_no, cost0[p], -1])
            lot_rows.append((f"L-{lot_no:06d}", world.products.product_id[p], None, world.days[0].date(),
                             world.days[min(nd - 1, int(max(5, shelf[p] * 0.6)))].date(), r * 10, cost0[p]))
            move_rows.append((f"L-{lot_no:06d}", world.products.product_id[p], world.days[0].date(), "Opening stock", r * 10))

    for d in range(nd):
        pol = policy_b if (policy_b is not None and switch_idx is not None and d >= switch_idx) else policy
        for p in range(npr):
            # 1. receive
            arrived = [x for x in pipeline[p] if x[0] <= d]
            if arrived:
                pipeline[p] = [x for x in pipeline[p] if x[0] > d]
                for a in arrived:
                    lot_no += 1
                    ub = d + max(2, int(shelf[p] - rng.integers(0, 3)))
                    lots[p].append([ub, a[1], lot_no, a[3], d])
                    lot_rows.append((f"L-{lot_no:06d}", world.products.product_id[p], world.suppliers.supplier_id[prod_sup[p]],
                                     world.days[d].date(), world.days[min(nd - 1, ub)].date() if ub < nd else None, a[1], a[3]))
                    move_rows.append((f"L-{lot_no:06d}", world.products.product_id[p], world.days[d].date(), "Purchase receipt", a[1]))
            # 2. expire
            expired_qty = expired_val = 0.0
            keep = []
            for L in lots[p]:
                if L[0] < d and L[1] > 0:
                    expired_qty += L[1]; expired_val += L[1] * L[3]
                    move_rows.append((f"L-{L[2]:06d}", world.products.product_id[p], world.days[d].date(), "Expired write-off", -L[1]))
                else:
                    keep.append(L)
            lots[p] = keep
            waste_value += expired_val
            # 3. fulfil FEFO
            need = dem[p, d]
            got = 0.0
            if need > 0:
                lots[p].sort(key=lambda L: L[0])
                for L in lots[p]:
                    if need - got <= 1e-9:
                        break
                    take = min(L[1], need - got)
                    if take > 0:
                        L[1] -= take; got += take
                        move_rows.append((f"L-{L[2]:06d}", world.products.product_id[p], world.days[d].date(), "Sale", -take))
                lots[p] = [L for L in lots[p] if L[1] > 1e-9]
            short = max(0.0, need - got)
            if need > 0:
                fill[p, d] = got / need
            price_ref = float(world.products.unit_price[p])
            stockout_units += short
            stockout_value += short * price_ref
            lost_margin += short * (price_ref - cost0[p] * world.cost_mult[p, d])
            on_hand = sum(L[1] for L in lots[p])
            holding_value += on_hand * cost0[p]
            out_rows.append((world.days[d].date(), world.products.product_id[p], need, got, on_hand,
                             expired_qty, expired_val, short > 1e-9))
            # 4. order
            dow = world.days[d].dayofweek
            qty = 0.0
            eff_base = min(int(base_lead[prod_sup[p]]), 2) if shelf[p] <= 14 else int(base_lead[prod_sup[p]])  # local fresh supply is next-day to 2-day
            lead_now = int(eff_base + world.lead_extra[prod_sup[p], d] + jitter[p, d])
            if (d, p) in forced:
                r = float(max(dem[p, max(0, d - 28):d].mean() if d > 0 else 0.0, 1.0))
                qty = r * 7 * forced[(d, p)]
            elif dow in REVIEW_DOWS and d >= 14 and not pol is None:
                gap = 2 if dow in (0, 2) else 3
                ctx = PolicyCtx(d, p, dem[p, :d], on_hand, sum(x[1] for x in pipeline[p]), lead_now, int(shelf[p]),
                                float(cost0[p] * world.cost_mult[p, d]), gap,
                                [(L[0], L[1]) for L in sorted(lots[p], key=lambda L: L[0])])
                qty = pol(ctx)
            if qty > 0:
                po_no += 1
                c = float(cost0[p] * world.cost_mult[p, d])
                pipeline[p].append([d + max(1, lead_now), qty, po_no, c])
                po_rows.append((f"PO-{po_no:06d}", world.suppliers.supplier_id[prod_sup[p]], world.products.product_id[p],
                                world.days[d].date(), world.days[min(nd - 1, d + max(1, lead_now))].date(), qty, qty, c))

    lots_df = pd.DataFrame(lot_rows, columns=["lot_id", "product_id", "supplier_id", "received_date", "use_by_date", "qty_received", "unit_cost"])
    mv_df = pd.DataFrame(move_rows, columns=["lot_id", "product_id", "moved_on", "movement_type", "qty_change"])
    po_df = pd.DataFrame(po_rows, columns=["po_id", "supplier_id", "product_id", "created_on", "expected_on", "qty_ordered", "qty_received", "unit_cost"])
    out_df = pd.DataFrame(out_rows, columns=["day", "product_id", "demand", "fulfilled", "on_hand_end", "expired_qty", "expired_value", "stockout"])
    return SimResult(lots_df, mv_df, po_df, out_df, fill, waste_value, stockout_units, stockout_value, lost_margin, holding_value)
