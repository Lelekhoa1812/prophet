"""Synthetic twin: a calibrated meat-and-fresh wholesaler with four years of history and labelled shocks.

Why: staging holds ~8 weeks, so seasonality, promotions and 'did the alert catch a real problem'
cannot be tested on it. The twin's demand is drawn from a known process, shocks are injected on
purpose and recorded as truth, so every agent can be scored against what really happened.
Calibrated on the staging profile (category mix, price spread, basket size, customer breadth).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from ..calendar_vic import flags as cal_flags
import pandas as pd

# category: (weight, median price $/unit, margin, shelf_life_days, annual season amplitude, peak day-of-year)
CATEGORIES = {
    "BOX MEAT": (0.30, 14.5, 0.14, 160, 0.10, 350),
    "BEEF": (0.14, 22.0, 0.20, 14, 0.18, 20),      # BBQ season peak, summer
    "CHICKEN": (0.12, 9.5, 0.17, 9, 0.08, 20),
    "LAMB": (0.08, 24.0, 0.19, 12, 0.30, 105),     # Easter / Christmas lumps
    "PORK": (0.08, 13.0, 0.18, 12, 0.22, 355),     # Christmas ham
    "SAUSAGES": (0.07, 11.0, 0.22, 21, 0.20, 20),
    "SEAFOOD": (0.08, 32.0, 0.24, 5, 0.35, 355),   # twin-only: short shelf life, Christmas peak
    "SMALLGOODS": (0.05, 18.0, 0.25, 35, 0.15, 355),
    "PRODUCE": (0.08, 6.5, 0.28, 6, 0.25, 15),     # twin-only: fast ageing
}

SEGMENTS = {
    # name: (weight, basket skus (lo,hi), per-sku daily rate scale, dow profile Mon..Sun, school-holiday sensitivity)
    "Restaurant": (0.40, (3, 14), 0.050, [0.7, 0.9, 1.0, 1.3, 1.5, 0.9, 0.2], 0.0),
    "Butcher": (0.20, (8, 30), 0.060, [1.3, 1.1, 1.0, 1.0, 1.2, 0.8, 0.0], 0.0),
    "Supermarket": (0.12, (12, 40), 0.080, [1.2, 1.0, 1.0, 1.1, 1.2, 0.5, 0.0], 0.0),
    "Institution": (0.16, (3, 10), 0.070, [1.1, 1.1, 1.1, 1.1, 0.9, 0.0, 0.0], 0.55),
    "Caterer": (0.12, (3, 12), 0.040, [0.6, 0.8, 1.0, 1.1, 1.4, 1.4, 0.4], 0.0),
}

# Calendar effects by segment: multiplier on demand on each kind of day (1.0 = no effect).
# SYNTHETIC ASSUMPTIONS, not measured trade data. Round 3 (19 Nov) raised the Christmas, Easter and Cup effects so
# that a known event is visible in a 4-week forecast window. Replace with real history before any client use.
# Logic: applied as prod(1 + (effect - 1) * flag), so overlapping days (Cup day is also a public holiday) stack.
CAL_EFFECTS = {
    #              public  xmas_run_up  easter_run_up  cup_day
    "Restaurant":  (0.40,   1.20,        1.05,          0.70),
    "Butcher":     (0.30,   1.90,        1.80,          1.00),
    "Supermarket": (0.50,   1.60,        1.50,          1.00),
    "Institution": (0.20,   1.00,        1.00,          1.00),
    "Caterer":     (0.60,   1.50,        1.00,          2.00),
}
CAL_KEYS = ["public", "xmas_run_up", "easter_run_up", "cup_day"]


@dataclass
class TwinConfig:
    seed: int = 7
    start: str = "2022-11-01"
    end: str = "2026-11-06"
    n_products: int = 300
    n_customers: int = 250
    n_suppliers: int = 12
    shocks: bool = True
    extra: dict = field(default_factory=dict)


@dataclass
class TwinWorld:
    cfg: TwinConfig
    days: pd.DatetimeIndex
    products: pd.DataFrame
    customers: pd.DataFrame
    suppliers: pd.DataFrame
    pairs: pd.DataFrame          # one row per (customer, product): base rate, typical qty, price
    demand: np.ndarray           # (pairs, days) qty, after shocks
    price: np.ndarray            # (pairs, days) sell price
    truth: list[dict]
    supplier_lead: dict          # supplier_id -> base lead days
    cost_mult: np.ndarray        # (products, days) supplier cost multiplier
    lead_extra: np.ndarray       # (suppliers, days) extra lead days (supplier delays)
    forced_orders: list[dict]    # over-buy events: {day, product_idx, qty}
    expected: np.ndarray = None  # (pairs, days) true expected demand before noise and keying errors


def _season(cat: str, doy: np.ndarray) -> np.ndarray:
    amp, peak = CATEGORIES[cat][4], CATEGORIES[cat][5]
    return 1.0 + amp * np.cos(2 * np.pi * (doy - peak) / 365.25)


def build_world(cfg: TwinConfig | None = None) -> TwinWorld:
    cfg = cfg or TwinConfig()
    rng = np.random.default_rng(cfg.seed)
    days = pd.date_range(cfg.start, cfg.end, freq="D")
    nd = len(days)
    doy = days.day_of_year.values
    dow = days.day_of_week.values

    # ---- products
    cats = list(CATEGORIES)
    cat_w = np.array([CATEGORIES[c][0] for c in cats]); cat_w /= cat_w.sum()
    pc = rng.choice(cats, size=cfg.n_products, p=cat_w)
    price = np.array([CATEGORIES[c][1] for c in pc]) * rng.lognormal(0, 0.45, cfg.n_products)
    margin = np.array([CATEGORIES[c][2] for c in pc]) * rng.uniform(0.7, 1.3, cfg.n_products)
    cost = price * (1 - margin)
    shelf = np.array([CATEGORIES[c][3] for c in pc])
    pop = rng.pareto(1.3, cfg.n_products) + 1.0           # long-tail popularity
    products = pd.DataFrame({
        "product_id": [f"P-{i + 1:04d}" for i in range(cfg.n_products)], "category": pc,
        "uom": np.where(rng.random(cfg.n_products) < 0.6, "KG", "EA"),
        "is_catch_weight": rng.random(cfg.n_products) < 0.45,
        "shelf_life_days": shelf, "unit_cost": cost.round(2), "unit_price": price.round(2), "popularity": pop,
    })
    products["code"] = products.product_id
    products["display_name"] = products.category + " · " + products.code

    # ---- customers
    seg_names = list(SEGMENTS)
    seg_w = np.array([SEGMENTS[s][0] for s in seg_names]); seg_w /= seg_w.sum()
    cs = rng.choice(seg_names, size=cfg.n_customers, p=seg_w)
    customers = pd.DataFrame({
        "customer_id": [f"C-{i + 1:04d}" for i in range(cfg.n_customers)], "segment": cs,
        "zone": rng.choice(["Metro", "Country 1", "Country 2", "Bayside"], size=cfg.n_customers, p=[.5, .2, .1, .2]),
        "tier": rng.uniform(0.93, 1.08, cfg.n_customers),
        "scale": rng.lognormal(0, 0.6, cfg.n_customers),
    })
    customers["code"] = customers.customer_id
    customers["display_name"] = customers.segment + " account " + customers.code
    # Accounts join over time: 30% start after the first ~2 months, spread over the first three years.
    created_idx = np.zeros(cfg.n_customers, dtype=int)
    late = rng.choice(cfg.n_customers, int(cfg.n_customers * 0.30), replace=False)
    created_idx[late] = rng.integers(60, int(nd * 0.75), len(late))
    customers["created_on"] = [days[i].date() for i in created_idx]

    # Product lifecycle: 8% launch during the first 60% of the span, 4% are discontinued in the last 30%.
    launch_idx = np.zeros(cfg.n_products, dtype=int)
    retire_idx = np.full(cfg.n_products, nd)
    new = rng.choice(cfg.n_products, int(cfg.n_products * 0.08), replace=False)
    launch_idx[new] = rng.integers(60, int(nd * 0.6), len(new))
    old = rng.choice(np.setdiff1d(np.arange(cfg.n_products), new), int(cfg.n_products * 0.04), replace=False)
    retire_idx[old] = rng.integers(int(nd * 0.7), nd - 30, len(old))
    day_ix = np.arange(nd)[None, :]
    pmask = ((day_ix >= launch_idx[:, None]) & (day_ix < retire_idx[:, None])).astype(float)

    # Cost drift: monthly random walk, mild inflation, clipped to a band of 0.85x to 1.35x of the starting cost.
    months = nd // 30 + 2
    logd = np.cumsum(rng.normal(0.004, 0.012, (cfg.n_products, months)), axis=1)
    drift_m = np.clip(np.exp(logd), 0.85, 1.35)
    month_ix = np.minimum(((days - days[0]).days // 30).values, months - 1)
    drift = drift_m[:, month_ix]                                          # (products, days)

    # Shock counts scale with the span, so events per year stay the same as the 2-year twin.
    kscale = max(1.0, nd / 730.0)

    suppliers = pd.DataFrame({"supplier_id": [f"S-{i + 1:04d}" for i in range(cfg.n_suppliers)]})
    suppliers["code"] = suppliers.supplier_id
    suppliers["display_name"] = "Supplier " + suppliers.supplier_id
    supplier_lead = {sid: int(rng.integers(2, 6)) for sid in suppliers.supplier_id}
    prod_supplier = rng.integers(0, cfg.n_suppliers, cfg.n_products)
    products["supplier_id"] = suppliers.supplier_id.values[prod_supplier]

    # ---- (customer, product) pairs
    rows = []
    pw = pop / pop.sum()
    for ci, c in customers.iterrows():
        lo, hi = SEGMENTS[c.segment][1]
        k = int(min(cfg.n_products, rng.integers(lo, hi + 1)))
        chosen = rng.choice(cfg.n_products, size=k, replace=False, p=pw)
        for pi in chosen:
            rate = SEGMENTS[c.segment][2] * c.scale * rng.lognormal(0, 0.5) * min(3.0, pop[pi] ** 0.35)
            typ_qty = max(1.0, np.round(rng.lognormal(np.log(6), 0.8) * (1.0 if products.uom[pi] == "EA" else 1.6)))
            rows.append((ci, pi, rate, typ_qty, price[pi] * c.tier))
    pairs = pd.DataFrame(rows, columns=["ci", "pi", "rate", "typ_qty", "price"])
    npairs = len(pairs)

    # ---- demand intensity, then shocks
    seg_dow = np.array([SEGMENTS[customers.segment[ci]][3] for ci in pairs.ci])      # (pairs, 7)
    dow_f = seg_dow[:, dow]                                                           # (pairs, days)
    season = np.stack([_season(pc[pi], doy) for pi in pairs.pi.values])               # (pairs, days)
    trend = 1.0 + 0.10 * np.linspace(0, 2, nd)[None, :]                                # +10% a year
    cf = cal_flags(days)
    seg_hol = np.array([SEGMENTS[customers.segment[ci]][4] for ci in pairs.ci])[:, None]
    school = 1.0 - seg_hol * cf["school"][None, :]
    F = np.stack([cf[k] for k in CAL_KEYS])                                             # (4, days)
    E = np.array([CAL_EFFECTS[customers.segment[ci]] for ci in pairs.ci])                # (pairs, 4)
    cal_mult = np.prod(1.0 + (E[:, :, None] - 1.0) * F[None, :, :], axis=1)             # (pairs, days)

    # promotions: ~6% of SKUs get 2-week price cut + uplift, a few times a year
    promo = np.ones((cfg.n_products, nd))
    promo_price = np.ones((cfg.n_products, nd))
    truth: list[dict] = []
    n_promo = max(8, cfg.n_products // 12)
    for pi in rng.choice(cfg.n_products, n_promo, replace=False):
        for _ in range(int(rng.integers(1, 4))):
            s0 = int(rng.integers(30, nd - 20)); ln = int(rng.integers(7, 15))
            up = float(rng.uniform(1.4, 2.4))
            promo[pi, s0:s0 + ln] = up
            promo_price[pi, s0:s0 + ln] = 0.88
            truth.append(dict(kind="PROMO", entity_type="product", entity_id=products.product_id[pi],
                              start=days[s0].date(), end=days[min(nd - 1, s0 + ln)].date(), magnitude=up,
                              note=f"Planned promotion, demand x{up:.1f}, price -12%"))
    promo_pairs = promo[pairs.pi.values]
    pp_pairs = promo_price[pairs.pi.values]

    # Category calendar: which holidays move which lines (Easter lamb and seafood, Christmas pork, Australia Day sausages).
    CAT_CAL = {"SEAFOOD": {"easter_run_up": 1.5, "xmas_run_up": 1.3}, "LAMB": {"easter_run_up": 1.6},
               "PORK": {"xmas_run_up": 1.5}, "BOX MEAT": {"xmas_run_up": 1.2}, "SAUSAGES": {"public": 1.3},
               "BEEF": {"public": 1.2}, "PRODUCE": {"xmas_run_up": 1.2}}
    cat_cal = np.ones((cfg.n_products, nd))
    for pi_, cat_ in enumerate(pc):
        for key, f_ in CAT_CAL.get(cat_, {}).items():
            cat_cal[pi_] *= 1.0 + (f_ - 1.0) * cf[key]
    # Accounts only buy after they join.
    joined = (np.arange(nd)[None, :] >= created_idx[pairs.ci.values][:, None]).astype(float)

    mean = (pairs.rate.values[:, None] * dow_f * season * trend * school * cal_mult * promo_pairs
            * pmask[pairs.pi.values] * cat_cal[pairs.pi.values] * joined)
    cost_mult = drift.copy()
    lead_extra = np.zeros((cfg.n_suppliers, nd))
    # Sell price follows 70% of the cost drift (pass-through), so margins move but do not collapse.
    price_arr = pairs.price.values[:, None] * pp_pairs * (1.0 + 0.7 * (drift[pairs.pi.values] - 1.0))
    forced_orders: list[dict] = []

    def D(i):
        return days[i].date()

    if cfg.shocks:
        scored_from = int(nd * 0.55)      # leave the first ~13 months clean for context history
        ev = 0

        def pick_day(lo=None, span=0):
            lo = scored_from if lo is None else lo
            return int(rng.integers(lo, nd - span - 14))

        # demand surge
        for _ in range(int(round(10 * kscale))):
            pi = int(rng.choice(cfg.n_products, p=pw)); s0 = pick_day(span=8); ln = int(rng.integers(4, 9))
            mult = float(rng.uniform(1.8, 3.2))
            mean[pairs.pi.values == pi, s0:s0 + ln] *= mult
            truth.append(dict(kind="DEMAND_SURGE", entity_type="product", entity_id=products.product_id[pi],
                              start=D(s0), end=D(s0 + ln), magnitude=mult, note=f"Unplanned surge x{mult:.1f}"))
        # demand collapse
        for _ in range(int(round(5 * kscale))):
            pi = int(rng.choice(cfg.n_products, p=pw)); s0 = pick_day(span=14); ln = int(rng.integers(7, 15))
            mult = float(rng.uniform(0.15, 0.4))
            mean[pairs.pi.values == pi, s0:s0 + ln] *= mult
            truth.append(dict(kind="DEMAND_COLLAPSE", entity_type="product", entity_id=products.product_id[pi],
                              start=D(s0), end=D(s0 + ln), magnitude=mult, note=f"Demand falls to x{mult:.2f}"))
        # lost customer: stops ordering entirely
        big = customers.sort_values("scale", ascending=False).index[:40].values
        for ci in rng.choice(big, min(len(big), int(round(6 * kscale))), replace=False):
            s0 = pick_day(span=0)
            mean[pairs.ci.values == ci, s0:] *= 0.0
            truth.append(dict(kind="CUSTOMER_LOST", entity_type="customer", entity_id=customers.customer_id[ci],
                              start=D(s0), end=D(nd - 1), magnitude=0.0, note="Customer stops ordering"))
        # customer slowdown (churn signal before loss)
        for ci in rng.choice(big, min(len(big), int(round(6 * kscale))), replace=False):
            s0 = pick_day(span=21); ln = int(rng.integers(14, 28))
            mean[pairs.ci.values == ci, s0:s0 + ln] *= 0.35
            truth.append(dict(kind="CUSTOMER_SLOWDOWN", entity_type="customer", entity_id=customers.customer_id[ci],
                              start=D(s0), end=D(s0 + ln), magnitude=0.35, note="Orders drop to ~35% for weeks"))
        # supplier delay
        for _ in range(int(round(6 * kscale))):
            si = int(rng.integers(0, cfg.n_suppliers)); s0 = pick_day(span=8); ln = int(rng.integers(3, 8))
            extra = int(rng.integers(3, 7))
            lead_extra[si, s0:s0 + ln] += extra
            truth.append(dict(kind="SUPPLIER_DELAY", entity_type="supplier", entity_id=suppliers.supplier_id[si],
                              start=D(s0), end=D(s0 + ln), magnitude=extra, note=f"Deliveries {extra} days late"))
        # supplier price spike, sell price not updated -> margin erosion
        for _ in range(int(round(8 * kscale))):
            pi = int(rng.choice(cfg.n_products, p=pw)); s0 = pick_day(span=30); ln = int(rng.integers(14, 45))
            pct = float(rng.uniform(0.10, 0.28))
            cost_mult[pi, s0:] *= (1 + pct)       # persists, like a real price rise
            truth.append(dict(kind="COST_SPIKE", entity_type="product", entity_id=products.product_id[pi],
                              start=D(s0), end=D(nd - 1), magnitude=pct, note=f"Supplier cost +{pct * 100:.0f}%, sell price unchanged"))
        # sell price below floor for one customer-product (keying/contract error)
        for _ in range(int(round(8 * kscale))):
            r = int(rng.integers(0, npairs)); s0 = pick_day(span=20); ln = int(rng.integers(5, 20))
            price_arr[r, s0:s0 + ln] *= float(rng.uniform(0.62, 0.8))
            truth.append(dict(kind="PRICE_BELOW_FLOOR", entity_type="pair",
                              entity_id=f"{customers.customer_id[pairs.ci[r]]}|{products.product_id[pairs.pi[r]]}",
                              start=D(s0), end=D(s0 + ln), magnitude=0.7, note="Sell price keyed far below normal"))
        # over-buy: a PO of ~3x normal cover arrives for a perishable line -> expiry risk
        perish = np.where(shelf <= 14)[0]
        for _ in range(int(round(8 * kscale))):
            pi = int(rng.choice(perish)); s0 = pick_day(span=3)
            forced_orders.append(dict(day=s0, product_idx=int(pi), mult=float(rng.uniform(3.0, 5.0))))
            truth.append(dict(kind="OVERBUY", entity_type="product", entity_id=products.product_id[pi],
                              start=D(s0), end=D(min(nd - 1, s0 + int(shelf[pi]))), magnitude=0.0,
                              note="Over-order of a perishable line, waste risk"))

    # ---- realise demand (Poisson occurrence x typical quantity), then keying errors
    occur = rng.random(mean.shape) < (1 - np.exp(-np.clip(mean, 0, 8)))
    qty = np.maximum(1.0, np.round(pairs.typ_qty.values[:, None] * rng.lognormal(0, 0.35, mean.shape)))
    demand = np.where(occur, qty, 0.0)
    # a bump in promos/surges also lifts quantity, not only frequency
    demand *= np.sqrt(np.maximum(1.0, mean / np.maximum(pairs.rate.values[:, None] * dow_f * season, 1e-6)))
    demand = np.round(demand)
    boost = np.sqrt(np.maximum(1.0, mean / np.maximum(pairs.rate.values[:, None] * dow_f * season, 1e-6)))
    expected = (1 - np.exp(-np.clip(mean, 0, 8))) * pairs.typ_qty.values[:, None] * np.exp(0.35 ** 2 / 2) * boost

    if cfg.shocks:
        scored_from = int(nd * 0.55)
        hits = np.argwhere(demand[:, scored_from:] > 0)
        for r, c in hits[rng.choice(len(hits), min(len(hits), int(round(10 * kscale))), replace=False)]:
            c += scored_from
            f = float(rng.choice([10, 10, 100]))
            demand[r, c] *= f
            truth.append(dict(kind="KEYING_ERROR", entity_type="pair",
                              entity_id=f"{customers.customer_id[pairs.ci[r]]}|{products.product_id[pairs.pi[r]]}",
                              start=D(c), end=D(c), magnitude=f, note=f"Quantity keyed x{int(f)}"))

    # Bounds: stacked cost spikes on top of drift are capped so no product's cost leaves 0.85x to 1.45x of its start.
    cost_mult = np.clip(cost_mult, 0.85, 1.45)

    for i, t in enumerate(sorted(truth, key=lambda t: t["start"])):
        t["event_id"] = f"E-{i + 1:03d}"

    return TwinWorld(cfg, days, products, customers, suppliers, pairs, demand, price_arr, truth,
                     supplier_lead, cost_mult, lead_extra, forced_orders, expected)
