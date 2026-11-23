"""Synthetic demand generator: seeded entity x item series with a known future, for scoring forecasts offline.

What it makes
Entities (customers, sites, accounts) each buy a basket of items. Every (entity, item) pair is a series with a stable
id. Demand has a trend, annual and weekly seasonality, calendar effects, item launches and retirements, intermittent
zero periods (low-rate series occur only sometimes) and labelled shock events. A simple stock model per item then caps
recorded sales at the units available (see censor.py), so the record under-states demand in short periods. True demand
over the horizon is kept alongside, so a forecast can be scored against what really happens.

Ported from the legacy twin generator. Removed: domain wording, supplier lead and delay shocks, prices, margins, cost
drift, expiry lots, keying errors (they would break the censoring invariant), and the regional public-holiday calendar,
replaced by three generic windows. Shock counts, shapes and magnitudes are kept. Every effect size is a SYNTHETIC
assumption, not measured data. Do not present results from this generator as client results.

Dataset shape (SyntheticDataset). S = series, I = items, T = history periods, H = horizon periods.
  series_ids, entity_ids, item_ids  (S,) str. series_ids are "ENT-0001|ITM-0001": stable for a given seed and layout.
  item_of_series                    (S,) int. Row of `items` for each series.
  history_index, future_index       DatetimeIndex of period start dates (lengths T and H).
  observed                          (S, T) int64. Recorded sales after censoring. This is the forecaster's input.
  censored                          (S, T) bool. True where observed < latent demand (the record under-states demand).
  latent_history                    (S, T) int64. Uncensored history demand. For analysis, not for forecasting.
  available                         (I, T) float. Units available to sell per item per period (the cap).
  stockout                          (I, T) bool. True where total demand exceeded the units available.
  truth_future                      (S, H) int64. Realised demand in the horizon. The scoring target.
  truth_mean_future                 (S, H) float. Expected demand in the horizon. A noise-free reference.
  events                            list of dict. Each has event_id, kind, scope ("item", "entity" or "series"),
                                    entity_id, item_id (or None), start, end (dates), magnitude, note.
  entities, items                   DataFrames of per-entity and per-item attributes.
Periods are days (freq="D") or weeks starting on `start` (freq="W"). The engine always runs daily, then sums to weeks.
Use `SyntheticDataset.to_long()` for a long table of one row per series and period.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

from .censor import censor_series, simulate_stock

# Item classes: (weight, annual amplitude, peak day-of-year). Collapsed from nine legacy categories into seasonal
# shapes. Names describe the seasonal shape only.
ITEM_CLASSES: dict[str, tuple[float, float, int]] = {
    "steady": (0.30, 0.10, 350),
    "summer_a": (0.14, 0.18, 20),
    "summer_b": (0.12, 0.08, 20),
    "summer_c": (0.07, 0.20, 20),
    "spring": (0.08, 0.30, 105),
    "year_end_a": (0.08, 0.22, 355),
    "year_end_b": (0.08, 0.35, 355),
    "year_end_c": (0.05, 0.15, 355),
    "fast_cycle": (0.08, 0.25, 15),
}

# Extra calendar sensitivity per item class: multiplier on the "run_up" and "mid_year" windows (1.0 = no effect).
ITEM_CLASS_CALENDAR: dict[str, dict[str, float]] = {
    "steady": {"run_up": 1.2},
    "summer_a": {"mid_year": 1.2},
    "summer_c": {"mid_year": 1.3},
    "spring": {"mid_year": 1.6},
    "year_end_a": {"run_up": 1.5},
    "year_end_b": {"run_up": 1.3, "mid_year": 1.5},
    "fast_cycle": {"run_up": 1.2},
}


class EntityType(NamedTuple):
    weight: float
    basket: tuple[int, int]          # items per entity (lo, hi)
    rate: float                      # base daily purchase rate per item
    weekday: list[float]             # Mon..Sun demand profile
    closure_sensitivity: float       # share of demand lost during the recess window


ENTITY_TYPES: dict[str, EntityType] = {
    "frequent": EntityType(0.40, (3, 14), 0.050, [0.7, 0.9, 1.0, 1.3, 1.5, 0.9, 0.2], 0.0),
    "bulk": EntityType(0.20, (8, 30), 0.060, [1.3, 1.1, 1.0, 1.0, 1.2, 0.8, 0.0], 0.0),
    "broad": EntityType(0.12, (12, 40), 0.080, [1.2, 1.0, 1.0, 1.1, 1.2, 0.5, 0.0], 0.0),
    "scheduled": EntityType(0.16, (3, 10), 0.070, [1.1, 1.1, 1.1, 1.1, 0.9, 0.0, 0.0], 0.55),
    "periodic": EntityType(0.12, (3, 12), 0.040, [0.6, 0.8, 1.0, 1.1, 1.4, 1.4, 0.4], 0.0),
}

# Calendar effect by entity type: (run_up multiplier, mid_year multiplier). SYNTHETIC, not measured trade data.
CAL_EFFECTS: dict[str, tuple[float, float]] = {
    "frequent": (1.2, 1.05),
    "bulk": (1.9, 1.8),
    "broad": (1.6, 1.5),
    "scheduled": (1.0, 1.0),
    "periodic": (1.5, 2.0),
}

# Generic calendar windows, as inclusive day-of-year ranges. Replaces the legacy holiday calendar.
RUN_UP_DOY = (330, 358)     # late-year run-up
MID_YEAR_DOY = (100, 115)   # mid-year peak
CLOSURE_DOY = (190, 203)    # two-week recess


@dataclass
class SyntheticDataset:
    seed: int
    freq: str
    series_ids: np.ndarray
    entity_ids: np.ndarray
    item_ids: np.ndarray
    item_of_series: np.ndarray
    history_index: pd.DatetimeIndex
    future_index: pd.DatetimeIndex
    observed: np.ndarray
    censored: np.ndarray
    latent_history: np.ndarray
    available: np.ndarray
    stockout: np.ndarray
    truth_future: np.ndarray
    truth_mean_future: np.ndarray
    events: list[dict]
    entities: pd.DataFrame
    items: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def to_long(self) -> pd.DataFrame:
        """One row per series and history period. Item-level columns are looked up from the item of each series."""
        n_series, n_periods = self.observed.shape
        return pd.DataFrame({
            "series_id": np.repeat(self.series_ids, n_periods),
            "entity_id": np.repeat(self.entity_ids, n_periods),
            "item_id": np.repeat(self.item_ids, n_periods),
            "period": np.tile(self.history_index.values, n_series),
            "observed": self.observed.ravel(),
            "latent_demand": self.latent_history.ravel(),
            "censored": self.censored.ravel(),
            "item_available": self.available[self.item_of_series].ravel(),
            "item_stockout": self.stockout[self.item_of_series].ravel(),
        })


def _season(amp: float, peak: int, doy: np.ndarray) -> np.ndarray:
    return 1.0 + amp * np.cos(2 * np.pi * (doy - peak) / 365.25)


def calendar_flags(days: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """0/1 flags per day for the three generic calendar windows."""
    doy = np.asarray(days.day_of_year)

    def window(lo_hi: tuple[int, int]) -> np.ndarray:
        return ((doy >= lo_hi[0]) & (doy <= lo_hi[1])).astype(float)

    return {"run_up": window(RUN_UP_DOY), "mid_year": window(MID_YEAR_DOY), "closure": window(CLOSURE_DOY)}


def generate(seed: int, n_entities: int, n_periods: int, horizon: int, *, freq: Literal["D", "W"] = "D",
             n_items: int | None = None, start: str = "2022-01-03", shocks: bool = True) -> SyntheticDataset:
    """Generate a reproducible dataset: `n_periods` of history and `horizon` periods of known future.

    The same arguments always give the same output. `start` should be a Monday when freq="W".
    """
    if n_entities < 1 or n_periods < 1 or horizon < 1:
        raise ValueError("n_entities, n_periods and horizon must all be at least 1")
    if freq not in ("D", "W"):
        raise ValueError(f"freq must be 'D' or 'W', got {freq!r}")
    k = 7 if freq == "W" else 1
    n_total = n_periods + horizon
    nd = n_total * k
    if nd < 180:
        raise ValueError("the daily span must be at least 180 days so lifecycle and shock windows fit")
    n_items = max(20, n_entities) if n_items is None else int(n_items)
    if n_items < 1:
        raise ValueError("n_items must be at least 1")

    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=nd, freq="D")
    period_dates = pd.date_range(start, periods=n_total, freq=f"{k}D")
    doy = days.day_of_year.values.astype(float)
    dow = days.day_of_week.values
    day_ix = np.arange(nd)
    cf = calendar_flags(days)
    events: list[dict] = []   # day indices here; converted to period dates at the end

    # ---- items: class, popularity, launches and retirements
    class_names = list(ITEM_CLASSES)
    class_w = np.array([ITEM_CLASSES[c][0] for c in class_names])
    item_class = rng.choice(len(class_names), size=n_items, p=class_w / class_w.sum())
    popularity = rng.pareto(1.3, n_items) + 1.0
    pw = popularity / popularity.sum()
    launch = np.zeros(n_items, dtype=int)
    retire = np.full(n_items, nd)
    new = rng.choice(n_items, int(n_items * 0.08), replace=False)
    launch[new] = rng.integers(60, int(nd * 0.6), len(new))
    old = rng.choice(np.setdiff1d(np.arange(n_items), new), int(n_items * 0.04), replace=False)
    retire[old] = rng.integers(int(nd * 0.7), nd - 30, len(old))
    live = ((day_ix[None, :] >= launch[:, None]) & (day_ix[None, :] < retire[:, None])).astype(float)

    season_by_class = np.stack([_season(ITEM_CLASSES[c][1], ITEM_CLASSES[c][2], doy) for c in class_names])
    item_season = season_by_class[item_class]                                  # (items, days)
    item_cal = np.ones((n_items, nd))
    for i, code in enumerate(item_class):
        for key, mult in ITEM_CLASS_CALENDAR.get(class_names[code], {}).items():
            item_cal[i] *= 1.0 + (mult - 1.0) * cf[key]

    # promotions: about one item in twelve gets one to three uplift windows of one to two weeks
    promo = np.ones((n_items, nd))
    for i in rng.choice(n_items, min(n_items, max(8, n_items // 12)), replace=False):   # at most every item when n_items < 8
        for _ in range(int(rng.integers(1, 4))):
            s0 = int(rng.integers(30, nd - 20))
            ln = int(rng.integers(7, 15))
            up = float(rng.uniform(1.4, 2.4))
            promo[i, s0:s0 + ln] = up
            events.append(dict(kind="PROMOTION", scope="item", entity=None, item=i, s0=s0, e0=s0 + ln - 1,
                               magnitude=up, note=f"Planned promotion, demand x{up:.1f}"))

    # ---- entities: type, scale, late joiners
    type_names = list(ENTITY_TYPES)
    type_w = np.array([ENTITY_TYPES[t].weight for t in type_names])
    ent_type = rng.choice(len(type_names), size=n_entities, p=type_w / type_w.sum())
    ent_scale = rng.lognormal(0, 0.6, n_entities)
    created_idx = np.zeros(n_entities, dtype=int)
    late = rng.choice(n_entities, int(n_entities * 0.30), replace=False)
    created_idx[late] = rng.integers(60, int(nd * 0.75), len(late))

    # ---- series: one per (entity, item) pair drawn from the entity's basket
    ent_l, item_l, rate_l, typ_l = [], [], [], []
    for e in range(n_entities):
        et = ENTITY_TYPES[type_names[ent_type[e]]]
        k_items = int(min(n_items, rng.integers(et.basket[0], et.basket[1] + 1)))
        for i in rng.choice(n_items, size=k_items, replace=False, p=pw):
            rate = et.rate * ent_scale[e] * rng.lognormal(0, 0.5) * min(3.0, popularity[i] ** 0.35)
            typ = max(1.0, float(np.round(rng.lognormal(np.log(6), 0.8))))
            ent_l.append(e)
            item_l.append(int(i))
            rate_l.append(rate)
            typ_l.append(typ)
    ent_idx = np.array(ent_l, dtype=int)
    item_idx = np.array(item_l, dtype=int)
    rate = np.array(rate_l)
    typ = np.array(typ_l)
    n_series = len(ent_idx)

    # ---- demand intensity (before shocks)
    weekday = np.array([ENTITY_TYPES[type_names[c]].weekday for c in ent_type])          # (entities, 7)
    dow_f = weekday[ent_idx][:, dow]                                                      # (series, days)
    season = item_season[item_idx]                                                        # (series, days)
    trend = 1.0 + 0.10 * (day_ix / 365.25)                                                # +10% a year
    eff = np.array([CAL_EFFECTS[type_names[c]] for c in ent_type])[ent_idx]              # (series, 2)
    cal_mult = (1.0 + (eff[:, 0:1] - 1.0) * cf["run_up"][None, :]) * \
               (1.0 + (eff[:, 1:2] - 1.0) * cf["mid_year"][None, :])
    sens = np.array([ENTITY_TYPES[type_names[c]].closure_sensitivity for c in ent_type])[ent_idx]
    closure_mult = 1.0 - sens[:, None] * cf["closure"][None, :]
    joined = (day_ix[None, :] >= created_idx[ent_idx][:, None]).astype(float)           # buy only after joining
    base0 = rate[:, None] * dow_f * season                                                # no trend, for the boost
    mean = (base0 * trend[None, :] * closure_mult * cal_mult * promo[item_idx] * live[item_idx]
            * item_cal[item_idx] * joined)

    kscale = max(1.0, nd / 730.0)                                                         # events per year stay fixed
    scored_from = int(nd * 0.55)

    def pick_day(lo: int, span: int) -> int:
        hi = nd - span - 14
        if hi <= lo:
            hi = lo + 1
        return int(rng.integers(lo, hi))

    if shocks:
        for _ in range(int(round(10 * kscale))):                                          # demand surge
            i = int(rng.choice(n_items, p=pw))
            s0 = pick_day(scored_from, 8)
            ln = int(rng.integers(4, 9))
            mult = float(rng.uniform(1.8, 3.2))
            mean[item_idx == i, s0:s0 + ln] *= mult
            events.append(dict(kind="DEMAND_SURGE", scope="item", entity=None, item=i, s0=s0, e0=s0 + ln - 1,
                               magnitude=mult, note=f"Unplanned surge x{mult:.1f}"))
        for _ in range(int(round(5 * kscale))):                                           # demand collapse
            i = int(rng.choice(n_items, p=pw))
            s0 = pick_day(scored_from, 14)
            ln = int(rng.integers(7, 15))
            mult = float(rng.uniform(0.15, 0.4))
            mean[item_idx == i, s0:s0 + ln] *= mult
            events.append(dict(kind="DEMAND_COLLAPSE", scope="item", entity=None, item=i, s0=s0, e0=s0 + ln - 1,
                               magnitude=mult, note=f"Demand falls to x{mult:.2f}"))
        big = np.argsort(-ent_scale)[:40]                                                 # the largest entities
        for e in rng.choice(big, min(len(big), int(round(6 * kscale))), replace=False):  # entity stops ordering
            s0 = pick_day(scored_from, 0)
            mean[ent_idx == e, s0:] *= 0.0
            events.append(dict(kind="ENTITY_LOST", scope="entity", entity=int(e), item=None, s0=s0, e0=nd - 1,
                               magnitude=0.0, note="Entity stops ordering"))
        for e in rng.choice(big, min(len(big), int(round(6 * kscale))), replace=False):  # slowdown before loss
            s0 = pick_day(scored_from, 21)
            ln = int(rng.integers(14, 28))
            mean[ent_idx == e, s0:s0 + ln] *= 0.35
            events.append(dict(kind="ENTITY_SLOWDOWN", scope="entity", entity=int(e), item=None, s0=s0,
                               e0=s0 + ln - 1, magnitude=0.35, note="Orders drop to about 35% for weeks"))

    # ---- realise demand: Poisson-like occurrence x typical quantity, then the quantity boost
    occur = rng.random(mean.shape) < (1.0 - np.exp(-np.clip(mean, 0, 8)))
    qty = np.maximum(1.0, np.round(typ[:, None] * rng.lognormal(0, 0.35, mean.shape)))
    boost = np.sqrt(np.maximum(1.0, mean / np.maximum(base0, 1e-6)))
    demand = np.round(np.where(occur, qty, 0.0) * boost)
    expected = (1.0 - np.exp(-np.clip(mean, 0, 8))) * typ[:, None] * np.exp(0.35 ** 2 / 2) * boost

    if k > 1:
        demand = demand.reshape(n_series, n_total, k).sum(axis=2)
        expected = expected.reshape(n_series, n_total, k).sum(axis=2)

    hist = demand[:, :n_periods].astype(np.int64)
    truth_future = demand[:, n_periods:].astype(np.int64)
    truth_mean = expected[:, n_periods:]

    # ---- supply and censoring, on the period grid of the output
    item_hist = np.zeros((n_items, n_periods))
    np.add.at(item_hist, item_idx, hist)
    lead = rng.integers(1, 4, n_items)
    run = simulate_stock(item_hist, lead, window=28 if freq == "D" else 4)
    observed, censored = censor_series(hist, item_idx, run)

    # ---- identifiers, attribute tables and events
    entity_ids = np.array([f"ENT-{e + 1:04d}" for e in range(n_entities)])
    item_ids = np.array([f"ITM-{i + 1:04d}" for i in range(n_items)])
    series_ids = np.array([f"{entity_ids[e]}|{item_ids[i]}" for e, i in zip(ent_idx, item_idx)])

    def period_date(day: int) -> date:
        return period_dates[min(day, nd - 1) // k].date()

    ev_rows = []
    for ev in sorted(events, key=lambda ev: ev["s0"]):
        item_id = item_ids[ev["item"]] if ev["item"] is not None else None
        entity_id = entity_ids[ev["entity"]] if ev["entity"] is not None else None
        ev_rows.append(dict(kind=ev["kind"], scope=ev["scope"], entity_id=entity_id, item_id=item_id,
                            start=period_date(ev["s0"]), end=period_date(ev["e0"]),
                            magnitude=ev["magnitude"], note=ev["note"]))
    for n, row in enumerate(ev_rows, start=1):
        row["event_id"] = f"E-{n:03d}"

    entities = pd.DataFrame({
        "entity_id": entity_ids,
        "archetype": [type_names[c] for c in ent_type],
        "scale": ent_scale,
        "joined_on": [days[i].date() for i in created_idx],
    })
    items = pd.DataFrame({
        "item_id": item_ids,
        "item_class": [class_names[c] for c in item_class],
        "popularity": popularity,
        "lead_periods": lead,
        "launched_on": [days[min(i, nd - 1)].date() for i in launch],
        "retired_on": [days[i].date() if i < nd else None for i in retire],
    })

    return SyntheticDataset(
        seed=seed,
        freq=freq,
        series_ids=series_ids,
        entity_ids=entity_ids[ent_idx],
        item_ids=item_ids[item_idx],
        item_of_series=item_idx,
        history_index=pd.DatetimeIndex(period_dates[:n_periods]),
        future_index=pd.DatetimeIndex(period_dates[n_periods:]),
        observed=observed,
        censored=censored,
        latent_history=hist,
        available=run.available,
        stockout=run.short,
        truth_future=truth_future,
        truth_mean_future=truth_mean,
        events=ev_rows,
        entities=entities,
        items=items,
        meta={"n_items": n_items, "n_series": n_series, "start": start, "shocks": shocks, "kscale": kscale},
    )
