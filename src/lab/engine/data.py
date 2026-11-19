"""Load demand series from a dataset as dense matrices (products x periods)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..store import open_dataset


@dataclass
class Demand:
    product_ids: list[str]
    periods: pd.DatetimeIndex
    qty: np.ndarray                 # (products, periods) units ordered
    revenue: np.ndarray             # (products, periods) $ ordered
    category: dict
    freq: str


def load_demand(dataset: str, freq: str = "D") -> Demand:
    con = open_dataset(dataset, read_only=True)
    df = con.execute("""
        select l.product_id, l.delivery_date as d, sum(l.qty) as qty, sum(l.qty*coalesce(l.unit_price,0)) as rev
        from order_lines l where not coalesce(l.cancelled,false) and l.delivery_date is not null group by 1,2""").df()
    cats = dict(con.execute("select product_id, category from products").fetchall())
    con.close()
    df["d"] = pd.to_datetime(df.d)
    lo, hi = df.d.min(), df.d.max()
    if freq == "W":
        lo = lo - pd.Timedelta(days=lo.dayofweek)
        df["p"] = df.d - pd.to_timedelta(df.d.dt.dayofweek, unit="D")
        periods = pd.date_range(lo, hi, freq="W-MON")
        # only count complete trailing weeks
        if hi.dayofweek < 4 and len(periods):
            periods = periods[:-1]
    else:
        df["p"] = df.d
        periods = pd.date_range(lo, hi, freq="D")
    ids = sorted(df.product_id.unique())
    pi = {p: i for i, p in enumerate(ids)}
    pj = {t: j for j, t in enumerate(periods)}
    qty = np.zeros((len(ids), len(periods))); rev = np.zeros_like(qty)
    for r in df.itertuples():
        j = pj.get(r.p)
        if j is not None:
            qty[pi[r.product_id], j] += r.qty; rev[pi[r.product_id], j] += r.rev
    return Demand(ids, periods, qty, rev, cats, freq)


def load_promo_covariate(dataset: str, dem: Demand, extra: int = 0) -> np.ndarray:
    """Planned-promotion covariate per product and period (uplift x share of period covered).

    Promotions are planned and known ahead, so they are legitimate future covariates.
    Returns (products, periods + extra).
    """
    con = open_dataset(dataset, read_only=True)
    ev = con.execute("select entity_id, start_date, end_date, magnitude from truth_events where kind='PROMO'").df()
    con.close()
    n = len(dem.periods) + extra
    step = 7 if dem.freq == "W" else 1
    starts = list(dem.periods) + [dem.periods[-1] + pd.Timedelta(days=step * (i + 1)) for i in range(extra)]
    out = np.zeros((len(dem.product_ids), n))
    pi = {p: i for i, p in enumerate(dem.product_ids)}
    for r in ev.itertuples():
        i = pi.get(r.entity_id)
        if i is None:
            continue
        a, b = pd.Timestamp(r.start_date), pd.Timestamp(r.end_date)
        for j, st in enumerate(starts):
            en = st + pd.Timedelta(days=step - 1)
            overlap = (min(en, b) - max(st, a)).days + 1
            if overlap > 0:
                out[i, j] = max(out[i, j], (r.magnitude - 1.0) * overlap / step)
    return out


def load_expected(dataset: str, dem: Demand) -> np.ndarray | None:
    """True expected demand per product and period (twin only), the noise floor for scoring."""
    con = open_dataset(dataset, read_only=True)
    n = con.execute("select count(*) from truth_expected").fetchone()[0]
    if not n:
        con.close()
        return None
    df = con.execute("select product_id, day, expected_qty from truth_expected").df()
    con.close()
    df["day"] = pd.to_datetime(df.day)
    if dem.freq == "W":
        df["p"] = df.day - pd.to_timedelta(df.day.dt.dayofweek, unit="D")
    else:
        df["p"] = df.day
    g = df.groupby(["product_id", "p"]).expected_qty.sum()
    out = np.zeros_like(dem.qty)
    pi = {p: i for i, p in enumerate(dem.product_ids)}
    pj = {t: j for j, t in enumerate(dem.periods)}
    for (pid, per), v in g.items():
        if pid in pi and per in pj:
            out[pi[pid], pj[per]] = v
    return out
