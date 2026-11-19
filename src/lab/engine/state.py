"""Business state at a point in time: demand history, stock, pipeline, prices. Cached per dataset."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA
from ..store import open_dataset
from .forecast import Forecast, TimesFMEngine

CACHE_DIR = DATA / "cache"
CACHE_DIR.mkdir(exist_ok=True)


@dataclass
class WorldData:
    name: str
    products: pd.DataFrame
    customers: pd.DataFrame
    lines: pd.DataFrame            # order lines, delivery_date datetime64
    lots: pd.DataFrame
    moves: pd.DataFrame
    pos: pd.DataFrame
    days: pd.DatetimeIndex
    pidx: dict
    demand: np.ndarray             # (products, days) daily units ordered
    first: pd.Timestamp
    last: pd.Timestamp
    reveal: bool = False


_WORLDS: dict[str, WorldData] = {}


def get_world(name: str, reload: bool = False) -> WorldData:
    if name in _WORLDS and not reload:
        return _WORLDS[name]
    con = open_dataset(name, read_only=True)
    q = lambda s: con.execute(s).df()
    products = q("select * from products")
    customers = q("select * from customers")
    lines = q("select * from order_lines where not coalesce(cancelled,false) and delivery_date is not null")
    lines["delivery_date"] = pd.to_datetime(lines.delivery_date)
    lots = q("select * from lots"); lots["received_date"] = pd.to_datetime(lots.received_date); lots["use_by_date"] = pd.to_datetime(lots.use_by_date)
    moves = q("select * from stock_movements"); moves["moved_on"] = pd.to_datetime(moves.moved_on)
    pos = q("select * from purchase_lines"); pos["created_on"] = pd.to_datetime(pos.created_on); pos["expected_on"] = pd.to_datetime(pos.expected_on)
    con.close()
    first, last = lines.delivery_date.min(), lines.delivery_date.max()
    days = pd.date_range(first, last, freq="D")
    pidx = {p: i for i, p in enumerate(products.product_id)}
    demand = np.zeros((len(products), len(days)))
    di = (lines.delivery_date - first).dt.days.values
    pi = lines.product_id.map(pidx).values
    ok = ~pd.isna(pi)
    np.add.at(demand, (pi[ok].astype(int), di[ok]), lines.qty.values[ok])
    w = WorldData(name, products, customers, lines, lots, moves, pos, days, pidx, demand, first, last)
    _WORLDS[name] = w
    return w


def week_start(d: pd.Timestamp) -> pd.Timestamp:
    return (d - pd.Timedelta(days=d.dayofweek)).normalize()


def weekly_matrix(w: WorldData, as_of: pd.Timestamp) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Complete weeks strictly before the week containing as_of: (products, weeks)."""
    end = week_start(as_of)                          # exclusive
    start = week_start(w.first)
    weeks = pd.date_range(start, end - pd.Timedelta(days=7), freq="7D")
    if len(weeks) == 0:
        return np.zeros((len(w.products), 0)), weeks
    out = np.zeros((len(w.products), len(weeks)))
    for j, ws in enumerate(weeks):
        a = (ws - w.first).days; b = a + 7
        a0, b0 = max(a, 0), min(b, len(w.days))
        if b0 > a0:
            out[:, j] = w.demand[:, a0:b0].sum(axis=1)
    return out, weeks


def stock_at(w: WorldData, as_of: pd.Timestamp) -> pd.DataFrame:
    """Remaining quantity per lot at the start of as_of, joined to product, use-by, cost."""
    m = w.moves[w.moves.moved_on < as_of].groupby("lot_id").qty_change.sum().rename("qty")
    lots = w.lots.set_index("lot_id").join(m, how="inner")
    lots = lots[lots.qty > 1e-6]
    lots = lots[(lots.received_date < as_of)]
    return lots.reset_index()


def pipeline_at(w: WorldData, as_of: pd.Timestamp) -> pd.DataFrame:
    p = w.pos
    return p[(p.created_on < as_of) & (p.expected_on >= as_of)]


def unit_cost_at(w: WorldData, as_of: pd.Timestamp) -> pd.Series:
    p = w.pos[w.pos.created_on < as_of].sort_values("created_on").groupby("product_id").unit_cost.last()
    base = w.products.set_index("product_id").unit_cost
    return p.reindex(base.index).fillna(base)


class ForecastService:
    """Weekly and daily forecasts at an as-of date, cached on disk so replays are cheap."""

    def __init__(self, w: WorldData):
        self.w = w

    def _path(self, kind: str, as_of: pd.Timestamp, horizon: int) -> Path:
        return CACHE_DIR / f"{self.w.name}_{kind}_{as_of:%Y%m%d}_{horizon}.pkl"

    def weekly(self, as_of: pd.Timestamp, horizon: int = 4) -> dict:
        path = self._path("wk", week_start(as_of), horizon)
        if path.exists():
            return pickle.loads(path.read_bytes())
        mat, weeks = weekly_matrix(self.w, as_of)
        active = np.where(mat[:, -min(26, mat.shape[1]):].sum(axis=1) > 0)[0] if mat.shape[1] else np.array([], dtype=int)
        res = {"weeks": weeks, "active": active, "mat": mat, "future_start": week_start(as_of)}
        if len(active) and mat.shape[1] >= 4:
            fc = TimesFMEngine.get().predict([mat[i] for i in active], horizon)
            res["fc"] = fc
        else:
            res["fc"] = None
        path.write_bytes(pickle.dumps(res))
        return res

    def weekday_profile(self, as_of: pd.Timestamp, product: int, weeks: int = 12, k: float = 2.0) -> np.ndarray:
        """Mon..Sun share of demand for one product over the last `weeks` weeks, mean 1.

        Motivation vs Logic: the twin and real trade both sell very unevenly by weekday (median busiest day 1.7x
        the quietest). Logic: count units per weekday, then shrink toward flat with k pseudo-weeks of flat history
        (k=2) so a product with few orders does not get an invented pattern, while a real weekly rhythm survives.
        """
        end = int(self.w.days.searchsorted(as_of))
        start = max(0, end - 7 * weeks)
        seg = self.w.demand[product, start:end]
        dows = np.asarray(self.w.days[start:end].day_of_week)
        S = np.array([seg[dows == d].sum() for d in range(7)], dtype=float)
        m = S.mean()
        if m <= 0 or len(seg) == 0:
            return np.ones(7)
        return (S + k * m) / ((1.0 + k) * m)

    def daily_path(self, as_of: pd.Timestamp, product: int, days_ahead: int) -> dict | None:
        """Expected units per day for the next days_ahead days (q10, q50, q90, point).

        Motivation vs Logic: the weekly TimesFM median was spread evenly over each day, so the daily view was flat
        and expiry maths ignored that Saturday sells more than Sunday. Logic: each day takes its week's forecast
        total divided by 7, times the product's weekday share from weekday_profile.
        """
        res = self.weekly(as_of, 4)
        fc: Forecast | None = res["fc"]
        if fc is None:
            return None
        pos = np.where(res["active"] == product)[0]
        if len(pos) == 0:
            return None
        k = pos[0]
        shape = self.weekday_profile(as_of, product)
        out = {"point": [], "q10": [], "q50": [], "q90": [], "dates": []}
        for j in range(days_ahead):
            day = as_of + pd.Timedelta(days=j)
            wk = int((day - res["future_start"]).days // 7)
            if wk < 0 or wk >= fc.point.shape[1]:
                break
            f = shape[day.dayofweek] / 7.0
            out["dates"].append(day)
            out["point"].append(float(fc.point[k, wk] * f))
            out["q10"].append(float(fc.q[k, wk, 0] * f))
            out["q50"].append(float(fc.q[k, wk, 4] * f))
            out["q90"].append(float(fc.q[k, wk, 8] * f))
        return out

    def daily_demand_quantiles(self, as_of: pd.Timestamp, product: int, days_ahead: int) -> dict | None:
        """Projected demand over the next days_ahead days (point, q10, q50, q90), summed from daily_path."""
        path = self.daily_path(as_of, product, days_ahead)
        if path is None:
            return None
        return {key: float(sum(path[key])) for key in ("point", "q10", "q50", "q90")}


def customer_weekly(w: WorldData, as_of: pd.Timestamp) -> tuple[np.ndarray, list, pd.DatetimeIndex]:
    """Weekly revenue per customer for complete weeks before as_of's week: (customers, weeks)."""
    end = week_start(as_of); start = week_start(w.first)
    weeks = pd.date_range(start, end - pd.Timedelta(days=7), freq="7D")
    cids = list(w.customers.customer_id)
    ci = {c: i for i, c in enumerate(cids)}
    L = w.lines[w.lines.delivery_date < end]
    out = np.zeros((len(cids), len(weeks)))
    if len(weeks) and len(L):
        wk = ((L.delivery_date - start).dt.days // 7).values
        rows = L.customer_id.map(ci).values
        ok = ~pd.isna(rows) & (wk >= 0) & (wk < len(weeks))
        np.add.at(out, (rows[ok].astype(int), wk[ok]), (L.qty * L.unit_price.fillna(0)).values[ok])
    return out, cids, weeks


def customer_fc(fs: "ForecastService", as_of: pd.Timestamp, horizon: int = 2) -> dict:
    path = CACHE_DIR / f"{fs.w.name}_cust_{week_start(as_of):%Y%m%d}_{horizon}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    mat, cids, weeks = customer_weekly(fs.w, as_of)
    res = {"weeks": weeks, "cids": cids, "mat": mat, "future_start": week_start(as_of), "fc": None, "active": np.array([], dtype=int)}
    if mat.shape[1] >= 8:
        active = np.where(((mat[:, -12:] > 0).sum(axis=1) >= 3))[0]
        if len(active):
            res["active"] = active
            res["fc"] = TimesFMEngine.get().predict([mat[i] for i in active], horizon)
    path.write_bytes(pickle.dumps(res))
    return res


def cost_weekly(w: WorldData, as_of: pd.Timestamp) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Volume-weighted weekly purchase unit cost per product, forward-filled: (products, weeks)."""
    end = week_start(as_of); start = week_start(w.first)
    weeks = pd.date_range(start, end - pd.Timedelta(days=7), freq="7D")
    out = np.full((len(w.products), len(weeks)), np.nan)
    P = w.pos[(w.pos.created_on < end) & w.pos.unit_cost.notna()]
    if len(P) and len(weeks):
        wk = ((P.created_on - start).dt.days // 7).values
        pi = P.product_id.map(w.pidx).values
        ok = ~pd.isna(pi) & (wk >= 0) & (wk < len(weeks))
        num = np.zeros_like(out); den = np.zeros_like(out)
        np.add.at(num, (pi[ok].astype(int), wk[ok]), (P.unit_cost * P.qty_ordered).values[ok])
        np.add.at(den, (pi[ok].astype(int), wk[ok]), P.qty_ordered.values[ok])
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(den > 0, num / den, np.nan)
        for i in range(out.shape[0]):               # forward-fill gaps
            last = np.nan
            for j in range(out.shape[1]):
                if np.isnan(out[i, j]): out[i, j] = last
                else: last = out[i, j]
    return out, weeks


def cost_fc(fs: "ForecastService", as_of: pd.Timestamp, horizon: int = 2) -> dict:
    path = CACHE_DIR / f"{fs.w.name}_cost_{week_start(as_of):%Y%m%d}_{horizon}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    mat, weeks = cost_weekly(fs.w, as_of)
    res = {"weeks": weeks, "mat": mat, "future_start": week_start(as_of), "fc": None, "active": np.array([], dtype=int)}
    if mat.shape[1] >= 10:
        active = np.where(~np.isnan(mat[:, -10:]).any(axis=1))[0]
        if len(active):
            res["active"] = active
            res["fc"] = TimesFMEngine.get().predict([np.nan_to_num(mat[i]) for i in active], horizon)
    path.write_bytes(pickle.dumps(res))
    return res
