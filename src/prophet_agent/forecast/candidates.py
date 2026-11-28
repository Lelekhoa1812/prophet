"""Candidate forecasts for the combination: simple rules, TimesFM, and the market and structural models of each domain.

Motivation vs Logic
Motivation (Mission 10, owner decisions 2026-11-28): the centre line is earned on the backtest, and where a market prices the
target its view enters for the whole horizon. The combination (forecast.ensemble) needs every candidate as a function that
forecasts from any origin with only what was known there.
Logic: a candidate is a point path from an origin plus a band. Rules and market paths take an empirical band: the quantiles
of their own errors at each step ahead, over windows that ended before the origin (actual minus path, or the log ratio for a
price), added to the path. The same errors give a bias-corrected variant whose centre moves by their median, which is how a
market path's term premium or a rule's lag is removed without assuming its size. TimesFM keeps its own band and is called
once for all origins. Domain candidates read outside data through ``fetch`` and use only observations dated on or before the
origin's cutoff (the end of the last known period):
- price or index level with an implied-volatility index (gold: GVZ, S&P 500: VIX): the cost-of-carry forward (spot grown at
  the 3-month Treasury bill rate) with the implied volatility as the width;
- the Australian cash rate: the market path from bank bill rates (1, 3 and 6 months, RBA F1) and government bond yields (2, 3
  and 5 years, RBA F2), turned into forward rates between those tenors;
- Australian unemployment: a direct regression of the change over each horizon on the recent change, the gap to the
  ten-year mean and the 12-month growth of job vacancies (ABS), and the RBA's own published forecast path (SMP archive).
Measured basis for including each: see GOAL.md, Mission 10 evidence. Every candidate is scored by the combination; none is
trusted because of what it is.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import re
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.stats import norm

from prophet_agent.forecast.ensemble import Candidate
from prophet_agent.forecast.metrics import QS

Z = norm.ppf(QS)
LOOKBACK = 120          # past origins whose errors make a rule's band
MIN_ERRORS = 8          # fewer errors at a step than this: a normal band from the one-step changes instead
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
RBA_CSV = "https://www.rba.gov.au/statistics/tables/csv/{table}-data.csv"
ABS_JV = "https://data.api.abs.gov.au/rest/data/ABS,JV,1.0/M1.7.TOT.20.AUS.Q?format=csvfilewithlabels"
RBA_SMP_ARCHIVE = "https://www.rba.gov.au/statistics/xls/smp-forecast-archive.xlsx"

Fetch = Callable[[str], bytes]


# ---------------------------------------------------------------- dates and outside series

def _date(text: str) -> dt.date:
    return dt.date.fromisoformat(str(text)[:10])


def cutoffs(dates: list[str]) -> list[dt.date]:
    """The last day each value's information covers. Dates that are all the first of a month label whole months (a resampled
    close is the month's last value), so the cutoff is that month's last day; otherwise the date itself."""
    ds = [_date(d) for d in dates]
    if ds and all(d.day == 1 for d in ds):
        out = []
        for d in ds:
            nxt = dt.date(d.year + (d.month == 12), d.month % 12 + 1, 1)
            out.append(nxt - dt.timedelta(days=1))
        return out
    return ds


def step_years(dates: list[str]) -> float:
    ds = [_date(d) for d in dates[-60:]]
    if len(ds) < 3:
        return 1 / 12
    return float(np.median(np.diff([d.toordinal() for d in ds]))) / 365.25


class Dated:
    """An outside series as sorted (date, value) pairs, read as of a cutoff."""

    def __init__(self, rows: list[tuple[str, float]], label: str):
        rows = sorted((_date(d), float(v)) for d, v in rows if v is not None and np.isfinite(v))
        self.days = np.array([d.toordinal() for d, _ in rows])
        self.vals = np.array([v for _, v in rows])
        self.label = label

    def asof(self, day: dt.date) -> float:
        k = int(np.searchsorted(self.days, day.toordinal(), side="right")) - 1
        if k < 0:
            raise ValueError(f"{self.label} has no value on or before {day}")
        return float(self.vals[k])

    def lag_days(self, day: dt.date) -> int:
        k = int(np.searchsorted(self.days, day.toordinal(), side="right")) - 1
        return day.toordinal() - int(self.days[k]) if k >= 0 else 10 ** 6


def parse_fred(body: bytes, label: str) -> Dated:
    rows = []
    for r in csv.reader(io.StringIO(body.decode("utf-8-sig", errors="replace"))):
        if len(r) >= 2 and re.match(r"\d{4}-\d{2}-\d{2}$", r[0]):
            try:
                rows.append((r[0], float(r[1])))
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"no observations in {label}")
    return Dated(rows, label)


def merged(daily: "Dated | None", monthly: "Dated | None", label: str) -> "Dated | None":
    """A daily series with its monthly companion's earlier observations in front of it (RBA F1 and F1.1)."""
    if monthly is None or not len(monthly.days):
        return daily
    if daily is None or not len(daily.days):
        return monthly
    early = monthly.days < daily.days[0]
    rows = [(dt.date.fromordinal(int(d)).isoformat(), float(v)) for d, v in zip(monthly.days[early], monthly.vals[early])]
    rows += [(dt.date.fromordinal(int(d)).isoformat(), float(v)) for d, v in zip(daily.days, daily.vals)]
    return Dated(rows, label)


def parse_rba(body: bytes) -> dict[str, Dated]:
    from prophet_agent.opendata.market_benchmark import parse_f1
    return {sid: Dated(rows, sid) for sid, rows in parse_f1(body).items() if rows}


# ---------------------------------------------------------------- bands from past errors

class _PathBand:
    """A point-path function turned into quantile forecasts with a band from its own past errors."""

    def __init__(self, y: np.ndarray, path: Callable[[int, int], np.ndarray], log: bool, recentre: bool):
        self.y, self.path, self.log, self.recentre = y, path, log, recentre
        self.memo: dict[tuple[int, int], np.ndarray] = {}

    def _p(self, s: int, h: int) -> np.ndarray:
        key = (s, h)
        if key not in self.memo:
            self.memo[key] = np.asarray(self.path(s, h), dtype=float)
        return self.memo[key]

    def __call__(self, t: int, h: int) -> np.ndarray:
        y, centre = self.y, self._p(t, h)
        if not np.isfinite(centre).all() or (self.log and (centre <= 0).any()):
            raise ValueError("the path is not defined here")
        errs: list[list[float]] = [[] for _ in range(h)]
        for s in range(max(2, t - LOOKBACK), t):
            try:
                p = self._p(s, h)
            except (ValueError, IndexError):
                continue
            for k in range(1, h + 1):
                if s + k > t or not np.isfinite(p[k - 1]):
                    break
                a = y[s + k - 1]
                if self.log:
                    if a > 0 and p[k - 1] > 0:
                        errs[k - 1].append(float(np.log(a / p[k - 1])))
                else:
                    errs[k - 1].append(float(a - p[k - 1]))
        hist = y[:t]
        steps = np.diff(np.log(hist[hist > 0])) if self.log else np.diff(hist)
        sd1 = float(np.std(steps[-60:], ddof=1)) if len(steps) > 2 else 0.0
        q = np.empty((h, len(QS)))
        for k in range(h):
            e = errs[k]
            if len(e) >= MIN_ERRORS:
                qq = np.quantile(e, QS)
                if not self.recentre:
                    qq = qq - np.median(e)
            else:
                qq = sd1 * np.sqrt(k + 1) * Z
            q[k] = qq
        q = np.maximum.accumulate(np.sort(q, axis=1), axis=1)
        return np.exp(np.log(centre)[:, None] + q) if self.log else centre[:, None] + q


def banded(name: str, kind: str, y: np.ndarray, path: Callable[[int, int], np.ndarray], *, log: bool,
           recentre: bool = False, note: str = "") -> Candidate:
    return Candidate(name, kind, _PathBand(y, path, log, recentre), note)


# ---------------------------------------------------------------- rules

def rule_candidates(y: np.ndarray, *, log: bool, season: int | None, mean_reverting: bool) -> list[Candidate]:
    def naive(t, h):
        return np.full(h, y[t - 1])

    def drift(t, h):
        x = y[max(0, t - 61):t]
        if log:
            x = x[x > 0]
            g = np.mean(np.diff(np.log(x))) if len(x) > 2 else 0.0
            return y[t - 1] * np.exp(g * np.arange(1, h + 1))
        d = np.mean(np.diff(x)) if len(x) > 2 else 0.0
        return y[t - 1] + d * np.arange(1, h + 1)

    out = [banded("naive", "rule", y, naive, log=log, note="the last value, held"),
           banded("drift", "rule", y, drift, log=log, note="the mean change of the last five years, continued")]
    if season and season >= 2:
        m = int(season)

        def snaive(t, h):
            if t < m:
                raise ValueError("shorter than one season")
            return np.array([y[t - m + (k % m)] for k in range(h)])
        out.append(banded("seasonal_naive", "rule", y, snaive, log=log, note=f"the same period {m} steps earlier"))
    if mean_reverting:
        def ar1(t, h):
            x = y[max(0, t - 120):t]
            if len(x) < 24:
                raise ValueError("too short to fit")
            b, a = np.polyfit(x[:-1], x[1:], 1)
            b = float(np.clip(b, -0.999, 0.999))
            out_, cur = [], y[t - 1]
            for _ in range(h):
                cur = a + b * cur
                out_.append(cur)
            return np.array(out_)
        out.append(banded("mean_reversion", "rule", y, ar1, log=False,
                          note="AR(1) on the last ten years: the pull towards the recent mean"))
    return out


def timesfm_candidate(y: np.ndarray, engine: Any) -> Candidate:
    cache: dict[int, np.ndarray] = {}

    def prepare(origins: list[int], h: int) -> None:
        todo = [t for t in origins if t not in cache and t >= 8]
        for i in range(0, len(todo), 64):
            part = todo[i:i + 64]
            fc = engine.forecast_batch([y[:t] for t in part], h)
            for j, t in enumerate(part):
                cache[t] = np.asarray(fc.q[j], dtype=float)

    def predict(t: int, h: int) -> np.ndarray:
        if t not in cache:
            prepare([t], h)
        return cache[t][:h]

    return Candidate("timesfm", "model", predict, "TimesFM 3 on the target's own history", prepare=prepare)


def driver_candidate(y: np.ndarray, drivers: dict[str, np.ndarray], engine: Any) -> Candidate:
    """TimesFM with the adopted outside drivers as past covariates (known only up to each origin), so an adopted driver
    reaches the forecast and competes in the combination (Mission 10, M10-5)."""
    from prophet_agent.forecast.control import Control

    def predict(t: int, h: int) -> np.ndarray:
        if t < 24:
            raise ValueError("too short for covariates")
        c = Control(past={k: np.asarray(v[:t], dtype=float) for k, v in drivers.items()})
        return np.asarray(engine.forecast_controlled(y[:t], h, c).q, dtype=float)
    return Candidate("timesfm_drivers", "driver", predict,
                     "TimesFM 3 with the adopted drivers as past covariates: " + ", ".join(sorted(drivers)))


# ---------------------------------------------------------------- domain candidates

def carry_candidate(y: np.ndarray, cut: list[dt.date], years: float, iv: Dated, rate: Dated, label: str) -> Candidate:
    """The cost-of-carry forward, spot grown at the bill rate, with the implied volatility index as the band."""
    def predict(t: int, h: int) -> np.ndarray:
        day = cut[t - 1]
        if iv.lag_days(day) > 10:
            raise ValueError(f"{iv.label} has no value within ten days of {day}")
        vol = iv.asof(day) / 100.0
        r = rate.asof(day) / 100.0
        k = np.arange(1, h + 1) * years
        centre = np.log(y[t - 1]) + r * k
        return np.exp(centre[:, None] + vol * np.sqrt(k)[:, None] * Z[None, :])
    return Candidate("market_forward", "market", predict,
                     f"forward price (spot grown at the 3-month bill rate) with {label} as the width")


BAB = (("FIRMMBAB30D", 1), ("FIRMMBAB90D", 3), ("FIRMMBAB180D", 6))
BONDS = (("FCMYGBAG2D", 24), ("FCMYGBAG3D", 36), ("FCMYGBAG5D", 60))


def market_path(knots: list[tuple[int, float]], h: int) -> np.ndarray:
    """Monthly forward rates from average rates to given maturities (months): the forward between two knots is
    (b * R_b - a * R_a) / (b - a); before the first knot its own rate; after the last, flat."""
    knots = sorted(knots)
    fwd = np.empty(h)
    prev_m, prev_r = 0, None
    seg = []
    for m, r in knots:
        f = r if prev_r is None else (m * r - prev_m * prev_r) / (m - prev_m)
        seg.append((prev_m, m, f))
        prev_m, prev_r = m, r
    for k in range(1, h + 1):
        f = next((f for a, b, f in seg if a < k <= b), seg[-1][2])
        fwd[k - 1] = f
    return fwd


def cash_rate_candidates(y: np.ndarray, cut: list[dt.date], tables: dict[str, Dated]) -> list[Candidate]:
    def path(t: int, h: int) -> np.ndarray:
        day = cut[t - 1]
        knots = []
        for sid, m in BAB + BONDS:
            s = tables.get(sid)
            if s is not None and s.lag_days(day) <= 10:
                knots.append((m, s.asof(day)))
        if not any(m <= 6 for m, _ in knots):
            raise ValueError(f"no bank bill rates within ten days of {day}")
        return market_path(knots, h)
    return [banded("market_path", "market", y, path, log=False,
                   note="the forward path in bank bill rates (1, 3, 6 months) and bond yields (2, 3, 5 years), as priced"),
            banded("market_path_adjusted", "market", y, path, log=False, recentre=True,
                   note="the same path moved by the median of its own past errors at each step (term premium, bill spread)")]


def vacancy_candidate(y: np.ndarray, cut: list[dt.date], jv: Dated) -> Candidate:
    """Direct regression, per step ahead, of the change in unemployment on its 3-month change, its gap to the ten-year mean
    and the 12-month log growth of job vacancies (a quarter's vacancies are counted known two months after it ends)."""
    n = len(y)

    def feats(i: int) -> np.ndarray | None:
        if i < 120:
            return None
        day = cut[i]
        known = day - dt.timedelta(days=62)
        try:
            v_now, v_year = jv.asof(known), jv.asof(known - dt.timedelta(days=365))
        except ValueError:
            return None
        return np.array([1.0, y[i] - y[i - 3], y[i] - np.mean(y[i - 119:i + 1]), 100 * np.log(v_now / v_year)])

    F = [feats(i) for i in range(n)]

    def path(t: int, h: int) -> np.ndarray:
        x0 = F[t - 1]
        if x0 is None:
            raise ValueError("no vacancy data at this origin")
        out = np.empty(h)
        for k in range(1, h + 1):
            rows = [(F[i], y[i + k] - y[i]) for i in range(120, t - k) if F[i] is not None]
            if len(rows) < 48:
                raise ValueError("too few complete windows to fit")
            X = np.array([r[0] for r in rows])
            b = np.linalg.lstsq(X, np.array([r[1] for r in rows]), rcond=None)[0]
            out[k - 1] = y[t - 1] + float(x0 @ b)
        return out
    return banded("vacancy_model", "structural", y, path, log=False,
                  note="direct regression on the 3-month change, the gap to the ten-year mean and vacancy growth (ABS)")


def parse_smp_archive(body: bytes, sheet: str = "UR") -> list[tuple[dt.date, dt.date, float]]:
    """(publication month, target quarter-end month, value) from the RBA's SMP forecast archive workbook."""
    import openpyxl
    mon = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}

    def my(text: Any) -> dt.date | None:
        m = re.match(r"^([A-Z][a-z]{2})-(\d{4})$", str(text or "").strip())
        return dt.date(int(m.group(2)), mon[m.group(1)], 1) if m and m.group(1) in mon else None

    wb = openpyxl.load_workbook(io.BytesIO(body), read_only=True, data_only=True)
    rows = list(wb[sheet].iter_rows(values_only=True))
    header = next((r for r in rows if sum(1 for c in r[1:] if my(c)) >= 2), None)
    if header is None:
        raise ValueError("the SMP archive has no row of publication months")
    pubs = {j: my(c) for j, c in enumerate(header) if j > 0 and my(c)}
    out = []
    for r in rows:
        tgt = my(r[0]) if r else None
        if tgt is None or r is header:
            continue
        for j, pub in pubs.items():
            if j < len(r) and isinstance(r[j], (int, float)):
                out.append((pub, tgt, float(r[j])))
    return out


def rba_path_candidate(y: np.ndarray, cut: list[dt.date], dates: list[str],
                       archive: list[tuple[dt.date, dt.date, float]]) -> Candidate:
    """The RBA's latest published forecast path at the origin, by month: linear between its quarter-end targets, from the
    last known value to the first target, flat after the last. A vintage counts as known from the end of its month."""
    vintages: dict[dt.date, list[tuple[dt.date, float]]] = {}
    for pub, tgt, v in archive:
        vintages.setdefault(pub, []).append((tgt, v))
    pubs = sorted(vintages)
    labels = [_date(d) for d in dates]

    def month_index(d: dt.date) -> int:
        return d.year * 12 + d.month - 1

    def path(t: int, h: int) -> np.ndarray:
        day = cut[t - 1]
        known = [p for p in pubs if (dt.date(p.year + (p.month == 12), p.month % 12 + 1, 1) - dt.timedelta(days=1)) <= day]
        if not known:
            raise ValueError("no RBA forecast published by this origin")
        tg = sorted(vintages[known[-1]])
        base = month_index(labels[t - 1])
        xs = [base] + [month_index(d) for d, _ in tg if month_index(d) > base]
        vs = [y[t - 1]] + [v for d, v in tg if month_index(d) > base]
        if len(xs) < 2:
            raise ValueError("the latest RBA path ends before this origin")
        ahead = [base + k for k in range(1, h + 1)]
        return np.interp(ahead, xs, vs)
    return banded("rba_forecast", "structural", y, path, log=False,
                  note="the RBA's latest Statement on Monetary Policy forecast path known at the origin")


# ---------------------------------------------------------------- assembly

LEVEL_LOG_DOMAINS = ("precious_metal", "equity_index", "commodity", "currency")
MEAN_REVERTING = ("labour_market", "policy_rate")


def regime_labels(dates: list[str], *, domain: str, place: str | None,
                  fetch: Fetch | None) -> tuple[list[str | None] | None, str, list[str]]:
    """The risk regime known at every origin, from the index registered for the domain (opendata.risk_indices): labels for
    t = 0..n, the index label, and notes. None when the domain has no index, the series has no dates or the index failed."""
    from prophet_agent.forecast import regimes
    from prophet_agent.opendata import risk_indices
    key = risk_indices.domain_index(domain, place)
    if key is None or fetch is None or not dates:
        return None, "", []
    try:
        idx = risk_indices.fetch_index(key, fetch)
    except Exception as exc:          # the band then stays unconditioned, and the note says why
        return None, "", [f"the {risk_indices.INDICES[key][0]} index could not be read: {str(exc)[:120]}"]
    d = Dated(idx["rows"], idx["label"])
    return regimes.labels_for(cutoffs(dates), d.days, d.vals), idx["label"], []


def build(values: Any, dates: list[str], *, domain: str, place: str | None, target: str = "", engine: Any = None,
          fetch: Fetch | None = None, season: int | None = None,
          drivers: dict[str, Any] | None = None) -> tuple[list[Candidate], list[str], bool]:
    """The candidates for one target, the notes on any that could not be built, and whether scoring is relative."""
    y = np.asarray(values, dtype=float)
    log = domain in LEVEL_LOG_DOMAINS and bool((y > 0).all())
    notes: list[str] = []
    cands = rule_candidates(y, log=log, season=season, mean_reverting=domain in MEAN_REVERTING)
    if engine is not None:
        cands.append(timesfm_candidate(y, engine))
        usable = {}
        for k, v in (drivers or {}).items():
            arr = np.asarray(v, dtype=float) if isinstance(v, (list, tuple, np.ndarray)) else None
            if arr is not None and arr.shape == y.shape and np.isfinite(arr).all():
                usable[str(k)] = arr
            else:
                notes.append(f"driver {k} does not cover every value of the series, so it was left out")
        if usable:
            cands.append(driver_candidate(y, usable, engine))
    if fetch is None or not dates or len(dates) != len(y):
        if fetch is not None:
            notes.append("the series has no dates, so no outside candidate could be aligned to it")
        return cands, notes, log
    cut = cutoffs(dates)
    yrs = step_years(dates)
    try:
        if domain in ("precious_metal", "equity_index") and target in ("gold", "spx"):
            ivid, label = ("GVZCLS", "GVZ (gold options)") if target == "gold" else ("VIXCLS", "VIX (S&P 500 options)")
            iv = parse_fred(fetch(FRED_CSV.format(id=ivid)), ivid)
            bill = parse_fred(fetch(FRED_CSV.format(id="DTB3")), "DTB3")
            cands.append(carry_candidate(y, cut, yrs, iv, bill, label))
        elif domain == "policy_rate" and place == "Australia":
            tables = {**parse_rba(fetch(RBA_CSV.format(table="f1"))), **parse_rba(fetch(RBA_CSV.format(table="f2")))}
            for table in ("f1.1", "f2.1"):            # monthly tables reach back to 1990 for bills, 2013 for bonds
                try:
                    monthly = parse_rba(fetch(RBA_CSV.format(table=table)))
                except Exception as exc:
                    notes.append(f"RBA {table} could not be read: {str(exc)[:120]}")
                    continue
                for sid, _m in BAB + BONDS:
                    tables[sid] = merged(tables.get(sid), monthly.get(sid[:-1]), sid)
            if abs(yrs - 1 / 12) > 0.02:
                notes.append("the market path is monthly; the series is not, so it was left out")
            else:
                cands.extend(cash_rate_candidates(y, cut, tables))
        elif domain == "labour_market" and place == "Australia":
            if abs(yrs - 1 / 12) > 0.02:
                notes.append("the vacancy model and the RBA path are monthly; the series is not, so they were left out")
            else:
                jv_rows = []
                for r in csv.DictReader(io.StringIO(fetch(ABS_JV).decode("utf-8-sig", errors="replace"))):
                    m = re.match(r"(\d{4})-Q([1-4])$", r.get("TIME_PERIOD", ""))
                    if m and r.get("OBS_VALUE"):
                        y_, q_ = int(m.group(1)), int(m.group(2))
                        end = dt.date(y_ + (q_ == 4), (3 * q_) % 12 + 1, 1) - dt.timedelta(days=1)
                        jv_rows.append((end.isoformat(), float(r["OBS_VALUE"])))
                cands.append(vacancy_candidate(y, cut, Dated(jv_rows, "job vacancies")))
                try:
                    cands.append(rba_path_candidate(y, cut, dates, parse_smp_archive(fetch(RBA_SMP_ARCHIVE))))
                except Exception as exc:
                    notes.append(f"the RBA forecast path could not be read: {str(exc)[:160]}")
    except Exception as exc:
        notes.append(f"the {domain} market or structural candidates could not be built: {str(exc)[:200]}")
    return cands, notes, log
