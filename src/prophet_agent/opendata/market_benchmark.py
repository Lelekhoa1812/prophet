"""Market benchmarks a forecast is compared with: the market-implied path of the RBA cash rate, and the options-implied range
of the S&P 500 (VIX) or gold (GVZ).

Motivation vs Logic
Motivation (round 20261128-063203 audits): every mission ignored the market's own forecast. AURATE drew a centre rising to
5.44% while its lanes had read that the market prices cuts; SPX delivered a 24-month band of about plus or minus 12% (56%
coverage in its backtest) beside a VIX of 26.6 in its own sources. A professional forecast starts from what the market
prices and says why it differs.
Logic:
- au_cash_rate: the RBA's F1 table (keyless CSV). F1 stopped publishing overnight index swap (OIS) rates in December 2022;
  its 1, 3 and 6-month bank bill rates are current. A bank bill rate is the expected average cash rate over its term plus a
  credit spread, so the spread of each tenor is measured as the median of bank bill less OIS over the dates both were
  published (2011 to 2022), and subtracted. The implied average cash rate over 1, 3 and 6 months gives forward points for
  months 0-1, 1-3 and 3-6. The result says it is an estimate with that method.
- vix / gvz: the newest close of the 30-day implied volatility index (Cboe's VIX history CSV; FRED's GVZCLS). A risk-neutral
  lognormal range around a given level at each horizon: level times exp(+-1.2816 sigma sqrt(T)), where sigma is the average
  volatility over the horizon as today's level reverts to the index's long-run median with a three-month half-life.
Every request goes through market._get (netguard, size and time caps). Parsing is pure and tested offline.
"""
from __future__ import annotations

import csv
import io
import math
import statistics
from datetime import date, datetime
from typing import Any, Callable

from prophet_agent.opendata import market

SOURCES = {
    "au_cash_rate": "https://www.rba.gov.au/statistics/tables/csv/f1-data.csv",
    "vix": "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "gvz": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GVZCLS",
}
WHAT = {"vix": "CBOE Volatility Index (30-day implied volatility of S&P 500 options)",
        "gvz": "CBOE Gold ETF Volatility Index (30-day implied volatility of GLD options)"}
Z90 = 1.2815515655446004
TENORS = (1, 3, 6)
BAB_IDS = {1: "FIRMMBAB30D", 3: "FIRMMBAB90D", 6: "FIRMMBAB180D"}
OIS_IDS = {1: "FIRMMOIS1D", 3: "FIRMMOIS3D", 6: "FIRMMOIS6D"}
TARGET_ID = "FIRMMCRTD"


def _float(text: str) -> float | None:
    try:
        v = float(str(text).strip())
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _iso(text: str) -> str | None:
    for fmt in ("%d-%b-%Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_f1(body: bytes) -> dict[str, list[tuple[str, float]]]:
    """F1's series by Series ID, each a list of (ISO date, value) in file order. The header rows are read by their first
    cell; a data row starts with a date."""
    rows = list(csv.reader(io.StringIO(body.decode("utf-8-sig", errors="replace"))))
    ids = next((r for r in rows if r and r[0].strip() == "Series ID"), None)
    if ids is None:
        raise ValueError("the F1 file has no Series ID row; the RBA may have changed its layout")
    out: dict[str, list[tuple[str, float]]] = {sid.strip(): [] for sid in ids[1:] if sid.strip()}
    for r in rows:
        day = _iso(r[0]) if r else None
        if not day:
            continue
        for k, sid in enumerate(ids[1:], start=1):
            if k < len(r) and sid.strip() and (v := _float(r[k])) is not None:
                out[sid.strip()].append((day, v))
    return out


def cash_rate_path(series: dict[str, list[tuple[str, float]]]) -> dict[str, Any]:
    """The cash rate target now and the market-implied path for the next six months, from F1's series."""
    target = series.get(TARGET_ID) or []
    if not target:
        raise ValueError("F1 holds no cash rate target values")
    bab = {m: dict(series.get(BAB_IDS[m]) or []) for m in TENORS}
    common = sorted(set.intersection(*(set(v) for v in bab.values()))) if all(bab.values()) else []
    if not common:
        raise ValueError("F1 holds no date with all three bank bill rates")
    as_of = common[-1]
    spreads: dict[int, dict] = {}
    for m in TENORS:
        ois = dict(series.get(OIS_IDS[m]) or [])
        diffs = [bab[m][d] - ois[d] for d in ois if d in bab[m]]
        if len(diffs) < 250:
            raise ValueError(f"too few dates with both the {m}-month bank bill and OIS rates to measure their spread")
        both = sorted(d for d in ois if d in bab[m])
        spreads[m] = {"median": round(statistics.median(diffs), 4), "n": len(diffs), "from": both[0], "to": both[-1],
                      "p10": round(statistics.quantiles(diffs, n=10)[0], 4), "p90": round(statistics.quantiles(diffs, n=10)[-1], 4)}
    avg = {m: round(bab[m][as_of] - spreads[m]["median"], 4) for m in TENORS}
    fwd = [{"months": "0-1", "rate": avg[1]},
           {"months": "1-3", "rate": round((3 * avg[3] - avg[1]) / 2, 4)},
           {"months": "3-6", "rate": round((6 * avg[6] - 3 * avg[3]) / 3, 4)}]
    target_day, target_now = [(d, v) for d, v in target if d <= as_of][-1]
    for p in fwd:
        p["vs_target_bp"] = round(100 * (p["rate"] - target_now))
    decisions = []
    for (d0, v0), (d1, v1) in zip(target, target[1:]):
        if abs(v1 - v0) > 1e-9:
            decisions.append({"date": d1, "from": v0, "to": v1})
    return {"target": {"date": target_day, "value": target_now}, "as_of": as_of,
            "bank_bills": {f"{m}m": bab[m][as_of] for m in TENORS}, "spread_to_ois": {f"{m}m": spreads[m] for m in TENORS},
            "implied_average": {f"{m}m": avg[m] for m in TENORS}, "implied_forwards": fwd,
            "recent_decisions": decisions[-6:],
            "method": ("bank bill rate less the median bank bill to OIS spread of the same tenor (the dates F1 published "
                       "both); the forward for months 1-3 is (3 x 3m - 1m) / 2 and for months 3-6 (6 x 6m - 3 x 3m) / 3. An "
                       "estimate: the spread moves, between its 10th and 90th percentiles in spread_to_ois.")}


def parse_close(body: bytes, kind: str) -> list[tuple[str, float]]:
    """(ISO date, close) rows of the Cboe VIX history CSV or a FRED graph CSV, oldest first."""
    reader = csv.reader(io.StringIO(body.decode("utf-8-sig", errors="replace")))
    head = next(reader, [])
    cols = [c.strip().upper() for c in head]
    col = cols.index("CLOSE") if "CLOSE" in cols else 1
    out = []
    for r in reader:
        if len(r) > col and (day := _iso(r[0])) and (v := _float(r[col])) is not None:
            out.append((day, v))
    if not out:
        raise ValueError(f"the {kind} file holds no closes")
    return sorted(out)


HALF_LIFE_MONTHS = 3.0


def term_vol(vol_now: float, vol_long: float, months: float, half_life: float = HALF_LIFE_MONTHS) -> float:
    """Average implied volatility over the next ``months`` when today's level reverts to the long-run level with the given
    half-life (the usual mean-reverting variance term structure, applied to the volatility)."""
    k = math.log(2) / half_life
    t = max(months, 1e-9)
    return vol_long + (vol_now - vol_long) * (1 - math.exp(-k * t)) / (k * t)


def implied_range(level: float, vol_pct: float, months: list[int], vol_long_pct: float | None = None) -> list[dict]:
    """The risk-neutral 10 to 90 range around ``level`` at each horizon in months, from an annualised volatility in per
    cent. With ``vol_long_pct`` the volatility reverts toward it (HALF_LIFE_MONTHS) instead of holding flat."""
    rows = []
    for m in months:
        vol = term_vol(vol_pct, vol_long_pct, m) if vol_long_pct is not None else vol_pct
        s = vol / 100.0 * math.sqrt(m / 12.0)
        rows.append({"months": m, "vol_pct": round(vol, 2), "q10": round(level * math.exp(-Z90 * s), 4),
                     "q90": round(level * math.exp(Z90 * s), 4), "width_log": round(2 * Z90 * s, 4)})
    return rows


def benchmark(kind: str, level: float | None = None, months: list[int] | None = None,
              get: Callable[[str], bytes] | None = None) -> dict[str, Any]:
    """One benchmark. ``get`` returns the body of a URL (tests pass a stub); the default goes through market._get."""
    if kind not in SOURCES:
        raise ValueError(f"kind must be one of {sorted(SOURCES)}")
    url = SOURCES[kind]
    fetch = get or (lambda u: market._get(u, timeout_s=market.DEFAULT_TIMEOUT_S, resolver=None)[0])
    try:
        body = fetch(url)
    except market.MarketError as exc:
        raise ValueError(f"{url} did not answer: {exc}") from None
    if kind == "au_cash_rate":
        out = cash_rate_path(parse_f1(body))
        return {"kind": kind, "label": "market-implied", "source_url": url, **out}
    rows = parse_close(body, kind)
    day, vol = rows[-1]
    year_ago = [v for d, v in rows if d >= (date.fromisoformat(day).replace(year=date.fromisoformat(day).year - 1)).isoformat()]
    long_run = round(statistics.median(v for _, v in rows), 2)
    out = {"kind": kind, "label": "options-implied (risk-neutral)", "source_url": url, "what": WHAT[kind],
           "as_of": day, "implied_vol_pct": vol, "long_run_median_pct": long_run, "history_from": rows[0][0],
           "past_year": {"min": min(year_ago), "median": round(statistics.median(year_ago), 2), "max": max(year_ago)},
           "note": (f"today's 30-day implied volatility reverts toward its long-run median ({long_run}) with a "
                    f"{HALF_LIFE_MONTHS:g}-month half-life over longer horizons; a risk-neutral range carries no risk "
                    "premium and no drift, so it checks the width of a forecast range, not its centre")}
    if level is not None:
        if not isinstance(level, (int, float)) or isinstance(level, bool) or level <= 0:
            raise ValueError("level must be the current level of the index or price, a positive number")
        out["level"] = float(level)
        out["implied_range"] = implied_range(float(level), vol, months or [1, 3, 6, 12, 24], long_run)
    return out
