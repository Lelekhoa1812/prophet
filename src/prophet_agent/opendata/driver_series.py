"""Named driver series for each forecast domain, fetched keyless from FRED and aligned to a target series without look-ahead.

Motivation vs Logic
Motivation (round 20261128-110827, owner decision 2026-11-28): AUUNEMP tried to finish 17 times and SPX 3 times; neither
scored an outside driver against its target, although the blocking checks named the step every time. The owner chose that
the harness runs the core driver test itself and the model interprets it. That needs data the harness can fetch without a
model choosing it: named series per domain and place.
Logic: CANDIDATES lists, per domain (driver_families.DOMAINS) and place, up to five FRED series ids with the driver family
each measures. Each was fetched on 2026-11-28 and holds data to within two months of that date (the OECD business and
consumer confidence series for Australia stop in 2023 and 2024, so they are left out). ``aligned`` turns a driver's dated
values into one value per target period: the mean of the driver's observations inside that period, then shifted one period
later, so the value at period t is what was known by the end of period t-1 (publication lags of a month or less cannot leak
the period being forecast). Periods before the driver starts are None; a gap inside the driver carries the last value.
"""
from __future__ import annotations

import bisect
from datetime import date
from typing import Any

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"


def _d(family: str, sid: str, label: str) -> dict[str, str]:
    return {"family": family, "id": sid, "label": label, "url": FRED_CSV.format(id=sid)}


_US_RATES = _d("monetary_policy", "FEDFUNDS", "US federal funds rate (monthly)")
_US_UNEMP = _d("global_spillover", "UNRATE", "US unemployment rate (monthly)")
_BRENT = _d("commodities_energy", "DCOILBRENTEU", "Brent crude oil price (daily)")
_USD = _d("us_dollar", "DTWEXBGS", "US dollar broad trade-weighted index (daily)")
_US10Y = _d("financial_conditions", "DGS10", "US 10-year Treasury yield (daily)")
_VIX = _d("volatility", "VIXCLS", "CBOE VIX, S&P 500 30-day implied volatility (daily)")

CANDIDATES: dict[str, dict[str, list[dict[str, str]]]] = {
    "labour_market": {
        "Australia": [_d("monetary_policy", "IRSTCI01AUM156N", "Australia interbank overnight rate (monthly, OECD)"),
                      _d("financial_conditions", "IRLTLT01AUM156N", "Australia 10-year government bond yield (monthly, OECD)"),
                      _US_UNEMP, _BRENT],
        "*": [_US_UNEMP, _US_RATES, _BRENT, {**_USD, "family": "financial_conditions"}],
    },
    "policy_rate": {
        "Australia": [_d("labour_market", "LRUNTTTTAUM156S", "Australia unemployment rate (monthly, OECD harmonised)"),
                      _d("financial_conditions", "IRLTLT01AUM156N", "Australia 10-year government bond yield (monthly, OECD)"),
                      {**_US_RATES, "family": "global_spillover"}, _BRENT],
        "*": [{**_US_RATES, "family": "global_spillover"}, _US10Y, _BRENT],
    },
    "precious_metal": {
        "*": [_d("real_rates", "DFII10", "US 10-year real yield, TIPS (daily)"), _USD,
              _d("inflation", "T10YIE", "US 10-year breakeven inflation (daily)"),
              {**_VIX, "family": "risk_appetite"}, {**_BRENT, "family": "inflation"}],
    },
    "equity_index": {
        "*": [{**_US10Y, "family": "monetary_policy"}, _d("financial_conditions", "T10Y2Y", "US 10-year less 2-year yield (daily)"),
              _VIX, {**_USD, "family": "flows_positioning"}, {**_BRENT, "family": "inflation"}],
    },
    "currency": {
        "*": [{**_USD, "family": "global_spillover"}, _US10Y, _BRENT, {**_VIX, "family": "sentiment"}],
    },
    "commodity": {
        "*": [{**_USD, "family": "financial_conditions"}, {**_US10Y, "family": "monetary_policy"},
              {**_BRENT, "family": "global_spillover"}, {**_VIX, "family": "sentiment"}],
    },
}
LAG_PERIODS = 1


def candidates(domain: str, place: str | None) -> list[dict[str, str]]:
    """The driver series for a domain and place (the place's own list, else the domain's general one), or []."""
    by_place = CANDIDATES.get(domain) or {}
    return [dict(c) for c in (by_place.get(place or "") or by_place.get("*") or [])]


def aligned(target_dates: list[str], rows: list[tuple[str, float]], lag: int = LAG_PERIODS) -> list[float | None]:
    """One value per target period: the mean of the driver inside [date_i, date_i+1) (the last period runs one median step),
    shifted ``lag`` periods later. None where the driver has no value yet."""
    days = [date.fromisoformat(str(d)[:10]) for d in target_dates]
    if len(days) < 2:
        raise ValueError("the target needs at least two dates to align a driver")
    steps = sorted((b - a).days for a, b in zip(days, days[1:]))
    step = steps[len(steps) // 2] or 1
    edges = [d.toordinal() for d in days] + [days[-1].toordinal() + step]
    pts = sorted((date.fromisoformat(str(d)[:10]).toordinal(), float(v)) for d, v in rows)
    xs = [p[0] for p in pts]
    means: list[float | None] = []
    for lo, hi in zip(edges, edges[1:]):
        a, b = bisect.bisect_left(xs, lo), bisect.bisect_left(xs, hi)
        vals = [pts[k][1] for k in range(a, b)]
        means.append(sum(vals) / len(vals) if vals else None)
    first = next((k for k, v in enumerate(means) if v is not None), None)
    if first is not None:                        # a gap inside the driver carries its last value, never one from later
        last = None
        for k in range(first, len(means)):
            if means[k] is None:
                means[k] = last
            last = means[k]
    return [None] * lag + means[:len(means) - lag] if lag else means


def change_corr(target: list[float], driver: list[float]) -> float | None:
    """Pearson correlation of period-on-period changes of the target and the (already lagged) driver, or None."""
    dt = [b - a for a, b in zip(target, target[1:])]
    dd = [b - a for a, b in zip(driver, driver[1:])]
    n = len(dt)
    if n < 8:
        return None
    mt, md = sum(dt) / n, sum(dd) / n
    st = sum((x - mt) ** 2 for x in dt) ** 0.5
    sd = sum((x - md) ** 2 for x in dd) ** 0.5
    if st == 0 or sd == 0:
        return None
    return round(sum((x - mt) * (y - md) for x, y in zip(dt, dd)) / (st * sd), 3)


def describe(c: dict[str, Any]) -> str:
    return f"{c['label']} [{c['id']}]"
