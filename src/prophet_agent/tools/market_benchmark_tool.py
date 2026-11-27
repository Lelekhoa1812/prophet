"""market_benchmark: the market's own forecast to compare a forecast with (opendata/market_benchmark.py).

Motivation vs Logic
Motivation (round 20261128-063203 audits): AURATE drew a centre rising to 5.44% beside lane reports that the market prices
cuts, and SPX delivered a band of about plus or minus 12% at 24 months while its own sources quoted a VIX of 26.6. The
market's forecast is the first benchmark a professional reads.
Logic: one keyless file per kind, parsed into a short result: the cash rate path the bank bill curve implies (au_cash_rate),
or the options-implied range at each horizon around the given level (vix for the S&P 500, gvz for gold). The source URL is
in the result, so a driver_map benchmark can cite it and the finish checks can find it.
"""
from __future__ import annotations

from typing import Any

from prophet_agent.opendata import market_benchmark as mb
from prophet_agent.runtime.runner import ArgumentRejected, Tool

ALLOWED = {"kind", "level", "months"}


def _run(args: dict, _key: str) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise ArgumentRejected("the arguments must be an object")
    extra = sorted(set(args) - ALLOWED)
    if extra:
        raise ArgumentRejected(f"unknown arguments {extra}; allowed: {sorted(ALLOWED)}")
    kind = str(args.get("kind") or "").strip().lower()
    if kind not in mb.SOURCES:
        raise ArgumentRejected(f"kind must be one of {sorted(mb.SOURCES)}: au_cash_rate (RBA cash rate path implied by bank "
                               "bills), vix (S&P 500 options), gvz (gold options)")
    months = args.get("months")
    if months is not None and (not isinstance(months, list) or not months or len(months) > 12
                               or not all(isinstance(m, int) and not isinstance(m, bool) and 1 <= m <= 120 for m in months)):
        raise ArgumentRejected("months must be a list of up to 12 whole months from 1 to 120, for example [1, 3, 6, 12, 24]")
    level = args.get("level")
    if kind != "au_cash_rate" and level is None:
        raise ArgumentRejected("level is required for vix and gvz: the newest level of the index or price (for example the "
                               "last S&P 500 close in the downloaded series), so the implied range is in its units")
    return mb.benchmark(kind, level=level, months=months)


def market_benchmark_tool() -> Tool:
    return Tool("market_benchmark", _run, idempotent=True)


MARKET_BENCHMARK_SPEC = {
    "name": "market_benchmark",
    "title": "Market benchmark",
    "activity": "Reading what the market prices",
    "description": "The market's own forecast, to compare yours with and to cite as a benchmark. kind au_cash_rate: the RBA "
                   "cash rate target now, its recent decisions, and the path the bank bill curve implies for months 0-1, "
                   "1-3 and 3-6 (bank bill rates less their measured spread to OIS; an estimate). kind vix (S&P 500) or gvz "
                   "(gold) with level (the newest level of the index or price): today's 30-day implied volatility and the "
                   "options-implied 10 to 90 range at each horizon in months. A forecast range much narrower than the "
                   "implied range claims more certainty than the options market; the finish check compares them. Cite the "
                   "result's source_url in driver_map benchmarks.",
    "args": {
        "kind": "au_cash_rate, vix or gvz",
        "level": "vix and gvz: the newest level of the target (for example the last close of the downloaded series)",
        "months": "optional list of horizons in months for the implied range (default [1, 3, 6, 12, 24])",
    },
}
