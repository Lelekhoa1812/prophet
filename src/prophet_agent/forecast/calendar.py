"""Trading calendars read from the data: which exchange's holidays a business-day series skips.

Motivation vs Logic
Motivation: a series observed on trading days only has gaps on weekends and on its market's holidays. Mapping forecast steps to
dates with a plain Monday-to-Friday calendar puts a step on a holiday (live: a WTI forecast labelled 26 Nov 2026, US Thanksgiving,
as trading day 7), so every later date is off by one and the "20 trading days" the person asked for ends a day early.
Logic: for each candidate exchange calendar, take its weekday holidays inside the history's span and measure the share that the
history does not contain. The calendar whose holidays are missing most often, with enough holidays to judge, is the series' own
calendar. Its future holidays are then skipped. When no calendar is clearly followed, plain weekdays are used and that is said.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

CANDIDATES = ("NYSE", "ECB", "LSE", "JPX", "TSX", "ASX", "HKEX", "SSE")
MIN_HOLIDAYS = 8          # weekday holidays inside the history needed before a calendar can be judged
MIN_ABSENT_SHARE = 0.8    # share of those holidays the history must skip for the calendar to count as followed


def _weekday_holidays(code: str, years: Iterable[int]) -> set[date]:
    import holidays
    try:
        cal = holidays.financial_holidays(code, years=list(years))
    except (NotImplementedError, KeyError, AttributeError):
        return set()
    return {d for d in cal if d.weekday() < 5}


def infer_calendar(observed: Iterable[date]) -> dict:
    """The exchange calendar a business-day history follows, measured. Returns {"calendar": code or None, "absent_share": float,
    "holidays_checked": int, "scores": {code: share}}. ``observed`` is the dates that carry a value."""
    days = sorted({d if isinstance(d, date) else d.date() for d in observed})
    if len(days) < 60:
        return {"calendar": None, "absent_share": 0.0, "holidays_checked": 0, "scores": {}, "reason": "fewer than 60 dates"}
    first, last, have = days[0], days[-1], set(days)
    years = range(first.year, last.year + 1)
    best, scores = None, {}
    for code in CANDIDATES:
        hols = [d for d in _weekday_holidays(code, years) if first <= d <= last]
        if len(hols) < MIN_HOLIDAYS:
            continue
        share = sum(d not in have for d in hols) / len(hols)
        scores[code] = round(share, 3)
        if share >= MIN_ABSENT_SHARE and (best is None or share > best[1] or (share == best[1] and len(hols) > best[2])):
            best = (code, share, len(hols))
    if best is None:
        return {"calendar": None, "absent_share": 0.0, "holidays_checked": 0, "scores": scores,
                "reason": "no exchange calendar's holidays are consistently missing; plain weekdays used"}
    return {"calendar": best[0], "absent_share": round(best[1], 3), "holidays_checked": best[2], "scores": scores}


def next_trading_days(last: date, n: int, calendar: str | None,
                      closed_weekdays: frozenset[int] = frozenset({5, 6})) -> list[date]:
    """The ``n`` days after ``last`` on which the series trades: not a closed weekday (Saturday and Sunday by default; Monday
    is 0, Sunday is 6) and not a holiday of ``calendar`` when one is given."""
    years = range(last.year, last.year + 3 + n // 200)
    skip = _weekday_holidays(calendar, years) if calendar else set()
    out, d = [], last
    while len(out) < n:
        d += timedelta(days=1)
        if d.weekday() not in closed_weekdays and d not in skip:
            out.append(d)
    return out
