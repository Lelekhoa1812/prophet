"""Event features on a series' own calendar: an indicator, leads, lags and counts for each event type.

Motivation vs Logic
Motivation: a dated event (earnings, a fixture, a rate decision) explains a spike only if the forecast can see where it falls
relative to each period. Writing that by hand in sandbox code is where the calendar goes wrong: on a business-day series a Friday
event leads the next period by one step, not by three calendar days.
Logic: periods are the series' dates plus the horizon's periods. The indicator is the share of a period inside an event, as
prophet_lib.event_covariate does, so a one-day event is 1.0 in a daily series. A lead (lag) k is 1 when an event of that type
falls in the next (previous) k periods, counted in the series' own periods. A count is the number of that type's events that
start in the period. A type with no event in the table is not built, and a feature that is constant over history plus horizon is
dropped, since a constant covariate tells the model nothing.
This file is copied beside prophet_lib into the sandbox and is imported by the server for the covariate check, so it depends on
pandas and numpy only.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

MAX_LEAD = 30  # periods; a lead or lag longer than this is refused by the caller


def period_starts(index: Any, horizon: int, freq: str | None = None) -> pd.DatetimeIndex:
    """Start dates of the history plus ``horizon`` further periods (the same rule as prophet_lib)."""
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if len(idx) < 3:
        raise ValueError("need at least 3 dates to know the period length")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = freq or pd.infer_freq(idx)
    if f is None:
        step = pd.Series(idx).diff().dropna().median()
        return idx.append(pd.DatetimeIndex([idx[-1] + step * (i + 1) for i in range(horizon)]))
    off = pd.tseries.frequencies.to_offset(f)
    return idx.append(pd.date_range(idx[-1] + off, periods=horizon, freq=off)) if horizon else idx


def _is_business(starts: pd.DatetimeIndex) -> bool:
    """True for a business-day calendar: every period starts on a weekday and the typical step is one business day."""
    if (starts.day_of_week >= 5).any() or len(starts) < 3:
        return False
    return float(pd.Series(starts).diff().dropna().median() / pd.Timedelta(days=1)) < 2


def _event_frame(events: pd.DataFrame) -> pd.DataFrame:
    """Rows with type, start and end as timestamps. An empty or missing end date means a one-day event."""
    for col in ("date", "type"):
        if col not in events.columns:
            raise ValueError(f"the event table needs a {col!r} column")
    start = pd.to_datetime(events["date"].astype(str).str.strip(), errors="coerce")
    if "end_date" in events.columns:
        raw_end = events["end_date"].astype(str).str.strip()
        end = pd.to_datetime(raw_end.where(raw_end != ""), errors="coerce")
    else:
        end = pd.Series(pd.NaT, index=events.index)
    end = end.fillna(start)
    out = pd.DataFrame({"type": events["type"].astype(str).str.strip(), "start": start, "end": end})
    return out.dropna(subset=["start"]).reset_index(drop=True)


def _window(hit: np.ndarray, k: int, ahead: bool) -> np.ndarray:
    """1 where an event period falls in the next k periods (ahead) or the previous k (not ahead); the period itself is excluded."""
    n = len(hit)
    csum = np.concatenate([[0], np.cumsum(hit.astype(int))])
    out = np.zeros(n)
    for i in range(n):
        lo, hi = (i + 1, min(n, i + k + 1)) if ahead else (max(0, i - k), i)
        out[i] = 1.0 if hi > lo and csum[hi] - csum[lo] > 0 else 0.0
    return out


def event_features(index: Any, events: pd.DataFrame, horizon: int, freq: str | None = None, *,
                   types: Iterable[str] | None = None, leads: Iterable[int] = (), lags: Iterable[int] = (),
                   counts: bool = False) -> dict[str, np.ndarray]:
    """Event features as long as the history plus ``horizon`` periods, keyed by name.

    For each event type T present (optionally only ``types``): ``ev_T`` is the share of each period inside a T event; with
    ``leads`` k, ``ev_T_leadK`` is 1 when a T event falls in the next K periods; with ``lags`` k, ``ev_T_lagK`` is 1 when one
    fell in the previous K; with ``counts``, ``ev_T_count`` is the number of T events starting in the period. Features that
    are constant over the whole span are dropped.

    Motivation vs Logic
    Motivation: the planner needs the same feature whether it is built in the server (to test it) or in the sandbox (to hand to
    the forecast), and a lead that counts calendar days instead of series periods misplaces every weekend.
    Logic: the event day sets the period it sits in, and the window counts series periods. On a business-day calendar the
    share counts business days, so a Friday event is 1.0 in its Friday and the weekend is not a period. An event on a day
    that is in no period (a weekend date on a business-day series, a date outside the span) adds nothing; it is not moved.
    Events outside the span are ignored, not clipped, so a future event (a scheduled fixture) enters through the horizon and
    a past one through history.
    """
    leads, lags = [int(k) for k in leads], [int(k) for k in lags]
    for k in leads + lags:
        if not 1 <= k <= MAX_LEAD:
            raise ValueError(f"leads and lags must be whole periods from 1 to {MAX_LEAD}, got {k}")
    starts = period_starts(index, horizon, freq)
    ends = starts[1:].append(pd.DatetimeIndex([starts[-1] + (starts[-1] - starts[-2])]))
    business = _is_business(starts)
    if business:
        # A business-day period spans its weekend, so its share counts business days: a Friday event is 1.0 in its Friday.
        day_lo, day_hi = starts.to_numpy().astype("datetime64[D]"), ends.to_numpy().astype("datetime64[D]")
        period_days = np.busday_count(day_lo, day_hi).astype(float)
        period_days = np.where(period_days > 0, period_days, 1.0)
    else:
        span = (ends - starts).total_seconds().to_numpy()
    ev = _event_frame(events)
    if types is not None:
        ev = ev[ev["type"].isin(list(types))]
    out: dict[str, np.ndarray] = {}
    for kind, grp in ev.groupby("type"):
        share = np.zeros(len(starts))
        for a, b in zip(grp["start"], grp["end"]):
            if business:
                lo = np.maximum(day_lo, a.to_datetime64().astype("datetime64[D]"))
                hi = np.minimum(day_hi, (b + pd.Timedelta(days=1)).to_datetime64().astype("datetime64[D]"))
                share += np.clip(np.busday_count(lo, np.maximum(hi, lo)), 0, None) / period_days
            else:
                hi = b + pd.Timedelta(days=1)
                overlap = np.minimum(ends.to_numpy(), hi.to_datetime64()) - np.maximum(starts.to_numpy(), a.to_datetime64())
                share += np.clip(overlap / np.timedelta64(1, "s"), 0, None) / span
        share = np.clip(share, 0, 1)
        name = f"ev_{kind}"
        out[name] = share
        hit = share > 0
        for k in leads:
            out[f"{name}_lead{k}"] = _window(hit, k, ahead=True)
        for k in lags:
            out[f"{name}_lag{k}"] = _window(hit, k, ahead=False)
        if counts:
            cnt = np.zeros(len(starts))
            pos = starts.searchsorted(grp["start"].to_numpy(), side="right") - 1   # the period holding each event's start
            ok = (pos >= 0) & (pos < len(starts))
            np.add.at(cnt, pos[ok], 1)
            out[f"{name}_count"] = cnt
    return {k: v for k, v in out.items() if np.ptp(v) > 0}
