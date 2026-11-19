"""Victorian calendar: public holidays and trading events that are known in advance.

Why: a forecast that does not know Christmas or Good Friday is wrong on those weeks, and those weeks decide
the buy. These dates are hand-entered for 2024 to 2026 and are approximate (observed-day rules are simplified).
Verify against the official Victorian calendar before any client use.

Logic: each day gets flags. The twin uses the flags to generate demand effects. The forecast covariate
(calendar_channel) is built from the same flags, so the lab result is an optimistic upper bound for the
calendar idea: real effect sizes are not known yet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Victorian public holidays (approximate, 2024 to 2026).
PUBLIC_HOLIDAYS = {
    "2024-11-05": "Melbourne Cup", "2024-12-25": "Christmas Day", "2024-12-26": "Boxing Day",
    "2025-01-01": "New Year's Day", "2025-01-27": "Australia Day (observed)", "2025-03-10": "Labour Day",
    "2025-04-18": "Good Friday", "2025-04-21": "Easter Monday", "2025-04-25": "Anzac Day",
    "2025-06-09": "King's Birthday", "2025-11-04": "Melbourne Cup", "2025-12-25": "Christmas Day",
    "2025-12-26": "Boxing Day", "2026-01-01": "New Year's Day", "2026-01-26": "Australia Day",
    "2026-03-09": "Labour Day", "2026-04-03": "Good Friday", "2026-04-06": "Easter Monday",
    "2026-04-25": "Anzac Day", "2026-06-08": "King's Birthday", "2026-11-03": "Melbourne Cup",
    "2026-12-25": "Christmas Day", "2026-12-28": "Boxing Day (observed)",
}
EASTER_SUNDAY = {2024: "2024-03-31", 2025: "2025-04-20", 2026: "2026-04-05", 2027: "2027-03-28"}

# Victorian school holidays (approximate windows).
SCHOOL_HOLIDAYS = [
    ("2024-12-21", "2025-01-28"), ("2025-04-05", "2025-04-21"), ("2025-07-05", "2025-07-20"),
    ("2025-09-20", "2025-10-05"), ("2025-12-20", "2026-01-27"), ("2026-04-04", "2026-04-19"),
    ("2026-07-04", "2026-07-19"), ("2026-09-19", "2026-10-04"),
]


def _dates(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(days).normalize()


def flags(days: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Daily 0/1 flags for each calendar effect, aligned to `days`."""
    d = _dates(days)
    out = {k: np.zeros(len(d)) for k in ["public", "xmas_run_up", "easter_run_up", "cup_day", "school"]}
    ph = {pd.Timestamp(k) for k, v in PUBLIC_HOLIDAYS.items()}
    out["public"][np.isin(d, list(ph))] = 1.0
    out["cup_day"][np.isin(d, [pd.Timestamp(k) for k, v in PUBLIC_HOLIDAYS.items() if v == "Melbourne Cup"])] = 1.0
    out["xmas_run_up"][(d.month == 12) & (d.day >= 15) & (d.day <= 23)] = 1.0
    for y, sun in EASTER_SUNDAY.items():
        s = pd.Timestamp(sun)
        out["easter_run_up"][(d >= s - pd.Timedelta(days=4)) & (d <= s - pd.Timedelta(days=2))] = 1.0
    for a, b in SCHOOL_HOLIDAYS:
        out["school"][(d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))] = 1.0
    return out


def calendar_channel(days: pd.DatetimeIndex, step: int = 1) -> np.ndarray:
    """One known-future covariate channel: share of each period that is a holiday, run-up or school-holiday day.

    The forecast wrapper takes one covariate channel per series, so the flags are summed into one value.
    `days` is the start of each period; `step` is its length in days (1 daily, 7 weekly).
    """
    day_idx = pd.date_range(days[0], periods=len(days) * step, freq="D")
    f = flags(day_idx)
    special = np.maximum.reduce([f["public"], f["xmas_run_up"], f["easter_run_up"], f["cup_day"]]) + f["school"]
    return special.reshape(len(days), step).mean(axis=1)
