"""Focused tests: weekday shape of daily demand and its use in the sell-down maths (business rules)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab.engine.state import ForecastService  # noqa: E402


def _stub(demand_row, weekly_total=70.0, start="2026-05-04", n_days=84):
    days = pd.date_range(start, periods=n_days, freq="D")
    w = SimpleNamespace(days=days, demand=np.array([demand_row], dtype=float))
    fc = SimpleNamespace(point=np.full((1, 4), weekly_total), q=np.stack([np.full((1, 4), v) for v in
                                                                         np.linspace(0.5, 1.5, 9)], axis=-1))
    svc = SimpleNamespace(w=w)
    svc.weekly = lambda as_of, horizon: {"fc": fc, "active": np.array([0]), "future_start": pd.Timestamp("2026-07-27")}
    svc.weekday_profile = lambda as_of, product, weeks=12, k=2.0: ForecastService.weekday_profile(svc, as_of, product, weeks, k)
    return svc, days


def test_weekday_profile_has_mean_one_and_follows_history():
    # Sales only Mon to Fri, nothing on weekends, for 12 weeks.
    row = [10, 10, 10, 10, 10, 0, 0] * 12
    svc, days = _stub(row, start="2026-05-04", n_days=len(row))
    shape = ForecastService.weekday_profile(svc, pd.Timestamp("2026-07-27"), 0)
    assert abs(shape.mean() - 1.0) < 1e-9
    assert shape[5] < 0.8 and shape[0] > 1.05  # Saturday clearly below Monday after shrinkage


def test_weekday_profile_is_flat_without_sales():
    svc, _ = _stub([0.0] * 84)
    assert np.allclose(ForecastService.weekday_profile(svc, pd.Timestamp("2026-07-27"), 0), 1.0)


def test_daily_path_sums_to_the_weekly_total_and_follows_weekdays():
    row = [10, 10, 10, 10, 10, 0, 0] * 12
    svc, _ = _stub(row, weekly_total=70.0)
    path = ForecastService.daily_path(svc, pd.Timestamp("2026-07-27"), 0, 7)  # a full Monday-to-Sunday week
    assert len(path["point"]) == 7
    assert abs(sum(path["point"]) - 70.0) < 1e-6
    assert path["point"][5] < path["point"][0]  # Saturday lower than Monday
    assert path["q50"][5] < path["q50"][0]


def test_daily_path_is_not_flat_when_history_is_weekly_shaped():
    row = [10, 10, 10, 10, 10, 0, 0] * 12
    svc, _ = _stub(row, weekly_total=70.0)
    path = ForecastService.daily_path(svc, pd.Timestamp("2026-07-27"), 0, 7)
    assert np.ptp(path["point"]) > 3  # Monday about 11, Saturday about 7
