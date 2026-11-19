"""Focused tests for the alert digest (business rule) and the Victorian calendar flags."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab.api.main import PER_OWNER, _apply_digest  # noqa: E402
from lab.calendar_vic import calendar_channel, flags  # noqa: E402


def _p(i, owner, agent, dollars, action="sell_down_offer"):
    return {"proposal_id": f"P{i}", "owner": owner, "agent": agent, "dollars_at_stake": dollars, "action": {"type": action}}


def test_each_owner_sees_top_n_and_rest_is_digested():
    props = [_p(i, "sales", "freshness", 1000 - i) for i in range(8)]
    shown, digest = _apply_digest(props, limit=50)
    assert len(shown) == PER_OWNER
    assert [p["proposal_id"] for p in shown] == [f"P{i}" for i in range(PER_OWNER)]
    assert digest == [{"owner": "sales", "agent": "freshness", "count": 3,
                       "dollars": round(sum(1000 - i for i in range(PER_OWNER, 8)), 2)}]


def test_order_entry_stops_are_never_digested():
    props = [_p(i, "order_desk", "customer", 500 - i, "hold_order_line") for i in range(9)]
    shown, digest = _apply_digest(props, limit=50)
    assert len(shown) == 9 and digest == []


def test_limit_moves_overflow_into_digest_without_losing_items():
    props = [_p(i, "buyer", "shortfall", 100 - i) for i in range(4)]
    shown, digest = _apply_digest(props, limit=2)
    assert len(shown) == 2
    assert sum(g["count"] for g in digest) == 2


def test_calendar_flags_cover_christmas_run_up_and_easter():
    f = flags(pd.date_range("2025-12-20", "2025-12-26", freq="D"))
    assert f["xmas_run_up"].tolist() == [1, 1, 1, 1, 0, 0, 0]  # run-up is 15 to 23 Dec
    assert f["public"][5] == 1 and f["public"][6] == 1  # 25 and 26 Dec
    e = flags(pd.date_range("2026-04-01", "2026-04-06", freq="D"))
    assert e["easter_run_up"].sum() == 3  # Wed, Thu, Fri before Easter Sunday 5 April


def test_weekly_calendar_channel_is_a_share_of_the_week():
    weeks = pd.DatetimeIndex(["2025-12-22"])  # Monday; 22-23 Dec run-up, 25-26 public, then school holidays from 20 Dec
    val = calendar_channel(weeks, 7)[0]
    assert 1.5 < val < 1.7  # 2 run-up + 2 public + 7 school days on top = 11 / 7
