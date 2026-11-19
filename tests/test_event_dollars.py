"""Dollar truth for customer and margin events, checked on hand-made order lines (no twin needed)."""
import pandas as pd
import pytest

from lab.sim import event_dollars as ed


def L(rows):
    df = pd.DataFrame(rows, columns=["delivery_date", "qty", "unit_price", "unit_cost"])
    df["delivery_date"] = pd.to_datetime(df.delivery_date)
    return df


START = pd.Timestamp("2026-06-01")


def test_keying_value_is_the_overstated_quantity_at_price():
    lines = L([("2026-06-01", 100.0, 4.0, 2.0)])
    assert ed.keying_dollars(lines, 10) == pytest.approx(100 * 0.9 * 4.0)
    assert ed.keying_dollars(lines, 1) == 0.0


def test_price_below_floor_values_only_the_discount_on_event_lines():
    pre = [(START - pd.Timedelta(days=d), 10.0, 10.0, 6.0) for d in (1, 10, 20)]
    ev = [("2026-06-03", 20.0, 7.0, 6.0), ("2026-06-04", 5.0, 10.0, 6.0)]      # second line is at the normal price
    lines = L(pre + ev)
    got = ed.price_below_floor_dollars(lines, START, pd.Timestamp("2026-06-10"))
    assert got == pytest.approx(20 * (10.0 - 7.0))


def test_price_below_floor_without_history_is_unscored_not_zero():
    lines = L([("2026-06-03", 20.0, 7.0, 6.0)])
    assert ed.price_below_floor_dollars(lines, START, pd.Timestamp("2026-06-10")) is None


def test_cost_rise_counts_only_the_rise_inside_the_horizon():
    pre = [(START - pd.Timedelta(days=d), 10.0, 12.0, 5.0) for d in (2, 9, 16)]
    after = [(START + pd.Timedelta(days=3), 10.0, 12.0, 6.0), (START + pd.Timedelta(days=200), 10.0, 12.0, 9.0)]
    lines = L(pre + after)
    got = ed.cost_rise_dollars(lines, START, pd.Timestamp("2027-01-01"))
    assert got == pytest.approx(10 * (6.0 - 5.0))            # the day-200 line is past the horizon


def test_customer_lost_is_run_rate_times_horizon():
    pre = [(START - pd.Timedelta(days=d), 2.0, 50.0, 0.0) for d in range(1, 57)]   # 100 a day for 56 days
    lines = L(pre)
    assert ed.customer_lost_dollars(lines, START) == pytest.approx(100.0 * ed.HORIZON_DAYS)


def test_customer_slowdown_is_the_lost_share_of_the_run_rate():
    pre = [(START - pd.Timedelta(days=d), 2.0, 50.0, 0.0) for d in range(1, 57)]
    lines = L(pre)
    end = START + pd.Timedelta(days=13)                       # 14 days
    assert ed.customer_slowdown_dollars(lines, START, end) == pytest.approx(100.0 * 0.65 * 14)


def test_customer_with_no_history_is_unscored():
    assert ed.customer_lost_dollars(L([]), START) is None
