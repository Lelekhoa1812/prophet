"""M9-4: event features on the series' own calendar (indicator, leads, lags, counts), with a business-day calendar where a
weekend is not a period. Offline; the table is written by hand."""
import numpy as np
import pandas as pd
import pytest

from prophet_agent.sandbox import prophet_events as pe


def table(*rows):
    """rows of (date, type) or (date, end_date, type)."""
    out = []
    for r in rows:
        if len(r) == 2:
            out.append({"date": r[0], "end_date": "", "type": r[1]})
        else:
            out.append({"date": r[0], "end_date": r[1] or "", "type": r[2]})
    return pd.DataFrame(out, columns=["date", "end_date", "type"])


# 2026-01-02 is a Friday. Ten business days: Fri 2, Mon 5 .. Fri 9, Mon 12 .. Thu 15.
BIZ = pd.bdate_range("2026-01-02", periods=10)


def test_a_friday_event_is_one_in_its_own_business_period_not_a_third_of_the_weekend():
    f = pe.event_features(BIZ, table(("2026-01-09", "earnings")), horizon=3)
    assert f["ev_earnings"][5] == 1.0          # Fri 9 Jan is the sixth business period (index 5)
    assert f["ev_earnings"][[4, 6]].sum() == 0.0
    assert len(f["ev_earnings"]) == 13          # history 10 plus horizon 3


def test_leads_and_lags_count_series_periods_so_a_friday_event_leads_monday_by_one():
    f = pe.event_features(BIZ, table(("2026-01-09", "earnings")), horizon=3, leads=[1, 2], lags=[1])
    assert f["ev_earnings_lead1"][4] == 1.0    # Thursday 8 Jan: the event is in the next period (Friday)
    assert f["ev_earnings_lead1"][5] == 0.0    # the event's own period is not its own lead
    assert f["ev_earnings_lead2"][[3, 4]].tolist() == [1.0, 1.0]
    assert f["ev_earnings_lead2"][2] == 0.0
    assert f["ev_earnings_lag1"][6] == 1.0    # Monday 12 Jan: the event was the previous period
    assert f["ev_earnings_lag1"][7] == 0.0


def test_a_lead_reaches_into_the_horizon_when_the_event_is_scheduled_there():
    f = pe.event_features(BIZ, table(("2026-01-14", "fixture")), horizon=3, leads=[3])
    # 14 Jan is index 8 (history is indices 0..9, so the event sits in history); its leads are the 3 periods before it.
    assert f["ev_fixture"][8] == 1.0
    assert f["ev_fixture_lead3"][5:8].tolist() == [1.0, 1.0, 1.0]
    assert f["ev_fixture_lead3"][8] == 0.0 and f["ev_fixture_lead3"][4] == 0.0
    # An event scheduled inside the horizon (Tue 20 Jan, index 12) is a lead for the periods before it.
    g = pe.event_features(BIZ, table(("2026-01-20", "fixture")), horizon=3, leads=[2])
    assert g["ev_fixture"][12] == 1.0 and g["ev_fixture_lead2"][10:12].tolist() == [1.0, 1.0]


def test_counts_give_the_number_of_events_that_start_in_each_period():
    f = pe.event_features(BIZ, table(("2026-01-05", "policy"), ("2026-01-05", "policy"), ("2026-01-07", "policy")),
                          horizon=1, counts=True)
    assert f["ev_policy_count"][1] == 2.0 and f["ev_policy_count"][3] == 1.0
    assert f["ev_policy_count"].sum() == 3.0


def test_a_multi_day_event_is_shared_across_its_business_periods():
    f = pe.event_features(BIZ, table(("2026-01-05", "2026-01-07", "holiday")), horizon=0)
    assert f["ev_holiday"][[1, 2, 3]].tolist() == [1.0, 1.0, 1.0]
    assert f["ev_holiday"][0] == 0.0 and f["ev_holiday"][4] == 0.0


def test_a_daily_calendar_uses_calendar_days():
    daily = pd.date_range("2026-01-01", periods=10, freq="D")
    f = pe.event_features(daily, table(("2026-01-04", "weather")), horizon=2)
    assert f["ev_weather"][3] == 1.0 and f["ev_weather"].sum() == 1.0


def test_events_outside_the_span_add_nothing_and_constant_features_are_dropped():
    f = pe.event_features(BIZ, table(("2025-06-01", "product"), ("2026-03-01", "product")), horizon=3)
    assert f == {}                                   # no feature: nothing falls in history plus horizon


def test_types_filter_and_separate_features_per_type():
    f = pe.event_features(BIZ, table(("2026-01-06", "earnings"), ("2026-01-07", "weather")), horizon=0,
                          types=["weather"])
    assert set(f) == {"ev_weather"}


def test_leads_and_lags_are_bounded_in_periods():
    with pytest.raises(ValueError, match="from 1 to 30"):
        pe.event_features(BIZ, table(("2026-01-06", "earnings")), horizon=0, leads=[31])
    with pytest.raises(ValueError, match="from 1 to 30"):
        pe.event_features(BIZ, table(("2026-01-06", "earnings")), horizon=0, lags=[0])


def test_the_table_needs_the_date_and_type_columns():
    with pytest.raises(ValueError, match="'type' column"):
        pe.event_features(BIZ, pd.DataFrame({"date": ["2026-01-06"]}), horizon=0)


def test_the_business_rule_is_only_for_weekday_series():
    assert pe._is_business(BIZ)
    assert not pe._is_business(pd.date_range("2026-01-01", periods=10, freq="D"))
    assert not pe._is_business(pd.date_range("2026-01-02", periods=10, freq="W-FRI"))
