"""Synthetic generator: reproducibility, ground-truth shape, censoring invariants and non-negativity. Offline only."""
import re

import numpy as np
import pytest

from prophet_agent.synth import generate

N_ENT, N_PER, HORIZON = 25, 200, 28


@pytest.fixture(scope="module")
def data():
    return generate(7, N_ENT, N_PER, HORIZON)


def test_same_seed_gives_identical_output():
    a = generate(7, N_ENT, N_PER, HORIZON)
    b = generate(7, N_ENT, N_PER, HORIZON)
    for name in ("series_ids", "observed", "censored", "latent_history", "available", "stockout",
                 "truth_future", "truth_mean_future", "item_of_series"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.events == b.events


def test_different_seed_gives_different_output():
    a = generate(7, N_ENT, N_PER, HORIZON)
    b = generate(8, N_ENT, N_PER, HORIZON)
    assert not np.array_equal(a.truth_future, b.truth_future)
    assert not np.array_equal(a.observed, b.observed)


@pytest.mark.parametrize("freq, n_periods, horizon", [("D", 200, 28), ("W", 60, 8)])
def test_truth_future_length_equals_horizon(freq, n_periods, horizon):
    d = generate(3, N_ENT, n_periods, horizon, freq=freq)
    assert d.truth_future.shape == (len(d.series_ids), horizon)
    assert d.truth_mean_future.shape == (len(d.series_ids), horizon)
    assert len(d.future_index) == horizon
    assert d.observed.shape == (len(d.series_ids), n_periods)
    assert len(d.history_index) == n_periods


def test_censoring_never_reports_sales_above_available_supply(data):
    for item in range(data.available.shape[0]):
        rows = data.item_of_series == item
        total_observed = data.observed[rows].sum(axis=0)
        assert np.all(total_observed <= data.available[item] + 1e-9), item


def test_censored_flag_marks_exactly_the_under_stated_periods(data):
    assert np.array_equal(data.censored, data.observed < data.latent_history)
    assert np.all(data.observed <= data.latent_history)
    # Under-statement only happens while the item is short of stock.
    rows, cols = np.nonzero(data.censored)
    assert np.all(data.stockout[data.item_of_series[rows], cols])
    assert data.censored.any(), "the default configuration should contain some censoring"


def test_no_negative_demand_or_supply(data):
    for arr in (data.latent_history, data.observed, data.truth_future, data.truth_mean_future, data.available):
        assert np.all(arr >= 0)
    assert np.all(data.truth_future == np.round(data.truth_future))


def test_series_ids_are_stable_unique_and_neutral(data):
    assert len(set(data.series_ids)) == len(data.series_ids)
    pattern = re.compile(r"^ENT-\d{4}\|ITM-\d{4}$")
    assert all(pattern.match(s) for s in data.series_ids)
    again = generate(7, N_ENT, N_PER, HORIZON)
    assert np.array_equal(data.series_ids, again.series_ids)


def test_weekly_history_is_seven_days_apart():
    d = generate(5, N_ENT, 60, 8, freq="W")
    assert np.all(np.diff(d.history_index.values) == np.timedelta64(7, "D"))
    assert np.all(np.diff(d.future_index.values) == np.timedelta64(7, "D"))


def test_long_table_has_one_row_per_series_and_period(data):
    long = data.to_long()
    assert len(long) == len(data.series_ids) * N_PER
    row = long.iloc[N_PER + 5]                      # second series, sixth period
    assert row.series_id == data.series_ids[1]
    assert row.observed == data.observed[1, 5]


def test_events_reference_known_ids_and_valid_windows(data):
    entity_ids = set(data.entities.entity_id)
    item_ids = set(data.items.item_id)
    event_ids = [ev["event_id"] for ev in data.events]
    assert len(set(event_ids)) == len(event_ids)
    for ev in data.events:
        assert ev["entity_id"] is None or ev["entity_id"] in entity_ids
        assert ev["item_id"] is None or ev["item_id"] in item_ids
        assert ev["start"] <= ev["end"]
        assert ev["scope"] in {"item", "entity", "series"}


def test_series_are_intermittent(data):
    assert (data.latent_history == 0).mean() > 0


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        generate(1, N_ENT, N_PER, HORIZON, freq="M")
    with pytest.raises(ValueError):
        generate(1, N_ENT, 10, 5)                   # span shorter than 180 days
    with pytest.raises(ValueError):
        generate(1, 0, N_PER, HORIZON)
