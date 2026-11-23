"""The generator accepts fewer than eight items and keeps its invariants (GOAL.md, known defect).

Motivation vs Logic
Motivation: generate() drew max(8, n_items // 12) promoted items without replacement, so any n_items under 8 raised
"Cannot take a larger sample than population" instead of generating a small panel for a quick test.
Logic: the promotion count is capped at the number of items. For n_items of eight or more nothing changes. Synthetic data only.
"""
import numpy as np
import pytest

from prophet_agent.synth import generate


@pytest.mark.parametrize("n_items", [1, 3, 5, 7])
def test_a_panel_with_fewer_than_eight_items_is_generated(n_items):
    d = generate(2, 4, 200, 28, n_items=n_items)
    assert d.available.shape[0] == n_items
    assert d.observed.shape == (len(d.series_ids), 200)
    assert d.truth_future.shape == (len(d.series_ids), 28)


@pytest.mark.parametrize("n_items", [1, 5])
def test_a_small_panel_keeps_the_censoring_and_non_negativity_invariants(n_items):
    d = generate(2, 4, 200, 28, n_items=n_items)
    for item in range(n_items):
        rows = d.item_of_series == item
        assert np.all(d.observed[rows].sum(axis=0) <= d.available[item] + 1e-9), item
    assert np.all(d.observed <= d.latent_history)
    assert np.all(d.observed >= 0) and np.all(d.truth_future >= 0)


def test_a_panel_of_eight_items_still_generates():
    d = generate(2, 4, 200, 28, n_items=8)
    assert d.available.shape[0] == 8
