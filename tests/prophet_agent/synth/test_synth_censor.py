"""Stock-out censoring: the planner's stock run and the capped, flagged observed sales. Offline only."""
import numpy as np
import pytest

from prophet_agent.synth.censor import StockRun, censor_series, simulate_stock


def test_simulated_sales_are_the_smaller_of_demand_and_units_on_hand():
    rng = np.random.default_rng(0)
    demand = rng.poisson(5, size=(4, 60)).astype(float)
    demand[:, 30:34] *= 4                                   # a spike that outruns stock
    run = simulate_stock(demand, lead=np.array([1, 2, 3, 1]))
    assert np.all(run.sold >= 0)
    assert np.all(run.sold <= run.available + 1e-9)
    assert np.allclose(run.sold, np.minimum(demand, run.available))
    assert np.array_equal(run.short, demand > run.available)
    assert run.short.any(), "the spike should cause at least one shortage"


def test_lead_times_must_be_whole_periods_of_at_least_one():
    with pytest.raises(ValueError):
        simulate_stock(np.ones((2, 10)), lead=np.array([0, 1]))


def test_censoring_caps_shared_stock_and_flags_only_the_short_series():
    # Item 0 is shared by series 0 and 1: demand 6 + 4 = 10 against 5 units on hand.
    # Item 1 is series 2 alone, fully stocked.
    latent = np.array([[6.0], [4.0], [2.0]])
    run = StockRun(demand=np.array([[10.0], [2.0]]),
                   available=np.array([[5.0], [10.0]]),
                   sold=np.array([[5.0], [2.0]]),
                   short=np.array([[True], [False]]))
    observed, censored = censor_series(latent, np.array([0, 0, 1]), run)
    assert observed.tolist() == [[3], [2], [2]]             # floor(6*0.5), floor(4*0.5), unchanged
    assert observed[:2].sum() <= run.available[0, 0]
    assert censored.tolist() == [[True], [True], [False]]
    assert np.all(observed <= latent)


def test_censoring_rejects_latent_demand_that_does_not_match_the_run():
    run = StockRun(demand=np.array([[10.0]]), available=np.array([[5.0]]),
                   sold=np.array([[5.0]]), short=np.array([[True]]))
    with pytest.raises(ValueError):
        censor_series(np.array([[6.0]]), np.array([0]), run)
