"""censor_series refuses a non-integer latent demand instead of truncating it and flagging it as censored (GOAL.md, known defect).

Motivation vs Logic
Motivation: the observed sales were cast to int64, so a latent demand of 4.4 on an item with stock was recorded as 4, and
censored = observed < latent then flagged it as under-stated although nothing was short. The docstring already requires
whole-number demand; the code did not check it.
Logic: demand is counts, so a non-integer or non-finite latent is a caller error and is refused with the reason. Whole
numbers behave as before. Synthetic data only.
"""
import numpy as np
import pytest

from prophet_agent.synth.censor import StockRun, censor_series


def _unshort_run(demand):
    d = np.asarray(demand, dtype=float)
    return StockRun(demand=d, available=np.full_like(d, 100.0), sold=d, short=np.zeros(d.shape, dtype=bool))


def test_a_fractional_latent_demand_is_refused_not_truncated_and_flagged():
    # Nothing is short here, so the old code recorded 4.4 as 4 and flagged it as censored with no stock-out.
    with pytest.raises(ValueError, match="whole"):
        censor_series(np.array([[4.4]]), np.array([0]), _unshort_run([[4.4]]))


def test_a_non_finite_latent_demand_is_refused():
    with pytest.raises(ValueError, match="whole"):
        censor_series(np.array([[np.nan, 5.0]]), np.array([0]), _unshort_run([[5.0, 5.0]]))


def test_whole_number_latent_demand_is_still_accepted_and_capped():
    # Item 0 is short: 6 + 4 = 10 demand against 5 units, so the observed sales are the floor of each share.
    latent = np.array([[6.0], [4.0], [2.0]])
    run = StockRun(demand=np.array([[10.0], [2.0]]), available=np.array([[5.0], [10.0]]),
                   sold=np.array([[5.0], [2.0]]), short=np.array([[True], [False]]))
    observed, censored = censor_series(latent, np.array([0, 0, 1]), run)
    assert observed.tolist() == [[3], [2], [2]]
    assert censored.tolist() == [[True], [True], [False]]
