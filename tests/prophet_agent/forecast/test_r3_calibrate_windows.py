"""A2-12 (audit round 3): coverage intervals use the held-out windows as the independent units, and the adaptive method is
not labelled as the DtACI mixture it does not implement."""
import inspect

import numpy as np

from prophet_agent.forecast import calibrate as cal
from prophet_agent.tools import calibrate_tool as ct


def _bands(hits_per_window):
    """Windows of 10 points whose coverage is the given shares; band [0, 1] holds an actual of 0.5, 2.0 is outside."""
    acts, qs = [], []
    for h in hits_per_window:
        a = np.where(np.arange(10) < round(h * 10), 0.5, 2.0)
        q = np.zeros((10, 9)); q[:, 8] = 1.0
        acts.append(a); qs.append(q)
    return np.stack(acts), np.stack(qs)


def test_the_interval_is_a_t_interval_over_window_coverages():
    a, q = _bands([0.6, 0.8, 1.0, 0.7, 0.9])
    lo, hi = cal._window_interval(a, q)
    per = np.array([0.6, 0.8, 1.0, 0.7, 0.9])
    from scipy import stats
    half = stats.t.ppf(0.95, 4) * per.std(ddof=1) / np.sqrt(5)
    assert lo == round(per.mean() - half, 3) and hi == round(min(1.0, per.mean() + half), 3)
    assert cal._window_interval(*_bands([0.8])) == [0.0, 1.0]


def test_calibrate_reports_the_window_interval_and_the_tool_shows_it():
    class Engine:
        def forecast(self, hist, h):
            from prophet_agent.forecast.types import Forecast
            last = float(hist[-1]); qs = np.array([last + z for z in np.linspace(-1.3, 1.3, 9)])
            return Forecast(point=np.full(h, last), q=np.tile(qs, (h, 1)))
    rng = np.random.default_rng(4)
    x = 100 + np.cumsum(rng.normal(0, 1.0, 300))
    out = cal.calibrate(x, 10, engine=Engine(), n_origins=12)
    lo, hi = out["heldout_raw_interval90"]
    assert lo <= out["heldout_coverage_raw"] <= hi
    lo, hi = out["heldout_calibrated_interval90"]
    assert lo <= out["heldout_coverage_calibrated"] <= hi
    assert "windows are the independent units" in out["interval_method"]


def test_the_adaptive_method_is_not_called_dtaci():
    doc = inspect.getdoc(cal.aci_scales)
    assert "expert selection" in doc
    assert "not its weighted mixture" in cal.__doc__
