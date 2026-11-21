"""build_drivers and detect_anomalies (real sandbox runs; the series is written by an earlier analysis run, as in production)."""
from __future__ import annotations

import pytest

from prophet_agent.sandbox.runner import run_python
from prophet_agent.tools.drivers_tool import _build, _detect

SERIES_CODE = '''
import numpy as np, pandas as pd, prophet_lib as pl
idx = pd.date_range("2025-12-01", periods=60, freq="D")
rng = np.random.default_rng(1)
y = 100 + 10 * np.sin(np.arange(60) * 2 * np.pi / 7) + rng.normal(0, 2, 60)
y[{spike}] *= 8
pl.save_series(pd.Series(y, index=idx), "out/series.json")
'''


def _source(spike=()) -> dict:
    r = run_python(SERIES_CODE.format(spike=list(spike)))
    assert r.ok, r.stderr
    return {"run_id": r.run_id, "file": "out/series.json"}


def test_build_drivers_writes_covariates_of_history_plus_horizon_aligned_to_dates():
    src = _source()
    out = _build({"source": src, "horizon": 10, "events": [{"name": "promo", "start": "2026-01-05", "end": "2026-01-07"}],
                  "holidays": {"country": "AU", "subdiv": "VIC"}}, "k")
    assert out["history_points"] == 60 and out["horizon"] == 10
    assert set(out["covariate_names"]) == {"promo", "public_holiday"}
    assert out["covariates"]["promo"]["history_periods_active"] == 3
    assert out["covariates"]["public_holiday"]["history_periods_active"] >= 1          # 25 Dec and 1 Jan in the window
    # The written file is the forecast input: its covariates are as long as history plus horizon.
    import json
    from prophet_agent.sandbox.runner import SANDBOX_ROOT
    doc = json.loads((SANDBOX_ROOT / out["run_id"] / out["file"]).read_text())
    assert len(doc["covariates"]["promo"]) == 70 and len(doc["series"]) == 60 and len(doc["dates"]) == 60
    assert doc["covariates"]["promo"][35] == 1.0          # 2026-01-05 is index 35 of 2025-12-01


def test_build_drivers_refuses_bad_arguments_with_the_fix():
    src = _source()
    with pytest.raises(ValueError, match="ends before it starts"):
        _build({"source": src, "horizon": 3, "events": [{"name": "x", "start": "2026-02-02", "end": "2026-02-01"}]}, "k")
    with pytest.raises(ValueError, match="lowercase"):
        _build({"source": src, "horizon": 3, "events": [{"name": "Big Sale", "start": "2026-02-02"}]}, "k")
    with pytest.raises(ValueError, match="horizon"):
        _build({"source": src, "horizon": 0, "events": [{"name": "x", "start": "2026-02-02"}]}, "k")
    with pytest.raises(ValueError, match="give events"):
        _build({"source": src, "horizon": 3}, "k")
    with pytest.raises(ValueError, match="two-letter"):
        _build({"source": src, "horizon": 3, "holidays": {"country": "Australia"}}, "k")
    with pytest.raises(ValueError, match="unknown arguments"):
        _build({"source": src, "horizon": 3, "events": [{"name": "x", "start": "2026-02-02"}], "code": "x"}, "k")


def test_build_drivers_needs_saved_dates():
    r = run_python("import prophet_lib as pl\npl.save_series([1.0, 2.0, 3.0, 4.0, 5.0], 'out/series.json')")
    with pytest.raises(ValueError, match="no saved dates"):
        _build({"source": {"run_id": r.run_id, "file": "out/series.json"}, "horizon": 2,
                "events": [{"name": "x", "start": "2026-02-02"}]}, "k")


def test_detect_anomalies_reports_candidates_with_dates_and_ranges():
    src = _source(spike=(10, 40))
    out = _detect({"source": src}, "k")
    assert out["points"] == 60 and out["dated"] is True
    assert [10, 11] in out["ignore_ranges"] and [40, 41] in out["ignore_ranges"]
    by_index = {c["index"]: c for c in out["candidates"]}
    assert by_index[10]["date"] == "2025-12-11"
    assert by_index[40]["score"] > 6


def test_detect_anomalies_stays_quiet_on_a_clean_series():
    out = _detect({"source": _source(), "season": 7}, "k")
    assert out["ignore_ranges"] == []


def test_detect_anomalies_refuses_bad_threshold():
    with pytest.raises(ValueError, match="threshold"):
        _detect({"source": _source(), "threshold": 1}, "k")


def test_built_drivers_feed_the_forecast_control_by_name():
    from prophet_agent.forecast.control import build_control
    from prophet_agent.tools.series_io import source_covariates
    out = _build({"source": _source(), "horizon": 10, "events": [{"name": "promo", "start": "2026-01-05", "end": "2026-01-07"}]}, "k")
    args = {"source": {"run_id": out["run_id"], "file": out["file"]}, "future_covariates": ["promo"]}
    control = build_control(args, out["history_points"], out["horizon"], source_covariates(args))
    assert control.future["promo"].shape == (70,)
    assert control.future["promo"].sum() == 3
