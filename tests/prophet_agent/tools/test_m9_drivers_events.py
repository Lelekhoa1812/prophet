"""M9-4: build_drivers takes dated events (from a research call), builds event features on the series calendar, and keeps a
feature only when the rolling backtest shows it helps. Real sandbox runs; the series is synthetic with a known earnings effect
(+25 on event days), and every figure is synthetic."""
import csv
import io
import json

import numpy as np
import pandas as pd
import pytest

from prophet_agent.sandbox.runner import SANDBOX_ROOT, run_python
from prophet_agent.tools.drivers_tool import _build

HISTORY, HORIZON = 300, 10
EVENT_DAYS = [40, 103, 166, 229, 292, 305]     # 305 is inside the horizon: a scheduled event
NULL_DAYS = [7, 19, 58, 77, 120, 151, 188, 210, 233, 260, 279, 301, 306]


def _series_source() -> dict:
    rng = np.random.default_rng(100)
    event = np.zeros(HISTORY)
    event[[d for d in EVENT_DAYS if d < HISTORY]] = 1.0
    y = np.zeros(HISTORY)
    y[0] = 100.0
    eps = rng.normal(0, 3, HISTORY)
    for t in range(1, HISTORY):
        y[t] = 100 + 0.6 * (y[t - 1] - 100) + 25 * event[t] + eps[t]
    code = ("import pandas as pd, prophet_lib as pl\n"
            f"idx = pd.bdate_range('2024-01-01', periods={HISTORY})\n"
            f"pl.save_series(pd.Series({[float(v) for v in y]!r}, index=idx), 'out/series.json')\n")
    r = run_python(code)
    assert r.ok, r.stderr
    return {"run_id": r.run_id, "file": "out/series.json"}


def _day(i: int) -> str:
    return str(pd.bdate_range("2024-01-01", periods=HISTORY + HORIZON)[i].date())


def _earnings(i: int, url: str) -> dict:
    return {"date": _day(i), "event": "Company will report quarterly results this day", "source_url": url,
            "how_sure": "high", "date_found": "stated in text"}


def _other(i: int, url: str) -> dict:
    return {"date": _day(i), "event": "Statistics bureau publishes the monthly series", "source_url": url,
            "how_sure": "high", "date_found": "stated in text"}


@pytest.fixture(scope="module")
def source():
    return _series_source()


def _events() -> list[dict]:
    rows = [_earnings(d, f"https://ir.example/{d}") for d in EVENT_DAYS]
    rows.append(_earnings(EVENT_DAYS[0], "https://news.example/second-source"))       # the same event from a second source
    rows += [_other(d, f"https://stats.example/{d}") for d in NULL_DAYS]
    return rows


def test_dated_events_become_kept_features_with_verdicts_and_the_table_written(source):
    out = _build({"source": source, "horizon": HORIZON, "dated_events": _events()}, "k")
    names = set(out["covariate_names"])
    assert "ev_earnings" in names and "ev_other" not in names
    block = out["event_features"]
    assert block["kept"] == ["ev_earnings"] and block["checked"] is True
    assert block["verdicts"]["ev_earnings"]["kept"] is True
    assert block["verdicts"]["ev_earnings"]["improvement_vs_best_rule"] > 0
    assert block["verdicts"]["ev_other"]["kept"] is False
    assert block["rows"] == len(EVENT_DAYS) + len(NULL_DAYS)     # the duplicate earnings day merged into one row
    assert block["table_file"] == "out/event_table.csv"
    doc = json.loads((SANDBOX_ROOT / out["run_id"] / out["file"]).read_text())
    assert len(doc["covariates"]["ev_earnings"]) == HISTORY + HORIZON
    assert doc["covariates"]["ev_earnings"][305] == 1.0          # the scheduled event in the horizon is a known-ahead value
    table = list(csv.DictReader(io.StringIO((SANDBOX_ROOT / out["run_id"] / "out/event_table.csv").read_text())))
    first = [r for r in table if r["date"] == _day(EVENT_DAYS[0])][0]
    assert first["source_count"] == "2"
    assert set(first["source_urls"].split(" | ")) == {f"https://ir.example/{EVENT_DAYS[0]}", "https://news.example/second-source"}


def test_check_false_keeps_every_built_feature_and_says_so(source):
    out = _build({"source": source, "horizon": HORIZON, "dated_events": _events(),
                  "event_features": {"check": False}}, "k")
    assert {"ev_earnings", "ev_other"} <= set(out["covariate_names"])
    assert out["event_features"]["checked"] is False and out["event_features"]["verdicts"] is None


def test_leads_lags_and_counts_are_named_by_type_and_added(source):
    out = _build({"source": source, "horizon": HORIZON, "dated_events": _events(),
                  "event_features": {"types": ["earnings"], "leads": [5], "lags": [1], "counts": True, "check": False}}, "k")
    assert {"ev_earnings", "ev_earnings_lead5", "ev_earnings_lag1", "ev_earnings_count"} <= set(out["covariate_names"])
    assert "ev_other" not in out["covariate_names"]


def test_refused_rows_are_counted_with_their_reasons_and_the_rest_used(source):
    bad = {"date": "2026-02-30", "event": "Results", "source_url": "https://x.example/1"}
    out = _build({"source": source, "horizon": HORIZON, "dated_events": _events() + [bad]}, "k")
    assert out["event_features"]["refused"] == 1
    assert "ISO date" in out["event_features"]["refused_reasons"][0]["reason"]
    assert "ev_earnings" in out["covariate_names"]


def test_a_null_event_list_builds_no_driver_and_the_measured_figures_are_given(source):
    with pytest.raises(ValueError, match="no event feature beat the simple rules"):
        _build({"source": source, "horizon": HORIZON, "dated_events": [_other(d, f"https://s.example/{d}") for d in NULL_DAYS]},
               "k")


def test_event_features_need_dated_events_and_valid_options(source):
    with pytest.raises(ValueError, match="needs dated_events"):
        _build({"source": source, "horizon": HORIZON, "events": [{"name": "promo", "start": "2026-01-05"}],
                "event_features": {"check": False}}, "k")
    with pytest.raises(ValueError, match="at least medium"):
        _build({"source": source, "horizon": HORIZON,
                "dated_events": [{"date": _day(40), "event": "Results due", "source_url": "https://x.example/1",
                                  "date_found": "inferred"}]}, "k")
    with pytest.raises(ValueError, match="types must be"):
        _build({"source": source, "horizon": HORIZON, "dated_events": _events(),
                "event_features": {"types": ["sport"]}}, "k")
    with pytest.raises(ValueError, match="at most 10 whole periods|from 1 to 30"):
        _build({"source": source, "horizon": HORIZON, "dated_events": _events(),
                "event_features": {"leads": [31]}}, "k")
    with pytest.raises(ValueError, match="event_features must be an object"):
        _build({"source": source, "horizon": HORIZON, "dated_events": _events(),
                "event_features": {"window": 3}}, "k")
    with pytest.raises(ValueError, match="list of event objects"):
        _build({"source": source, "horizon": HORIZON, "dated_events": "earnings on the 3rd"}, "k")


def test_a_feature_name_that_clashes_with_a_window_is_refused_by_the_sandbox(source):
    with pytest.raises(ValueError, match="same name as another driver"):
        _build({"source": source, "horizon": HORIZON, "dated_events": _events(), "event_features": {"check": False},
                "events": [{"name": "ev_earnings", "start": _day(HISTORY + 2)}]}, "k")
