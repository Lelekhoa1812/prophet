"""The report's centre option 'blend': the equal-weight blend of TimesFM's centre and the best simple rule is delivered only when
the tool's own backtest on the same windows says it beats both. The engines are fakes (see test_equal_blend.py for the series)."""
from __future__ import annotations

import numpy as np
import pytest

from prophet_agent.forecast import timesfm as timesfm_mod
from prophet_agent.forecast.types import Forecast
from prophet_agent.tools import report_tool as RT

LEVELS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]


class _Engine:
    checkpoint = "stub-checkpoint"
    global_context = 16

    def __init__(self, x, mode):
        self.x, self.mode = np.asarray(x, dtype=float), mode

    def forecast_batch(self, contexts, horizon):
        pts = []
        for ctx in contexts:
            y = self.x[len(ctx):len(ctx) + horizon]
            pts.append({"mirror": 2 * y - 100, "perfect": y.copy()}[self.mode])
        point = np.stack(pts)
        return Forecast(point, np.repeat(point[..., None], 9, axis=-1))


def _alternating(n=60):
    return [90.0 if i % 2 == 0 else 110.0 for i in range(n)]


def _fc(point):
    """A TimesFM forecast as _one returns it: the centre, and nine quantile lines around it."""
    point = [float(v) for v in point]
    return {"point": point, "quantiles": {lv: [p + (i - 4) for p in point] for i, lv in enumerate(LEVELS)},
            "config": {"history_points": 60}, "health": {"verdict": "ok"}}


@pytest.fixture()
def engine(monkeypatch):
    """Returns a function that makes TimesFMEngine.get() return a fake engine for the given series and mode."""
    def use(x, mode):
        eng = _Engine(x, mode)
        monkeypatch.setattr(timesfm_mod.TimesFMEngine, "get", classmethod(lambda cls: eng))
        return eng
    return use


def test_the_blend_is_delivered_when_its_own_backtest_says_it_wins(engine):
    x = _alternating()
    engine(x, "mirror")
    out = RT._blend_centre(_fc([80.0]), x, {"season": 1, "horizon": 1})
    assert out["point"] == [90.0]                          # (TimesFM 80 + rule 100) / 2, exact
    assert out["centre"]["rule"] == "blend" and out["centre"]["with"] == "mean_last_season"
    assert out["blend_check"]["adopt"] is True


def test_the_blend_is_not_delivered_when_timesfm_wins_and_the_centre_is_unchanged(engine):
    x = _alternating()
    engine(x, "perfect")
    base = _fc([80.0])
    out = RT._blend_centre(base, x, {"season": 1, "horizon": 1})
    assert out["point"] == [80.0] and "centre" not in out and out["quantiles"] == base["quantiles"]
    assert out["blend_check"]["adopt"] is False


def test_the_centre_text_names_the_blend_and_its_selection_caveat(engine):
    x = _alternating()
    engine(x, "mirror")
    fc = RT._blend_centre(_fc([80.0]), x, {"season": 1, "horizon": 1})
    note = RT._method_note({}, fc["centre"])
    assert "equal-weight blend" in note and "the average of the last season" in note
    assert "selected on these windows" in note.lower() and "not an unbiased" in note
    assert "more accurate" not in note.lower()


def test_the_blend_text_says_the_tool_checked_it_and_does_not_claim_a_caller_choice(engine):
    x = _alternating()
    engine(x, "mirror")
    note = RT._method_note({}, RT._blend_centre(_fc([80.0]), x, {"season": 1, "horizon": 1})["centre"])
    assert "set for this pack by the caller" not in note
    assert "this tool's backtest" in note


def test_the_other_centre_rules_keep_their_text_and_shape(engine):
    out = RT._centre(_fc([80.0]), [5.0, 8.0, 12.0], "naive")
    assert set(out["centre"]) == {"rule", "reason"} and out["centre"]["rule"] == "naive"
    assert "set for this pack by the caller" in RT._method_note({}, out["centre"])


def test_a_series_too_short_for_the_backtest_keeps_timesfm_and_says_why(engine):
    engine([1.0, 2.0, 3.0], "perfect")
    out = RT._blend_centre(_fc([80.0]), [1.0, 2.0, 3.0], {"horizon": 1})
    assert out["point"] == [80.0] and out["blend_check"]["adopt"] is False and out["blend_check"]["reason"]


def test_intermittent_demand_is_not_blended_because_the_centre_is_expected_demand(engine):
    fc = _fc([5.0])
    fc["median_point"] = [0.0]
    out = RT._blend_centre(fc, [0.0] * 20 + [4.0], {"horizon": 1})
    assert out["point"] == [5.0] and out["blend_check"]["adopt"] is False


def test_the_observation_carries_the_blend_check_for_the_planner(engine):
    x = _alternating()
    engine(x, "perfect")
    fc = RT._blend_centre(_fc([80.0]), x, {"season": 1, "horizon": 1})
    obs = RT.observe({"forecast": fc, "run_id": "r", "files": [], "manifest": "m"})
    assert obs["blend_check"]["adopt"] is False


def test_a_per_series_blend_on_a_panel_is_refused_with_the_whole_panel_form():
    # Changed in round 5 (live electricity rep10): _panel maps centre "blend" to global_blend before it gets here; a per-series
    # blend is still refused, and the message names the form that works
    with pytest.raises(ValueError, match="as one string"):
        RT._panel_centres("blend", ["A", "B"])
    with pytest.raises(ValueError, match="series on its own"):
        RT._panel_centres({"A": "blend"}, ["A", "B"])


def test_the_observation_keeps_the_delivered_centre_so_the_gate_can_read_it():
    # Live (WTI rep3, 2026-11-22): forecast_report was called with centre mean_last_season, yet baseline_compared still said the
    # centre was TimesFM's. observe() passed the forecast through _compact, which keeps only config, health and the lines, so the
    # gate never saw the centre the pack drew.
    from prophet_agent.runtime import contracts
    fc = RT._centre(_fc([80.0, 81.0]), [5.0, 8.0, 12.0], "naive")
    obs = RT.observe({"forecast": fc, "run_id": "r", "files": [], "manifest": "m"})
    assert obs["forecast"]["centre"]["rule"] == "naive"
    rec = contracts._forecast_record({"kind": "tool", "tool": "forecast_report", "result": obs, "args": {}})
    assert rec["centre"]["rule"] == "naive"
