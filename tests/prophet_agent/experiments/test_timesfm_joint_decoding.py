"""Offline tests for scripts/experiments/timesfm_joint_decoding.py: window cutting, the gap policy, MASE, the arm
bookkeeping and the paired summary. A stub forecaster stands in for TimesFM, so no model is loaded."""
from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "experiments" / "timesfm_joint_decoding.py"
_spec = importlib.util.spec_from_file_location("timesfm_joint_decoding", SCRIPT)
exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp)


class StubForecaster:
    """Stands in for TimesFMForecaster. Each row's point forecast is the row's first history value, so the test can see
    which entity each output row belongs to. Records the history length it was given for every call."""

    def __init__(self):
        self.history_lengths: list[int] = []
        self.joint_calls: list[int] = []
        self.symmetric_calls = 0

    @staticmethod
    def _out(ids, h):
        point = np.repeat(np.asarray(ids, dtype=float)[:, None], h, axis=1)
        q = point[..., None] + np.linspace(-1.0, 1.0, 9)[None, None, :]
        return point, q

    def per_entity(self, histories, horizon):
        self.history_lengths += [len(x) for x in histories]
        return self._out([x[0] for x in histories], horizon)

    def per_entity_symmetric(self, histories, horizon):
        self.symmetric_calls += 1
        return self._out([x[0] for x in histories], horizon)

    def joint(self, matrix, horizon):
        self.joint_calls.append(matrix.shape[0])
        return self._out(matrix[:, 0], horizon)


def _panel(n_entities=5, n_days=40):
    # Row i holds i*1000 + day, so the first value identifies the entity.
    return np.array([[i * 1000.0 + d for d in range(n_days)] for i in range(n_entities)])


def test_origins_are_non_overlapping_blocks_ending_on_the_last_day():
    starts = exp.origins_for(1461, 14, 6)
    assert len(starts) == 6
    assert starts[-1] + 14 == 1461
    assert all(b - a == 14 for a, b in zip(starts, starts[1:]))
    assert starts[0] >= exp.MIN_HISTORY


def test_origins_refuse_too_little_history():
    with pytest.raises(ValueError, match="history"):
        exp.origins_for(60, 14, 6)


def test_each_window_sees_only_history_strictly_before_its_origin():
    X = _panel(n_entities=3, n_days=60)
    stub = StubForecaster()
    result = exp.run_origin(stub, X, start=40, horizon=7, arms=("per_entity",))
    assert stub.history_lengths == [40, 40, 40]
    assert np.array_equal(result["actual"], X[:, 40:47])


def test_gap_policy_fills_interior_gaps_causally_and_drops_bad_entities(tmp_path):
    first = dt.date(2020, 1, 1)
    days = [(first + dt.timedelta(days=i)).isoformat() for i in range(40)]  # 40 contiguous days
    lines = ["entity,timestamp,value"]
    for name, skip in [("A", set()), ("B", {"2020-01-10"}), ("C", {"2020-01-01"}),
                       ("D", {"2020-01-05", "2020-01-06", "2020-01-07"})]:
        for i, day in enumerate(days):
            if day not in skip:
                lines.append(f"{name},{day},{float(i)}")
    path = tmp_path / "panel.csv"
    path.write_text("\n".join(lines) + "\n")

    entities, dates, X, log = exp.load_panel(path)

    assert entities == ["A", "B"]
    assert set(log["dropped"]) == {"C", "D"}
    assert X.shape == (2, 40)
    # B's missing 10 January takes the value of 9 January (the last observed day), never a later one.
    assert X[1, 9] == X[1, 8] == 8.0
    assert log["filled_days"] == {"A": 0, "B": 1}


def test_mase_scale_uses_season_seven_then_falls_back_to_lag_one():
    weekly = np.arange(28, dtype=float) + np.tile(np.arange(7, dtype=float), 4)  # seasonal diff = 7 each day
    assert exp.mase_scale(weekly) == pytest.approx(7.0)
    repeating = np.tile(np.arange(7, dtype=float), 4)  # seasonal diffs are all zero, lag-1 diffs are not
    assert exp.mase_scale(repeating) == pytest.approx(float(np.mean(np.abs(np.diff(repeating)))))
    assert np.isnan(exp.mase_scale(np.full(20, 3.0)))


def test_mase_per_entity_matches_the_definition():
    actual = np.array([[1.0, 3.0], [2.0, 2.0]])
    pred = np.array([[2.0, 1.0], [2.0, 4.0]])
    scale = np.array([2.0, 4.0])
    # entity 0: mean(|1-2|, |3-1|) = 1.5 over scale 2 = 0.75; entity 1: mean(0, 2) = 1 over scale 4 = 0.25
    assert exp.mase_per_entity(actual, pred, scale) == pytest.approx([0.75, 0.25])


def test_finalise_floors_only_non_negative_histories_and_sorts_quantiles():
    nonneg = np.array([0.0, 1.0, 2.0, 3.0] * 10)
    signed = np.array([-5.0, 4.0, -6.0, 7.0] * 10)
    point = np.array([[-1.0, 2.0], [-1.0, 2.0]])
    q = np.zeros((2, 2, 9))
    q[..., :] = np.array([3.0, 2.0, 1.0, 0.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    out_point, out_q = exp.finalise(point, q, [nonneg, signed])
    assert out_point[0, 0] == 0.0          # non-negative history: floored
    assert out_point[1, 0] == -1.0         # signed history: left alone
    assert np.all(np.diff(out_q[0], axis=-1) >= 0)
    assert out_q[0].min() == 0.0 and out_q[1].min() == -1.0


def test_every_arm_runs_and_joint_rows_come_back_in_panel_order():
    X = _panel(n_entities=5, n_days=40)
    stub = StubForecaster()
    result = exp.run_origin(stub, X, start=30, horizon=7)
    assert set(result["forecasts"]) == set(exp.ARMS)
    assert set(result["timings"]) == set(exp.ARMS)
    assert stub.joint_calls == [5]                       # one joint context by default
    assert np.array_equal(result["forecasts"]["joint"][0][:, 0], np.arange(5) * 1000.0)
    assert np.array_equal(result["forecasts"]["per_entity"][0][:, 0], np.arange(5) * 1000.0)
    assert stub.symmetric_calls == 1


def test_joint_group_size_splits_entities_in_order_and_reassembles_them():
    X = _panel(n_entities=5, n_days=40)
    stub = StubForecaster()
    result = exp.run_origin(stub, X, start=30, horizon=7, arms=("joint",), group_size=2)
    assert stub.joint_calls == [2, 2, 1]
    assert np.array_equal(result["forecasts"]["joint"][0][:, 0], np.arange(5) * 1000.0)


def test_scoring_reports_mase_and_band_hits_for_every_arm():
    X = _panel(n_entities=3, n_days=40)
    stub = StubForecaster()
    result = exp.run_origin(stub, X, start=30, horizon=7, arms=("per_entity", "naive"))
    scored = exp.score_origin(result)
    assert scored["per_entity"]["mase"].shape == (3,)
    assert scored["naive"]["inside"].shape == (3, 7)
    assert scored["per_entity"]["inside"].dtype == bool


def test_summary_pairs_entities_and_labels_every_number_as_measured():
    mase = {
        "per_entity": np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        "joint": np.array([[0.5, 0.5], [0.9, 0.9], [0.4, 0.4]]),
    }
    inside = {arm: np.ones((3, 2, 14), dtype=bool) for arm in mase}
    wall = {"per_entity": 1.0, "joint": 0.5}
    summary = exp.summarise("beijing", 14, ["a", "b"], ["e1", "e2", "e3"], mase, inside, wall,
                            paired_pairs=[("joint", "per_entity")], dropped={"x": "reason"})
    assert "measured" in summary["provenance"] and "2 origins" in summary["provenance"]
    assert "horizon 14" in summary["provenance"]
    paired = summary["paired"]["joint_minus_per_entity"]
    assert paired["mean_mase_difference"] == pytest.approx(-0.4, abs=1e-4)
    assert paired["entities_where_first_arm_lower"] == 3
    assert paired["sign_flip"]["method"] == "exact"
    assert summary["arms"]["joint"]["coverage_80_pooled"] == 1.0
    assert summary["entities_dropped_at_load"] == {"x": "reason"}
