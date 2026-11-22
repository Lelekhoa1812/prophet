"""panel_backtest tool: argument checks with clear reasons, a compact observed view, and registration."""
import json

import numpy as np
import pytest

from prophet_agent.forecast.types import Forecast
from prophet_agent.tools import series_io
from prophet_agent.tools import panel_tool as pt

RUN = "a" * 32
Z = np.array([-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816])


class Engine:
    def forecast(self, series, horizon):
        last = float(series[-1])
        return Forecast(np.full(horizon, last, dtype=np.float32),
                        (last + Z[None, :] * np.ones((horizon, 1))).astype(np.float32))


def _panel(entities=8, n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    shape = 100 + 10 * np.sin(2 * np.pi * t / 7) + 0.05 * t
    return {f"sku{i}": [float(v) for v in (5 * (i + 1)) * (shape + rng.normal(0, 3, n))] for i in range(entities)}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A sandbox root under tmp_path holding one saved panel, so nothing is read from the real data folder."""
    monkeypatch.setattr(series_io, "SANDBOX_ROOT", tmp_path)
    out = tmp_path / RUN / "out"
    out.mkdir(parents=True)
    (out / "series.json").write_text(json.dumps({"series": _panel(), "dates": {}}))
    (out / "one.json").write_text(json.dumps({"series": {"only": [1.0, 2.0] * 30}, "dates": {}}))
    return tmp_path


def _args(**kw):
    base = {"source": {"run_id": RUN, "file": "out/series.json"}, "horizon": 6, "n_origins": 3, "season": 7}
    base.update(kw)
    return base


def _run(args):
    return pt._run(args, "k", lambda: Engine())


def test_a_valid_call_returns_the_table_the_reading_and_the_basis(sandbox):
    res = _run(_args())
    assert res["overall"]["timesfm"] >= 0 and "blend" in res["overall"]
    assert res["reading"].endswith(".") and "rolling origins" in res["reading"]
    assert res["basis"].startswith("measured")
    view = pt.observe(res)
    assert view["entities"] == 8 and len(view["per_entity"]) == 8
    assert {"overall", "wins", "best_overall", "coverage_80", "reading"} <= set(view)


def test_observe_keeps_at_most_thirty_entity_rows(sandbox):
    (sandbox / RUN / "out" / "series.json").write_text(json.dumps({"series": _panel(entities=31, n=60), "dates": {}}))
    view = pt.observe(_run(_args(horizon=2, n_origins=3)))
    assert len(view["per_entity"]) == 30 and view["entities"] == 31
    assert "first 30 of 31" in view["per_entity_note"]


def test_entities_narrow_the_panel_and_unknown_names_are_refused(sandbox):
    res = _run(_args(entities=["sku1", "sku3"]))
    assert sorted(res["per_entity"]) == ["sku1", "sku3"]
    with pytest.raises(ValueError, match="not in the file"):
        _run(_args(entities=["sku1", "nope"]))


@pytest.mark.parametrize("bad, message", [
    ({"values": [1.0, 2.0]}, "panel_backtest needs source"),
    ({"source": None}, "panel_backtest needs source"),
    ({"horizon": 0}, "horizon must be"),
    ({"horizon": True}, "horizon must be"),
    ({"horizon": 6, "n_origins": 2}, "n_origins must be"),
    ({"n_origins": 13}, "n_origins must be"),
    ({"season": 1}, "season must be"),
    ({"entities": []}, "entities must be"),
    ({"entities": ["sku1", "sku1"]}, "must not repeat"),
    ({"entities": [""]}, "entities must be"),
    ({"entities": ["sku%d" % i for i in range(201)]}, "entities must be"),
    ({"bogus": 1}, "unknown arguments"),
    ({"source": {"run_id": "not-hex", "file": "out/series.json"}}, "run_id"),
    ({"source": {"run_id": RUN, "file": "out/missing.json"}}, "no file"),
    ({"source": {"run_id": RUN, "file": "out/one.json"}}, "at least 2"),
])
def test_bad_arguments_are_refused_with_the_fix(sandbox, bad, message):
    args = {k: v for k, v in {**_args(), **bad}.items() if v is not None}   # a None value means "leave it out"
    with pytest.raises(ValueError, match=message):
        _run(args)


def test_the_tool_is_idempotent_and_registered_with_its_title_and_spec():
    from prophet_agent.tools import default_tools
    from prophet_agent.tools.panel_tool import PANEL_BACKTEST_SPEC, panel_backtest_tool
    tool = panel_backtest_tool(lambda: Engine())
    assert tool.name == "panel_backtest" and tool.idempotent is True and tool.observe is pt.observe
    assert PANEL_BACKTEST_SPEC["title"] == "Panel backtest"
    assert PANEL_BACKTEST_SPEC["activity"] == "Comparing a global model with TimesFM across entities"
    registered = default_tools()["panel_backtest"]
    assert registered.title == "Panel backtest" and "source" in registered.doc and "horizon" in registered.doc


def test_observe_says_how_to_see_the_entities_it_leaves_out_and_summarises_all_of_them(sandbox):
    # Live (electricity panel, 60 clients, 2026-11-22): the planner saw the first 30 rows, said the rest were clipped, and
    # re-ran the same call six times (the stall rescue then ended the step). The view must say that the rest are summarised
    # and how to see named ones, and it must give a per-model median over every entity, not only the rows shown.
    (sandbox / RUN / "out" / "series.json").write_text(json.dumps({"series": _panel(entities=40, n=60), "dates": {}}))
    view = pt.observe(_run(_args(horizon=2, n_origins=3)))
    assert "entities" in view["per_entity_note"] and "same result" in view["per_entity_note"]
    assert set(view["median_mase_all_entities"]) >= {"naive", "timesfm"}
    assert all(isinstance(v, float) for v in view["median_mase_all_entities"].values())


def test_observe_counts_the_entities_on_which_each_model_beats_naive_over_all_of_them(sandbox):
    # Live (electricity panel, 60 clients, 2026-11-23): the answer was asked for each client against naive, and only 30 rows
    # were visible, so the comparison was called missing. The view must count every entity, not only the rows it shows.
    (sandbox / RUN / "out" / "series.json").write_text(json.dumps({"series": _panel(entities=40, n=60), "dates": {}}))
    res = _run(_args(horizon=2, n_origins=3))
    view = pt.observe(res)
    out_of = sum(1 for m in res["per_entity"].values() if isinstance(m.get("naive"), float))
    assert view["beats_naive_out_of"] == 40 and out_of == 40
    for model, n in view["beats_naive_on"].items():
        assert model != "naive" and 0 <= n <= 40
        expected = sum(1 for m in res["per_entity"].values() if m[model] < m["naive"])
        assert n == expected
    assert "timesfm" in view["beats_naive_on"]
