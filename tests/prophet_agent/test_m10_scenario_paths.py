"""Mission 10, M10-6: a scenario can follow a combination candidate's path, so its shift is traced to a model output."""
import importlib
import json

import numpy as np
import pytest

from prophet_agent.forecast import scenarios as sc

ft = importlib.import_module("prophet_agent.tools.fundamentals_tool")


def test_shift_path_moves_the_scenario_by_step():
    scen = sc.validate([{"name": "market path", "probability": 0.4, "shift_path": [0.0, 0.1, 0.3], "evidence": "web_research S1"}], 3, None)
    assert scen[0]["kind"] == "path"
    knots = {0.1: [3.9, 3.8, 3.7], 0.5: [4.0, 4.0, 4.0], 0.9: [4.1, 4.2, 4.3]}
    out = sc.mix(knots, scen, seed=1)
    assert out["scenarios"][0]["median"] == pytest.approx([4.0, 4.1, 4.3])
    assert out["weighted_last_shift"] == pytest.approx(0.4 * 0.3)
    with pytest.raises(ValueError, match="exactly one"):
        sc.validate([{"name": "x", "probability": 0.2, "shift": 1, "shift_path": [0, 0, 0]}], 3, None)
    with pytest.raises(ValueError, match="3 numbers"):
        sc.validate([{"name": "x", "probability": 0.2, "shift_path": [0, 0]}], 3, None)


def test_path_of_reads_the_pack_members(tmp_path, monkeypatch):
    run = "a" * 32
    (tmp_path / run / "data").mkdir(parents=True)
    (tmp_path / run / "data" / "members.json").write_text(json.dumps({"market_path": [4.1, 4.2, 4.5]}))
    monkeypatch.setattr(ft, "SANDBOX_ROOT", tmp_path)
    d = {"point": [4.0, 4.0, 4.0]}
    out = ft._member_paths([{"name": "priced path", "probability": 0.3, "path_of": "market_path"},
                            {"name": "hike", "probability": 0.2, "shift": 0.25}], run, d)
    assert out[0]["shift_path"] == pytest.approx([0.1, 0.2, 0.5]) and "path_of" not in out[0]
    assert "market_path" in out[0]["why"] and run in out[0]["evidence"]
    assert out[1] == {"name": "hike", "probability": 0.2, "shift": 0.25}
    with pytest.raises(ValueError, match="not a candidate"):
        ft._member_paths([{"name": "x", "probability": 0.1, "path_of": "nope"}], run, d)
