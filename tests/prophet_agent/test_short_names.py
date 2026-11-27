"""Short names (owner, 2026-11-27): each sub-agent, lane and code run carries a name of 1 to 4 words saying its purpose. The
planner's name is kept (cleaned); without one, a name is derived from the brief or the code. Offline: a scripted brain."""
from __future__ import annotations

import pytest

from prophet_agent.runtime import names


def test_a_given_name_is_cut_to_four_words_and_loses_trailing_filler():
    assert names.clean("RBA policy path and market pricing") == "RBA policy path"
    assert names.clean("  gold   price drivers ") == "gold price drivers"
    assert names.clean("") == "" and names.clean(None) == ""
    assert len(names.clean("x" * 80)) <= names.MAX_CHARS


def test_a_task_name_is_the_planners_else_its_first_clause():
    assert names.for_task("anything", "club injury news") == "Club injury news"
    assert names.for_task("Find the RBA decisions since 2024: dates and moves") == "RBA decisions since 2024"
    assert names.for_task("", fallback="Lane 3") == "Lane 3"
    for task in ("Research the driving series for the goal, then report", "angle two: market pricing"):
        assert 1 <= len(names.for_task(task).split()) <= 4


def test_a_code_name_is_the_planners_else_its_comment_else_its_main_call():
    assert names.for_code("print(1)", "Backtest seasonal baselines") == "Backtest seasonal baselines"
    assert names.for_code("# Fit a seasonal baseline to the monthly series\nx = 1") == "Fit seasonal baseline monthly"
    assert names.for_code("import pandas as pd\ndf = pd.read_csv('in/x.csv')") == "Load the data"
    assert names.for_code("x = 1") == "Python analysis"
    for code in ("#!/usr/bin/env python\n# -*- coding: utf-8 -*-\nbacktest(df)",):
        assert names.for_code(code) == "Backtest models"


def test_sandbox_python_accepts_a_name_and_still_refuses_other_arguments():
    from prophet_agent.tools.sandbox_tool import SANDBOX_PYTHON_SPEC, make_run
    run = make_run({})
    assert run({"code": "print(2)", "name": "Print two"}, "n1")["ok"]
    with pytest.raises(ValueError, match="allowed: code, carry, name"):
        run({"code": "print(1)", "label": "x"}, "n2")
    assert "name" in SANDBOX_PYTHON_SPEC["args"]


def test_the_plan_schema_takes_a_name_on_tasks_and_lanes():
    from prophet_agent.runtime.goal import PLAN_SCHEMA, _specs
    props = PLAN_SCHEMA["properties"]
    assert "name" in props and "name" in props["tasks"]["items"]["properties"]
    assert "name" in props["lanes"]["items"]["properties"]
    assert _specs({"action": "delegate", "task": "t", "name": "Gold drivers"})[0]["name"] == "Gold drivers"


def test_the_planner_prompt_asks_for_names():
    from prophet_agent.prompts import current
    text = current("planner").text
    assert "26c. Names" in text and "sandbox_python call a `name` of 1 to 4 words" in text


def test_the_panel_shows_each_agents_name_and_each_code_runs_name(tmp_path):
    from fastapi.testclient import TestClient

    from prophet_agent.api import activity
    from prophet_agent.api.app import create_app
    from prophet_agent.api.service import MissionService
    from prophet_agent.tools import default_tools
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("_panel", Path(__file__).parent / "api" / "test_activity_panel.py")
    panel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel)
    LaneBrain = panel.LaneBrain

    class Named(LaneBrain):
        def __init__(self):
            super().__init__()
            self.lead[0]["lanes"][0]["name"] = "RBA decision history"

    service = MissionService(tmp_path / "m.sqlite", lambda: Named(), default_tools())
    client = TestClient(create_app(service), base_url="http://127.0.0.1:8000")
    mid = client.post("/api/missions", json={"goal": "What will the RBA cash rate be?"}).json()["id"]
    service.wait(mid, timeout=20)
    agents = {a["agent_id"]: a for a in client.get(f"/api/missions/{mid}/activity").json()["agents"]}
    assert agents["a1"]["name"] == "RBA decision history"
    assert agents["a2"]["name"] == "Angle two", "a lane without a name is named from its brief"
    assert client.get(f"/api/missions/{mid}/agents/a1").json()["name"] == "RBA decision history"
    assert activity._code_title({"code": "x = 1", "name": "Scan the panel"}, "sandbox_python") == "Scan the panel"
    assert activity._code_title({"code": "# load the series\nx = 1"}, "sandbox_python") == "Load series"
    assert activity._code_title({}, "forecast_report") == "Forecast report"
