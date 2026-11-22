"""Round 3 follow-up (A3-06, A4-08, A3-08): the automatic researcher brief carries the data's file and last date, and the
research trigger fires on pollution goals. Motivation: web_research now refuses to search a regional family without a
region and places dated events against data_end, but the harness brief named neither, so a Beijing PM2.5 mission was
never researched and an electricity researcher would get region_unknown on every call."""
from prophet_agent.runtime import goal as g
from prophet_agent.runtime.runner import Tool


def _inspect_full():
    return {"kind": "tool", "tool": "inspect_data", "args": {"file": "PRSA_Data_Beijing.csv"},
            "result": {"file": "PRSA_Data_Beijing.csv", "run_id": "ab" * 16,
                       "profile": {"rows": 10, "columns": [
                           {"name": "date", "kind": "time", "first": "2013-03-01 00:00:00", "last": "2017-02-28 23:00:00"},
                           {"name": "PM2.5", "kind": "number"}]}}}


def _inspect_compact():
    return {"kind": "tool", "tool": "inspect_data", "args": {},
            "result": {"file": "load.csv", "columns": ["ts: time, first=2011-01-01 00:00:00, last=2014-12-31 23:45:00; e.g. []",
                                                       "MT_001: number, min=0.0; e.g. []"]}}


def test_data_facts_read_the_full_profile_and_the_compact_view():
    assert g._data_facts([_inspect_full()]) == {"files": ["PRSA_Data_Beijing.csv"], "data_end": "2017-02-28"}
    assert g._data_facts([_inspect_compact()]) == {"files": ["load.csv"], "data_end": "2014-12-31"}
    assert g._data_facts([]) == {"files": [], "data_end": None}


def test_pollution_goals_trigger_research():
    for text in ("Forecast PM2.5 for the next 7 days", "forecast pm10 at each station", "air quality outlook next week",
                 "forecast pollution levels"):
        assert g.EXTERNAL_GOAL.search(text), text


def test_the_researcher_brief_names_the_file_last_date_and_region_rule():
    tools = {"web_research": Tool("web_research", lambda a, k: {}, idempotent=True),
             "sandbox_python": Tool("sandbox_python", lambda a, k: {}, idempotent=False)}
    plan = g._harness_plan("Forecast PM2.5 for the next 7 days", [_inspect_full()], tools)
    assert plan and plan["auto"] == "research"
    task = plan["task"]
    assert "PRSA_Data_Beijing.csv" in task and "data_end='2017-02-28'" in task
    assert "region" in task and "horizon_end" in task
