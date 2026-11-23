"""Repeat round 6 (2026-11-23): analysts start with the lead's tool results, so an identical call is not run again.

Live Beijing (3c379255): four analysts each ran the lead's inspect_data again and then read the same pages, and the step
stalled. Every series here is synthetic.
"""
from __future__ import annotations

from prophet_agent.runtime import goal as G


def _tool(name, args, result, **extra):
    return {"kind": "tool", "tool": name, "args": args, "result": result, **extra}


def test_a_new_analyst_starts_with_the_lead_tool_results_marked_seeded():
    lead = [_tool("inspect_data", {"file": "stations.csv"}, {"rows": 100}),
            {"kind": "planner", "note": "not a tool result"}]
    seeded = G.seed_history(lead)
    assert len(seeded) == 1 and seeded[0]["seeded"] is True and seeded[0]["tool"] == "inspect_data"
    assert "seeded" not in lead[0]                              # the lead's own entries are not changed


def test_pointers_rejected_and_failed_calls_are_not_seeded():
    lead = [_tool("inspect_data", {"file": "a.csv"}, {"rows": 1}),
            _tool("inspect_data", {"file": "a.csv"}, {"rows": 1}, repeat_of=1),
            _tool("timesfm_forecast", {"horizon": 3}, {"point": [1]}, rejected=True),
            {"kind": "tool", "tool": "backtest_series", "args": {}, "error": "no file"}]
    assert [h["tool"] for h in G.seed_history(lead)] == ["inspect_data"]


def test_only_the_most_recent_results_are_seeded():
    lead = [_tool("inspect_data", {"file": f"f{i}.csv"}, {"i": i}) for i in range(G.SEED_RESULTS + 5)]
    seeded = G.seed_history(lead)
    assert len(seeded) == G.SEED_RESULTS
    assert seeded[-1]["args"] == {"file": f"f{G.SEED_RESULTS + 4}.csv"}


def test_an_identical_call_to_a_seeded_result_is_recognised_as_a_repeat():
    # The analyst's own history begins with the seeded lead result, so _repeat_of finds the earlier identical call.
    seeded = G.seed_history([_tool("inspect_data", {"file": "beijing.csv"}, {"rows": 738})])
    assert G._repeat_of("inspect_data", {"file": "beijing.csv"}, seeded) == 1
    assert G._repeat_of("inspect_data", {"file": "other.csv"}, seeded) is None


def test_a_seeded_pointer_points_at_a_result_the_analyst_can_read():
    seeded = G.seed_history([_tool("inspect_data", {"file": "beijing.csv"}, {"rows": 738})])
    obs = G._repeat_obs("inspect_data", {"file": "beijing.csv"}, seeded, 1)
    assert obs["repeat_of"] == 1 and "was not run again" in obs["note"]
