"""Round 3, runtime goal 1 (A5-01): a ledger written before the lane column existed gets its lanes from the step names when it
is opened. A sub-agent's steps are named "act:d1.<agent>:<n>"; their calls are labelled delegated, every other call lead."""
from __future__ import annotations

from prophet_agent.runtime.store import Store


def test_an_old_ledger_gets_its_lanes_from_the_step_names(tmp_path):
    path = tmp_path / "old.sqlite"
    s = Store(path)
    mid = s.create_mission("g", {})
    with s._tx() as c:
        for idx, name in ((2, "act:d0:2"), (3, "act:d1.a1:3")):
            c.execute("INSERT INTO steps (mission_id, idx, name, status, output_json, attempts, updated) "
                      "VALUES (?,?,?,?,?,?,?)", (mid, idx, name, "done", None, 1, 0.0))
            c.execute("INSERT INTO tool_calls (idem_key, mission_id, step_idx, tool, args_json, status, updated) "
                      "VALUES (?,?,?,?,?,?,?)", (f"k{idx}", mid, idx, "a", "{}", "done", 0.0))
        c.execute("ALTER TABLE tool_calls DROP COLUMN lane")
        c.execute("PRAGMA user_version=0")
    s.close()

    reopened = Store(path)
    assert reopened.count_tool_calls(mid, lane="lead") == 1
    assert reopened.count_tool_calls(mid, lane="delegated") == 1
    reopened.close()
