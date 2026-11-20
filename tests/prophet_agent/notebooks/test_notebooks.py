"""Notebook layer tests: schema upgrade, sources and profiling, derived tables and lineage, turns, cancel, and routes.

The sandbox runs for real (it is a child process, no network); nothing calls NIM or TimesFM.
"""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.notebooks import Notebooks
from prophet_agent.runtime.store import NotebookBusy, SCHEMA_VERSION, Store, _DDL
from prophet_agent.sandbox.runner import run_python
from prophet_agent.tools import default_tools
from prophet_agent.tools.notebook_tools import NotebookBinding

CSV = b"date,units,region\n2026-01-05,10,North\n2026-01-12,12,North\n2026-01-19,9,South\n"


class NoBrain:
    def decide(self, messages, schema, name):
        raise AssertionError("no planner call expected in these tests")


@pytest.fixture()
def service(tmp_path):
    svc = MissionService(tmp_path / "p.sqlite", lambda: NoBrain(), default_tools(), tools_factory=default_tools)
    yield svc
    svc.store.close()


def _write_out(name: str, text: str) -> str:
    res = run_python(f"import pathlib; pathlib.Path('out/{name}').write_text({text!r}, encoding='utf-8')", timeout_s=30)
    assert res.ok, res.stderr
    return res.run_id


def test_v1_database_is_upgraded_in_place(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(_DDL)
    conn.execute("INSERT INTO missions (id, goal, status, budget_json, created, updated) VALUES "
                 "('m1', 'old goal', 'completed', '{}', 1, 1)")
    conn.execute("PRAGMA user_version=1")
    conn.commit()
    conn.close()

    store = Store(path)
    cols = {r["name"] for r in store.conn.execute("PRAGMA table_info(missions)")}
    assert {"notebook_id", "seq", "parent_turn_id", "cancel_requested"} <= cols
    assert store.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert store.mission("m1")["notebook_id"] is None, "an old mission stays a one-off mission"
    store.close()


def test_one_active_turn_per_notebook_and_sequence_numbers(tmp_path):
    store = Store(tmp_path / "t.sqlite")
    first = store.create_mission("q1", {}, notebook_id="nb1")
    with pytest.raises(NotebookBusy):
        store.create_mission("q2", {}, notebook_id="nb1")
    store.set_status(first, "completed")
    second = store.create_mission("q2", {}, notebook_id="nb1")
    assert (store.mission(first)["seq"], store.mission(second)["seq"]) == (1, 2)
    assert store.create_mission("other notebook", {}, notebook_id="nb2") is not None
    store.close()


def test_cancel_of_a_waiting_turn_closes_it_and_its_question(service):
    nb = service.notebooks.create("Sales")
    mid = service.create_turn(nb["id"], "which horizon?")
    service.store.set_status(mid, "waiting_user")
    assert service.cancel(mid) == "cancelled"
    assert service.store.mission(mid)["status"] == "cancelled"
    assert service.cancel(mid) == "cancelled", "cancelling again changes nothing and does not fail"


def test_cancel_of_a_finished_turn_is_a_no_op(service):
    mid = service.create("finished goal")
    service.store.set_status(mid, "completed")
    assert service.cancel(mid) == "completed"


def test_source_is_stored_once_by_hash_and_profiled(service):
    nb = service.notebooks.create("Sales")
    src = service.notebooks.add_source(nb["id"], "sales 2026.csv", CSV, profile_async=False)
    assert src["sha256"] and src["name"] == "sales 2026.csv"
    service.notebooks.profile_source(src["id"])
    done = service.notebooks.source(src["id"])
    assert done["status"] == "ready", done
    sheet = done["card"]["sheets"][0]
    assert sheet["rows"] == 3
    assert any(c.startswith("units:") for c in sheet["columns"])

    again = service.notebooks.add_source(nb["id"], "copy of sales.csv", CSV, profile_async=False)
    assert again["id"] == src["id"] and again["duplicate"] is True
    assert len(service.notebooks.sources(nb["id"])) == 1


def test_first_source_names_an_untitled_notebook_only(service):
    untitled = service.notebooks.create()
    service.notebooks.add_source(untitled["id"], "Store_sales 2026.csv", CSV, profile_async=False)
    assert service.notebooks.get(untitled["id"])["title"] == "Store sales 2026"
    service.notebooks.add_source(untitled["id"], "returns.csv", b"a,b\n1,2\n", profile_async=False)
    assert service.notebooks.get(untitled["id"])["title"] == "Store sales 2026"

    named = service.notebooks.create("Quarterly review")
    service.notebooks.add_source(named["id"], "other.csv", CSV, profile_async=False)
    assert service.notebooks.get(named["id"])["title"] == "Quarterly review"


def test_excluded_source_is_left_out_of_the_turn(service):
    nb = service.notebooks.create("Sales")
    src = service.notebooks.add_source(nb["id"], "a.csv", CSV, profile_async=False)
    service.notebooks.profile_source(src["id"])
    service.notebooks.set_source(nb["id"], src["id"], included=False)
    assert "a.csv" not in service.notebooks.context(nb["id"])
    service.notebooks.set_source(nb["id"], src["id"], included=True)
    assert "a.csv" in service.notebooks.context(nb["id"])


def test_save_table_makes_immutable_versions_with_lineage(service):
    nb = service.notebooks.create("Sales")
    run1 = _write_out("table.csv", "region,units\nNorth,22\nSouth,9\n")
    v1 = service.notebooks.promote(nb["id"], run1, "", "by_region", "units by region", turn_id="t1")
    assert v1["version"] == 1 and v1["rows"] == 2 and v1["columns"] == ["region", "units"]

    same = service.notebooks.promote(nb["id"], run1, "out/table.csv", "by_region")
    assert same["version"] == 1 and same["unchanged"] is True

    run2 = _write_out("table.csv", "region,units\nNorth,22\nSouth,9\nEast,4\n")
    v2 = service.notebooks.promote(nb["id"], run2, "out/table.csv", "by_region")
    assert v2["version"] == 2 and v2["rows"] == 3

    versions = service.notebooks.table_versions(nb["id"], "by_region")
    assert [v["version"] for v in versions] == [1, 2]
    assert versions[1]["lineage"]["parent_version"] == 1 and versions[1]["lineage"]["run_id"] == run2
    assert service.notebooks.table_path(nb["id"], "by_region", 1).read_text().count("\n") == 3, "v1 file is kept intact"
    assert service.notebooks.tables(nb["id"])[0]["version"] == 2


@pytest.mark.parametrize("name", ["Bad Name", "1st", "x" * 41, "../up"])
def test_save_table_refuses_bad_names(service, name):
    nb = service.notebooks.create("Sales")
    run = _write_out("table.csv", "a,b\n1,2\n")
    with pytest.raises(ValueError):
        service.notebooks.promote(nb["id"], run, "out/table.csv", name)


def test_save_table_refuses_files_outside_the_run_outputs(service):
    nb = service.notebooks.create("Sales")
    run = _write_out("table.csv", "a,b\n1,2\n")
    with pytest.raises(ValueError):
        service.notebooks.promote(nb["id"], run, "../program.py", "t")
    with pytest.raises(ValueError):
        service.notebooks.promote(nb["id"], "not-a-run", "out/table.csv", "t")


def test_notebook_turn_gets_its_files_tables_and_save_tool(service):
    nb = service.notebooks.create("Sales")
    src = service.notebooks.add_source(nb["id"], "sales.csv", CSV, profile_async=False)
    service.notebooks.profile_source(src["id"])
    run = _write_out("table.csv", "region,units\nNorth,22\n")
    service.notebooks.promote(nb["id"], run, "out/table.csv", "by_region", "units by region")

    mid = service.create_turn(nb["id"], "how many units in total?")
    mission = service.store.mission(mid)
    assert mission["seq"] == 1 and mission["notebook_id"] == nb["id"]
    tools = service.tools_for(mission)
    assert "save_table" in tools, "a notebook turn may keep tables"
    assert "inspect_data" in tools and "sandbox_python" in tools

    ctx = service.files_context(mission)
    assert "sales.csv" in ctx and "by_region v1" in ctx

    solo = service.create("a one-off question about the forecast")
    assert "save_table" not in service.tools_for(service.store.mission(solo))


def test_save_table_tool_needs_a_notebook(service):
    from prophet_agent.tools.notebook_tools import save_table_tool
    with pytest.raises(ValueError, match="inside a notebook"):
        save_table_tool(None).fn({"run_id": "x", "file": "out/a.csv", "name": "a"}, "k")


class GatedBrain:
    """Holds the first planner decision until the test releases it, so a turn stays active for the busy check."""

    def __init__(self):
        import threading
        self.gate = threading.Event()

    def decide(self, messages, schema, name):
        if name == "plan":
            self.gate.wait(10)
            return {"action": "finish", "rationale": "done"}
        return {"verdict": "met", "reason": "ok", "answer": "ok"}


def test_notebook_routes(tmp_path, monkeypatch):
    monkeypatch.setattr("prophet_agent.nim.transport.load_keys", lambda: ["nvapi-test-placeholder"])
    brain = GatedBrain()
    service = MissionService(tmp_path / "r.sqlite", lambda: brain, default_tools(), tools_factory=default_tools)
    client = TestClient(create_app(service))
    nb = client.post("/api/notebooks", json={"title": "Store sales"}).json()
    assert client.get(f"/api/notebooks/{nb['id']}").status_code == 200
    assert client.get("/api/notebooks/" + "0" * 32).status_code == 404

    up = client.post(f"/api/notebooks/{nb['id']}/sources?name=sales.csv", content=CSV)
    assert up.status_code == 201, up.text
    service.notebooks.wait_profiled()
    detail = client.get(f"/api/notebooks/{nb['id']}").json()
    assert detail["sources"][0]["status"] == "ready"

    sid = detail["sources"][0]["id"]
    assert client.patch(f"/api/notebooks/{nb['id']}/sources/{sid}", json={"included": False}).json()["included"] is False

    empty = client.post(f"/api/notebooks/{nb['id']}/sources?name=empty.csv", content=b"")
    assert empty.status_code == 422

    turn = client.post(f"/api/notebooks/{nb['id']}/turns", json={"goal": "total units by region"})
    assert turn.status_code == 201, turn.text
    busy = client.post(f"/api/notebooks/{nb['id']}/turns", json={"goal": "and by month?"})
    assert busy.status_code == 409

    cancelled = client.post(f"/api/missions/{turn.json()['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelling", "the turn has a live runner, so it stops at its next step"
    brain.gate.set()
    service.wait(turn.json()["id"], 15)
    assert service.store.mission(turn.json()["id"])["status"] in ("cancelled", "completed")
    assert client.post("/api/missions/" + "f" * 32 + "/cancel").status_code == 404
    service.store.close()
