"""Round 6n, defect 4: a snapshot poll can ask for only the rows it has not seen.

The console polls a running mission while it runs. The snapshot used to load every event and every step, with each
step's full output, on each poll. A cursor (the last event seq and the last step index already seen) now returns only the
newer rows. Without a cursor the snapshot is the full one, as before, and the response keys do not change.
"""
from __future__ import annotations

import pytest

from prophet_agent.api import titles
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools

SUMMARY_KEYS = {"id", "goal", "status", "budget", "notebook_id", "seq", "datasets", "answer", "draft", "outcome",
                "progress", "events", "steps", "questions", "answered_questions"}


@pytest.fixture(autouse=True)
def no_title_model(monkeypatch):
    monkeypatch.setattr(titles, "name_in_background", lambda *a, **k: None)


def _service(tmp_path):
    return MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())


def _seed(service, mid, steps: range, events: range):
    for i in steps:
        service.store.finish_step(mid, i, f"tool:{i}", {"v": {"big": "x" * 500, "n": i}})
    for i in events:
        service.store.append_event(mid, "mission.status", {"n": i})


def test_a_cursor_returns_only_the_rows_after_it(tmp_path):
    service = _service(tmp_path)
    mid = service.store.create_mission("forecast demand", {})
    _seed(service, mid, range(3), range(4))
    first = service.snapshot(mid)
    seq_seen, step_seen = first["events"][-1]["seq"], first["steps"][-1]["idx"]
    assert [s["idx"] for s in first["steps"]] == [0, 1, 2]

    _seed(service, mid, range(3, 5), range(4, 6))
    second = service.snapshot(mid, after_seq=seq_seen, after_step=step_seen)
    assert [s["idx"] for s in second["steps"]] == [3, 4], "only the steps after the cursor"
    assert all(e["seq"] > seq_seen for e in second["events"]), "only the events after the cursor"
    assert [e["payload"]["n"] for e in second["events"] if e["kind"] == "mission.status"] == [4, 5]
    assert set(second) == set(first) == SUMMARY_KEYS, "the response keeps its keys"
    assert second["steps"][0]["output"] == {"big": "x" * 500, "n": 3}, "new steps keep their output"


def test_the_first_snapshot_without_a_cursor_is_the_full_one_and_does_not_change(tmp_path):
    service = _service(tmp_path)
    mid = service.store.create_mission("forecast demand", {})
    _seed(service, mid, range(2), range(3))
    first = service.snapshot(mid)
    assert [s["idx"] for s in first["steps"]] == [0, 1]
    assert [e["payload"]["n"] for e in first["events"] if e["kind"] == "mission.status"] == [0, 1, 2]

    _seed(service, mid, range(2, 3), range(3, 4))
    full = service.snapshot(mid)
    assert full["steps"][:2] == first["steps"], "earlier steps are the same rows"
    assert full["events"][:len(first["events"])] == first["events"], "earlier events are the same rows"
    assert [s["idx"] for s in full["steps"]] == [0, 1, 2]


def test_a_cursor_with_nothing_new_returns_empty_lists_and_the_same_summary(tmp_path):
    service = _service(tmp_path)
    mid = service.store.create_mission("forecast demand", {})
    _seed(service, mid, range(2), range(2))
    first = service.snapshot(mid)
    quiet = service.snapshot(mid, after_seq=first["events"][-1]["seq"], after_step=first["steps"][-1]["idx"])
    assert quiet["steps"] == [] and quiet["events"] == []
    assert quiet["status"] == first["status"] and quiet["goal"] == first["goal"]


def test_an_event_cursor_alone_still_returns_every_step(tmp_path):
    """Callers that send only the event cursor keep the old behaviour for steps."""
    service = _service(tmp_path)
    mid = service.store.create_mission("forecast demand", {})
    _seed(service, mid, range(3), range(2))
    assert [s["idx"] for s in service.snapshot(mid, after_seq=1)["steps"]] == [0, 1, 2]


def test_progress_and_outcome_do_not_depend_on_the_cursor(tmp_path):
    service = _service(tmp_path)
    mid = service.store.create_mission("forecast demand", {})
    _seed(service, mid, range(2), range(2))
    service.store.append_event(mid, "mission.progress", {"step": 2, "tool_calls": 2, "elapsed_s": 3.0, "checklist_done": 1,
                                                          "checklist_total": 4, "next": "fit the model"})
    service.store.set_status(mid, "checkpoint")
    service.store.append_event(mid, "mission.checkpoint", {"which": "steps"})
    full = service.snapshot(mid)
    cursored = service.snapshot(mid, after_seq=full["events"][-1]["seq"], after_step=full["steps"][-1]["idx"])
    assert full["progress"]["next"] == "fit the model"
    assert cursored["progress"] == full["progress"]
    assert full["outcome"]["reasons"] and full["outcome"]["reasons"][0]["code"] == "budget_spent"
    assert cursored["outcome"] == full["outcome"]
