"""An answer must resume the mission it belongs to, whichever service process records it.

Before the fix, answer() committed the answer and then called start(). The run that paused on the question can still be
alive for a moment after it has released the lease (it is finishing its pause). start() saw that live thread and raised
LeaseHeld: the answer was stored, nothing resumed the mission, and it stayed waiting_user. A lease taken by another process
also reached the caller as an unmapped LeaseLost.
"""
from __future__ import annotations

import threading
import time

import pytest

from prophet_agent.api.service import MissionService
from prophet_agent.api.sharing import PolicyRefused
from prophet_agent.nim.errors import NotConfigured
from prophet_agent.runtime.store import LeaseHeld


def _no_planner(*args, **kwargs):
    raise NotConfigured("no planner for this test")


def _ask_once(ctx):
    answer = ctx.step("ask", lambda c: c.ask({"question": "Which model?", "options": ["a", "b"]}))
    return {"status": "met", "answer": answer}


def _service(db_path):
    svc = MissionService(db_path, _no_planner, {})
    svc._build_program = lambda mission: _ask_once
    return svc


def _held_after_pause(svc, paused, gate):
    """Hold the worker thread after its run has paused and released the lease, before it decides whether to exit."""
    original = svc._next_part

    def held(mission_id):
        paused.set()
        gate.wait(10)
        return original(mission_id)

    svc._next_part = held


def _new_mission(svc):
    return svc.store.create_mission("which model", {"max_tool_calls": 5, "max_seconds": 60})


def test_an_answer_that_lands_as_the_run_pauses_still_resumes_the_mission(tmp_path):
    svc = _service(tmp_path / "one.sqlite")
    paused, gate = threading.Event(), threading.Event()
    _held_after_pause(svc, paused, gate)
    mid = _new_mission(svc)
    svc.start(mid)
    assert paused.wait(10)
    assert svc.store.mission(mid)["status"] == "waiting_user"
    qid = svc.store.open_questions(mid)[0]["id"]
    svc.answer(qid, {"Model": "a"})                 # must not raise, and must not leave the answer unused
    gate.set()
    svc.wait(mid, 10)
    assert svc.store.mission(mid)["status"] == "completed"
    assert svc.store.open_questions(mid) == []


def test_an_answer_through_a_second_service_resumes_the_mission(tmp_path):
    db = tmp_path / "shared.sqlite"
    first = _service(db)
    paused, gate = threading.Event(), threading.Event()
    _held_after_pause(first, paused, gate)
    mid = _new_mission(first)
    first.start(mid)
    assert paused.wait(10)
    second = _service(db)
    qid = second.store.open_questions(mid)[0]["id"]
    second.answer(qid, {"Model": "a"})
    second.wait(mid, 10)
    assert second.store.mission(mid)["status"] == "completed"
    gate.set()
    first.wait(mid, 10)
    assert first.store.mission(mid)["status"] == "completed"
    assert [e for e in first.store.events(mid) if e["kind"] == "service.error"] == []


def test_an_answer_the_notebook_no_longer_allows_is_refused_before_it_is_recorded(tmp_path):
    """A notebook's provider rule can change while a turn waits for an answer. The rule is checked before the answer is
    recorded: a resume refused after the commit would leave the answered mission waiting, with nothing to resume it."""
    svc = _service(tmp_path / "policy.sqlite")
    paused, gate = threading.Event(), threading.Event()
    _held_after_pause(svc, paused, gate)
    nb = svc.notebooks.create("policy")
    mid = svc.store.create_mission("which model", {"max_tool_calls": 5, "max_seconds": 60}, notebook_id=nb["id"])
    svc.start(mid)
    assert paused.wait(10)
    gate.set()
    svc.wait(mid, 10)
    qid = svc.store.open_questions(mid)[0]["id"]
    svc.set_sharing(nb["id"], {"providers": []})
    with pytest.raises(PolicyRefused):
        svc.answer(qid, {"Model": "a"})
    assert [q["id"] for q in svc.store.open_questions(mid)] == [qid]
    svc.set_sharing(nb["id"], {"providers": None})
    svc.answer(qid, {"Model": "a"})
    svc.wait(mid, 10)
    assert svc.store.mission(mid)["status"] == "completed"


def test_an_answer_refused_by_a_lease_another_process_holds_is_a_conflict(tmp_path):
    """A service whose earlier lease on the mission lapsed still has that owner on record. Its answer must be refused
    as a conflict (409 at the API), not raised as LeaseLost, and the answer must not be recorded."""
    db = tmp_path / "stale.sqlite"
    first, second = _service(db), _service(db)
    mid = _new_mission(first)
    first.store.set_status(mid, "running")
    key = "question-tool-call"
    first.store.tool_start(mid, 0, "ask_user", {"question": "Which?"}, key)
    qid = first.store.open_question(mid, key, {"question": "Which model?"})
    second.store.acquire_lease(mid, "second-old", 0.0)
    time.sleep(0.01)
    first.store.acquire_lease(mid, "first-live", 300)
    with pytest.raises(LeaseHeld):
        second.answer(qid, {"Model": "a"})
    assert [q["id"] for q in first.store.open_questions(mid)] == [qid]
