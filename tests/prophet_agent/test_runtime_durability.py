"""Crash-injection tests for the durable runtime (GOAL G6) and the pause/resume path used by questions (G8).

A ``Crash`` is a BaseException, so the runner's cleanup does not run, which is what a killed process looks like:
the lease and the mission status stay as they were. Each test starts a fresh Runtime on the same SQLite file, as a
restarted process would.
"""
from __future__ import annotations

import pytest

from prophet_agent.runtime.runner import Ctx, Runtime, Tool
from prophet_agent.runtime.store import LeaseHeld, Store, idem_key


class Crash(BaseException):
    """Simulates a process kill: not caught by any ``except Exception`` in the runtime."""


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "prophet.sqlite")
    yield s
    s.close()


def _mission(store, budget=None):
    return store.create_mission("forecast demand", budget or {})


class FakeExternal:
    """Stands in for a system with side effects. Idempotent tools rely on it deduplicating by key."""

    def __init__(self):
        self.writes: dict[str, dict] = {}
        self.calls = 0

    def create_order(self, args: dict, key: str) -> dict:
        self.calls += 1
        self.writes.setdefault(key, args)
        return {"order": key[:8], "qty": args["qty"]}

    def send_email(self, args: dict, key: str) -> dict:
        self.calls += 1
        return {"sent": True}


def test_completed_steps_are_not_rerun_after_crash(store):
    mid = _mission(store)
    runs = {"a": 0, "b": 0}

    def program(ctx: Ctx):
        def a(c):
            runs["a"] += 1
            return 1

        def b(c):
            runs["b"] += 1
            raise Crash()

        x = ctx.step("a", a)
        return ctx.step("b", lambda c: b(c)) + x

    with pytest.raises(Crash):
        Runtime(store, owner="r1").run(mid, program)

    # Restart: step a is replayed from the ledger, step b runs again because it never finished.
    def program_ok(ctx: Ctx):
        x = ctx.step("a", lambda c: runs.__setitem__("a", runs["a"] + 1) or 1)
        return ctx.step("b", lambda c: runs.__setitem__("b", runs["b"] + 1) or 1) + x

    # The lease from the crashed run is still held, so use a new store handle with the lease expired.
    store.conn.execute("UPDATE missions SET lease_until=0 WHERE id=?", (mid,))
    result = Runtime(store, owner="r2").run(mid, program_ok)
    assert result == {"status": "completed", "result": 2}
    assert runs == {"a": 1, "b": 2}, "step a must not run again; step b ran in the crashed run and again on resume"


def test_idempotent_tool_resumes_and_external_side_effect_happens_once(store):
    mid = _mission(store)
    ext = FakeExternal()
    crash_once = {"armed": True}

    def create(args, key):
        ext.create_order(args, key)
        if crash_once["armed"]:
            crash_once["armed"] = False
            raise Crash()          # killed after the write, before the result is recorded
        return {"ok": True}

    tool = Tool("create_order", create, idempotent=True)

    def program(ctx: Ctx):
        return ctx.step("order", lambda c: c.call(tool, {"qty": 3}))

    with pytest.raises(Crash):
        Runtime(store, owner="r1").run(mid, program)
    store.conn.execute("UPDATE missions SET lease_until=0 WHERE id=?", (mid,))

    result = Runtime(store, owner="r2").run(mid, program)
    assert result["status"] == "completed"
    assert ext.calls == 2, "the tool re-ran because its outcome was not recorded"
    assert len(ext.writes) == 1, "the external system saw one key, so the side effect happened once"


def test_non_idempotent_tool_with_unknown_outcome_is_not_rerun(store):
    mid = _mission(store)
    ext = FakeExternal()

    def send(args, key):
        ext.send_email(args, key)
        raise Crash()

    tool = Tool("send_email", send, idempotent=False)

    def program(ctx: Ctx):
        return ctx.step("notify", lambda c: c.call(tool, {"to": "team"}))

    with pytest.raises(Crash):
        Runtime(store, owner="r1").run(mid, program)
    store.conn.execute("UPDATE missions SET lease_until=0 WHERE id=?", (mid,))

    result = Runtime(store, owner="r2").run(mid, program)
    assert result["status"] == "outcome_unknown"
    assert ext.calls == 1, "a non-idempotent call must never be repeated automatically"
    assert store.mission(mid)["status"] == "outcome_unknown"


def test_recorded_tool_result_is_returned_without_reexecution(store):
    mid = _mission(store)
    count = {"n": 0}

    def fetch(args, key):
        count["n"] += 1
        return {"value": 42}

    tool = Tool("fetch", fetch)

    def program(ctx: Ctx):
        a = ctx.step("s", lambda c: c.call(tool, {"id": 1}))
        return ctx.step("t", lambda c: c.call(tool, {"id": 1}))["value"] + a["value"]

    assert Runtime(store, owner="r1").run(mid, program)["result"] == 84
    assert count["n"] == 2, "the same args in two different steps are two calls, by design"

    # Re-running a completed mission replays every step and every call from the ledger.
    assert Runtime(store, owner="r2").run(mid, program)["result"] == 84
    assert count["n"] == 2


def test_live_lease_blocks_a_second_runtime(store):
    mid = _mission(store)
    store.acquire_lease(mid, "first", ttl_s=3600)
    with pytest.raises(LeaseHeld):
        store.acquire_lease(mid, "second", ttl_s=3600)


def test_expired_lease_is_taken_over(store):
    mid = _mission(store)
    store.acquire_lease(mid, "first", ttl_s=-1)
    store.acquire_lease(mid, "second", ttl_s=60)   # no exception: the first lease had already expired


def test_tool_budget_stops_the_mission(store):
    mid = _mission(store, budget={"max_tool_calls": 2})
    tool = Tool("ping", lambda args, key: {"ok": True})

    def program(ctx: Ctx):
        out = []
        for i in range(5):
            out.append(ctx.step(f"p{i}", lambda c, i=i: c.call(tool, {"i": i})))
        return out

    result = Runtime(store, owner="r1").run(mid, program)
    assert result == {"status": "budget_exhausted", "which": "max_tool_calls"}
    assert store.mission(mid)["status"] == "budget_exhausted"


def test_tool_exception_is_recorded_and_mission_fails_cleanly(store):
    mid = _mission(store)

    def boom(args, key):
        raise ValueError("bad token nvapi-LEAKED987654321 in response")

    tool = Tool("boom", boom)
    result = Runtime(store, owner="r1").run(mid, lambda ctx: ctx.step("x", lambda c: c.call(tool, {})))
    assert result["status"] == "failed"
    assert "nvapi-LEAKED987654321" not in result["error"]
    rows = store.conn.execute("SELECT status, error FROM tool_calls WHERE mission_id=?", (mid,)).fetchall()
    assert rows[0]["status"] == "failed"
    assert "nvapi-LEAKED987654321" not in rows[0]["error"]


def test_ask_pauses_then_resumes_from_the_answer_without_asking_again(store):
    mid = _mission(store)
    payload = {"questions": [{"header": "Horizon", "question": "How far ahead?",
                              "options": [{"label": "4 weeks"}, {"label": "12 weeks"}], "multiSelect": False}]}

    def program(ctx: Ctx):
        answers = ctx.step("ask", lambda c: c.ask(payload))
        return {"horizon": answers["Horizon"]}

    first = Runtime(store, owner="r1").run(mid, program)
    assert first["status"] == "waiting_user"
    assert store.mission(mid)["status"] == "waiting_user"
    assert len(store.open_questions(mid)) == 1, "one open question, not one per run"

    # A second run before the answer must not open a duplicate question.
    again = Runtime(store, owner="r2").run(mid, program)
    assert again["status"] == "waiting_user"
    assert len(store.open_questions(mid)) == 1

    qid = store.open_questions(mid)[0]["id"]
    store.answer_question(qid, {"Horizon": "12 weeks"})

    done = Runtime(store, owner="r3").run(mid, program)
    assert done == {"status": "completed", "result": {"horizon": "12 weeks"}}


def test_idempotency_key_is_stable_across_argument_order():
    assert idem_key("m", 0, "t", {"a": 1, "b": 2}) == idem_key("m", 0, "t", {"b": 2, "a": 1})
    assert idem_key("m", 0, "t", {"a": 1}) != idem_key("m", 1, "t", {"a": 1})


def test_start_never_downgrades_a_finished_call(store):
    mid = _mission(store)
    key = idem_key(mid, 0, "t", {})
    store.tool_start(mid, 0, "t", {}, key)
    store.tool_set(key, "done", {"ok": 1})
    store.tool_start(mid, 0, "t", {}, key)
    rec = store.tool_get(key)
    assert rec.status == "done" and rec.result == {"ok": 1}
