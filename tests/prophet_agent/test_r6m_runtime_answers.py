"""Round 6m (2026-11-23): an answer is never lost or re-asked, whatever the timing of the pause.

Audit (runtime runner.py and store.py, questions): (1) an answer that arrived after the question was opened but before
the run recorded its pause left the mission in waiting_user with nothing open to answer, so nothing resumed it. (2) An
answer that landed between the first check of the call and the write that marks it waiting was overwritten, and the
question was asked again. Both are closed by deciding the pause and the question in the store, in one transaction.
All missions here are synthetic.
"""
import pytest

from prophet_agent.runtime.runner import Ctx, Runtime
from prophet_agent.runtime.store import Store

PAYLOAD = {"questions": [{"header": "Horizon", "question": "How far ahead?",
                          "options": [{"label": "4 weeks"}, {"label": "12 weeks"}], "multiSelect": False}]}


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "prophet.sqlite")
    yield s
    s.close()


def _program(ctx: Ctx):
    answers = ctx.step("ask", lambda c: c.ask(PAYLOAD))
    return {"horizon": answers["Horizon"]}


def _questions(store, mid):
    return store.conn.execute("SELECT id, status FROM questions WHERE mission_id=? ORDER BY created", (mid,)).fetchall()


def test_an_answer_given_while_the_run_is_pausing_is_used_and_the_mission_is_not_stuck(store):
    mid = store.create_mission("forecast", {})
    original_open = store.open_question
    answered = {}

    def open_then_answer(mission_id, tool_key, payload):
        qid = original_open(mission_id, tool_key, payload)
        if not answered:
            answered["qid"] = qid
            store.answer_question(qid, {"Horizon": "12 weeks"})      # the person answers before the pause is recorded
        return qid

    store.open_question = open_then_answer
    result = Runtime(store, owner="r1").run(mid, _program)
    assert answered, "the answer landed while the question was being opened"
    assert result == {"status": "completed", "result": {"horizon": "12 weeks"}}
    assert store.mission(mid)["status"] == "completed"
    assert store.open_questions(mid) == []


def test_an_answer_given_while_the_ask_is_checking_is_used_and_not_asked_again(store):
    mid = store.create_mission("forecast", {})
    assert Runtime(store, owner="r1").run(mid, _program)["status"] == "waiting_user"
    qid = store.open_questions(mid)[0]["id"]
    store.answer_question(qid, {"Horizon": "12 weeks"})

    real_get = store.tool_get
    seen = {"n": 0}

    def first_read_is_stale(key):
        # The first check of the call sees it before the answer, as if the answer landed a moment later.
        seen["n"] += 1
        return None if seen["n"] == 1 else real_get(key)

    store.tool_get = first_read_is_stale
    result = Runtime(store, owner="r2").run(mid, _program)
    assert result == {"status": "completed", "result": {"horizon": "12 weeks"}}
    assert len(_questions(store, mid)) == 1, "the answered question is not replaced by a new one"


def test_a_second_answer_is_refused_and_the_first_is_kept(store):
    mid = store.create_mission("forecast", {})
    Runtime(store, owner="r1").run(mid, _program)
    qid = store.open_questions(mid)[0]["id"]
    store.answer_question(qid, {"Horizon": "4 weeks"})
    with pytest.raises(ValueError):
        store.answer_question(qid, {"Horizon": "12 weeks"})
    assert store.answered_questions(mid)[0]["answers"] == {"Horizon": "4 weeks"}
    assert Runtime(store, owner="r2").run(mid, _program) == {"status": "completed",
                                                            "result": {"horizon": "4 weeks"}}


def test_a_cancelled_question_cannot_be_answered_and_does_not_resume_the_mission(store):
    mid = store.create_mission("forecast", {})
    Runtime(store, owner="r1").run(mid, _program)
    qid = store.open_questions(mid)[0]["id"]
    store.close_open_questions(mid)
    with pytest.raises(ValueError):
        store.answer_question(qid, {"Horizon": "12 weeks"})
    assert store.answered_questions(mid) == []


def test_each_question_and_answer_is_recorded_once_in_the_event_log(store):
    mid = store.create_mission("forecast", {})
    Runtime(store, owner="r1").run(mid, _program)
    Runtime(store, owner="r2").run(mid, _program)            # a second run while waiting opens no duplicate
    qid = store.open_questions(mid)[0]["id"]
    store.answer_question(qid, {"Horizon": "12 weeks"})
    Runtime(store, owner="r3").run(mid, _program)
    kinds = [e["kind"] for e in store.events(mid) if e["kind"].startswith("question.")]
    assert kinds == ["question.opened", "question.answered"]
