"""Mission 9 (M9-7): a malformed ask_user call is repaired or explained in one step.

Live, Nvidia 9e3408a4 (2026-11-26): an ask_user call failed with "invalid questions: questions must be a list of 1 to 4 items"
because the planner sent one question object. These tests cover the repairs (a single object, a JSON string, a wrapper, more than
four questions) and the explanation for a shape that cannot be repaired. Scripted planner only; no model is called.
"""
from __future__ import annotations

import json

from prophet_agent.runtime.goal import goal_program
from prophet_agent.runtime.questions import MAX_QUESTIONS, expected_shape, repair_questions, validate_questions
from prophet_agent.runtime.runner import Runtime
from tests.prophet_agent.test_goal_loop import ScriptedBrain, finish, met, store  # noqa: F401  (pytest fixture)
from tests.prophet_agent.test_goal_loop import run

HORIZON = "How far ahead should I forecast?"
RATIONALE = "high-risk: the horizon sets every delivered number and the data cannot say it"


def _question(text: str = HORIZON, label: str = "4 weeks") -> dict:
    return {"question": text, "header": "Horizon", "multiSelect": False,
            "options": [{"label": label, "description": "short"}, {"label": "12 weeks", "description": "quarter"}]}


def _ask(questions) -> dict:
    return {"action": "ask_user", "rationale": RATIONALE, "questions": questions}


def test_a_single_question_object_is_wrapped_in_a_list():
    raw, notes = repair_questions(_question())
    assert raw == [_question()]
    assert validate_questions({"questions": raw}) == []
    assert any("wrapped" in n for n in notes)


def test_a_payload_object_or_a_json_string_is_decoded():
    assert repair_questions({"questions": [_question()]})[0] == [_question()]
    assert repair_questions({"questions": _question()})[0] == [_question()]
    raw, notes = repair_questions(json.dumps([_question()]))
    assert raw == [_question()] and any("decoded" in n for n in notes)


def test_more_than_four_questions_keep_the_first_four_with_a_note():
    many = [_question(f"Which option for item {i}?") for i in range(6)]
    raw, notes = repair_questions(many)
    assert len(raw) == MAX_QUESTIONS and raw == many[:MAX_QUESTIONS]
    assert validate_questions({"questions": raw}) == []
    assert any("6 questions were sent" in n and "first 4" in n for n in notes)


def test_a_list_of_the_right_size_is_untouched():
    raw, notes = repair_questions([_question()])
    assert raw == [_question()] and notes == []


def test_an_unrepairable_shape_is_explained_with_the_expected_shape():
    raw, notes = repair_questions("what horizon do you want?")
    assert raw == "what horizon do you want?" and notes == []
    shape = expected_shape()
    assert '"questions":[{' in shape and '"options"' in shape and "1 to 4" in shape
    assert validate_questions({"questions": []})


def test_the_planner_single_object_pauses_for_the_answer_in_one_step(store):
    brain = ScriptedBrain(plans=[_ask(_question()), finish(), finish(), finish()], evals=[met("12-week forecast")])
    program = goal_program("forecast demand", {}, brain)
    mid, first = run(store, program)
    assert first["status"] == "waiting_user", "the single object is repaired and asked, not refused"
    qid = store.open_questions(mid)[0]["id"]
    store.answer_question(qid, {"Horizon": "12 weeks"})
    done = Runtime(store, owner="r2").run(mid, program)
    assert done["status"] == "completed"


def test_an_unrepairable_ask_returns_the_shape_in_the_observation(store):
    bad = _ask("how far ahead?")
    # A premature finish is refused before the evaluator (round 3, A5-04), so the script carries spare finishes.
    brain = ScriptedBrain(plans=[bad, finish(), finish(), finish()], evals=[met()])
    program = goal_program("forecast demand", {}, brain)
    _, out = run(store, program)
    assert out["status"] == "completed", "an invalid question is a planner observation, not a pause"
    error = out["result"]["evidence"][0]["error"]
    assert "invalid questions" in error and "Send the questions as a list" in error and '"options"' in error
