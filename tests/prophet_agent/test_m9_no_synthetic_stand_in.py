"""Mission 9 (task 2): no offer of invented data in place of a requested real series.

Live (Nvidia baseline, 2026-11-26): the planner asked the person "Use synthetic data based on known NVDA price ranges". The
refusal is in the question validator, the goal's ask_user path and the answer's contract check. A question about a synthetic twin
for a test the person asked for stays allowed. Scripted planner only; no model is called.
"""
from __future__ import annotations

import pytest

from prophet_agent.runtime.contracts import blocking, check_answer, check_answer_real_series
from prophet_agent.runtime.goal import goal_program
from prophet_agent.runtime.questions import invented_data_phrase, synthetic_substitute_problems, validate_for_goal
from prophet_agent.runtime.runner import Runtime
from tests.prophet_agent.test_goal_loop import ScriptedBrain, finish, met, store  # noqa: F401  (pytest fixture)
from tests.prophet_agent.test_goal_loop import run

GOAL = "forecast NVDA daily closes for the next two months"
RATIONALE = "high-risk: the price series sets every delivered number and the data cannot settle it"
SYNTHETIC = "Use synthetic data based on known NVDA price ranges"


def _question(*labels_and_descriptions) -> dict:
    return {"question": "Which NVDA series should the forecast use?", "header": "Series", "multiSelect": False,
            "options": [{"label": lab, "description": desc} for lab, desc in labels_and_descriptions]}


REAL = ("Yahoo chart JSON", "The public price endpoint for NVDA daily closes")


# ---------------------------------------------------------------- the question validator

def test_an_option_that_offers_synthetic_data_in_place_of_the_real_series_is_refused():
    payload = {"questions": [_question((SYNTHETIC, "Simulated closes"), REAL)]}
    problems = synthetic_substitute_problems(payload, GOAL)
    assert len(problems) == 1
    assert "questions[0].options[0]" in problems[0] and "Remove that option" in problems[0]
    assert "keep researching public sources" in problems[0]
    assert validate_for_goal(payload, GOAL) == problems


@pytest.mark.parametrize("label, description", [
    ("Simulated prices", "random walk fitted to the last year"),
    ("Approximate closes", "rounded from memory of the price ranges"),
    ("Made-up history", "a plausible series"),
    ("Fabricated values", "filled in by the assistant"),
    ("Use approximate NVDA prices", "from known ranges"),
])
def test_other_invented_or_approximate_data_offers_are_refused(label, description):
    assert synthetic_substitute_problems({"questions": [_question((label, description), REAL)]}, GOAL)


def test_a_synthetic_twin_for_a_test_the_person_asked_for_is_allowed():
    payload = {"questions": [_question(("Synthetic twin", "a simulated series for the backtest"), REAL)]}
    assert synthetic_substitute_problems(payload, "run a backtest on a synthetic twin of NVDA demand") == []
    assert synthetic_substitute_problems(payload, "simulate the series to test the forecast") == []


def test_real_source_options_and_refusals_of_synthetic_data_are_not_flagged():
    payload = {"questions": [_question(("Yahoo chart JSON", "Public daily closes"),
                                       ("Do not use synthetic data", "Stop and report the gap"))]}
    assert synthetic_substitute_problems(payload, GOAL) == []
    assert invented_data_phrase("We did not use synthetic data.") is None
    assert invented_data_phrase("The 80% band is approximate and wide.") is None


# ---------------------------------------------------------------- the goal's ask_user path

def _ask(question: dict) -> dict:
    return {"action": "ask_user", "rationale": RATIONALE, "questions": [question]}


def test_the_planner_offer_of_a_synthetic_series_is_refused_and_no_question_is_asked(store):
    brain = ScriptedBrain(plans=[_ask(_question((SYNTHETIC, "Simulated closes"), REAL)), finish(), finish(), finish()],
                          evals=[met("forecast from the real series")])
    program = goal_program(GOAL, {}, brain)
    mid, first = run(store, program)
    assert first["status"] != "waiting_user", "the offer of invented data was put to the person"
    assert store.open_questions(mid) == []


def test_the_same_question_without_the_synthetic_option_is_asked(store):
    brain = ScriptedBrain(plans=[_ask(_question(("Yahoo chart JSON", "Public daily closes"),
                                                ("Stooq CSV", "Public daily closes, older"))), finish()],
                          evals=[met()])
    program = goal_program(GOAL, {}, brain)
    mid, first = run(store, program)
    assert first["status"] == "waiting_user"
    assert store.open_questions(mid)


# ---------------------------------------------------------------- the answer's contract check

def test_an_answer_that_used_synthetic_data_in_place_of_the_requested_series_is_blocked():
    answer = "The forecast uses synthetic NVDA prices generated from the known price ranges, so the range is wide."
    check = check_answer_real_series(answer, GOAL)
    assert check is not None and check["ok"] is False and check["severity"] == "block"
    assert check["name"] == "answer_real_series"
    assert any(c["name"] == "answer_real_series" for c in blocking(check_answer(answer, [], goal=GOAL)))


def test_approximate_data_as_the_series_is_blocked_but_an_honest_answer_is_not():
    assert check_answer_real_series("We used approximate closing prices for March to June.", GOAL) is not None
    assert check_answer_real_series("The series is the public daily close from Yahoo; the band is approximate.", GOAL) is None
    assert check_answer_real_series("No synthetic data was used. The closes are from the public endpoint.", GOAL) is None


def test_a_goal_that_asks_for_a_synthetic_test_series_is_not_blocked():
    answer = "The backtest used a synthetic twin of the demand series with censoring."
    assert check_answer_real_series(answer, "backtest a synthetic twin of demand") is None
