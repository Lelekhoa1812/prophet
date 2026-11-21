"""A decision the person has made is never asked again: not after compaction, not when the wording changes, not in a later
part of the work. Regression for the live run where the same spike was asked four times (mission 8beef7ad, steps 18-27)."""
from __future__ import annotations

import pytest

from prophet_agent.runtime.goal import Memo, _already_asked, _decisions, _evidence_block, goal_program
from prophet_agent.runtime.runner import Runtime
from prophet_agent.runtime.store import Store


class ScriptedBrain:
    def __init__(self, plans=(), evals=()):
        self.plans, self.evals = list(plans), list(evals)
        self.calls = {"plan": 0, "eval": 0}
        self.plan_messages: list[list[dict]] = []

    def decide(self, messages, schema, name):
        self.calls[name] += 1
        if name == "plan":
            self.plan_messages.append(messages)
        queue = self.plans if name == "plan" else self.evals
        if not queue:
            raise AssertionError(f"brain asked for {name} but the script is exhausted")
        return queue.pop(0)


def ask(question, header, options=("Model it as a promotion", "Ignore it")):
    return {"action": "ask_user", "rationale": "decision", "questions": [
        {"question": question, "header": header, "multiSelect": False,
         "options": [{"label": o, "description": ""} for o in options]}]}


def finish(why="done"):
    return {"action": "finish", "rationale": why}


def met(answer="ok"):
    return {"verdict": "met", "reason": "enough", "answer": answer}


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "decisions.sqlite")
    yield s
    s.close()


def _opened(store, mid):
    return [e for e in store.events(mid, 0) if e["kind"] == "question.opened"]


def _run_until_answered(store, program, mid, runtime_owner="r1"):
    first = Runtime(store, owner=runtime_owner).run(mid, program)
    assert first["status"] == "waiting_user"
    return store.open_questions(mid)[0]["id"]


def test_repeated_question_with_same_header_is_refused_and_the_answer_stands(store):
    q = "The 2025-06-03 spike is an extreme outlier. Should I ignore it as a data error?"
    brain = ScriptedBrain(plans=[ask(q, "Outlier"), ask(q, "Outlier"), finish()], evals=[met()])
    program = goal_program("forecast sales", {}, brain)
    mid = store.create_mission("forecast sales", {})
    qid = _run_until_answered(store, program, mid)
    store.answer_question(qid, {"Outlier": "Ignore it"})
    done = Runtime(store, owner="r2").run(mid, program)

    assert done["status"] == "completed"
    assert len(_opened(store, mid)) == 1, "the repeated question must not open a second pause"
    refused = [h for h in done["result"]["evidence"] if h.get("kind") == "ask_user" and h.get("error")]
    assert refused and "already answered" in refused[0]["error"]


def test_same_question_with_new_header_and_near_identical_wording_is_refused(store):
    q1 = "The March and August 2025 spikes do not match any public holiday. Treat them as unrecorded promotions?"
    q2 = "The March and August 2025 spikes do not match any public holiday. Should I treat them as unrecorded promotions?"
    brain = ScriptedBrain(plans=[ask(q1, "March/August"), ask(q2, "Spikes"), finish()], evals=[met()])
    program = goal_program("forecast sales", {}, brain)
    mid = store.create_mission("forecast sales", {})
    qid = _run_until_answered(store, program, mid)
    store.answer_question(qid, {"March/August": "Model it as a promotion"})
    done = Runtime(store, owner="r2").run(mid, program)

    assert done["status"] == "completed"
    assert len(_opened(store, mid)) == 1


def test_a_genuinely_new_decision_still_pauses(store):
    q1 = "The March spikes: are they promotions, or should they be ignored as unexplained?"
    q2 = "Which currency is the sales column in: AUD or USD?"
    brain = ScriptedBrain(plans=[ask(q1, "March"), ask(q2, "Currency", ("AUD", "USD")), finish()], evals=[met()])
    program = goal_program("forecast sales", {}, brain)
    mid = store.create_mission("forecast sales", {})
    qid = _run_until_answered(store, program, mid)
    store.answer_question(qid, {"March": "Model it as a promotion"})
    second = Runtime(store, owner="r2").run(mid, program)

    assert second["status"] == "waiting_user", "a new decision is asked, it is not refused"
    assert len(_opened(store, mid)) == 2


def test_the_planner_is_shown_each_answer_verbatim_on_the_next_plan(store):
    q = "How far ahead should I forecast?"
    brain = ScriptedBrain(plans=[ask(q, "Horizon", ("4 weeks", "12 weeks")), finish()], evals=[met()])
    program = goal_program("forecast demand", {}, brain)
    mid = store.create_mission("forecast demand", {})
    qid = _run_until_answered(store, program, mid)
    store.answer_question(qid, {"Horizon": "12 weeks"})
    Runtime(store, owner="r2").run(mid, program)

    last_plan = brain.plan_messages[-1][-1]["content"]
    assert "Decisions the person has already made" in last_plan
    assert "How far ahead should I forecast?" in last_plan and "12 weeks" in last_plan


def test_decisions_are_kept_when_memory_has_folded_their_steps():
    history = [
        {"kind": "ask_user", "questions": [{"question": "Ignore the 2025-06-03 spike as a data error?", "header": "Outlier"}],
         "answers": {"Outlier": "Ignore as data error"}},
        {"kind": "tool", "tool": "sandbox_python", "args": {}, "result": {"ok": True}},
        {"kind": "tool", "tool": "backtest_series", "args": {}, "result": {"ok": True}},
    ]
    # The compactor folded everything up to the end, and its prose did not mention the decision.
    memo = Memo(text="Goal: forecast sales. Data loaded and checked.", covered=len(history))
    block = _evidence_block(history, memo)
    assert "Ignore as data error" in block
    assert block.index("Decisions the person has already made") < block.index("Memory of earlier steps")


def test_an_old_ledger_answer_without_stored_questions_still_appears_in_decisions():
    history = [{"kind": "ask_user", "answers": {"Horizon": "12 weeks"}}]
    assert _decisions(history) and "12 weeks" in _decisions(history)[0]


def test_already_asked_keeps_only_questions_not_yet_answered():
    history = [{"kind": "ask_user", "answers": {"A": "x"}, "questions": [
        {"question": "Should the 2025-06-03 spike be ignored as a data error?", "header": "A"}]}]
    new = [{"question": "Should the 2025-06-03 spike be ignored as a data error?", "header": "B"},
           {"question": "What is the forecast horizon in weeks?", "header": "C"}]
    open_qs, repeated = _already_asked(new, history)
    assert [q["header"] for q in open_qs] == ["C"]
    assert repeated and "2025-06-03" in repeated[0]


def test_questions_per_mission_are_capped_and_the_run_still_finishes(store):
    """Regression for the fake-workbook run (mission 0c7432c7): the planner asked 109 questions in one mission. Past the
    cap the question is refused as an observation and the work decides with what it has."""
    from prophet_agent.runtime.goal import MAX_ASKS

    topics = [
        "How many weeks ahead should the forecast go?",
        "Which currency is the sales column recorded in?",
        "Should the warehouse split be kept as a separate series?",
        "Is the Tuesday dip a public holiday in your region?",
        "Do returns count as negative sales or get dropped?",
        "Should promotion weeks be shown as a separate line?",
        "Which store group should the total cover?",
        "Do you want prediction intervals at 80 or 95 percent?",
        "Is the field labelled units in cases or in single items?",
    ]
    assert len(topics) > MAX_ASKS
    plans = [ask(t, f"H{i}") for i, t in enumerate(topics)] + [finish()]
    brain = ScriptedBrain(plans=plans, evals=[met()])
    program = goal_program("forecast sales", {}, brain)
    mid = store.create_mission("forecast sales", {})
    out = Runtime(store, owner="r1").run(mid, program)
    while out["status"] == "waiting_user":
        q = store.open_questions(mid)[0]
        header = q["payload"]["questions"][0]["header"]
        store.answer_question(q["id"], {header: "Ignore it"})
        out = Runtime(store, owner="r2").run(mid, program)

    assert out["status"] == "completed" and out["result"]["status"] == "met"
    assert len(_opened(store, mid)) == MAX_ASKS
    refused = [h for h in out["result"]["evidence"] if "no more questions" in str(h.get("error", ""))]
    assert refused, "the question past the cap is refused, not asked"


def _answered(question, header):
    return {"kind": "ask_user", "answers": {header: "Forecast 30 days (current)"}, "questions": [
        {"question": question, "header": header}]}


def test_reworded_question_about_the_same_dates_is_the_same_decision():
    # Live run 04 (mission f73bc4c5): the promo window was asked again with new wording and the same dates, after the
    # person had chosen to keep the 30-day window. The re-ask is refused; a different question about another date is not.
    first = ("The future promo calendar covers 2026-03-09 to 2026-03-14, but the 30-day forecast from 2025-12-30 ends on "
             "2026-01-28. How should I handle this?")
    again = ("The future promotion calendar you provided covers 2026-03-09 to 2026-03-14, but the 30-day forecast from "
             "2025-12-30 ends on 2026-01-29. Extend the forecast to cover the promotion period?")
    other = "The 2025-06-03 spike (2002.1) is flagged as a major anomaly. Is this a real event to keep?"
    history = [_answered(first, "Promo window")]
    open_qs, repeated = _already_asked([
        {"question": again, "header": "Promo horizon"},
        {"question": other, "header": "Spike"}], history)
    assert [q["header"] for q in open_qs] == ["Spike"]
    assert len(repeated) == 1


def test_unrelated_numbers_do_not_make_a_question_a_repeat():
    history = [_answered("The 2025-06-03 spike is an outlier. Keep it?", "Spike")]
    open_qs, repeated = _already_asked([
        {"question": "The series has 1200 rows from 2024-01-01 to 2025-04-30. Which date order applies?",
         "header": "Order"}], history)
    assert len(open_qs) == 1 and repeated == []


def test_planner_never_asks_the_person_about_how_tools_store_data():
    # Live run 09 (300k rows): the planner asked the person to choose between re-creating 300 series in one run or
    # forecasting a subset, because files did not survive between sandbox runs. That is the agent's problem to solve.
    from prophet_agent.prompts import current

    text = " ".join(current("planner").text.split())
    assert "How the tools store and pass data is never a question for the person" in text
    assert "rebuild what a later run needs, in one run, or batch the work" in text
