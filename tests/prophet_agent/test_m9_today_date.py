"""Mission 9 (M9-7): the planner is given today's date, and a resumed run reads the date the first run stored.

Live, Nvidia 9e3408a4 (2026-11-26): the planner used data_end 2026-07-24 because no prompt said what today was. The date is read
once in the contracts step (the step ledger), so a replay sees the same line and makes the same decisions. Scripted brain only.
"""
from __future__ import annotations

import time

from prophet_agent.runtime import goal as G
from prophet_agent.runtime.goal import goal_program
from prophet_agent.runtime.runner import Runtime
from tests.prophet_agent.test_goal_loop import finish, met, run, store  # noqa: F401  (pytest fixture)
from tests.prophet_agent.test_r7_stall_progress import RecordingBrain

HORIZON = "How far ahead should I forecast?"


def _ask() -> dict:
    return {"action": "ask_user", "rationale": "high-risk: the horizon sets every delivered number and the data cannot say it",
            "questions": [{"question": HORIZON, "header": "Horizon", "multiSelect": False,
                           "options": [{"label": "4 weeks", "description": "short"}, {"label": "12 weeks", "description": "long"}]}]}


def test_dated_puts_the_date_first_and_keeps_the_context():
    dated = G._dated("Some context.", "2026-11-26")
    assert dated.startswith("The date today is 2026-11-26.") and dated.endswith("Some context.")
    assert G._dated("", "2026-11-26").count("\n") == 0
    assert G._dated("Some context.", None) == "Some context."


def test_the_planner_prompt_carries_today(store):  # noqa: F811
    brain = RecordingBrain(plans=[finish(), finish(), finish()], evals=[met()])
    _, out = run(store, goal_program("forecast demand", {}, brain))
    assert out["status"] == "completed"
    today = time.strftime("%Y-%m-%d")
    assert brain.prompts and all(f"The date today is {today}" in p for p in brain.prompts)


def test_a_resumed_run_reads_the_stored_date(store, monkeypatch):  # noqa: F811
    real = time.strftime("%Y-%m-%d")
    brain = RecordingBrain(plans=[_ask(), finish(), finish(), finish()], evals=[met("12-week forecast")])
    program = goal_program("forecast demand", {}, brain)
    mid, first = run(store, program)
    assert first["status"] == "waiting_user"
    # The clock moves on before the person answers. The resumed run must not read the new date.
    monkeypatch.setattr(G.time, "strftime", lambda fmt, *a: "2031-01-01" if fmt == "%Y-%m-%d" else time.strftime(fmt, *a))
    store.answer_question(store.open_questions(mid)[0]["id"], {"Horizon": "12 weeks"})
    done = Runtime(store, owner="r2").run(mid, program)
    assert done["status"] == "completed"
    assert store.step_record(mid, 0)["v"]["today"] == real
    assert brain.prompts and f"The date today is {real}" in brain.prompts[-1]
    assert "2031-01-01" not in brain.prompts[-1]
