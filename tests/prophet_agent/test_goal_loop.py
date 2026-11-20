"""Goal-loop tests with a scripted brain (GOAL G7) and the ask/pause path (G8), run on the durable runtime.

The brain is a queue of decisions. Each test checks the outcome and, where it matters, how many decisions were asked
for, so a resumed run is shown not to ask the model again for work it already finished.
"""
from __future__ import annotations

import pytest

from prophet_agent.runtime.goal import GoalConfig, goal_program
from prophet_agent.runtime.questions import validate_questions
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import Store


class Crash(BaseException):
    pass


class ScriptedBrain:
    def __init__(self, plans=(), evals=()):
        self.plans = list(plans)
        self.evals = list(evals)
        self.calls = {"plan": 0, "eval": 0}

    def decide(self, messages, schema, name):
        self.calls[name] += 1
        queue = self.plans if name == "plan" else self.evals
        if not queue:
            raise AssertionError(f"brain asked for {name} but the script is exhausted")
        return queue.pop(0)


def call(tool, args=None, why="gather"):
    return {"action": "call_tool", "tool": tool, "args": args or {}, "rationale": why}


def finish(why="done"):
    return {"action": "finish", "rationale": why}


def met(answer="ok"):
    return {"verdict": "met", "reason": "enough", "answer": answer}


def not_met():
    return {"verdict": "not_met", "reason": "need more"}


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "goal.sqlite")
    yield s
    s.close()


def _tool(name, fn=None, idempotent=True):
    return Tool(name, fn or (lambda args, key: {"echo": args}), idempotent=idempotent)


def run(store, program, budget=None, owner="r1"):
    mid = store.create_mission("forecast", budget or {})
    return mid, Runtime(store, owner=owner).run(mid, program)


def test_met_after_one_tool_call(store):
    brain = ScriptedBrain(plans=[call("ping", {"x": 1}), finish()], evals=[met("pong")])
    program = goal_program("check", {"ping": _tool("ping")}, brain)
    _, out = run(store, program)
    assert out["status"] == "completed"
    assert out["result"]["status"] == "met"
    assert out["result"]["answer"] == "pong"
    assert brain.calls == {"plan": 2, "eval": 1}, "the evaluator runs once, when the planner says it is done"


def test_not_met_then_met_after_two_iterations(store):
    brain = ScriptedBrain(plans=[call("a"), finish(), call("b"), finish()], evals=[not_met(), met()])
    program = goal_program("g", {"a": _tool("a"), "b": _tool("b")}, brain)
    _, out = run(store, program)
    assert out["result"]["status"] == "met"
    assert out["result"]["iterations"] == 4
    assert out["result"]["evidence"][2]["kind"] == "evaluation", "a not-met verdict goes back to the planner as evidence"


def test_impossible_verdict_stops_the_loop(store):
    brain = ScriptedBrain(plans=[call("a"), finish()], evals=[{"verdict": "impossible", "reason": "no data for that region"}])
    program = goal_program("g", {"a": _tool("a")}, brain)
    _, out = run(store, program)
    assert out["result"]["status"] == "impossible"
    assert out["result"]["reason"] == "no data for that region"


def test_stall_detected_when_the_same_call_returns_the_same_result(store):
    plans = [call("a", {"k": 1})] * 5
    brain = ScriptedBrain(plans=plans, evals=[])
    program = goal_program("g", {"a": _tool("a", lambda args, key: {"same": True})}, brain,
                           GoalConfig(max_iterations=8, stall_window=3))
    mid, out = run(store, program)
    assert out["result"]["status"] == "stalled"
    assert out["status"] == "stalled", "a stalled goal must not be recorded as a completed mission"
    assert store.mission(mid)["status"] == "stalled"
    assert brain.calls["plan"] == 3, "the loop stops after the stall window, not after the iteration cap"


def test_goal_and_argument_names_survive_a_large_observation(store):
    seen = []

    class Recording(ScriptedBrain):
        def decide(self, messages, schema, name):
            seen.append((name, messages[-1]["content"]))
            return super().decide(messages, schema, name)

    big = {"pad": list(range(3000))}       # far over OBS_CHARS once serialised
    tool = Tool("a", lambda args, key: big, idempotent=True,
                doc="Echo a large payload. Arguments (use these names only): none.")
    brain = Recording(plans=[call("a"), finish(), finish()], evals=[not_met(), met()])
    goal = "Forecast marker ZX-42 for 8 weeks"
    _, out = run(store, goal_program(goal, {"a": tool}, brain))
    assert out["result"]["status"] == "met"
    plan_text = next(c for n, c in seen if n == "plan")
    eval_texts = [c for n, c in seen if n == "eval"]
    assert f"Goal:\n{goal}" in plan_text and "Arguments (use these names only)" in plan_text
    assert all(f"Goal:\n{goal}" in text for text in eval_texts), "the goal must never be clipped away"
    assert "...[clipped]" in eval_texts[-1], "the large observation is clipped on its own"


def test_iteration_budget_caps_an_endless_loop(store):
    counter = {"n": 0}

    def changing(args, key):
        counter["n"] += 1
        return {"n": counter["n"]}

    plans = [call("a")] * 4
    brain = ScriptedBrain(plans=plans, evals=[not_met()])     # only the last iteration is evaluated
    program = goal_program("g", {"a": _tool("a", changing)}, brain, GoalConfig(max_iterations=4))
    _, out = run(store, program)
    assert out["result"]["status"] == "iteration_budget"
    assert out["result"]["iterations"] == 4


def test_tool_budget_from_the_runtime_stops_the_mission(store):
    counter = {"n": 0}

    def changing(args, key):
        counter["n"] += 1
        return {"n": counter["n"]}

    brain = ScriptedBrain(plans=[call("a")] * 10, evals=[])
    program = goal_program("g", {"a": _tool("a", changing)}, brain, GoalConfig(max_iterations=10))
    _, out = run(store, program, budget={"max_tool_calls": 2})
    assert out == {"status": "budget_exhausted", "which": "max_tool_calls"}


def test_unknown_tool_is_an_observation_not_a_crash(store):
    brain = ScriptedBrain(plans=[call("nope"), call("a"), finish()], evals=[met()])
    program = goal_program("g", {"a": _tool("a")}, brain)
    _, out = run(store, program)
    assert out["result"]["status"] == "met"
    assert "unknown tool" in out["result"]["evidence"][0]["error"]


def test_tool_error_is_passed_to_the_planner_as_evidence(store):
    def broken(args, key):
        raise RuntimeError("upstream 500 with token nvapi-LEAKED987654321")

    brain = ScriptedBrain(plans=[call("a"), finish()], evals=[met()])
    program = goal_program("g", {"a": _tool("a", broken, idempotent=False)}, brain)
    _, out = run(store, program)
    assert out["result"]["status"] == "met"
    err = out["result"]["evidence"][0]["error"]
    assert "upstream 500" in err and "nvapi-LEAKED987654321" not in err


def test_invalid_questions_do_not_pause(store):
    bad = {"action": "ask_user", "rationale": "need horizon", "questions": [
        {"question": "How far ahead", "header": "This header is far too long", "multiSelect": False,
         "options": [{"label": "4 weeks", "description": ""}]}]}
    brain = ScriptedBrain(plans=[bad, finish()], evals=[met()])
    program = goal_program("g", {}, brain)
    _, out = run(store, program)
    assert out["status"] == "completed", "an invalid question is a planner observation, not a pause"
    assert "invalid questions" in out["result"]["evidence"][0]["error"]


def test_ask_pauses_then_resumes_from_the_answer(store):
    question = {"action": "ask_user", "rationale": "horizon", "questions": [
        {"question": "How far ahead should I forecast?", "header": "Horizon", "multiSelect": False,
         "options": [{"label": "4 weeks", "description": "short"}, {"label": "12 weeks", "description": "quarter"}]}]}
    brain = ScriptedBrain(plans=[question, finish()], evals=[met("12-week forecast")])
    program = goal_program("forecast demand", {}, brain)

    mid, first = run(store, program)
    assert first["status"] == "waiting_user"
    qid = store.open_questions(mid)[0]["id"]
    store.answer_question(qid, {"Horizon": "12 weeks"})

    done = Runtime(store, owner="r2").run(mid, program)
    assert done["status"] == "completed"
    assert done["result"]["status"] == "met"
    assert brain.calls == {"plan": 2, "eval": 1}, "plan:0 is not asked again after the answer"
    assert len(store.open_questions(mid)) == 0


def test_crash_mid_loop_resumes_without_repeating_decisions_or_tools(store):
    ran = {"tool": 0}
    crash_after = {"armed": True}

    def tool_fn(args, key):
        ran["tool"] += 1
        return {"v": args["i"]}

    class CrashingBrain(ScriptedBrain):
        plan_requests = 0

        def decide(self, messages, schema, name):
            if name == "plan":
                self.plan_requests += 1
                if self.plan_requests == 3 and crash_after["armed"]:
                    crash_after["armed"] = False
                    raise Crash()      # die while the third plan is in flight, before it is stored
            return super().decide(messages, schema, name)

    plans = [call("a", {"i": 0}), call("a", {"i": 1}), call("a", {"i": 2}), finish()]
    brain = CrashingBrain(plans=plans, evals=[met("three")])
    program = goal_program("g", {"a": _tool("a", tool_fn)}, brain)

    mid = store.create_mission("g", {})
    with pytest.raises(Crash):
        Runtime(store, owner="r1").run(mid, program)
    store.conn.execute("UPDATE missions SET lease_until=0 WHERE id=?", (mid,))

    out = Runtime(store, owner="r2").run(mid, program)
    assert out["result"]["status"] == "met"
    assert out["result"]["answer"] == "three"
    assert ran["tool"] == 3, "each tool ran once across the crash"
    # Decisions already stored are not requested again: four plans were answered in total, and one evaluation.
    assert brain.calls["plan"] == 4
    assert brain.calls["eval"] == 1


def test_question_validator_rules():
    good = {"questions": [{"question": "Pick one?", "header": "Pick", "multiSelect": False,
                           "options": [{"label": "A", "description": ""}, {"label": "B", "description": ""}]}]}
    assert validate_questions(good) == []
    assert validate_questions({"questions": []})
    assert validate_questions({"questions": [{**good["questions"][0], "options": [{"label": "A"}]}]})
    assert validate_questions({"questions": [{**good["questions"][0], "header": "x" * 13}]})
    assert validate_questions({"questions": [{**good["questions"][0], "question": "no mark"}]})
    five = [good["questions"][0]] * 5
    assert validate_questions({"questions": five})
    assert validate_questions({"questions": [{**good["questions"][0], "options": [
        {"label": "A", "description": ""}, {"label": "A", "description": ""}]}]})


def test_prompt_contracts_are_versioned_files_and_recorded_in_the_ledger(tmp_path):
    from prophet_agent import prompts
    for name, version in prompts.CURRENT.items():
        c = prompts.load(version)
        assert c.version == version and c.text and len(c.sha256) == 12
    store = Store(tmp_path / "c.sqlite")
    mid = store.create_mission("g", {"max_tool_calls": 3, "max_seconds": 30})

    class B:
        def decide(self, messages, schema, name):
            return {"action": "finish", "rationale": "r"} if name == "plan" else {"verdict": "met", "reason": "ok", "answer": "a"}

    Runtime(store, owner="t").run(mid, goal_program("g", {}, B()))
    recorded = next(s for s in store.steps(mid) if s["name"] == "contracts")["output"]
    assert recorded["planner"]["version"] == prompts.CURRENT["planner"]


def test_a_figure_missing_from_the_evidence_is_repaired_once(store):
    brain = ScriptedBrain(plans=[call("a"), finish()],
                          evals=[met("The total is 999 units."), met("The total is 42 units.")])
    program = goal_program("what is the total", {"a": _tool("a", lambda args, key: {"total": 42})}, brain)
    _, out = run(store, program)
    assert out["result"]["status"] == "met"
    assert out["result"]["answer"] == "The total is 42 units."
    assert brain.calls["eval"] == 2, "one judgement and one repair, no more"


def test_figures_still_unsupported_after_repair_make_a_partial_answer(store):
    brain = ScriptedBrain(plans=[call("a"), finish()], evals=[met("Total 999."), met("Total 998.")])
    program = goal_program("what is the total", {"a": _tool("a", lambda args, key: {"total": 42})}, brain)
    _, out = run(store, program)
    assert out["result"]["status"] == "partial"
    assert out["result"]["unverified"] == ["998"]
    assert out["status"] == "partial", "a partial answer is not recorded as met"


class WritingBrain(ScriptedBrain):
    """A brain that can also write free text, as the real one does, so the answer keeps its line breaks."""

    def __init__(self, plans=(), evals=(), writes=()):
        super().__init__(plans, evals)
        self.writes = list(writes)
        self.prompts = []

    def write(self, messages):
        self.prompts.append(messages[-1]["content"])
        return self.writes.pop(0)


def test_the_final_answer_is_written_as_free_text_with_its_line_breaks(store):
    flat = "## Totals  **42 units**  - North 42"        # what a schema-bound reply looks like: no newlines
    nice = "## Totals\n\n**42 units**\n\n- North 42"
    brain = WritingBrain(plans=[call("a"), finish()], evals=[met(flat)], writes=[nice])
    program = goal_program("totals", {"a": _tool("a", lambda args, key: {"north": 42})}, brain)
    _, out = run(store, program)
    assert out["result"]["status"] == "met"
    assert out["result"]["answer"] == nice and "\n" in out["result"]["answer"]
    assert flat in brain.prompts[0], "the verdict's answer goes to the writer as the draft"


def test_a_written_answer_with_a_stray_figure_is_rewritten_once(store):
    brain = WritingBrain(plans=[call("a"), finish()], evals=[met("North 42")],
                         writes=["North **42** of 999\n", "North **42**\n"])
    program = goal_program("totals", {"a": _tool("a", lambda args, key: {"north": 42})}, brain)
    _, out = run(store, program)
    assert out["result"]["status"] == "met" and out["result"]["answer"] == "North **42**\n"
    assert "999" in brain.prompts[1], "the rewrite is told which figure was not in the evidence"


def test_a_spent_tool_budget_ends_in_a_judgement_not_a_runtime_stop(store):
    seen, counter = [], {"n": 0}

    class Rec(ScriptedBrain):
        def decide(self, messages, schema, name):
            seen.append((name, messages[-1]["content"]))
            return super().decide(messages, schema, name)

    def changing(args, key):
        counter["n"] += 1
        return {"n": counter["n"]}

    brain = Rec(plans=[call("a", {"i": 0}), call("a", {"i": 1}), call("a", {"i": 2})],
                evals=[{"verdict": "not_met", "reason": "forecast missing", "answer": "So far 3 steps."}])
    program = goal_program("g", {"a": _tool("a", changing)}, brain, GoalConfig(max_iterations=10, tool_budget=3))
    _, out = run(store, program, budget={"max_tool_calls": 3})
    assert out["result"]["status"] == "partial", "the loop judges before the runtime has to stop it"
    assert out["result"]["answer"] == "So far 3 steps."
    plans = [text for name, text in seen if name == "plan"]
    assert "Budget: step 1 of 10; 3 steps left." in plans[0] and "Stop exploring" not in plans[0]
    assert "Stop exploring" in plans[2], "with two or fewer steps left the planner is told to finish"
    assert len(plans) == 3, "no fourth plan is requested once the calls are used"


def test_an_empty_repair_is_reported_as_partial_not_met(store):
    brain = ScriptedBrain(plans=[call("a"), finish()], evals=[met("Total 999."), {"verdict": "not_met", "reason": "x"}])
    program = goal_program("what is the total", {"a": _tool("a", lambda args, key: {"total": 42})}, brain)
    _, out = run(store, program)
    assert out["result"]["status"] == "partial"
    assert out["result"]["answer"] == ""


def test_a_spent_budget_gives_the_best_answer_so_far(store):
    brain = ScriptedBrain(plans=[call("a")], evals=[
        {"verdict": "not_met", "reason": "no forecast yet", "answer": "So far 7 units."}])
    program = goal_program("g", {"a": _tool("a", lambda args, key: {"n": 7})}, brain, GoalConfig(max_iterations=1))
    _, out = run(store, program)
    assert out["result"]["status"] == "partial"
    assert out["result"]["answer"] == "So far 7 units."
    assert out["result"]["reason"] == "no forecast yet"


def test_evaluator_question_pauses_once_then_judges_with_the_answer(store):
    question = {"verdict": "not_met", "reason": "which product", "questions": [
        {"question": "Which product should I total?", "header": "Product", "multiSelect": False,
         "options": [{"label": "Widgets", "description": ""}, {"label": "Gadgets", "description": ""}]}]}
    brain = ScriptedBrain(plans=[call("a"), finish()], evals=[question, met("5 units of widgets.")])
    program = goal_program("total units", {"a": _tool("a", lambda args, key: {"units": 5})}, brain)
    mid, first = run(store, program)
    assert first["status"] == "waiting_user"
    store.answer_question(store.open_questions(mid)[0]["id"], {"Product": "Widgets"})
    done = Runtime(store, owner="r2").run(mid, program)
    assert done["result"]["status"] == "met"
    assert done["result"]["answer"] == "5 units of widgets."
    assert brain.calls == {"plan": 2, "eval": 2}


def test_near_duplicate_options_are_rejected():
    base = {"question": "Which total?", "header": "Total", "multiSelect": False}
    dup = {**base, "options": [{"label": "Monthly totals", "description": ""}, {"label": "Monthly total", "description": ""}]}
    assert validate_questions({"questions": [dup]})
    distinct = {**base, "options": [{"label": "Monthly", "description": ""}, {"label": "Quarterly", "description": ""}]}
    assert validate_questions({"questions": [distinct]}) == []


def test_grounding_matches_figures_at_their_written_precision():
    from prophet_agent.runtime.grounding import unverified
    history = [{"kind": "tool", "result": {"share": 0.1234, "units": 1204}}]
    assert unverified("Units 1,204 and a share of 12.3%.", history, "g") == []
    assert unverified("Units 1,205.", history, "g") == ["1,205"], "a different figure is not a rounding"
    assert unverified("Share 9%.", history, "g") == ["9%"]
    assert unverified("Horizon 8 weeks.", history, "forecast 8 weeks") == [], "figures in the goal are allowed"


def test_grounding_does_not_read_dates_or_ranges_as_negative_numbers():
    from prophet_agent.runtime.grounding import figures
    assert figures("The week of 2026-01-19 and 12-14 units") == ["12", "14"]
    assert figures("Change was -19 and 5%") == ["-19", "5%"]


def test_grounding_normalises_unicode_hyphens_and_minus():
    from prophet_agent.runtime.grounding import figures
    assert figures("week of 2026‑01‑26 and a 4‑week view") == ["4"]
    assert figures("fell by −5 units") == ["-5"]


def test_grounding_treats_month_and_year_labels_as_labels():
    from prophet_agent.runtime.grounding import figures
    assert figures("Jan 2026 is 1,409; March 15, 2026 and 3 June 2027 follow") == ["1,409"]


def test_grounding_does_not_ask_the_evidence_for_a_bare_year():
    from prophet_agent.runtime.grounding import unverified
    assert unverified("Rose each year, 2023 to 2026, to 1,389 units", [{"n": 1389}], "") == []
    assert unverified("Sold 1,390 units in 2026", [{"n": 1389}], "") == ["1,390"]
