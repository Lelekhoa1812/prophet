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
    brain = ScriptedBrain(plans=plans, evals=[{"verdict": "not_met", "reason": "stuck"}])
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
    assert out["result"]["status"] == "checkpoint", "the step budget is a checkpoint, never a terminal failure"
    assert out["result"]["iterations"] == 4


def test_tool_budget_from_the_runtime_stops_the_mission(store):
    counter = {"n": 0}

    def changing(args, key):
        counter["n"] += 1
        return {"n": counter["n"]}

    brain = ScriptedBrain(plans=[call("a")] * 10, evals=[])
    program = goal_program("g", {"a": _tool("a", changing)}, brain, GoalConfig(max_iterations=10))
    mid, out = run(store, program, budget={"max_tool_calls": 2})
    assert out["status"] == "checkpoint" and out["which"] == "max_tool_calls", "a spent allowance is resumable, not terminal"
    assert store.mission(mid)["status"] == "checkpoint"


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


def test_a_figure_missing_from_the_evidence_is_gathered_before_it_is_repaired(store):
    brain = ScriptedBrain(plans=[call("a"), finish(), finish()],
                          evals=[met("The total is 999 units."), met("The total is 42 units.")])
    program = goal_program("what is the total", {"a": _tool("a", lambda args, key: {"total": 42})}, brain)
    _, out = run(store, program)
    assert out["result"]["status"] == "met"
    assert out["result"]["answer"] == "The total is 42 units."
    assert brain.calls["eval"] == 2, "one judgement that is sent back to gather, then a second judgement that passes"


def test_figures_still_unsupported_after_gathering_and_repair_make_a_partial_answer(store):
    brain = ScriptedBrain(plans=[call("a"), finish(), finish(), finish()],
                          evals=[met("Total 999."), met("Total 998."), met("Total 998."), met("Total 998.")])
    program = goal_program("what is the total", {"a": _tool("a", lambda args, key: {"total": 42})}, brain)
    _, out = run(store, program)
    assert out["result"]["status"] == "partial"
    assert out["result"]["unverified"] == ["998"]
    assert out["status"] == "partial", "a partial answer is not recorded as met"
    assert brain.calls["eval"] == 4, "two gathering rounds, then one repair judgement"


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


def test_a_written_answer_with_a_stray_figure_is_gathered_then_rewritten(store):
    brain = WritingBrain(plans=[call("a"), finish(), finish(), finish()], evals=[met("North 42"), met("North 42"), met("North 42")],
                         writes=["North **42** of 999\n", "North **42** of 999\n", "North **42**\n"])
    program = goal_program("totals", {"a": _tool("a", lambda args, key: {"north": 42})}, brain)
    _, out = run(store, program)
    assert out["result"]["status"] == "met" and out["result"]["answer"] == "North **42**\n"
    assert brain.calls["eval"] == 3, "the stray figure is sent back twice to gather, then the rewrite passes"


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
    assert out["result"]["status"] == "checkpoint", "a spent tool allowance is a checkpoint, not a partial answer"
    assert out["result"]["draft"] == "So far 3 steps."
    plans = [text for name, text in seen if name == "plan"]
    assert "Budget: step 1 of this part of the work; 3 steps left in it." in plans[0] and "Stop exploring" not in plans[0]
    assert "Stop exploring" in plans[2], "with two or fewer steps left the planner is told to finish"
    assert len(plans) == 3, "no fourth plan is requested once the calls are used"


def test_an_empty_repair_is_reported_as_partial_not_met(store):
    brain = ScriptedBrain(plans=[call("a"), finish(), finish(), finish()],
                          evals=[met("Total 999."), met("Total 999."), met("Total 999."), {"verdict": "met", "reason": "x", "answer": ""}])
    program = goal_program("what is the total", {"a": _tool("a", lambda args, key: {"total": 42})}, brain)
    _, out = run(store, program)
    assert out["result"]["status"] == "partial"
    assert out["result"]["answer"] == ""


def test_a_spent_step_budget_is_a_checkpoint_that_keeps_the_draft(store):
    brain = ScriptedBrain(plans=[call("a")], evals=[
        {"verdict": "not_met", "reason": "no forecast yet", "answer": "So far 7 units."}])
    program = goal_program("g", {"a": _tool("a", lambda args, key: {"n": 7})}, brain, GoalConfig(max_iterations=1))
    _, out = run(store, program)
    assert out["result"]["status"] == "checkpoint" and out["status"] == "checkpoint"
    assert out["result"]["draft"] == "So far 7 units." and "answer" not in out["result"]


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
    evidence = [{"kind": "tool", "result": {"n": 1389}}]
    assert unverified("Rose each year, 2023 to 2026, to 1,389 units", evidence, "") == []
    assert unverified("Sold 1,390 units in 2026", evidence, "") == ["1,390"]


def test_the_models_own_arguments_and_rationale_are_not_evidence():
    from prophet_agent.runtime.grounding import unverified
    history = [{"kind": "tool", "tool": "a", "args": {"horizon": 1390}, "result": {"n": 1}, "rationale": "use 1390"},
               {"kind": "evaluation", "verdict": "not_met", "missing": "a 1390 figure is needed"}]
    assert unverified("Sold 1,390 units", history, "") == ["1,390"], "a figure the model only typed in is not evidence"
    assert unverified("Sold 1 unit", history, "") == []


def test_a_partial_stalled_answer_is_grounded_and_its_unsupported_figures_listed(store):
    brain = WritingBrain(plans=[call("a")] * 3, evals=[{"verdict": "not_met", "reason": "stuck", "answer": "North 42 so far"}],
                         writes=["**North 99 so far.**"])
    program = goal_program("totals", {"a": _tool("a", lambda args, key: {"north": 42})}, brain)
    _, out = run(store, program)
    assert out["result"]["status"] == "partial"
    assert out["result"]["unverified"] == ["99"], "a partial answer is grounded too, not only a met one"


def test_a_met_verdict_without_an_answer_is_written_from_its_reason(store):
    brain = WritingBrain(plans=[call("a"), finish()], evals=[{"verdict": "met", "reason": "North sold 42 units"}],
                         writes=["**North sold 42 units.**"])
    program = goal_program("totals", {"a": _tool("a", lambda args, key: {"north": 42})}, brain)
    _, out = run(store, program)
    assert out["result"]["status"] == "met" and out["result"]["answer"] == "**North sold 42 units.**"
    assert "North sold 42 units" in brain.prompts[0]


def test_met_with_nothing_to_read_is_never_reported_as_met(store):
    brain = ScriptedBrain(plans=[call("a"), finish(), finish()],
                          evals=[{"verdict": "met", "reason": ""}, {"verdict": "met", "reason": "", "answer": "North 42"}])
    program = goal_program("totals", {"a": _tool("a", lambda args, key: {"north": 42})}, brain)
    _, out = run(store, program)
    assert out["result"]["status"] == "met" and out["result"]["answer"] == "North 42"   # it went round again and got one
    brain = ScriptedBrain(plans=[call("a"), finish()], evals=[{"verdict": "met", "reason": ""}])
    _, out = run(store, goal_program("totals", {"a": _tool("a", lambda args, key: {"north": 42})}, brain, GoalConfig(max_iterations=1)))
    assert out["result"]["status"] == "partial" and out["result"]["answer"] == ""


def test_a_stalled_run_gives_a_partial_answer_from_what_was_found(store):
    brain = WritingBrain(plans=[call("a")] * 3, evals=[{"verdict": "not_met", "reason": "stuck", "answer": "North 42 so far"}],
                         writes=["**North 42 so far.**"])
    program = goal_program("totals", {"a": _tool("a", lambda args, key: {"north": 42})}, brain)
    _, out = run(store, program)
    assert out["result"]["status"] == "partial" and out["result"]["answer"] == "**North 42 so far.**"
    assert "repeating" in out["result"]["reason"]


def test_grounding_ignores_ordinal_labels_but_not_suffixed_figures():
    from prophet_agent.runtime.grounding import figures
    assert figures("10th–90th percentile, 1st and 3rd place") == []
    assert figures("sold 1.2M units, 5k orders") == ["1.2M", "5k"]          # the scale stays on the token; matching reads it


def test_grounding_reads_a_counting_first_column_as_row_labels_only():
    from prophet_agent.runtime.grounding import figures
    counted = "| Week | Units |\n|---|---|\n| 1 | 0.46 |\n| 2 | 0.44 |\n| 3 | 0.44 |\n| 4 | 0.45 |"
    assert figures(counted) == ["0.46", "0.44", "0.44", "0.45"]
    # a first column that is not a plain 1..n count holds real figures and is still checked
    real = "| Units | Region |\n|---|---|\n| 5 | North |\n| 9 | South |\n| 12 | East |"
    assert figures(real) == ["5", "9", "12"]
    # too short to be a numbering: still checked
    assert figures("| 1 | 10 |\n| 2 | 20 |") == ["1", "10", "2", "20"]


class FixedWriterBrain(ScriptedBrain):
    def write(self, messages):
        self.calls["write"] = self.calls.get("write", 0) + 1
        return "Established: 7 units. Missing: the forecast."


def test_a_spent_step_budget_with_no_draft_is_a_checkpoint_with_the_reason(store):
    brain = FixedWriterBrain(plans=[call("a")], evals=[{"verdict": "not_met", "reason": "no forecast yet"}])
    program = goal_program("g", {"a": _tool("a", lambda args, key: {"n": 7})}, brain, GoalConfig(max_iterations=1))
    _, out = run(store, program)
    assert out["result"]["status"] == "checkpoint"
    assert out["result"]["reason"] == "no forecast yet" and out["result"]["draft"] == ""




def test_a_checkpoint_continues_from_the_stored_steps_with_a_new_part(store):
    """The core of "reached its limit": a part of the work ends in a checkpoint, and Continue runs the next part.
    The stored steps replay from the ledger (no model call repeats), then the loop goes on past the old end."""
    counter = {"n": 0}

    def changing(args, key):
        counter["n"] += 1
        return {"n": counter["n"]}

    mid = store.create_mission("g", {})
    first = ScriptedBrain(plans=[call("a"), call("a")], evals=[not_met()])
    out = Runtime(store, owner="r1").run(mid, goal_program("g", {"a": _tool("a", changing)}, first, GoalConfig(max_iterations=2)))
    assert out["result"]["status"] == "checkpoint" and out["result"]["segment"] == 0
    assert store.mission(mid)["status"] == "checkpoint"

    # Continue: part 1 has four steps; the model is asked only for the steps that are new.
    second = ScriptedBrain(plans=[call("a"), finish()], evals=[met("Total 3.")])
    cfg = GoalConfig(max_iterations=2, segment=1)
    out = Runtime(store, owner="r2").run(mid, goal_program("g", {"a": _tool("a", changing)}, second, cfg))
    assert out["result"]["status"] == "met", out
    assert second.calls["plan"] == 2, "the two stored plans of part 0 are not asked again"
    assert counter["n"] == 3, "the tool calls of part 0 are not repeated"
    assert any(h.get("kind") == "continued" for h in out["result"]["evidence"])


class CompactingBrain(ScriptedBrain):
    """A brain whose free-text writer doubles as the compactor. Records every planner prompt it is shown."""

    def __init__(self, plans=(), evals=()):
        super().__init__(plans, evals)
        self.writes = 0
        self.planner_prompts: list[str] = []

    def decide(self, messages, schema, name):
        if name == "plan":
            self.planner_prompts.append("\n".join(m["content"] for m in messages))
        return super().decide(messages, schema, name)

    def write(self, messages):
        self.writes += 1
        return "MEMO-1: the tool a was called for each earlier step."     # no figures: grounding would rightly reject them


def test_older_observations_fold_into_memory_and_the_record_is_kept(store):
    """Context management: once the view outgrows its budget, older steps fold into a memo the planner reads, while
    every observation stays in the stored record for replay and audit."""
    counter = {"n": 0}

    def changing(args, key):
        counter["n"] += 1
        return {"n": counter["n"], "pad": "x" * 200}

    brain = CompactingBrain(plans=[call("a")] * 5 + [finish()], evals=[met()])
    cfg = GoalConfig(view_chars=300, keep_recent=2, memory_chars=4000)
    _, out = run(store, goal_program("g", {"a": _tool("a", changing)}, brain, cfg))
    assert out["result"]["status"] == "met"
    assert brain.writes >= 1, "the older steps were folded by the compactor, not dropped"
    assert "Memory of earlier steps" in brain.planner_prompts[-1]
    assert "MEMO-1" in brain.planner_prompts[-1]
    tool_steps = [h for h in out["result"]["evidence"] if h.get("kind") == "tool"]
    assert len(tool_steps) == 5, "compaction never deletes an observation from the record"


def test_a_brain_that_cannot_write_cuts_older_steps_and_says_so(store):
    """Without a writer the planner is told how many steps were left out of its view; nothing is invented."""
    counter = {"n": 0}

    def changing(args, key):
        counter["n"] += 1
        return {"n": counter["n"], "pad": "x" * 200}

    class NoWriter(ScriptedBrain):          # no free-text writer: the brain cannot compact, so the view is cut instead
        def __init__(self, plans=(), evals=()):
            super().__init__(plans, evals)
            self.planner_prompts: list[str] = []

        def decide(self, messages, schema, name):
            if name == "plan":
                self.planner_prompts.append("\n".join(m["content"] for m in messages))
            return super().decide(messages, schema, name)

    brain = NoWriter(plans=[call("a")] * 5 + [finish()], evals=[met()])
    cfg = GoalConfig(view_chars=300, keep_recent=2)
    _, out = run(store, goal_program("g", {"a": _tool("a", changing)}, brain, cfg))
    assert out["result"]["status"] == "met"
    assert "left out of this view" in brain.planner_prompts[-1]
    assert "MEMO-1" not in brain.planner_prompts[-1], "nothing is invented for the cut steps"
    tool_steps = [h for h in out["result"]["evidence"] if h.get("kind") == "tool"]
    assert len(tool_steps) == 5, "the record is complete even when the view is cut"


def test_a_sub_agent_returns_only_its_report_to_the_planner(store):
    """Isolation: the sub-agent's own tool results never enter the parent's planner prompt; only its report does."""
    prompts_seen: list[str] = []

    class Recording(ScriptedBrain):
        def decide(self, messages, schema, name):
            if name == "plan":
                prompts_seen.append("\n".join(m["content"] for m in messages))
            return super().decide(messages, schema, name)

    brain = Recording(
        plans=[
            {"action": "delegate", "task": "find the trend", "tools": ["a"], "rationale": "isolate the search"},
            call("a"),        # the sub-agent's step
            finish(),         # the sub-agent is done
            finish(),         # the parent is done
        ],
        evals=[met("the trend is up"), met("final")],
    )
    tool = _tool("a", lambda args, key: {"marker": "SUB-EVIDENCE-7"})
    _, out = run(store, goal_program("g", {"a": tool}, brain))
    assert out["result"]["status"] == "met"
    parent_last = prompts_seen[-1]
    assert "the trend is up" in parent_last, "the report is what the parent reads"
    assert "SUB-EVIDENCE-7" not in parent_last, "the sub-agent's working stays out of the parent's context"
    sub = out["result"]["evidence"][0]
    assert sub["kind"] == "subagent" and sub["status"] and sub["steps"] == 2
    assert any("SUB-EVIDENCE-7" in str(h) for h in sub["evidence"]), "the working is kept in the record for audit"


def test_a_sub_agent_cannot_delegate_again(store):
    brain = ScriptedBrain(
        plans=[
            {"action": "delegate", "task": "outer task", "tools": [], "rationale": "split"},
            {"action": "delegate", "task": "inner task", "tools": [], "rationale": "split again"},
            finish(),
            finish(),
        ],
        evals=[met("inner"), met("outer")],
    )
    _, out = run(store, goal_program("g", {"a": _tool("a")}, brain))
    assert out["result"]["status"] == "met"
    sub_steps = out["result"]["evidence"][0]["evidence"]
    assert "cannot start another sub-agent" in sub_steps[0]["error"], "depth is capped at one level"


def test_format_options_that_differ_in_the_main_word_are_distinct():
    # Live run 06 (mixed day/month dates): the planner's correct question was refused 80 times in a row because
    # 'Day-first (DD/MM/YYYY)' and 'Month-first (MM/DD/YYYY)' were judged near-duplicates. The mission never paused.
    base = {"question": "Which order does the file use?", "header": "Date order", "multiSelect": False}
    day_month = {**base, "options": [
        {"label": "Day-first (DD/MM/YYYY)", "description": "01/02/2025 = 1 Feb"},
        {"label": "Month-first (MM/DD/YYYY)", "description": "01/02/2025 = 2 Jan"}]}
    assert validate_questions({"questions": [day_month]}) == []
    # Same choice written twice is still caught.
    reworded = {**base, "options": [{"label": "Day-first (DD/MM/YYYY)", "description": ""},
                                    {"label": "Day first (DD/MM/YYYY)", "description": ""}]}
    assert validate_questions({"questions": [reworded]})
    # Qualifiers that really differ make distinct choices.
    qualified = {**base, "options": [{"label": "Units (sold)", "description": ""}, {"label": "Units (returned)", "description": ""}]}
    assert validate_questions({"questions": [qualified]}) == []


def test_repeated_refused_questions_change_the_planner_note():
    # Live run 06: the same refused question was sent 80 times. Refusals now count as failures, and the note says so.
    from prophet_agent.runtime.goal import GoalConfig, _budget_note, _failure_streak
    refused = {"kind": "ask_user", "error": "invalid questions: options are near-duplicates"}
    history = [{"kind": "tool", "result": {}}, refused, refused]
    streak, err, kind = _failure_streak(history)
    assert (streak, kind) == (2, "ask_user") and "near-duplicates" in err
    note = _budget_note(3, GoalConfig(max_iterations=20), history)
    assert "questions were refused" in note and "Do not send that question again" in note
    # A tool failure keeps the old wording.
    tool_fail = [{"kind": "tool", "error": "boom"}, {"kind": "tool", "error": "boom"}]
    assert "calls failed" in _budget_note(2, GoalConfig(max_iterations=20), tool_fail)


def test_three_rejected_finishes_in_a_row_end_as_a_partial_answer_not_a_loop(store):
    # Regression: a planner said finish 90 times and the evaluator said not_met 89 times, so the mission never ended.
    gap = {"verdict": "not_met", "reason": "workbook not shown", "answer": "The forecast is established; the workbook is not."}
    brain = ScriptedBrain(plans=[finish(), finish(), finish()], evals=[dict(gap), dict(gap), dict(gap)])
    program = goal_program("g", {}, brain)
    _, out = run(store, program)
    assert out["result"]["status"] == "partial"
    assert "established" in out["result"]["answer"] and "workbook not shown" in out["result"]["reason"]
    assert brain.calls == {"plan": 3, "eval": 3}


def test_a_tool_call_between_finishes_resets_the_rejected_finish_count(store):
    brain = ScriptedBrain(plans=[finish(), finish(), call("a"), finish()], evals=[not_met(), not_met(), met("ok")])
    program = goal_program("g", {"a": _tool("a")}, brain)
    _, out = run(store, program)
    assert out["result"]["status"] == "met"


def test_a_run_id_the_model_made_up_is_replaced_by_this_missions_run_that_wrote_the_file(tmp_path, monkeypatch):
    from prophet_agent.runtime import goal

    real = "a" * 32
    (tmp_path / real / "out").mkdir(parents=True)
    (tmp_path / real / "out" / "series.json").write_text("{}")
    monkeypatch.setattr(goal, "SANDBOX_ROOT", tmp_path)
    history = [{"kind": "tool", "tool": "sandbox_python",
                "result": {"run_id": real, "outputs": ["out/series.json (12 bytes)"]}}]
    made_up = {"horizon": 48, "source": {"run_id": "b" * 32, "file": "out/series.json", "name": "series"}}
    fixed = goal._repair_source(made_up, history)
    assert fixed["source"]["run_id"] == real and fixed["source"]["name"] == "series"
    # an id that exists on disk is left alone, a file no run of this mission wrote is not guessed
    assert goal._repair_source({"source": {"run_id": real, "file": "out/series.json"}}, history)["source"]["run_id"] == real
    other = {"source": {"run_id": "c" * 32, "file": "out/other.json"}}
    assert goal._repair_source(other, history) == other
    assert goal._repair_source({"values": [1, 2]}, history) == {"values": [1, 2]}


def test_a_wrong_series_name_resolves_when_the_file_holds_only_one_series(tmp_path, monkeypatch):
    import json

    from prophet_agent.tools import series_io

    run = "d" * 32
    (tmp_path / run / "out").mkdir(parents=True)
    (tmp_path / run / "out" / "series.json").write_text(json.dumps({"series": [1.0, 2.0, 3.0]}))
    monkeypatch.setattr(series_io, "SANDBOX_ROOT", tmp_path)
    named = series_io.resolve_series({"source": {"run_id": run, "file": "out/series.json", "name": "total_load_kw"}})
    assert [n for n, _ in named] == ["series"]
    (tmp_path / run / "out" / "two.json").write_text(json.dumps({"series": {"a": [1.0], "b": [2.0]}}))
    with pytest.raises(ValueError, match="not in the file"):
        series_io.resolve_series({"source": {"run_id": run, "file": "out/two.json", "name": "zzz"}})


def test_the_report_observation_carries_the_run_id_so_the_files_can_be_found():
    from prophet_agent.tools.report_tool import observe

    point = [1.0 + i for i in range(48)]
    result = {"forecast": {"point": point, "health": {"verdict": "ok"}, "quantiles": {"0.1": point, "0.9": point},
                           "config": {"horizon": 48}},
              "run_id": "e" * 32, "manifest": "/x/manifest.json", "files": [{"path": "out/forecast.xlsx"}]}
    seen = observe(result)
    assert seen["run_id"] == "e" * 32 and seen["files"] == ["out/forecast.xlsx"] and "manifest" in seen


# -- the finish gate: deterministic checks run on the evidence --------------------------------------------------------

@pytest.fixture(autouse=True)
def _ledger_on_disk(tmp_path, monkeypatch):
    """The finish gate reads the analyst's ledger from the sandbox folder; point it at a folder holding a good one."""
    import json
    folder = tmp_path / "sandbox" / "runledger" / "out"
    folder.mkdir(parents=True)
    (folder / "ledger.json").write_text(json.dumps([{"stage": st, "finding": "f", "decision": "d"}
                                                    for st in ("time index", "coverage", "zeros", "outliers")]))
    monkeypatch.setattr("prophet_agent.runtime.goal.SANDBOX_ROOT", tmp_path / "sandbox")


def _gate_tools():
    fc = {"config": {"horizon": 3}, "health": {"verdict": "ok"}, "point": [1.0, 2.0, 3.0], "q10": [0.5, 1.5, 2.5],
          "q90": [1.5, 2.5, 3.5]}
    bt = {"horizon": 3, "best_baseline": "seasonal_naive",
          "models": {"naive": {"mase": 1.0}, "seasonal_naive": {"mase": 0.9}, "timesfm": {"mase": 0.8}}}
    return {"sandbox_python": _tool("sandbox_python", lambda a, k: {"ok": True, "stdout": "n=40", "run_id": "runledger",
                                                                      "outputs": ["out/ledger.json (300 bytes)", "out/scan.json (900 bytes)"]}),
            "timesfm_forecast": _tool("timesfm_forecast", lambda a, k: fc),
            "backtest_series": _tool("backtest_series", lambda a, k: bt)}


def test_a_forecast_never_compared_with_baselines_is_sent_back_before_it_counts_as_met(store):
    brain = ScriptedBrain(
        plans=[call("sandbox_python"), call("timesfm_forecast"), finish(), call("backtest_series"), finish()],
        evals=[met("forecast 1, 2, 3"), met("forecast 1, 2, 3")])
    _, out = run(store, goal_program("forecast", _gate_tools(), brain))
    assert out["result"]["status"] == "met"
    assert brain.calls["plan"] == 5, "the first finish was refused and a backtest was run before the second"
    refusals = [h for h in out["result"]["evidence"] if h.get("contract")]
    assert refusals and "baseline_compared" in refusals[0]["contract"]


def test_a_check_that_keeps_failing_leaves_the_answer_labelled_partial(store):
    brain = ScriptedBrain(plans=[call("sandbox_python"), call("timesfm_forecast"), finish(), finish(), finish()],
                          evals=[met("forecast 1, 2, 3")] * 3)
    _, out = run(store, goal_program("forecast", _gate_tools(), brain))
    assert out["result"]["status"] == "partial"
    assert "baseline_compared" in out["result"]["contract_failures"]
    assert out["result"]["answer"], "the answer is still delivered, labelled"


def test_a_forecast_before_any_data_inspection_is_blocked_until_it_is_run_again_after_inspection(store):
    brain = ScriptedBrain(
        plans=[call("timesfm_forecast"), call("backtest_series"), finish(), call("sandbox_python"),
               call("timesfm_forecast"), finish()],
        evals=[met("forecast 1, 2, 3"), met("forecast 1, 2, 3")])
    _, out = run(store, goal_program("forecast", _gate_tools(), brain))
    assert out["result"]["status"] == "met"
    refusals = [h for h in out["result"]["evidence"] if h.get("contract")]
    assert refusals and "data_inspected" in refusals[0]["contract"]


def test_a_forecast_without_the_analysis_ledger_is_sent_back_until_one_is_saved(store, tmp_path, monkeypatch):
    monkeypatch.setattr("prophet_agent.runtime.goal.SANDBOX_ROOT", tmp_path / "nowhere")      # no ledger anywhere
    brain = ScriptedBrain(plans=[call("sandbox_python"), call("backtest_series"), call("timesfm_forecast"), finish(), finish(),
                                 finish()], evals=[met("forecast 1, 2, 3")] * 3)
    _, out = run(store, goal_program("forecast", _gate_tools(), brain))
    assert out["result"]["status"] == "partial"
    assert out["result"]["contract_failures"] == ["analysis_ledger"]


def test_a_figure_marked_as_approximate_matches_a_printed_one_within_one_percent_and_an_unmarked_one_does_not():
    from prophet_agent.runtime.grounding import unverified
    history = [{"kind": "tool", "result": {"mean_band": 44512.3, "max_band": 70790.1, "bias": 0.1304}}]
    assert unverified("The band is about 44,500 kW wide, up to ~70,800 kW, and the bias is roughly 13%.", history, "g") == []
    assert unverified("The band is 44,500 kW wide.", history, "g") == ["44,500"], "unmarked figures stay exact"
    assert unverified("The band is about 45,900 kW wide.", history, "g") == ["45,900"], "one percent, not any rounding"
    assert unverified("It ran from 6,600 to 27,400 units.", history, "g") == ["6,600", "27,400"], "a range is not an approximation"


def test_the_call_count_includes_calls_a_sub_agent_made():
    """Audit 2026-11-22, item 7: the planner's budget must see the work a delegated sub-agent did."""
    from prophet_agent.runtime.goal import _calls_used
    history = [{"kind": "tool", "tool": "sandbox_python", "args": {}, "result": {"ok": True}},
               {"kind": "subagent", "agent_type": "analyst", "status": "met", "answer": "done", "evidence": [
                   {"kind": "tool", "tool": "web_fetch", "args": {}, "result": {"ok": True}},
                   {"kind": "tool", "tool": "sandbox_python", "args": {}, "result": {"ok": True}}]},
               {"kind": "tool", "tool": "sandbox_python", "error": "unknown tool 'x'", "args": {}}]
    assert _calls_used(history) == 3


def test_the_research_trigger_covers_retail_demand_weather_and_power_goals():
    """Audit 2026-11-22, item 9: these series are driven by outside things and must trigger the research step."""
    from prophet_agent.runtime.goal import EXTERNAL_GOAL
    for goal in ("Forecast weekly retail sales for the next 8 weeks", "Forecast electricity demand for Victoria",
                 "Forecast rainfall for next month", "Forecast tourism arrivals next year"):
        assert EXTERNAL_GOAL.search(goal), goal
    assert not EXTERNAL_GOAL.search("Forecast the number of tickets in our help desk queue")


def test_the_outside_context_heading_is_found_at_any_level_with_or_without_colon_or_bold():
    """Audit 2026-11-22, item 8: a heading written with a colon or in bold is still the outside-context section."""
    from prophet_agent.runtime.goal import OUTSIDE_HEAD, SECTION_END
    for head in ("## Outside context", "### Outside context:", "**Outside context**", "Outside context", "#### outside context:"):
        assert OUTSIDE_HEAD.search(head), head
    assert not OUTSIDE_HEAD.search("The outside context was not checked")
    body = "**Outside context**\nIMF cut rates.\n\n**Caveats**\nNone."
    start = OUTSIDE_HEAD.search(body).end()
    section = body[start:start + SECTION_END.search(body[start:]).start()]
    assert "IMF cut rates." in section and "Caveats" not in section      # the section stops at the next bold heading
