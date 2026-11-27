"""Round 7, goal audit: a repeated step with the same result is stopped early, whatever its code.

Live, 2026-11-24: four workflow runs ended partial with reason stalled ("Prophet kept repeating the same step"). GOAL.md lists the
open item "409 identical-output sandbox cells with different code": the stall key included the cell's code, so cells that printed
and wrote the same thing were never a repeat. Synthetic cells and results only. Nothing here runs the sandbox.
"""
from __future__ import annotations

import pytest

from prophet_agent.runtime import goal as G
from prophet_agent.runtime.goal import GoalConfig, goal_program
from prophet_agent.runtime.store import Store
from tests.prophet_agent.test_goal_loop import ScriptedBrain, _tool, call, met, run

GOAL = "Check the ledger was saved"
SAME_OUTPUT = "Ledger saved with 7 entries\n"
# Five cells that differ in text (layout, a comment, a variable, a split string) and print the same line.
SAME_OUTPUT_CODES = [
    "print('Ledger saved with 7 entries')",
    "x = 7\nprint('Ledger saved with 7 entries')",
    "print('Ledger saved' + ' with 7 entries')",
    "n = 3  # a note\nprint( 'Ledger saved with 7 entries' )",
    "y = 1\nprint('Ledger saved with 7 entries')  # again",
]


class RecordingBrain(ScriptedBrain):
    """A scripted brain that keeps the text of every planner prompt, so a test can read what the planner was told."""

    def __init__(self, plans=(), evals=()):
        super().__init__(plans, evals)
        self.prompts: list[str] = []

    def decide(self, messages, schema, name):
        if name == "plan":
            self.prompts.append(str(messages))
        return super().decide(messages, schema, name)


def partial(answer="found so far"):
    """An evaluator verdict that is not met and carries an answer, so a stall ends as partial."""
    return {"verdict": "not_met", "reason": "stuck", "answer": answer}


def cell(stdout_of):
    """A synthetic sandbox_python: the stdout it prints is chosen from its code, and each run gets its own run id."""
    count = {"n": 0}

    def run_cell(args, key):
        count["n"] += 1
        return {"ok": True, "returncode": 0, "timed_out": False, "stdout": stdout_of(args["code"]), "stderr": "",
                "run_id": f"{count['n']:032x}", "outputs": []}
    return _tool("sandbox_python", run_cell)


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "r7_stall.sqlite")
    yield s
    s.close()


def _sandbox_obs(code, stdout, step, stderr="", ok=True):
    """A stored sandbox observation as the loop keeps it: the planner's summary, with the step number it was given."""
    res = {"ok": ok, "returncode": 0 if ok else 1, "timed_out": False, "stdout": stdout, "stderr": stderr,
           "run_id": f"{step:032x}", "outputs": []}
    return {"kind": "tool", "tool": "sandbox_python", "args": {"code": code}, "result": res, "step": step}


# --- the bound, reproduced offline -------------------------------------------------------------------------------------

@pytest.mark.usefixtures("old_stop_gate")
def test_the_same_cell_with_the_same_output_stops_at_the_three_call_window(store):
    # The existing bound: three identical calls in a row stop the loop (stall_window = 3). Kept, and now named in the reason.
    brain = RecordingBrain(plans=[call("sandbox_python", {"code": "print(1)"})] * 5, evals=[partial()])
    _, out = run(store, goal_program(GOAL, {"sandbox_python": cell(lambda code: "1\n")}, brain,
                                     GoalConfig(max_iterations=8, stall_window=3)))
    assert brain.calls["plan"] == 3, "the loop stops on the third identical call"
    assert out["result"]["status"] == "partial"
    assert "sandbox_python" in out["result"]["reason"] and "kept repeating the same step" in out["result"]["reason"]


@pytest.mark.usefixtures("old_stop_gate")
def test_different_cells_with_the_same_output_stop_at_the_same_three_call_window(store):
    # Reproduction of the open item: the code differs on every call, the printed output does not. Before the fix nothing stopped
    # the loop before the iteration cap (six planner decisions here).
    brain = RecordingBrain(plans=[call("sandbox_python", {"code": c}) for c in SAME_OUTPUT_CODES], evals=[partial()])
    _, out = run(store, goal_program(GOAL, {"sandbox_python": cell(lambda code: SAME_OUTPUT)}, brain,
                                     GoalConfig(max_iterations=8, stall_window=3)))
    assert brain.calls["plan"] == 3, f"stopped after {brain.calls['plan']} planner decisions, not after three"
    assert out["result"]["status"] == "partial"
    assert out["result"]["reason"].startswith("The analysis kept repeating the same step (sandbox_python at step 1")


# --- the notice, one repeat before the stop -----------------------------------------------------------------------------

@pytest.mark.usefixtures("old_stop_gate")
def test_the_planner_is_told_after_one_repeat_that_the_result_already_exists(store):
    brain = RecordingBrain(plans=[call("sandbox_python", {"code": c}) for c in SAME_OUTPUT_CODES[:3]], evals=[partial()])
    run(store, goal_program(GOAL, {"sandbox_python": cell(lambda code: SAME_OUTPUT)}, brain,
                            GoalConfig(max_iterations=8, stall_window=3)))
    assert len(brain.prompts) == 3
    assert "Stall notice" not in brain.prompts[0] and "Stall notice" not in brain.prompts[1], "no repeat yet"
    notice = brain.prompts[2]
    assert "Stall notice" in notice and "already exists" in notice and "do not run this step again" in notice
    assert "sandbox_python at step 1" in notice, "the notice names the step that repeated"


def test_the_notice_names_the_next_unfinished_item_when_there_is_one():
    hist = [_sandbox_obs("print(1)", "1\n", 1), _sandbox_obs("print(1)  # b", "1\n", 2)]
    text = G._stall_notice(hist, "fit the forecast on the cleaned series")
    assert text is not None
    assert "already exists" in text and "must be done now" in text and "fit the forecast on the cleaned series" in text
    assert "must be done now" not in G._stall_notice(hist, None)
    assert G._stall_notice(hist[:1], "fit the forecast") is None, "one call is not a repeat"


# --- progress is not a stall -----------------------------------------------------------------------------------------------

def test_a_varied_sequence_that_makes_progress_is_not_flagged(store):
    # Six different cells, six different results, then the planner finishes. No notice, no stall.
    codes = [f"print({i})" for i in range(6)]
    brain = RecordingBrain(plans=[call("sandbox_python", {"code": c}) for c in codes] + [{"action": "finish", "rationale": "done"}],
                           evals=[met("all six steps done")])
    _, out = run(store, goal_program(GOAL, {"sandbox_python": cell(lambda code: f"step {code[6]} result\n")}, brain,
                                     GoalConfig(max_iterations=12, stall_window=3)))
    assert out["result"]["status"] == "met", out["result"]
    assert brain.calls["plan"] == 7
    assert not any("Stall notice" in p for p in brain.prompts)


def test_cells_that_print_different_figures_are_progress_even_with_the_same_code_shape():
    hist = [_sandbox_obs("print(total)", f"total {n}\n", n) for n in (1, 2, 3)]
    assert G._stalled(hist, 3) is False
    assert G._stall_notice(hist, None) is None


def test_a_different_output_from_a_cell_that_writes_a_different_file_is_progress():
    a = _sandbox_obs("write()", "saved\n", 1)
    b = _sandbox_obs("write()", "saved\n", 2)
    b["result"]["outputs"] = [{"path": "out/series.json", "sha256": "d2", "bytes": 10}]
    a["result"]["outputs"] = [{"path": "out/series.json", "sha256": "d1", "bytes": 10}]
    c = _sandbox_obs("write()", "saved\n", 3)
    c["result"]["outputs"] = [{"path": "out/series.json", "sha256": "d2", "bytes": 10}]
    assert G._stalled([a, b, c], 3) is False, "a new file digest is new evidence"


# --- the fingerprint ------------------------------------------------------------------------------------------------------

def test_the_outcome_ignores_run_ids_comments_and_layout_but_not_the_printed_figures():
    a = _sandbox_obs("x = 1\nprint(x)", "Ledger saved\n", 1)
    b = _sandbox_obs("x=1  # again\nprint( x )", "Ledger saved\n", 2)
    c = _sandbox_obs("x = 1\nprint(x)", "Ledger saved 8\n", 3)
    assert G._progress_key(a) == G._progress_key(b)
    assert G._progress_key(a) != G._progress_key(c)


def test_a_run_id_printed_in_the_output_does_not_make_the_same_output_new():
    a = _sandbox_obs("print(run)", "run 0123456789abcdef0123456789abcdef saved\n", 1)
    b = _sandbox_obs("print(run)  # b", "run fedcba9876543210fedcba9876543210 saved\n", 2)
    assert G._progress_key(a) == G._progress_key(b)


def test_silent_cells_say_nothing_about_progress_so_their_code_still_decides():
    # A cell that prints nothing and writes nothing gives no evidence, so the old rule (the code) stands: different code is not a stall.
    hist = [_sandbox_obs(f"x = {i}", "", i) for i in (1, 2, 3)]
    assert G._stalled(hist, 3) is False
    same = [_sandbox_obs("x = 1", "", i) for i in (1, 2, 3)]
    assert G._stalled(same, 3) is True


def test_a_pointer_carries_the_outcome_of_the_call_it_points_to():
    earlier = _sandbox_obs("print(1)", "1\n", 1)
    pointer = {"kind": "tool", "tool": "sandbox_python", "args": {"code": "print(1)"}, "repeat_of": 1,
               "sig": G._signature(earlier), "outcome": G._progress_key(earlier)}
    assert G._progress_key(pointer) == G._progress_key(earlier)
