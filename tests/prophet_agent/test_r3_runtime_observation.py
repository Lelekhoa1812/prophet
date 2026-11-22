"""Round 3, runtime goal 4 (A5-02): a stored observation is paged back with read_observation.

Before this, a result longer than OBS_CHARS was clipped for the planner with no way to read the rest: the full result sat in the
ledger and the planner could not reach it. Now a clipped observation names its step, and read_observation(step, page) returns the
full result in pages. The page reads from the stored history, so a resumed run reads the same text.
"""
from __future__ import annotations

import pytest

from prophet_agent.runtime.goal import GoalConfig, goal_program
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import Store, canonical


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "observation.sqlite")
    yield s
    s.close()


class Scripted:
    def __init__(self, plans, evals=()):
        self.plans, self.evals = list(plans), list(evals)
        self.plan_messages = []

    def decide(self, messages, schema, name):
        if name == "plan":
            self.plan_messages.append(messages[-1]["content"])
            queue = self.plans
        else:
            queue = self.evals
        if not queue:
            raise AssertionError(f"brain asked for {name} but the script is exhausted")
        return queue.pop(0)


def call(tool, args):
    return {"action": "call_tool", "tool": tool, "args": args, "rationale": "work"}


def finish():
    return {"action": "finish", "rationale": "done"}


def met(answer="ok"):
    return {"verdict": "met", "reason": "enough", "answer": answer}


def big(args, key):
    return {"ok": True, "text": "x" * 9000 + "END"}


def _run(store, brain):
    program = goal_program("analyse", {"a": Tool("a", big, idempotent=True)}, brain, GoalConfig(max_iterations=10))
    mid = store.create_mission("analyse", {"max_tool_calls": 20})
    return mid, Runtime(store, owner="r1").run(mid, program)


def test_a_clipped_observation_names_its_step_so_the_planner_can_page_it(store):
    brain = Scripted([call("a", {}), finish()], [met()])
    _run(store, brain)
    first_view = brain.plan_messages[1]
    assert "read_observation" in first_view and "step 1" in first_view, (
        "the clipped result must say how to read the rest")


def test_read_observation_returns_the_full_result_in_pages(store):
    # A page is 3000 characters, so a page plus its framing stays under the 4000-character observation cap and is not clipped again.
    brain = Scripted([call("a", {}), call("read_observation", {"step": 1, "page": 1}), finish()], [met()])
    _, out = _run(store, brain)
    ev = out["result"]["evidence"]
    page1 = ev[1]["result"]
    assert page1["step"] == 1 and page1["page"] == 1 and page1["pages"] >= 3, page1
    assert page1["next"] == 2 and len(page1["text"]) == 3000
    last = {"step": 1, "page": page1["pages"]}
    brain2 = Scripted([call("a", {}), call("read_observation", last), finish()], [met()])
    _, out2 = _run(store, brain2)
    page_last = out2["result"]["evidence"][1]["result"]
    assert "END" in page_last["text"] and "next" not in page_last, page_last
    whole = canonical(out2["result"]["evidence"][0]["result"])
    assert whole.startswith(page1["text"]) and whole.endswith(page_last["text"]), "the pages are the result, in order"


def test_reading_a_step_that_does_not_exist_is_an_error_not_a_crash(store):
    brain = Scripted([call("a", {}), call("read_observation", {"step": 99}), finish()], [met()])
    _, out = _run(store, brain)
    assert "no observation at step 99" in out["result"]["evidence"][1]["error"]


def test_a_resumed_run_reads_the_same_page_without_asking_the_brain(store):
    brain = Scripted([call("a", {}), call("read_observation", {"step": 1, "page": 2}), finish()], [met()])
    mid, first = _run(store, brain)

    class Silent:
        def decide(self, messages, schema, name):
            raise AssertionError(f"a resumed run asked for {name}")

    again = Runtime(store, owner="r2").run(mid, goal_program(
        "analyse", {"a": Tool("a", big, idempotent=True)}, Silent(), GoalConfig(max_iterations=10)))
    assert again["result"]["evidence"] == first["result"]["evidence"]
