"""Progress-aware stop gate and memory roll-up (owner, 2026-11-27: complex missions ended partial while the work was still
moving). A blocked answer goes back while the failing checks change; the same failing set twice stops it (CRITIC's rule).
Repeated rejections or a stall first turn into a change of strategy (a pivot), at most MAX_PIVOTS times."""
from __future__ import annotations

import pytest

from prophet_agent.runtime import goal as g
from prophet_agent.runtime.goal import GoalConfig, goal_program
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import Store


def _blocked(*names):
    return [{"name": n} for n in names]


def _evaluation(*names):
    return {"kind": "evaluation", "verdict": "not_met", "contract": list(names)}


def test_a_block_repeats_while_the_failing_checks_change():
    assert g._may_block_again([], _blocked("a"))
    assert g._may_block_again([_evaluation("a")], _blocked("a")), "one block with the same set is not yet unchanged twice"
    assert not g._may_block_again([_evaluation("a"), _evaluation("a")], _blocked("a"))
    assert g._may_block_again([_evaluation("a"), _evaluation("a")], _blocked("b")), "a new failing set is progress"
    assert g._may_block_again([_evaluation("a"), _evaluation("b")], _blocked("a"))


def test_blocks_stop_at_the_hard_cap_even_while_changing():
    history = [_evaluation(f"c{k}") for k in range(g.MAX_CONTRACT_BLOCKS)]
    assert not g._may_block_again(history, _blocked("new"))


def test_pivots_are_capped():
    history: list[dict] = []
    assert all(g._pivot(history, "stall", "same search") for _ in range(g.MAX_PIVOTS))
    assert not g._pivot(history, "stall")
    pivots = [h for h in history if h["kind"] == "pivot"]
    assert len(pivots) == g.MAX_PIVOTS and pivots[0]["repeated"] == "same search"
    assert "Change strategy" in pivots[0]["note"] and "lanes" in pivots[0]["note"]


def test_the_rollup_names_the_cut_steps_by_tool():
    cut = "3. web_search({'q': 'x'}) -> hits\n4. web_search({'q': 'y'})\n7. sub-agent a2: report\n9. run_python(code)\n"
    line = g._rollup(cut)
    assert "observations 3 to 9" in line and "web_search x2" in line and "sub-agent a2 x1" in line
    assert "read_observation" in line and len(line) <= g.ROLLUP_CHARS
    assert "read_observation" in g._rollup("free text with no stubs")


class Queue:
    def __init__(self, plans, evals):
        self.plans, self.evals, self.asked, self.seen = list(plans), list(evals), [], []

    def decide(self, messages, schema, name):
        self.asked.append(name)
        self.seen.append(messages[-1]["content"])
        queue = self.plans if name == "plan" else self.evals
        if not queue:
            raise AssertionError(f"no {name} scripted")
        return queue.pop(0)


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "gate.sqlite")
    yield s
    s.close()


def test_the_third_rejected_finish_pivots_before_the_write_up(store):
    tools = {"a": Tool("a", lambda args, key: {"ok": True}, idempotent=True)}
    finish = {"action": "finish", "rationale": "done"}
    no = {"verdict": "not_met", "reason": "the second source is not checked"}
    brain = Queue(plans=[finish] * 5, evals=[no] * 5)
    mid = store.create_mission("list three facts about the RBA", {})
    out = Runtime(store).run(mid, goal_program("list three facts about the RBA", tools, brain, GoalConfig(max_iterations=12)))
    assert out["status"] != "failed", out
    pivots = [h for h in out["result"]["evidence"] if h.get("kind") == "pivot"]
    assert len(pivots) == 1 and pivots[0]["why"].startswith("rejected finishes")
    assert brain.asked.count("eval") == g.MAX_REJECTED_FINISHES


def test_a_stall_pivots_before_it_stops_and_the_planner_sees_the_pivot(store):
    calls = []
    tools = {"a": Tool("a", lambda args, key: calls.append(1) or {"ok": True, "same": 1}, idempotent=True)}
    same = {"action": "call_tool", "tool": "a", "args": {"k": 1}, "rationale": "again"}
    brain = Queue(plans=[same] * 30, evals=[{"verdict": "partial", "reason": "stuck", "answer": "what was found"}] * 5)
    mid = store.create_mission("list three facts about the RBA", {})
    out = Runtime(store).run(mid, goal_program("list three facts about the RBA", tools, brain, GoalConfig(max_iterations=30)))
    assert out["status"] != "failed", out
    pivots = [h for h in out["result"]["evidence"] if h.get("kind") == "pivot"]
    assert len(pivots) == g.MAX_PIVOTS and all(p["why"] == "stall" for p in pivots)
    assert any("Change strategy" in text for text in brain.seen), "the pivot note reaches the planner"
    assert brain.asked.count("plan") < 30, "the run still stops once the pivots are spent"
