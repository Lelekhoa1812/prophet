"""Mission checklist, memory head and status events: the planner sees what is left to do, a cut memory keeps the person's
asks, and a long run can be followed from small status events."""
from __future__ import annotations

import json

from prophet_agent import prompts
from prophet_agent.runtime import contracts
from prophet_agent.runtime.goal import ASKS_KEEP_CHARS, GoalConfig, goal_program
from tests.prophet_agent.test_goal_loop import ScriptedBrain, _tool, call, finish, met, run, store  # noqa: F401


def _sandbox_ok(run_id="r1", outputs=("out/ledger.json", "out/scan.json")):
    return {"kind": "tool", "tool": "sandbox_python", "args": {}, "result": {"run_id": run_id, "outputs": list(outputs)}}


def _ledger(tmp_path, stages, run_id="r1"):
    folder = tmp_path / run_id / "out"
    folder.mkdir(parents=True)
    (folder / "ledger.json").write_text(json.dumps([{"stage": s, "finding": "f", "decision": "d"} for s in stages]))


def test_the_checklist_starts_with_everything_to_do_and_marks_the_first_item_next(tmp_path):
    items = contracts.progress([], run_root=tmp_path)
    assert [it["status"] for it in items] == ["todo"] * len(items)
    assert [it.get("next") for it in items].count(True) == 1 and items[0].get("next")


def test_the_checklist_ticks_items_from_the_evidence_and_counts_ledger_stages(tmp_path):
    _ledger(tmp_path, ["time index", "coverage"])
    history = [_sandbox_ok()]
    items = {it["item"].split(" (")[0].split(":")[0]: it for it in contracts.progress(history, run_root=tmp_path)}
    assert items["inspect the data"]["status"] == "done"
    ledger = next(it for k, it in items.items() if k.startswith("analysis ledger"))
    assert ledger["status"] == "todo" and ledger["detail"] == f"2 of {contracts.MIN_LEDGER_STAGES} recorded"
    assert ledger.get("next"), "the first undone item is the next one"


def test_a_flat_unexpected_forecast_leaves_the_health_item_undone(tmp_path):
    _ledger(tmp_path, ["a", "b", "c", "d"])
    fc = {"kind": "tool", "tool": "timesfm_forecast", "args": {}, "result": {"health": {"verdict": "flat_unexpected"},
                                                                          "point": [1.0], "config": {"horizon": 1}}}
    items = contracts.progress([_sandbox_ok(), fc], run_root=tmp_path)
    health = next(it for it in items if "health" in it["item"])
    assert health["status"] == "todo" and "flat_unexpected" in health["detail"]


def test_files_are_on_the_list_only_when_the_goal_asks_for_them(tmp_path):
    assert not any("charts" in it["item"] for it in contracts.progress([], run_root=tmp_path))
    assert any("charts" in it["item"] for it in contracts.progress([], run_root=tmp_path, wants_files=True))


def test_render_progress_marks_boxes_and_next(tmp_path):
    text = contracts.render_progress(contracts.progress([], run_root=tmp_path))
    assert "[ ]" in text and "<- next" in text and "Mission checklist" in text


def test_the_planner_sees_the_checklist_for_a_forecast_goal_and_status_events_are_written(store):
    class Seeing(ScriptedBrain):
        def __init__(self, plans, evals):
            super().__init__(plans, evals)
            self.prompts = []

        def decide(self, messages, schema, name):
            if name == "plan":
                self.prompts.append(messages[-1]["content"])
            return super().decide(messages, schema, name)

    brain = Seeing([call("a"), finish()], [met()])
    mid, out = run(store, goal_program("Forecast next 7 days of sales", {"a": _tool("a")}, brain))
    assert "Mission checklist" in brain.prompts[0]
    events = [e for e in store.events(mid) if e["kind"] == "mission.progress"]
    assert len(events) >= 2
    payload = events[0]["payload"]
    assert payload["checklist_total"] >= 4 and payload["next"]


def test_a_goal_that_is_not_a_forecast_gets_no_checklist(store):
    class Seeing(ScriptedBrain):
        prompts: list = []

        def decide(self, messages, schema, name):
            if name == "plan":
                self.prompts.append(messages[-1]["content"])
            return super().decide(messages, schema, name)

    brain = Seeing([finish()], [met()])
    Seeing.prompts = []
    run(store, goal_program("Describe the columns of the file", {"a": _tool("a")}, brain))
    assert all("Mission checklist" not in p for p in Seeing.prompts)


def test_a_memory_cut_to_fit_keeps_the_asks_at_its_head(store):
    asks = "ASKS: " + "keep this exactly. " * 40

    class Writer(ScriptedBrain):
        def __init__(self, plans, evals):
            super().__init__(plans, evals)
            self.prompts = []

        def decide(self, messages, schema, name):
            if name == "plan":
                self.prompts.append(messages[-1]["content"])
            return super().decide(messages, schema, name)

        def write(self, messages):
            return asks + "\nFOUND: " + "filler line. " * 600

    counter = {"n": 0}

    def changing(args, key):
        counter["n"] += 1
        return {"n": counter["n"], "pad": "x" * 200}

    brain = Writer([call("a")] * 5 + [finish()], [met()])
    cfg = GoalConfig(view_chars=300, keep_recent=2, memory_chars=3000)
    _, out = run(store, goal_program("g", {"a": _tool("a", changing)}, brain, cfg))
    assert out["result"]["status"] == "met"
    last = brain.prompts[-1]
    assert "keep this exactly. keep this exactly." in last, "the head of the memory (the asks) survived the cut"
    assert "[middle of the memory cut to fit]" in last


def test_compactor_v2_has_fixed_sections_and_is_current():
    c = prompts.current("compactor")
    assert c.version == "compactor.v2"
    for head in ("ASKS:", "FOUND:", "DONE AND DECIDED:", "ERRORS:", "OPEN:", "SUB-AGENTS:", "NEXT:"):
        assert head in c.text
    assert ASKS_KEEP_CHARS > 0


def test_a_crashed_analysis_cell_counts_as_a_failure_in_the_streak():
    from prophet_agent.runtime.goal import _budget_note, _failure_streak
    crash = {"kind": "tool", "tool": "sandbox_python", "args": {}, "result": {
        "ok": False, "stderr_tail": "Traceback (most recent call last):\n  File x\nKeyError: 'message'"}}
    history = [crash, dict(crash), dict(crash)]
    streak, err, kind = _failure_streak(history)
    assert streak == 3 and "KeyError: 'message'" in err and kind == "tool"
    note = _budget_note(0, GoalConfig(), history)
    assert "print the structure" in note.replace("only prints the structure", "print the structure")
    assert _failure_streak(history + [{"kind": "tool", "tool": "sandbox_python", "result": {"ok": True}}])[0] == 0


def test_a_forecast_whose_data_was_never_scanned_is_blocked(tmp_path):
    _ledger(tmp_path, ["a", "b", "c", "d"])
    fc = {"kind": "tool", "tool": "timesfm_forecast", "args": {}, "result": {"health": {"verdict": "ok"}, "point": [1.0],
                                                                          "config": {"horizon": 1}}}
    no_scan = [_sandbox_ok(outputs=("out/ledger.json",)), fc]
    blocked = {c["name"] for c in contracts.blocking(contracts.check_evidence(no_scan, run_root=tmp_path))}
    assert "data_scanned" in blocked
    with_scan = [_sandbox_ok(outputs=("out/ledger.json", "out/scan.json (900 bytes)")), fc]
    assert "data_scanned" not in {c["name"] for c in contracts.blocking(contracts.check_evidence(with_scan, run_root=tmp_path))}


def test_compaction_folds_in_batches_not_one_observation_per_step():
    from prophet_agent.runtime import goal

    class Ctx:
        def __init__(self):
            self.steps = []

        def step(self, name, fn):
            self.steps.append(name)
            return fn(None)

    class Brain:
        def write(self, messages):
            return "ASKS: x\nFOUND: y"

    big = {"kind": "tool", "tool": "sandbox_python", "args": {"code": "x" * 3000}, "result": {"ok": True, "stdout": "y" * 3000}}
    ctx, memo = Ctx(), goal.Memo() if hasattr(goal, "Memo") else None
    history = [dict(big) for _ in range(6)]       # 6 large observations: 2 older than the newest 4, view over its limit
    goal._compact(ctx, Brain(), "goal", history, memo, "t", view_chars=20000, keep_recent=4)
    assert ctx.steps == [], "two older observations wait for a batch while the view is not far over"
    history += [dict(big) for _ in range(2)]      # now 4 older observations
    goal._compact(ctx, Brain(), "goal", history, memo, "t", view_chars=20000, keep_recent=4)
    assert len(ctx.steps) == 1 and memo.covered == 4


def test_external_series_checklist_asks_for_outside_context_until_a_researcher_or_web_tool_ran(tmp_path):
    items = contracts.progress([], run_root=tmp_path, external=True)
    ctx = next(it for it in items if it["item"].startswith("outside context"))
    assert ctx["status"] == "todo"
    assert not any(it["item"].startswith("outside context") for it in contracts.progress([], run_root=tmp_path))
    for done in ({"kind": "subagent", "agent_type": "researcher"},
                 {"kind": "tool", "tool": "web_search", "args": {}, "error": "offline"}):   # an attempt counts
        items = contracts.progress([done], run_root=tmp_path, external=True)
        assert next(it for it in items if it["item"].startswith("outside context"))["status"] == "done"
    assert contracts.researched([{"kind": "subagent", "agent_type": "analyst"}]) is False


def test_calibrate_accepts_the_season_argument_the_backtest_takes():
    from prophet_agent.tools.calibrate_tool import ALLOWED
    assert "season" in ALLOWED


def test_playbook_and_note_reads_are_not_charged_to_the_call_budget(tmp_path):
    from prophet_agent.runtime.goal import _calls_used
    from prophet_agent.runtime.store import Store
    hist = [{"kind": "tool", "tool": "analyst_skill", "args": {}, "result": {}},
            {"kind": "tool", "tool": "notes", "args": {}, "result": {}},
            {"kind": "tool", "tool": "sandbox_python", "args": {}, "result": {}}]
    assert _calls_used(hist) == 1
    store = Store(tmp_path / "s.sqlite")
    m1 = store.create_mission("goal", {})
    for i, (tool, args) in enumerate([("analyst_skill", {"name": "a"}), ("analyst_skill", {"name": "b"}), ("sandbox_python", {"code": "x"})]):
        store.tool_start(m1, i, tool, args, f"k{i}")
    assert store.count_tool_calls(m1) == 3
    assert store.count_tool_calls(m1, exclude=("analyst_skill",)) == 1
    assert [a["name"] for a in store.tool_args(m1, "analyst_skill")] == ["a", "b"]
