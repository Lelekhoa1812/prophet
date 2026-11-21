"""Mission service: starts and resumes missions on background threads and exposes their state to the HTTP layer.

Motivation vs Logic
Motivation: a mission can run for minutes. The HTTP request that starts it must return at once, and an answer to a
question must resume the same mission without the browser holding it open.
Logic: each start or resume runs ``Runtime.run`` on a daemon thread. The durable store is the source of truth; the
service keeps no mission state of its own. A second start on a mission that is already running hits the lease and
returns a conflict rather than running twice. A notebook turn is a mission with a notebook link: its files and context
come from the notebook at the moment it starts, and it may keep tables for later turns.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from prophet_agent.cleanup import mission_run_ids, remove_datasets, remove_runs, unreferenced_datasets
from prophet_agent.datasets import DatasetStore
from prophet_agent.notebooks import Notebooks
from prophet_agent.settings import Settings
from prophet_agent.runtime.goal import GoalConfig, goal_program
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import LeaseHeld, Store
from prophet_agent.tools.notebook_tools import NotebookBinding

BrainFactory = Callable[[], Any]
ToolsFactory = Callable[..., dict[str, Tool]]
DEFAULT_BUDGET = {"max_tool_calls": 20, "max_seconds": 900}
AUTO_CONTINUE_PARTS = 2   # parts a mission may start by itself after the first; a proposal, set per mission with budget["auto_continue"]
# Statuses a person can continue or answer now from. The two legacy names are missions that stopped at a limit before
# checkpoints existed; their stored steps are intact, so they resume the same way instead of being stranded.
RESUMABLE_AT_CHECKPOINT = ("checkpoint", "budget_exhausted", "iteration_budget")


class MissionService:
    def __init__(self, db_path: str | Path, brain_factory: BrainFactory, tools: dict[str, Tool],
                 goal_cfg: GoalConfig | None = None, *, tools_factory: ToolsFactory | None = None,
                 datasets: DatasetStore | None = None, settings: Settings | None = None):
        self.store = Store(db_path)
        # When a settings store is given, its limits override goal_cfg at every start, so a change applies to the next part.
        # Without one, goal_cfg is used as passed. The fallback store keeps its files beside the database, so a test never
        # writes the real .env; it serves the settings endpoints only.
        self._use_settings = settings is not None
        self.settings = settings or Settings(data_root=Path(db_path).parent, dotenv=Path(db_path).parent / ".env")
        self.brain_factory = brain_factory
        self.tools = tools                      # the catalogue, for listings
        self.tools_factory = tools_factory      # per-mission tools (they need the mission's files); None: use `tools`
        self.datasets = datasets or DatasetStore(Path(db_path).parent / "datasets")
        self.notebooks = Notebooks(self.store, self.datasets, Path(db_path).parent / "notebooks")
        self.goal_cfg = goal_cfg or GoalConfig()
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        # Missions a previous process left running would otherwise block their notebook forever.
        self.interrupted_at_start = self.store.interrupt_orphans()
        self._stop = threading.Event()
        threading.Thread(target=self._sweep_loop, daemon=True, name="orphan-sweeper").start()

    # A restart inside a dead worker's lease window leaves its mission "running" with nobody behind it. The startup pass
    # cannot tell that from a live one, so the same check runs on a timer: once the lease lapses, the mission is marked
    # interrupted and can be resumed. Missions with a live thread here are never touched.
    SWEEP_EVERY_S = 30.0

    def sweep(self) -> list[str]:
        with self._lock:
            live = [mid for mid, t in self._threads.items() if t.is_alive()]
        return self.store.interrupt_orphans(exclude=live)

    def _sweep_loop(self) -> None:
        while not self._stop.wait(self.SWEEP_EVERY_S):
            try:
                self.sweep()
            except Exception as exc:                # a closed store at shutdown, or a busy database: try again next time
                if self._stop.is_set():
                    return
                logging.getLogger(__name__).warning("orphan sweep failed: %s", type(exc).__name__)

    # -- one-off missions ----------------------------------------------------------------------------------------

    def default_budget(self) -> dict:
        return self.settings.default_budget() if self._use_settings else dict(DEFAULT_BUDGET)

    def create(self, goal: str, budget: dict | None = None, dataset_ids: list[str] | None = None,
               model: str | None = None) -> str:
        budget = dict(budget or self.default_budget())
        if dataset_ids:
            for did in dataset_ids:
                self.datasets.get(did)          # KeyError for an unknown id, before any mission row exists
            budget["dataset_ids"] = list(dataset_ids)
        if model is not None:
            budget["model"] = model             # the catalog id the planner uses; absent means the default chain
        mission_id = self.store.create_mission(goal, budget)
        self._name(mission_id, goal)
        return mission_id

    def _name(self, mission_id: str, text: str) -> None:
        """Save a fallback name now and let the model improve it in the background."""
        from prophet_agent.api.titles import fallback_title, name_in_background

        self.set_title(mission_id, fallback_title(text), _quiet=True)
        name_in_background(text, lambda t: self.set_title(mission_id, t, _quiet=True))

    def set_title(self, mission_id: str, title: str, *, _quiet: bool = False) -> None:
        """Rename a quick forecast. Raises KeyError for an unknown one and ValueError for an empty name."""
        clean = " ".join(title.split())[:60]
        if not clean:
            raise ValueError("a name cannot be empty")
        with self.store.lock:
            cur = self.store.conn.execute("UPDATE missions SET title=? WHERE id=? AND notebook_id IS NULL", (clean, mission_id))
            if cur.rowcount == 0 and not _quiet:
                raise KeyError(mission_id)

    def set_pinned(self, mission_id: str, pinned: bool) -> None:
        with self.store.lock:
            cur = self.store.conn.execute("UPDATE missions SET pinned=? WHERE id=? AND notebook_id IS NULL", (1 if pinned else 0, mission_id))
            if cur.rowcount == 0:
                raise KeyError(mission_id)

    def delete(self, mission_id: str) -> None:
        """Permanently remove a quick forecast and its run history. A running one must be stopped first."""
        with self.store.lock:
            row = self.store.conn.execute("SELECT status, budget_json FROM missions WHERE id=? AND notebook_id IS NULL",
                                          (mission_id,)).fetchone()
            if row is None:
                raise KeyError(mission_id)
            if row["status"] in ("created", "running", "waiting_user"):
                raise ValueError("this forecast is still running; stop it first")
            dataset_ids = json.loads(row["budget_json"]).get("dataset_ids", [])
            with self.store._tx() as c:
                run_ids = mission_run_ids(c, [mission_id])
                _purge_missions(c, [mission_id])
                orphans = unreferenced_datasets(c, dataset_ids)
        # Files go after the commit: the run folder (outputs, copies of inputs) and any attached file nothing else uses.
        remove_runs(run_ids)
        remove_datasets(self.datasets.root, orphans)
        with self._lock:
            t = self._threads.get(mission_id)
            if t is None or not t.is_alive():
                self._threads.pop(mission_id, None)

    # -- notebook turns ------------------------------------------------------------------------------------------

    def create_turn(self, notebook_id: str, goal: str, model: str | None = None) -> str:
        """A new question in a notebook. Raises KeyError for an unknown notebook and NotebookBusy while one runs."""
        self.notebooks.get(notebook_id)
        last = self.store.conn.execute(
            "SELECT id FROM missions WHERE notebook_id=? ORDER BY seq DESC LIMIT 1", (notebook_id,)).fetchone()
        budget = self.default_budget()
        if model is not None:
            budget["model"] = model
        turn_id = self.store.create_mission(goal, budget, notebook_id=notebook_id,
                                            parent_turn_id=last["id"] if last else None)
        if last is None:
            self._name_notebook(notebook_id, goal)
        return turn_id

    def _name_notebook(self, notebook_id: str, goal: str) -> None:
        """Name an untitled notebook after its first question. A notebook already named (for example after its first file) keeps its name."""
        from prophet_agent.api.titles import fallback_title, name_in_background

        row = self.store.conn.execute("SELECT title FROM notebooks WHERE id=?", (notebook_id,)).fetchone()
        if row is None or row["title"] != "New notebook":
            return
        fallback = fallback_title(goal)
        with self.store._tx() as c:
            c.execute("UPDATE notebooks SET title=? WHERE id=? AND title='New notebook'", (fallback, notebook_id))

        def improve(title: str) -> None:
            # Replace the fallback only while it is still there, so a rename made in the meantime wins.
            with self.store._tx() as c:
                c.execute("UPDATE notebooks SET title=? WHERE id=? AND title=?", (title, notebook_id, fallback))

        name_in_background(goal, improve)

    # -- what a mission runs with --------------------------------------------------------------------------------

    def files_context(self, mission: dict) -> str:
        """What the planner is told before it reads the goal: the notebook's context for a turn, or the file names."""
        if mission.get("notebook_id"):
            return self.notebooks.context(mission["notebook_id"], before_seq=mission.get("seq"))
        ids = mission["budget"].get("dataset_ids", [])
        if not ids:
            return ""
        names = self.datasets.mount(ids)
        lines = [f"- {n} ({self.datasets.get(i)['bytes']:,} bytes)" for (n, _), i in zip(names.items(), ids)]
        return "Files the person attached (use these exact names; they are in in/ for analysis code):\n" + "\n".join(lines)

    def tools_for(self, mission: dict) -> dict[str, Tool]:
        if self.tools_factory is None:
            return self.tools
        nb = mission.get("notebook_id")
        if nb:
            mounts = self.notebooks.mounts(nb)
            mounts.update(self.notebooks.derived_mounts(nb))
            return self.tools_factory(mounts, NotebookBinding(self.notebooks, nb, mission["id"]))
        return self.tools_factory(self.datasets.mount(mission["budget"].get("dataset_ids", [])))

    # -- running -------------------------------------------------------------------------------------------------

    def _build_program(self, mission: dict) -> Callable:
        """The goal program for a mission as its budget stands now (segment, tool allowance, answer-now flag)."""
        budget = mission["budget"]
        base = self.settings.goal_config(self.goal_cfg) if self._use_settings else self.goal_cfg
        cfg = replace(base, tool_budget=budget.get("max_tool_calls"),
                      segment=int(budget.get("segment", 0)), finish_now=bool(budget.get("finish_now", False)))
        # A mission with a chosen model gets a brain for that model alone; without one the factory is called as before.
        model = budget.get("model")
        brain = self.brain_factory(model) if model else self.brain_factory()
        return goal_program(mission["goal"], self.tools_for(mission), brain, cfg, self.files_context(mission))

    def start(self, mission_id: str) -> None:
        """Run (or resume) the mission on a background thread. Raises LeaseHeld if it is already running."""
        mission = self.store.mission(mission_id)
        with self._lock:
            live = self._threads.get(mission_id)
            if live is not None and live.is_alive():
                raise LeaseHeld(f"mission {mission_id} is already running")
            program = self._build_program(mission)
            # A unique owner per start: two runtimes that share an owner string would each pass the lease check.
            runtime = Runtime(self.store, owner=f"api-{uuid.uuid4().hex}")
            t = threading.Thread(target=self._run, args=(runtime, mission_id, program), daemon=True)
            self._threads[mission_id] = t
            t.start()

    def _run(self, runtime: Runtime, mission_id: str, program: Callable) -> None:
        try:
            while True:
                runtime.run(mission_id, program)
                program = self._next_part(mission_id)
                if program is None:
                    return
                runtime = Runtime(self.store, owner=f"api-{uuid.uuid4().hex}")
        except LeaseHeld:
            return
        except Exception as exc:  # the runtime records its own failures; this guards against bugs in the service
            self.store.append_event(mission_id, "service.error", {"error": type(exc).__name__})

    def _next_part(self, mission_id: str) -> Callable | None:
        """A durable mission keeps going by itself: when a part ends at a checkpoint and the mission may still start parts on
        its own (``auto_continue``, default AUTO_CONTINUE_PARTS), the next part starts here with the stored steps replayed.
        None when the mission is done, waiting for the person, cancelled, or out of automatic parts. The person can still
        Continue or Answer now at the checkpoint that follows the last automatic part."""
        mission = self.store.mission(mission_id)
        if mission["status"] not in RESUMABLE_AT_CHECKPOINT:
            return None
        budget = dict(mission["budget"])
        used = int(budget.get("auto_continued", 0))
        if used >= int(budget.get("auto_continue", AUTO_CONTINUE_PARTS)):
            return None
        done = self._checklist_done(mission_id)
        if used >= 1 and done <= int(budget.get("auto_progress", -1)):
            # The last automatic part added nothing to the checklist: more of the same would only repeat failures.
            self.store.append_event(mission_id, "mission.auto_continue_stopped",
                                    {"reason": "no checklist progress in the last automatic part", "checklist_done": done})
            return None
        budget["auto_progress"] = done
        budget["auto_continued"] = used + 1
        budget["segment"] = int(budget.get("segment", 0)) + 1
        budget["max_tool_calls"] = self.store.count_tool_calls(mission_id) + int(self.default_budget()["max_tool_calls"])
        budget["finish_now"] = False
        self.store.set_budget(mission_id, budget)
        self.store.append_event(mission_id, "mission.auto_continue", {"part": budget["segment"] + 1,
                                                                      "automatic_parts_used": used + 1})
        return self._build_program(self.store.mission(mission_id))

    def _checklist_done(self, mission_id: str) -> int:
        """Checklist items done at the latest progress event (0 when the mission has none)."""
        done = 0
        for ev in self.store.events(mission_id):
            if ev["kind"] == "mission.progress":
                done = int(ev["payload"].get("checklist_done", done))
        return done

    def cancel(self, mission_id: str) -> str:
        """Ask a mission to stop. A running mission stops at its next step; one with no live runner (waiting for an
        answer, or never started) is closed here. Returns the resulting state: cancelled, cancelling, or the status of
        a mission that is no longer active."""
        mission = self.store.mission(mission_id)
        if not self.store.request_cancel(mission_id):
            return mission["status"]
        with self._lock:
            t = self._threads.get(mission_id)
            live = t is not None and t.is_alive()
        if live:
            return "cancelling"
        self.store.close_open_questions(mission_id)
        self.store.set_status(mission_id, "cancelled")
        return "cancelled"

    def answer(self, question_id: str, answers: dict) -> str:
        """Record the answer and resume the mission it belongs to. Returns the mission id."""
        info = self.store.answer_question(question_id, answers)
        self.start(info["mission_id"])
        return info["mission_id"]

    def continue_mission(self, mission_id: str) -> None:
        """Carry a checkpointed mission into its next part. The stored steps replay; the tool allowance is topped up by
        one part's worth from the calls already made, so a long session keeps going without a hard stop. Raises
        ValueError unless the mission is at a checkpoint, and LeaseHeld if it is already running."""
        mission = self.store.mission(mission_id)
        if mission["status"] not in RESUMABLE_AT_CHECKPOINT:
            raise ValueError("only a mission paused at a checkpoint can continue")
        budget = dict(mission["budget"])
        budget["segment"] = int(budget.get("segment", 0)) + 1
        budget["max_tool_calls"] = self.store.count_tool_calls(mission_id) + int(self.default_budget()["max_tool_calls"])
        budget["finish_now"] = False
        self.store.set_budget(mission_id, budget)
        self.start(mission_id)

    def answer_now(self, mission_id: str) -> None:
        """Stop gathering and write the best answer from the evidence so far (labelled partial). Only at a checkpoint."""
        mission = self.store.mission(mission_id)
        if mission["status"] not in RESUMABLE_AT_CHECKPOINT:
            raise ValueError("only a mission paused at a checkpoint can be answered now")
        budget = dict(mission["budget"])
        budget["finish_now"] = True
        self.store.set_budget(mission_id, budget)
        self.start(mission_id)

    def wait(self, mission_id: str, timeout: float | None = None) -> None:
        t = self._threads.get(mission_id)
        if t is not None:
            t.join(timeout)

    def snapshot(self, mission_id: str, after_seq: int = 0) -> dict:
        mission = self.store.mission(mission_id)
        return {
            "id": mission_id,
            "goal": mission["goal"],
            "status": mission["status"],
            "budget": mission["budget"],
            "notebook_id": mission.get("notebook_id"),
            "seq": mission.get("seq"),
            "datasets": [self.datasets.get(d) for d in mission["budget"].get("dataset_ids", []) if self._has(d)],
            "answer": mission["answer"],
            "events": self.store.events(mission_id, after_seq),
            "steps": self.store.steps(mission_id),
            "questions": self.store.open_questions(mission_id),
            "answered_questions": self.store.answered_questions(mission_id),
        }

    def _has(self, dataset_id: str) -> bool:
        try:
            self.datasets.get(dataset_id)
            return True
        except KeyError:
            return False

    def archive(self, mission_id: str, archived: bool = True) -> None:
        """Hide a finished quick forecast from the sidebar. Nothing is deleted; a running one cannot be hidden."""
        with self.store.lock:
            row = self.store.conn.execute("SELECT status FROM missions WHERE id=? AND notebook_id IS NULL", (mission_id,)).fetchone()
            if row is None:
                raise KeyError(mission_id)
            if row["status"] in ("created", "running", "waiting_user"):
                raise ValueError("this forecast is still running; stop it first")
            self.store.conn.execute("UPDATE missions SET archived=? WHERE id=?", (1 if archived else 0, mission_id))

    def list(self, limit: int = 20) -> list[dict]:
        with self.store.lock:
            rows = self.store.conn.execute(
                "SELECT id, goal, COALESCE(title, goal) AS title, status, created, updated, pinned FROM missions "
                "WHERE notebook_id IS NULL AND archived=0 ORDER BY pinned DESC, created DESC LIMIT ?", (int(limit),)).fetchall()
        return [{**dict(r), "pinned": bool(r["pinned"])} for r in rows]


def _purge_missions(c, mission_ids: list[str]) -> None:
    """Delete missions and everything that refers to them, children first. Caller holds the transaction."""
    for mid in mission_ids:
        c.execute("DELETE FROM questions WHERE mission_id=?", (mid,))
        c.execute("DELETE FROM tool_calls WHERE mission_id=?", (mid,))
        c.execute("DELETE FROM steps WHERE mission_id=?", (mid,))
        c.execute("DELETE FROM events WHERE mission_id=?", (mid,))
        c.execute("DELETE FROM missions WHERE id=?", (mid,))
