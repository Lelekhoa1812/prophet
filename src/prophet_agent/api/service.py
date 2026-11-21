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

import logging
import threading
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from prophet_agent.datasets import DatasetStore
from prophet_agent.notebooks import Notebooks
from prophet_agent.runtime.goal import GoalConfig, goal_program
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import LeaseHeld, Store
from prophet_agent.tools.notebook_tools import NotebookBinding

BrainFactory = Callable[[], Any]
ToolsFactory = Callable[..., dict[str, Tool]]
DEFAULT_BUDGET = {"max_tool_calls": 20, "max_seconds": 900}


class MissionService:
    def __init__(self, db_path: str | Path, brain_factory: BrainFactory, tools: dict[str, Tool],
                 goal_cfg: GoalConfig | None = None, *, tools_factory: ToolsFactory | None = None,
                 datasets: DatasetStore | None = None):
        self.store = Store(db_path)
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

    def create(self, goal: str, budget: dict | None = None, dataset_ids: list[str] | None = None) -> str:
        budget = dict(budget or DEFAULT_BUDGET)
        if dataset_ids:
            for did in dataset_ids:
                self.datasets.get(did)          # KeyError for an unknown id, before any mission row exists
            budget["dataset_ids"] = list(dataset_ids)
        return self.store.create_mission(goal, budget)

    # -- notebook turns ------------------------------------------------------------------------------------------

    def create_turn(self, notebook_id: str, goal: str) -> str:
        """A new question in a notebook. Raises KeyError for an unknown notebook and NotebookBusy while one runs."""
        self.notebooks.get(notebook_id)
        last = self.store.conn.execute(
            "SELECT id FROM missions WHERE notebook_id=? ORDER BY seq DESC LIMIT 1", (notebook_id,)).fetchone()
        return self.store.create_mission(goal, dict(DEFAULT_BUDGET), notebook_id=notebook_id,
                                         parent_turn_id=last["id"] if last else None)

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

    def start(self, mission_id: str) -> None:
        """Run (or resume) the mission on a background thread. Raises LeaseHeld if it is already running."""
        mission = self.store.mission(mission_id)
        with self._lock:
            live = self._threads.get(mission_id)
            if live is not None and live.is_alive():
                raise LeaseHeld(f"mission {mission_id} is already running")
            cfg = replace(self.goal_cfg, tool_budget=mission["budget"].get("max_tool_calls"))
            program = goal_program(mission["goal"], self.tools_for(mission), self.brain_factory(), cfg,
                                   self.files_context(mission))
            # A unique owner per start: two runtimes that share an owner string would each pass the lease check.
            runtime = Runtime(self.store, owner=f"api-{uuid.uuid4().hex}")
            t = threading.Thread(target=self._run, args=(runtime, mission_id, program), daemon=True)
            self._threads[mission_id] = t
            t.start()

    def _run(self, runtime: Runtime, mission_id: str, program: Callable) -> None:
        try:
            runtime.run(mission_id, program)
        except LeaseHeld:
            return
        except Exception as exc:  # the runtime records its own failures; this guards against bugs in the service
            self.store.append_event(mission_id, "service.error", {"error": type(exc).__name__})

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
                "SELECT id, goal, status, created, updated FROM missions WHERE notebook_id IS NULL AND archived=0 "
                "ORDER BY created DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]
