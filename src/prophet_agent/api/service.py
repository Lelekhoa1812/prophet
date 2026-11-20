"""Mission service: starts and resumes missions on background threads and exposes their state to the HTTP layer.

Motivation vs Logic
Motivation: a mission can run for minutes. The HTTP request that starts it must return at once, and an answer to a
question must resume the same mission without the browser holding it open.
Logic: each start or resume runs ``Runtime.run`` on a daemon thread. The durable store is the source of truth; the
service keeps no mission state of its own. A second start on a mission that is already running hits the lease and
returns a conflict rather than running twice.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from prophet_agent.datasets import DatasetStore
from prophet_agent.runtime.goal import GoalConfig, goal_program
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import LeaseHeld, Store

BrainFactory = Callable[[], Any]
ToolsFactory = Callable[[dict[str, Path]], dict[str, Tool]]


class MissionService:
    def __init__(self, db_path: str | Path, brain_factory: BrainFactory, tools: dict[str, Tool],
                 goal_cfg: GoalConfig | None = None, *, tools_factory: ToolsFactory | None = None,
                 datasets: DatasetStore | None = None):
        self.store = Store(db_path)
        self.brain_factory = brain_factory
        self.tools = tools                      # the catalogue, for listings
        self.tools_factory = tools_factory      # per-mission tools (they need the mission's files); None: use `tools`
        self.datasets = datasets or DatasetStore(Path(db_path).parent / "datasets")
        self.goal_cfg = goal_cfg or GoalConfig()
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def create(self, goal: str, budget: dict | None = None, dataset_ids: list[str] | None = None) -> str:
        budget = dict(budget or {"max_tool_calls": 12, "max_seconds": 600})
        if dataset_ids:
            for did in dataset_ids:
                self.datasets.get(did)          # KeyError for an unknown id, before any mission row exists
            budget["dataset_ids"] = list(dataset_ids)
        return self.store.create_mission(goal, budget)

    def files_context(self, mission: dict) -> str:
        """The person's files by the names the tools use, so the planner never has to guess them."""
        ids = mission["budget"].get("dataset_ids", [])
        if not ids:
            return ""
        names = self.datasets.mount(ids)
        lines = [f"- {n} ({self.datasets.get(i)['bytes']:,} bytes)" for (n, _), i in zip(names.items(), ids)]
        return "Files the person attached (use these exact names; they are in in/ for analysis code):\n" + "\n".join(lines)

    def tools_for(self, mission: dict) -> dict[str, Tool]:
        if self.tools_factory is None:
            return self.tools
        return self.tools_factory(self.datasets.mount(mission["budget"].get("dataset_ids", [])))

    def start(self, mission_id: str) -> None:
        """Run (or resume) the mission on a background thread. Raises LeaseHeld if it is already running."""
        mission = self.store.mission(mission_id)
        with self._lock:
            live = self._threads.get(mission_id)
            if live is not None and live.is_alive():
                raise LeaseHeld(f"mission {mission_id} is already running")
            program = goal_program(mission["goal"], self.tools_for(mission), self.brain_factory(), self.goal_cfg,
                                   self.files_context(mission))
            runtime = Runtime(self.store, owner=f"api-{mission_id[:8]}")
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

    def list(self, limit: int = 20) -> list[dict]:
        with self.store.lock:
            rows = self.store.conn.execute(
                "SELECT id, goal, status, created, updated FROM missions ORDER BY created DESC LIMIT ?",
                (int(limit),)).fetchall()
        return [dict(r) for r in rows]
