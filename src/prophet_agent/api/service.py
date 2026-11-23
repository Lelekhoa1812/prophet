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
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from prophet_agent.cleanup import mission_run_ids, remove_datasets, remove_runs, unreferenced_datasets
from prophet_agent.datasets import DatasetStore
from prophet_agent import mentions
from prophet_agent.notebooks import Notebooks
from prophet_agent.settings import Settings
from prophet_agent.runtime.goal import GoalConfig, goal_program
from prophet_agent.runtime.runner import DELEGATED_CALLS, UNCHARGED, Runtime, Tool
from prophet_agent.runtime.store import ACTIVE, LeaseHeld, LeaseLost, Store
from prophet_agent.tools.notebook_tools import NotebookBinding
from prophet_agent.api.outcome import outcome_for
from prophet_agent.api.sharing import DEFAULT_SHARING, PolicyRefused, tools_allowed
from prophet_agent.nim.providers import CATALOG, MODELS, PROVIDERS

BrainFactory = Callable[[], Any]
ToolsFactory = Callable[..., dict[str, Tool]]
DEFAULT_BUDGET = {"max_tool_calls": 20, "max_seconds": 900}
AUTO_CONTINUE_PARTS = 2   # parts a mission may start by itself after the first; a proposal, set per mission with budget["auto_continue"]
# Statuses a person can continue or answer now from. The two legacy names are missions that stopped at a limit before
# checkpoints existed; their stored steps are intact, so they resume the same way instead of being stranded.
RESUMABLE_AT_CHECKPOINT = ("checkpoint", "budget_exhausted", "iteration_budget")
# After a checkpoint caused by the language model service, the next automatic part waits this long first, doubling with each
# automatic part and capped at MODEL_SERVICE_WAIT_MAX_S. Live (2026-11-23): during a network outage both automatic parts
# restarted at once, failed again within seconds, and the missions sat at a checkpoint after the network came back.
MODEL_SERVICE_WAIT_S = 60.0
MODEL_SERVICE_WAIT_MAX_S = 300.0


class MissionService:
    def __init__(self, db_path: str | Path, brain_factory: BrainFactory, tools: dict[str, Tool],
                 goal_cfg: GoalConfig | None = None, *, tools_factory: ToolsFactory | None = None,
                 datasets: DatasetStore | None = None, settings: Settings | None = None):
        self.store = Store(db_path)
        # Per-notebook data sharing (A7-12). A row exists only once a notebook's setting differs from the default.
        with self.store._tx() as c:
            c.execute("CREATE TABLE IF NOT EXISTS notebook_sharing (notebook_id TEXT PRIMARY KEY, "
                      "web_research INTEGER NOT NULL DEFAULT 1, providers TEXT)")
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
        self.model_service_wait_s = MODEL_SERVICE_WAIT_S   # 0 in tests
        self._sleep, self._clock = time.sleep, time.monotonic
        self._threads: dict[str, threading.Thread] = {}
        # Missions whose worker has decided to exit and will start nothing more. A start may then begin a new worker while the
        # old one is still returning. Set and read only under self._lock.
        self._exiting: set[str] = set()
        self._lock = threading.Lock()
        # Missions a previous process left running would otherwise block their notebook forever.
        self.interrupted_at_start = self.store.interrupt_orphans()
        self._stop = threading.Event()
        self._sweeper = threading.Thread(target=self._sweep_loop, daemon=True, name="orphan-sweeper")
        self._sweeper.start()

    def close(self, timeout: float = 5.0) -> None:
        """Stop the orphan sweeper and wait for it to exit. Called when the API app shuts down, so the sweeper never runs
        against a store that its owner has finished with. Safe to call more than once."""
        self._stop.set()
        if self._sweeper is not threading.current_thread():
            self._sweeper.join(timeout)

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

    def _top_up(self, mission_id: str, budget: dict) -> None:
        """A new part's allowance on each lane, counted the way the runtime counts it (round 3, A5-01): the lead's charged
        calls plus one default part, and the sub-agents' calls plus their own grant. Counting every lane and read-only call
        here, as before, gave the lead an allowance that did not match the check it is held to."""
        lead = self.store.count_tool_calls(mission_id, exclude=UNCHARGED, lane="lead")
        budget["max_tool_calls"] = lead + int(self.default_budget()["max_tool_calls"])
        delegated = self.store.count_tool_calls(mission_id, exclude=UNCHARGED, lane="delegated")
        budget["max_delegated_calls"] = delegated + DELEGATED_CALLS

    def default_budget(self) -> dict:
        return self.settings.default_budget() if self._use_settings else dict(DEFAULT_BUDGET)

    def create(self, goal: str, budget: dict | None = None, dataset_ids: list[str] | None = None,
               model: str | None = None) -> str:
        budget = dict(budget or self.default_budget())
        # A file picked with @ is attached even when it was not in the attachment list; one that no longer exists is text.
        dataset_ids = self._with_mentioned_files(goal, dataset_ids)
        if dataset_ids:
            for did in dataset_ids:
                self.datasets.get(did)          # KeyError for an unknown id, before any mission row exists
            budget["dataset_ids"] = list(dataset_ids)
        if model is not None:
            budget["model"] = model             # the catalog id the planner uses; absent means the default chain
        mission_id = self.store.create_mission(goal, budget)
        self._name(mission_id, mentions.plain(goal))
        return mission_id

    def _with_mentioned_files(self, goal: str, dataset_ids: list[str] | None) -> list[str]:
        known = []
        for did in mentions.file_ids(goal):
            try:
                self.datasets.get(did)
            except KeyError:
                continue
            known.append(did)
        return mentions.merge_ids(dataset_ids or [], known)

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
        # The status and the lease are read in the same transaction as the purge, so a run that starts in another process
        # between a check and the delete cannot lose its ledger under it.
        with self.store._tx() as c:
            row = c.execute("SELECT status, lease_owner, lease_until, budget_json FROM missions "
                            "WHERE id=? AND notebook_id IS NULL", (mission_id,)).fetchone()
            if row is None:
                raise KeyError(mission_id)
            held = row["lease_owner"] is not None and row["lease_until"] > time.time()
            if row["status"] in ACTIVE or held:
                raise ValueError("this forecast is still running; stop it first")
            dataset_ids = json.loads(row["budget_json"]).get("dataset_ids", [])
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
        # The notebook's provider rule is checked before any row is written. With every provider allowed this is the model as sent.
        chosen = self._allowed_model(self.get_sharing(notebook_id), model)
        budget = self.default_budget()
        if chosen is not None:
            budget["model"] = chosen
        turn_id = self.store.create_mission(goal, budget, notebook_id=notebook_id,
                                            parent_turn_id=last["id"] if last else None)
        if last is None:
            self._name_notebook(notebook_id, mentions.plain(goal))
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
        if self.get_sharing(notebook_id)["providers"] is not None:
            return      # the naming model is the default provider, which this notebook may not allow: the fallback stays

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

    def mentions_context(self, mission: dict, tools: dict[str, Tool] | None = None) -> str:
        """What the planner is told about the skills, tools and files the person picked with / and @.

        Each pick is checked against what exists now: a playbook that is in the library, a tool the mission has, a file the
        person (or the notebook) holds. Anything else stays plain text and is never an error."""
        goal = mission["goal"]
        if not mentions.parse(goal):
            return ""
        from prophet_agent.skills import list_skills, read_skill

        titles = {s["name"]: s["name"] for s in list_skills()}
        catalogue = tools if tools is not None else self.tools_for(mission)
        nb = mission.get("notebook_id")

        def skill_body(name: str):
            if name not in titles:
                return None
            return mentions.humanise(name), read_skill(name)

        def tool_info(name: str):
            t = catalogue.get(name) or self.tools.get(name)
            return (t.title or t.name, t.doc or t.activity or t.title or t.name) if t else None

        def file_name(ident: str):
            if nb:
                try:
                    src = self.notebooks.source(ident)
                    if src["notebook_id"] == nb and not src["removed"]:
                        return src["name"]
                except KeyError:
                    pass
            try:
                return self.datasets.get(ident)["name"]
            except KeyError:
                return None

        return mentions.context_block(goal, skill_body=skill_body, tool_info=tool_info, file_name=file_name)

    def tools_for(self, mission: dict) -> dict[str, Tool]:
        """The mission's tools. A notebook with web research off never gets the research tools, whatever the factory offers."""
        tools = self._mission_tools(mission)
        nb = mission.get("notebook_id")
        return tools_allowed(tools, self.get_sharing(nb)) if nb else tools

    def _mission_tools(self, mission: dict) -> dict[str, Tool]:
        if self.tools_factory is None:
            return self.tools
        nb = mission.get("notebook_id")
        if nb:
            mounts = self.notebooks.mounts(nb)
            mounts.update(self.notebooks.derived_mounts(nb))
            return self.tools_factory(mounts, NotebookBinding(self.notebooks, nb, mission["id"]))
        return self.tools_factory(self.datasets.mount(mission["budget"].get("dataset_ids", [])))

    # -- per-notebook data sharing (A7-12) -----------------------------------------------------------------------

    def get_sharing(self, notebook_id: str) -> dict:
        """web_research (bool) and providers (sorted names, or None for every configured provider). KeyError for no notebook."""
        self.notebooks.get(notebook_id)
        with self.store.lock:
            row = self.store.conn.execute(
                "SELECT web_research, providers FROM notebook_sharing WHERE notebook_id=?", (notebook_id,)).fetchone()
        if row is None:
            return dict(DEFAULT_SHARING)
        return {"web_research": bool(row["web_research"]),
                "providers": json.loads(row["providers"]) if row["providers"] is not None else None}

    def set_sharing(self, notebook_id: str, patch: dict) -> dict:
        """Change only the keys in ``patch``. Already checked by sharing.normalise_patch. KeyError for no notebook."""
        current = self.get_sharing(notebook_id)
        current.update(patch)
        with self.store._tx() as c:
            c.execute("INSERT INTO notebook_sharing (notebook_id, web_research, providers) VALUES (?,?,?) "
                      "ON CONFLICT(notebook_id) DO UPDATE SET web_research=excluded.web_research, providers=excluded.providers",
                      (notebook_id, int(current["web_research"]),
                       None if current["providers"] is None else json.dumps(current["providers"])))
        return current

    def _allowed_model(self, sharing: dict, model: str | None) -> str | None:
        """The model a notebook turn runs on under its provider rule. With every provider allowed the choice is returned as
        sent (None stays None, so the default chain applies as before). A chosen model from a provider the notebook does not
        allow is refused. Without a choice, the first allowed provider that is turned on and ready supplies its first
        enabled model. Raises PolicyRefused."""
        allowed = sharing["providers"]
        if allowed is None:
            return model
        if model is not None:
            info = MODELS.get(model)
            if info is None:
                raise PolicyRefused("that model is not in the catalogue")
            if info.provider not in allowed:
                raise PolicyRefused(f"This notebook does not let {PROVIDERS[info.provider].label} receive its data. "
                                    "Choose a model from a provider it allows, or change the notebook's sharing settings.")
            return model
        for name in PROVIDERS:
            if name not in allowed or not self.settings.provider_enabled(name) or not self.settings.provider_ready(name):
                continue
            ids = [m.id for m in CATALOG if m.provider == name and self.settings.model_enabled(m.id)]
            if ids:
                return ids[0]
        raise PolicyRefused("No model provider this notebook allows is turned on and ready. Allow one in the notebook's "
                            "sharing settings, or set up its key in Settings.")

    def _sharing_budget(self, mission: dict) -> dict:
        """The mission's budget under its notebook's rules (a copy). Missions outside a notebook are not affected. Raises PolicyRefused."""
        budget = dict(mission["budget"])
        nb = mission.get("notebook_id")
        if nb:
            chosen = self._allowed_model(self.get_sharing(nb), budget.get("model"))
            if chosen is not None:
                budget["model"] = chosen
        return budget

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
        tools = self.tools_for(mission)
        context = "\n\n".join(p for p in (self.mentions_context(mission, tools), self.files_context(mission)) if p)
        return goal_program(mentions.plain(mission["goal"]), tools, brain, cfg, context)

    def start(self, mission_id: str) -> None:
        """Run (or resume) the mission on a background thread. Raises LeaseHeld if it is already running."""
        with self._lock:
            if self._busy(mission_id):
                raise LeaseHeld(f"mission {mission_id} is already running")
            program = self._resume_program(self.store.mission(mission_id))
            self._exiting.discard(mission_id)
            # A unique owner per start: two runtimes that share an owner string would each pass the lease check.
            runtime = Runtime(self.store, owner=f"api-{uuid.uuid4().hex}")
            t = threading.Thread(target=self._run, args=(runtime, mission_id, program), daemon=True)
            self._threads[mission_id] = t
            t.start()

    def _resume_program(self, mission: dict) -> Callable:
        """The program for the mission as it stands. The notebook's provider rule applies to every start, including a resume,
        and is checked before a thread exists. Raises PolicyRefused."""
        budget = self._sharing_budget(mission)
        if budget != mission["budget"]:
            self._store_budget(mission["id"], budget)
            mission = self.store.mission(mission["id"])
        return self._build_program(mission)

    def _busy(self, mission_id: str) -> bool:
        """True when this service's worker is running the mission, or a runtime in any process holds its lease now. The
        database answers the second part, because another process's runtime is invisible to this service's thread list.
        Caller holds self._lock."""
        t = self._threads.get(mission_id)
        if t is not None and t.is_alive() and mission_id not in self._exiting:
            return True
        with self.store.lock:
            row = self.store.conn.execute("SELECT lease_owner, lease_until FROM missions WHERE id=?", (mission_id,)).fetchone()
        return row is not None and row["lease_owner"] is not None and row["lease_until"] > time.time()

    def _refuse_if_running(self, mission_id: str) -> None:
        """Raise LeaseHeld before anything is written, when the mission is running here or in another process."""
        with self._lock:
            if self._busy(mission_id):
                raise LeaseHeld(f"mission {mission_id} is already running")

    def _store_budget(self, mission_id: str, budget: dict) -> None:
        """Write the budget through the lease fence. A lease another runtime holds is a conflict for the caller (LeaseHeld),
        not LeaseLost, which the API does not map."""
        try:
            self.store.set_budget(mission_id, budget)
        except LeaseLost:
            raise LeaseHeld(f"mission {mission_id} is being run by another runtime") from None

    def _run(self, runtime: Runtime, mission_id: str, program: Callable) -> None:
        try:
            while True:
                runtime.run(mission_id, program)
                program = self._next_part(mission_id) or self._after_pause(mission_id)
                if program is None:
                    return
                runtime = Runtime(self.store, owner=f"api-{uuid.uuid4().hex}")
        except (LeaseHeld, LeaseLost):
            return      # another runtime holds the mission now; it decides what happens next
        except Exception as exc:  # the runtime records its own failures; this guards against bugs in the service
            try:
                self.store.append_event(mission_id, "service.error", {"error": type(exc).__name__})
            except (LeaseHeld, LeaseLost):
                pass

    def _after_pause(self, mission_id: str) -> Callable | None:
        """Called by a worker whose run has stopped, just before it exits. A run that paused on a question releases the lease
        before it returns, so an answer recorded in that gap found this thread alive and was left to it: here the answer is
        resumed. Otherwise the worker retires. Both decisions are made under the lock that start() and answer() take, so an
        answer either lands before the decision (and is resumed here) or after it (and start() sees the worker retired).
        Returns the program to run next, or None after retiring."""
        me = threading.current_thread()
        with self._lock:
            mission = self.store.mission(mission_id)
            if mission["status"] == "waiting_user" and mission["cancel_requested"]:
                # A cancel arrived while the run was pausing. Nothing else will act on it, so it is closed here.
                self._close_unleased(mission_id)
            elif mission["status"] == "waiting_user" and not self.store.open_questions(mission_id):
                try:
                    return self._resume_program(mission)
                except PolicyRefused as exc:
                    self.store.append_event(mission_id, "mission.resume_refused", {"reason": str(exc)})
            if self._threads.get(mission_id) is me:
                self._exiting.add(mission_id)
        return None

    def _next_part(self, mission_id: str) -> Callable | None:
        """A durable mission keeps going by itself: when a part ends at a checkpoint and the mission may still start parts on
        its own (``auto_continue``, default AUTO_CONTINUE_PARTS), the next part starts here with the stored steps replayed.
        None when the mission is done, waiting for the person, cancelled, or out of automatic parts. The person can still
        Continue or Answer now at the checkpoint that follows the last automatic part."""
        mission = self.store.mission(mission_id)
        if mission["status"] not in RESUMABLE_AT_CHECKPOINT:
            return None
        try:
            budget = self._sharing_budget(mission)
        except PolicyRefused as exc:
            self.store.append_event(mission_id, "mission.auto_continue_stopped", {"reason": str(exc)})
            return None
        used = int(budget.get("auto_continued", 0))
        if used >= int(budget.get("auto_continue", AUTO_CONTINUE_PARTS)):
            return None
        done = self._checklist_done(mission_id)
        if used >= 1 and done <= int(budget.get("auto_progress", -1)):
            # The last automatic part added nothing to the checklist: more of the same would only repeat failures.
            self.store.append_event(mission_id, "mission.auto_continue_stopped",
                                    {"reason": "no checklist progress in the last automatic part", "checklist_done": done})
            return None
        if not self._wait_for_model_service(mission_id, used):
            return None
        budget["auto_progress"] = done
        budget["auto_continued"] = used + 1
        budget["segment"] = int(budget.get("segment", 0)) + 1
        self._top_up(mission_id, budget)
        budget["finish_now"] = False
        self._store_budget(mission_id, budget)
        self.store.append_event(mission_id, "mission.auto_continue", {"part": budget["segment"] + 1,
                                                                      "automatic_parts_used": used + 1})
        return self._build_program(self.store.mission(mission_id))

    def _wait_for_model_service(self, mission_id: str, used: int) -> bool:
        """When the part ended because the language model service failed, wait before the next automatic part, so a short
        outage does not spend the automatic parts in seconds. False when the mission left its checkpoint meanwhile (the person
        continued, answered now or cancelled), so this thread starts nothing."""
        last = next((e for e in reversed(self.store.events(mission_id)) if e["kind"] == "mission.checkpoint"), None)
        if last is None or (last.get("payload") or {}).get("which") != "model_service" or self.model_service_wait_s <= 0:
            return True
        wait = min(self.model_service_wait_s * (2 ** used), MODEL_SERVICE_WAIT_MAX_S)
        self.store.append_event(mission_id, "mission.auto_continue_waiting",
                                {"seconds": wait, "reason": "the language model service failed; waiting before trying again"})
        deadline = self._clock() + wait
        while (left := deadline - self._clock()) > 0:
            self._sleep(min(5.0, left))
            if self.store.mission(mission_id)["status"] not in RESUMABLE_AT_CHECKPOINT:
                return False
        return True

    def _checklist_done(self, mission_id: str) -> int:
        """Checklist items done at the latest progress event (0 when the mission has none)."""
        done = 0
        for ev in self.store.events(mission_id):
            if ev["kind"] == "mission.progress":
                done = int(ev["payload"].get("checklist_done", done))
        return done

    def cancel(self, mission_id: str) -> str:
        """Ask a mission to stop. A run that holds the mission (here, or in another process) stops at its next step and
        records the cancel itself; a mission that nothing holds (waiting for an answer, or never started) is closed here.
        Returns the resulting state: cancelled, cancelling, or the status of a mission that is no longer active."""
        mission = self.store.mission(mission_id)
        if not self.store.request_cancel(mission_id):
            return mission["status"]
        with self._lock:
            t = self._threads.get(mission_id)
            if t is not None and t.is_alive() and mission_id not in self._exiting:
                return "cancelling"
        return self._close_unleased(mission_id)

    def _close_unleased(self, mission_id: str) -> str:
        """Close an active mission that no runtime holds: its open questions and its status are cancelled in one transaction,
        so a runtime that takes the lease concurrently either comes before this write (and it is left alone) or after it.
        A lease held now, in any process, is left alone: its holder records the cancel at its next step. Never raises
        LeaseLost, because the write does not go through the lease fence."""
        now = time.time()
        with self.store._tx() as c:
            row = c.execute("SELECT status, lease_owner, lease_until FROM missions WHERE id=?", (mission_id,)).fetchone()
            if row is None:
                raise KeyError(mission_id)
            if row["status"] not in ACTIVE:
                return row["status"]
            if row["lease_owner"] is not None and row["lease_until"] > now:
                return "cancelling"
            c.execute("UPDATE questions SET status='cancelled' WHERE mission_id=? AND status='open'", (mission_id,))
            c.execute("UPDATE missions SET status='cancelled', updated=? WHERE id=?", (now, mission_id))
            self.store._emit(c, mission_id, "mission.status", {"status": "cancelled"})
        return "cancelled"

    def answer(self, question_id: str, answers: dict) -> str:
        """Record the answer and resume the mission it belongs to. Returns the mission id."""
        with self.store.lock:
            row = self.store.conn.execute("SELECT mission_id FROM questions WHERE id=?", (question_id,)).fetchone()
        if row is not None:
            # The notebook's provider rule is checked before the answer is recorded. A resume refused after the commit would
            # leave the answered mission waiting with nothing to resume it. Raises PolicyRefused, and nothing is written.
            self._sharing_budget(self.store.mission(row["mission_id"]))
        try:
            info = self.store.answer_question(question_id, answers)
        except LeaseLost:
            # This service still holds an old lease record for the mission, and another runtime holds it now: a conflict.
            raise LeaseHeld("the mission is being run by another runtime") from None
        mission_id = info["mission_id"]
        try:
            self.start(mission_id)
        except LeaseHeld:
            # Still being run: a worker here finishing its pause (it resumes the answer when it exits), or a runtime in another
            # process (its mark_waiting or its next replay of the question sees the answer). The answer is not lost.
            pass
        return mission_id

    def continue_mission(self, mission_id: str) -> None:
        """Carry a checkpointed mission into its next part. The stored steps replay; the tool allowance is topped up by
        one part's worth from the calls already made, so a long session keeps going without a hard stop. Raises
        ValueError unless the mission is at a checkpoint, and LeaseHeld if it is already running."""
        mission = self.store.mission(mission_id)
        if mission["status"] not in RESUMABLE_AT_CHECKPOINT:
            raise ValueError("only a mission paused at a checkpoint can continue")
        self._refuse_if_running(mission_id)
        budget = self._sharing_budget(mission)
        budget["segment"] = int(budget.get("segment", 0)) + 1
        self._top_up(mission_id, budget)
        budget["finish_now"] = False
        self._store_budget(mission_id, budget)
        self.start(mission_id)

    def answer_now(self, mission_id: str) -> None:
        """Stop gathering and write the best answer from the evidence so far (labelled partial). Only at a checkpoint."""
        mission = self.store.mission(mission_id)
        if mission["status"] not in RESUMABLE_AT_CHECKPOINT:
            raise ValueError("only a mission paused at a checkpoint can be answered now")
        self._refuse_if_running(mission_id)
        budget = self._sharing_budget(mission)
        budget["finish_now"] = True
        self._store_budget(mission_id, budget)
        self.start(mission_id)

    def wait(self, mission_id: str, timeout: float | None = None) -> None:
        t = self._threads.get(mission_id)
        if t is not None:
            t.join(timeout)

    def snapshot(self, mission_id: str, after_seq: int = 0) -> dict:
        mission = self.store.mission(mission_id)
        all_events = self.store.events(mission_id)
        return {
            "id": mission_id,
            "goal": mission["goal"],
            "status": mission["status"],
            "budget": mission["budget"],
            "notebook_id": mission.get("notebook_id"),
            "seq": mission.get("seq"),
            "datasets": [self.datasets.get(d) for d in mission["budget"].get("dataset_ids", []) if self._has(d)],
            "answer": mission["answer"],
            # Named reasons a mission stopped short (plain words), with the planner's own text under developer.
            "outcome": outcome_for(mission["status"], mission["answer"], mission["budget"], all_events),
            # The latest status line the runtime wrote: where the work stands, for the console's progress line.
            "progress": next((e["payload"] for e in reversed(all_events) if e["kind"] == "mission.progress"), None),
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
