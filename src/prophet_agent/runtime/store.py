"""Durable state for missions: SQLite in WAL mode with an event log, a step ledger, a tool-call ledger and questions.

Motivation vs Logic
Motivation: a mission can run for minutes, call paid models and tools, and be interrupted by a crash, a closed laptop
or a stopped process. Restarting must never repeat a side effect that already happened, and must never ask the model
to re-decide a step whose decision was recorded.
Logic: every decision and every tool call is written before the runtime moves on. Steps have a deterministic index,
so replay finds the recorded output by (mission, index). Tool calls are keyed by a hash of (mission, step, tool,
canonical arguments); a call already marked done returns its stored result and is never executed again. A call
left in "started" by a crash is re-run only if the tool declares itself idempotent; otherwise the runtime stops and
reports the outcome as unknown, rather than guessing. One runtime holds the mission lease at a time.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2

_DDL = """
CREATE TABLE IF NOT EXISTS missions (
  id TEXT PRIMARY KEY,
  goal TEXT NOT NULL,
  status TEXT NOT NULL,
  budget_json TEXT NOT NULL,
  created REAL NOT NULL,
  updated REAL NOT NULL,
  lease_owner TEXT,
  lease_until REAL NOT NULL DEFAULT 0,
  answer_json TEXT
);
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  mission_id TEXT NOT NULL REFERENCES missions(id),
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS events_by_mission ON events(mission_id, seq);
CREATE TABLE IF NOT EXISTS steps (
  mission_id TEXT NOT NULL REFERENCES missions(id),
  idx INTEGER NOT NULL,
  name TEXT NOT NULL,
  status TEXT NOT NULL,
  output_json TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  updated REAL NOT NULL,
  PRIMARY KEY (mission_id, idx)
);
CREATE TABLE IF NOT EXISTS tool_calls (
  idem_key TEXT PRIMARY KEY,
  mission_id TEXT NOT NULL REFERENCES missions(id),
  step_idx INTEGER NOT NULL,
  tool TEXT NOT NULL,
  args_json TEXT NOT NULL,
  status TEXT NOT NULL,
  result_json TEXT,
  error TEXT,
  updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS questions (
  id TEXT PRIMARY KEY,
  mission_id TEXT NOT NULL REFERENCES missions(id),
  tool_key TEXT NOT NULL REFERENCES tool_calls(idem_key),
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL,
  answers_json TEXT,
  created REAL NOT NULL,
  answered REAL
);
CREATE TABLE IF NOT EXISTS agent_notes (
  mission_id TEXT NOT NULL REFERENCES missions(id),
  scope TEXT NOT NULL,
  name TEXT NOT NULL,
  text TEXT NOT NULL,
  updated REAL NOT NULL,
  PRIMARY KEY (mission_id, scope, name)
);
CREATE TABLE IF NOT EXISTS lane_board (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  mission_id TEXT NOT NULL REFERENCES missions(id),
  sender TEXT NOT NULL,
  recipient TEXT NOT NULL,
  kind TEXT NOT NULL,
  topic TEXT NOT NULL DEFAULT '',
  text TEXT NOT NULL,
  idem_key TEXT UNIQUE,
  created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS lane_board_by_mission ON lane_board(mission_id, seq);
"""


# Schema v2 adds notebooks: persistent workspaces whose turns are missions. v1 databases are upgraded in place; their
# missions keep notebook_id NULL and run exactly as before.
_DDL_V2 = """
CREATE TABLE IF NOT EXISTS notebooks (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  instructions TEXT NOT NULL DEFAULT '',
  archived INTEGER NOT NULL DEFAULT 0,
  created REAL NOT NULL,
  updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY,
  notebook_id TEXT NOT NULL REFERENCES notebooks(id),
  dataset_id TEXT NOT NULL,
  name TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  bytes INTEGER NOT NULL,
  status TEXT NOT NULL,
  card_json TEXT,
  error TEXT,
  included INTEGER NOT NULL DEFAULT 1,
  removed INTEGER NOT NULL DEFAULT 0,
  created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS sources_by_notebook ON sources(notebook_id);
CREATE TABLE IF NOT EXISTS derived_tables (
  id TEXT PRIMARY KEY,
  notebook_id TEXT NOT NULL REFERENCES notebooks(id),
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  created REAL NOT NULL,
  UNIQUE(notebook_id, name)
);
CREATE TABLE IF NOT EXISTS derived_versions (
  id TEXT PRIMARY KEY,
  table_id TEXT NOT NULL REFERENCES derived_tables(id),
  version INTEGER NOT NULL,
  rows INTEGER NOT NULL,
  columns_json TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  file_path TEXT NOT NULL,
  lineage_json TEXT NOT NULL,
  turn_id TEXT,
  created REAL NOT NULL,
  UNIQUE(table_id, version)
);
"""
_ORGANISER_COLUMNS = {
    "missions": {"title": "TEXT", "pinned": "INTEGER NOT NULL DEFAULT 0",
                 # Checkpoints (docs/prophet/fork_rewind_copy.md). A rewind flags the turns it drops with the rewind's id and
                 # deletes nothing; a fork records the turn it was copied from. Readers that build a conversation skip flagged rows.
                 "dropped_by": "TEXT", "forked_from": "TEXT"},
    "notebooks": {"pinned": "INTEGER NOT NULL DEFAULT 0", "forked_from": "TEXT", "forked_at_seq": "INTEGER"},
    "derived_versions": {"dropped_by": "TEXT"},
    # The profiling run's id, so deleting the notebook removes that run folder with the rest.
    "sources": {"run_id": "TEXT"},
}

_MISSION_COLUMNS_V2 = {
    "notebook_id": "TEXT",
    "seq": "INTEGER",
    "parent_turn_id": "TEXT",
    "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
    "archived": "INTEGER NOT NULL DEFAULT 0",
    # The answer text while a free-text reply is still arriving. Not a step result: replay never reads it.
    "draft_text": "TEXT",
}
# One row per Rewind: which turn the conversation went back to, the text offered back as a draft, what was hidden, and
# whether Undo has restored it. A quick forecast (no notebook) is recorded with its mission id as ``to_turn_id``.
_DDL_REWINDS = """
CREATE TABLE IF NOT EXISTS rewinds (
  id TEXT PRIMARY KEY,
  notebook_id TEXT,
  to_turn_id TEXT NOT NULL,
  to_seq INTEGER,
  goal TEXT NOT NULL,
  dropped_json TEXT NOT NULL,
  tables_json TEXT NOT NULL DEFAULT '[]',
  created REAL NOT NULL,
  undone REAL
);
CREATE INDEX IF NOT EXISTS rewinds_by_notebook ON rewinds(notebook_id, created);
"""
ACTIVE = ("created", "running", "waiting_user")
# A turn that is "created" with no lease has not been started yet. The sweep leaves it alone until it has waited this long
# (round 6m): a fresh turn is about to start, and freeing its notebook would let a second turn begin beside it.
STARTUP_GRACE_S = 300.0


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def purge_missions(c: sqlite3.Connection, mission_ids: list[str]) -> None:
    """Delete these missions and every row that refers to them. The caller holds the transaction.

    A question refers to a tool call, so questions go first. Agent notes refer to the mission and must go before it.
    """
    for mid in mission_ids:
        c.execute("DELETE FROM questions WHERE mission_id=?", (mid,))
        c.execute("DELETE FROM tool_calls WHERE mission_id=?", (mid,))
        c.execute("DELETE FROM steps WHERE mission_id=?", (mid,))
        c.execute("DELETE FROM events WHERE mission_id=?", (mid,))
        c.execute("DELETE FROM agent_notes WHERE mission_id=?", (mid,))
        c.execute("DELETE FROM lane_board WHERE mission_id=?", (mid,))
        c.execute("DELETE FROM missions WHERE id=?", (mid,))


def idem_key(mission_id: str, step_idx: int, tool: str, args: Any) -> str:
    raw = f"{mission_id}:{step_idx}:{tool}:{canonical(args)}"
    return hashlib.sha256(raw.encode()).hexdigest()


class LeaseHeld(RuntimeError):
    """Another runtime holds the mission lease and it has not expired."""


class LeaseLost(RuntimeError):
    """This runtime's lease on a mission expired and was taken over. It must stop without writing further results."""


class NotebookBusy(RuntimeError):
    """A turn was started in a notebook that already has an active turn."""


class OutcomeUnknown(RuntimeError):
    """A non-idempotent tool started, the process died, and its outcome is not recorded."""


@dataclass(frozen=True)
class ToolRecord:
    idem_key: str
    status: str                 # started | waiting | done | failed
    result: Any
    error: str | None


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False, timeout=10.0)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        # Leases this process holds, by mission: the owner that last acquired each mission through this store (see _fence).
        self._owned: dict[str, str] = {}
        # The runtime on whose behalf the current call runs. Set only by LeaseBound, under the lock, for the length of one call.
        self._caller: str | None = None
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            found = int(self.conn.execute("PRAGMA user_version").fetchone()[0])
            if found > SCHEMA_VERSION:
                raise RuntimeError(f"database schema {found} is newer than this version of Prophet ({SCHEMA_VERSION})")
            self.conn.executescript(_DDL)
            have = {r["name"] for r in self.conn.execute("PRAGMA table_info(missions)")}
            for col, decl in _MISSION_COLUMNS_V2.items():
                if col not in have:
                    self.conn.execute(f"ALTER TABLE missions ADD COLUMN {col} {decl}")
            self.conn.executescript(_DDL_V2)
            self.conn.executescript(_DDL_REWINDS)
            # Organiser columns (sidebar title and pin) for quick forecasts and notebooks. Guarded, so older databases upgrade in place.
            for table, cols in _ORGANISER_COLUMNS.items():
                present = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
                for col, decl in cols.items():
                    if col not in present:
                        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
            # Round 3 (A5-01): each tool call belongs to the lead agent or to a delegated sub-agent, so the two allowances are
            # counted apart. Existing rows are labelled from their step names ("act:d1.*" is a sub-agent's step).
            present = {r["name"] for r in self.conn.execute("PRAGMA table_info(tool_calls)")}
            if "lane" not in present:
                self.conn.execute("ALTER TABLE tool_calls ADD COLUMN lane TEXT NOT NULL DEFAULT 'lead'")
                self.conn.execute("UPDATE tool_calls SET lane='delegated' WHERE EXISTS (SELECT 1 FROM steps s "
                                  "WHERE s.mission_id=tool_calls.mission_id AND s.idx=tool_calls.step_idx "
                                  "AND s.name LIKE 'act:d1.%')")
            self.conn.execute("CREATE INDEX IF NOT EXISTS missions_by_notebook ON missions(notebook_id, seq)")
            self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def _tx(self):
        store = self

        class _Tx:
            def __enter__(self):
                store.lock.acquire()
                try:
                    store.conn.execute("BEGIN IMMEDIATE")
                except BaseException:
                    store.lock.release()
                    raise
                return store.conn

            def __exit__(self, exc_type, exc, tb):
                try:
                    if exc_type:
                        store.conn.execute("ROLLBACK")
                    else:
                        try:
                            store.conn.execute("COMMIT")
                        except BaseException:
                            # A failed COMMIT (disk full, I/O error, busy) leaves the transaction open; roll it back so
                            # the shared connection is not wedged for every later write.
                            if store.conn.in_transaction:
                                store.conn.execute("ROLLBACK")
                            raise
                finally:
                    store.lock.release()
                return False

        return _Tx()

    # -- missions and leases -----------------------------------------------------------------------------------

    def create_mission(self, goal: str, budget: dict, *, notebook_id: str | None = None,
                       parent_turn_id: str | None = None) -> str:
        """A mission is one turn. In a notebook it gets the next sequence number, and only one turn may be active."""
        mission_id = uuid.uuid4().hex
        now = time.time()
        with self._tx() as c:
            seq = None
            if notebook_id is not None:
                busy = c.execute("SELECT 1 FROM missions WHERE notebook_id=? AND dropped_by IS NULL AND status IN (?,?,?)",
                                 (notebook_id, *ACTIVE)).fetchone()
                if busy:
                    raise NotebookBusy("this notebook is still working on the previous message")
                # Dropped turns keep their numbers; the next turn takes the number after the last one still in the conversation.
                seq = int(c.execute("SELECT COALESCE(MAX(seq),0)+1 FROM missions WHERE notebook_id=? AND dropped_by IS NULL",
                                    (notebook_id,)).fetchone()[0])
            c.execute("INSERT INTO missions (id, goal, status, budget_json, created, updated, notebook_id, seq, "
                      "parent_turn_id) VALUES (?,?,?,?,?,?,?,?,?)",
                      (mission_id, goal, "created", canonical(budget), now, now, notebook_id, seq, parent_turn_id))
            self._emit(c, mission_id, "mission.created", {"goal": goal, "budget": budget, "notebook_id": notebook_id})
        return mission_id

    def request_cancel(self, mission_id: str) -> bool:
        """Ask a running mission to stop at its next step. False if it is not active."""
        with self._tx() as c:
            cur = c.execute("UPDATE missions SET cancel_requested=1, updated=? WHERE id=? AND status IN (?,?,?)",
                            (time.time(), mission_id, *ACTIVE))
            return cur.rowcount == 1

    def is_cancel_requested(self, mission_id: str) -> bool:
        with self.lock:
            row = self.conn.execute("SELECT cancel_requested FROM missions WHERE id=?", (mission_id,)).fetchone()
        return bool(row and row["cancel_requested"])

    def close_open_questions(self, mission_id: str) -> None:
        with self._tx() as c:
            self._fence(c, mission_id)
            c.execute("UPDATE questions SET status='cancelled' WHERE mission_id=? AND status='open'", (mission_id,))

    def mission(self, mission_id: str) -> dict:
        with self.lock:
            row = self.conn.execute("SELECT * FROM missions WHERE id=?", (mission_id,)).fetchone()
        if row is None:
            raise KeyError(f"no mission {mission_id}")
        out = dict(row)
        out["budget"] = json.loads(out.pop("budget_json"))
        out["answer"] = json.loads(out.pop("answer_json") or "null")
        return out

    def set_status(self, mission_id: str, status: str) -> None:
        with self._tx() as c:
            self._fence(c, mission_id)
            c.execute("UPDATE missions SET status=?, updated=? WHERE id=?", (status, time.time(), mission_id))
            self._emit(c, mission_id, "mission.status", {"status": status})

    def set_budget(self, mission_id: str, budget: dict) -> None:
        with self._tx() as c:
            self._fence(c, mission_id)
            c.execute("UPDATE missions SET budget_json=?, updated=? WHERE id=?",
                      (canonical(budget), time.time(), mission_id))

    def set_answer(self, mission_id: str, answer: Any) -> None:
        with self._tx() as c:
            self._fence(c, mission_id)
            c.execute("UPDATE missions SET answer_json=?, draft_text=NULL, updated=? WHERE id=?",
                      (canonical(answer), time.time(), mission_id))

    def set_draft(self, mission_id: str, text: str) -> None:
        """Replace the text the console shows while a free-text answer is still being written.

        Empty text clears it. This is not a step and not an event: a resumed run does not read it, and it does not
        change the mission's updated time. ``set_answer`` clears it when the answer is stored.
        """
        with self._tx() as c:
            self._fence(c, mission_id)
            c.execute("UPDATE missions SET draft_text=? WHERE id=?", (text or None, mission_id))

    def _fence(self, c: sqlite3.Connection, mission_id: str) -> None:
        """Refuse a write to a mission whose lease this runtime no longer holds.

        Motivation (audit: runtime store.py:409-437, HIGH): writes did not check the lease owner, so a runtime whose lease
        had been taken over kept writing events, steps and tool results beside the new owner.
        Round 6m: the check compared the database owner with the owner that last acquired the mission through this store,
        which says nothing about who is writing. A stalled runtime on the same store passed once another runtime had
        taken the mission over; and a sweep that dropped the record unfenced a lapsed runtime altogether.
        Logic: ``_owned`` names the owner that holds the mission on this store. A write made by a runtime (``_caller`` set
        by LeaseBound) must come from that owner, and the database must still name it, both read in the write's own
        transaction. A write from outside the runner (the service's cancel, a clear) is refused only while a different
        runtime holds the lease; with no lease, or none held by anyone, it goes through as before.
        """
        expected = self._owned.get(mission_id)
        caller = self._caller
        if expected is None:
            if caller is None:
                return                                   # not leased through this store: an outside write, unchanged
            raise LeaseLost(f"mission {mission_id} is not leased to this runtime")
        row = c.execute("SELECT lease_owner, lease_until FROM missions WHERE id=?", (mission_id,)).fetchone()
        if row is None:
            raise LeaseLost(f"mission {mission_id} no longer exists")
        holder = row["lease_owner"]
        if caller is None:
            # A lapsed lease that another runtime took and then left is not a live holder: the record still names it
            # until a sweep clears it, but an outside write (cancel, budget) must not be refused by a stale name.
            live_foreign = holder is not None and holder != expected and row["lease_until"] > time.time()
            if live_foreign:
                raise LeaseLost(f"mission {mission_id} lease was taken over by another runtime")
            return
        if caller != expected or holder != expected:
            raise LeaseLost(f"mission {mission_id} lease was taken over by another runtime")

    def _fence_key(self, c: sqlite3.Connection, key: str) -> None:
        """The fence for a write keyed by a tool call: the mission is the one the call belongs to."""
        row = c.execute("SELECT mission_id FROM tool_calls WHERE idem_key=?", (key,)).fetchone()
        if row is not None:
            self._fence(c, row["mission_id"])

    def acquire_lease(self, mission_id: str, owner: str, ttl_s: float) -> None:
        now = time.time()
        with self._tx() as c:
            cur = c.execute(
                "UPDATE missions SET lease_owner=?, lease_until=? WHERE id=? "
                "AND (lease_owner IS NULL OR lease_owner=? OR lease_until<?)",
                (owner, now + ttl_s, mission_id, owner, now))
            if cur.rowcount != 1:
                raise LeaseHeld(f"mission {mission_id} is leased by another runtime")
            self._owned[mission_id] = owner

    def renew_lease(self, mission_id: str, owner: str, ttl_s: float) -> None:
        """Extend the lease. Raises LeaseLost when this runtime no longer holds it (it expired and another runtime
        took the mission), so the old runtime stops instead of running on beside the new one."""
        with self._tx() as c:
            cur = c.execute("UPDATE missions SET lease_until=? WHERE id=? AND lease_owner=?",
                            (time.time() + ttl_s, mission_id, owner))
            if cur.rowcount != 1:
                raise LeaseLost(f"mission {mission_id} lease was taken over by another runtime")

    def interrupt_orphans(self, exclude: tuple[str, ...] | list[str] = ()) -> list[str]:
        """At startup: a mission left as created or running by a process that died, whose lease has lapsed, is
        marked interrupted. That frees its notebook (active turns block new ones) and lets it be started again;
        a non-idempotent call left started is still stopped as outcome_unknown when the mission resumes.
        ``exclude`` names missions a live worker in this process is running, which are never touched."""
        now = time.time()
        with self._tx() as c:
            # Running with a lapsed or missing lease: its runtime died. Created: a lease that lapsed (a start that died after
            # taking it), or no lease and older than the startup grace (a turn whose start never came).
            rows = c.execute(
                "SELECT id FROM missions WHERE (status='running' AND (lease_owner IS NULL OR lease_until<?)) "
                "OR (status='created' AND ((lease_owner IS NULL AND updated<?) OR (lease_owner IS NOT NULL AND lease_until<?)))",
                (now, now - STARTUP_GRACE_S, now)).fetchall()
            ids = [r["id"] for r in rows if r["id"] not in exclude]
            for mid in ids:
                c.execute("UPDATE missions SET status='interrupted', lease_owner=NULL, lease_until=0, updated=? "
                          "WHERE id=?", (now, mid))
                self._emit(c, mid, "mission.status", {"status": "interrupted",
                                                      "reason": "the runtime stopped before this mission finished"})
        # The record of who held the lease is kept: a runtime that stalled past its lease is still fenced by the database
        # check in _fence, and must not regain the mission's writes because a sweep ran on this store (round 6m).
        return ids

    def release_lease(self, mission_id: str, owner: str) -> None:
        with self._tx() as c:
            c.execute("UPDATE missions SET lease_owner=NULL, lease_until=0 WHERE id=? AND lease_owner=?",
                      (mission_id, owner))
            if self._owned.get(mission_id) == owner:
                del self._owned[mission_id]

    # -- event log ---------------------------------------------------------------------------------------------

    def append_event(self, mission_id: str, kind: str, payload: dict) -> int:
        with self._tx() as c:
            self._fence(c, mission_id)
            return self._emit(c, mission_id, kind, payload)

    def append_event_once(self, mission_id: str, kind: str, payload: dict, key: str) -> int | None:
        """Write the event unless one of the same kind already carries the same ``payload[key]``. Check and write share a
        transaction, so two threads cannot both write it. Returns the new seq, or None when it was already there."""
        with self._tx() as c:
            self._fence(c, mission_id)
            for (old,) in c.execute("SELECT payload_json FROM events WHERE mission_id=? AND kind=?", (mission_id, kind)):
                if json.loads(old).get(key) == payload.get(key):
                    return None
            return self._emit(c, mission_id, kind, payload)

    def _emit(self, c: sqlite3.Connection, mission_id: str, kind: str, payload: dict) -> int:
        """Write an event inside the caller's transaction, so a state change and its event commit or fail together (round
        6m). A crash between the two used to leave a finished step with no event, or a waiting mission with no record."""
        cur = c.execute("INSERT INTO events (mission_id, kind, payload_json, ts) VALUES (?,?,?,?)",
                        (mission_id, kind, canonical(payload), time.time()))
        return int(cur.lastrowid)

    def events(self, mission_id: str, after_seq: int = 0) -> list[dict]:
        with self.lock:
            rows = self.conn.execute("SELECT seq, kind, payload_json, ts FROM events "
                                     "WHERE mission_id=? AND seq>? ORDER BY seq", (mission_id, after_seq)).fetchall()
        return [{"seq": r["seq"], "kind": r["kind"], "payload": json.loads(r["payload_json"]), "ts": r["ts"]}
                for r in rows]

    # -- steps -------------------------------------------------------------------------------------------------

    def step_output(self, mission_id: str, idx: int) -> Any | None:
        """Return the stored output of a finished step, or None if it has not finished."""
        with self.lock:
            row = self.conn.execute("SELECT status, output_json FROM steps WHERE mission_id=? AND idx=?",
                                    (mission_id, idx)).fetchone()
        if row is None or row["status"] != "done":
            return None
        return json.loads(row["output_json"])

    def step_record(self, mission_id: str, idx: int) -> dict | None:
        """The stored step at this position: name, status and output (``v``), or None if the position is empty."""
        with self.lock:
            row = self.conn.execute("SELECT name, status, output_json FROM steps WHERE mission_id=? AND idx=?",
                                    (mission_id, idx)).fetchone()
        if row is None:
            return None
        out = json.loads(row["output_json"]) if row["output_json"] else {}
        return {"name": row["name"], "status": row["status"], "v": out.get("v")}

    def steps(self, mission_id: str) -> list[dict]:
        with self.lock:
            rows = self.conn.execute("SELECT idx, name, status, output_json FROM steps WHERE mission_id=? "
                                     "ORDER BY idx", (mission_id,)).fetchall()
        return [{"idx": r["idx"], "name": r["name"], "status": r["status"],
                 "output": (json.loads(r["output_json"]) or {}).get("v") if r["output_json"] else None}
                for r in rows]

    def finish_step(self, mission_id: str, idx: int, name: str, output: Any) -> bool:
        """Record a finished step with its event, in one transaction. The first finished result at a position is the one
        kept: a later write there does not replace it (round 6m), so every replay returns what the ledger holds. Returns
        True when this call recorded the step, False when the position was already finished."""
        with self._tx() as c:
            self._fence(c, mission_id)
            cur = c.execute("INSERT INTO steps (mission_id, idx, name, status, output_json, attempts, updated) "
                            "VALUES (?,?,?,?,?,?,?) ON CONFLICT(mission_id, idx) DO NOTHING",
                            (mission_id, idx, name, "done", canonical(output), 1, time.time()))
            if cur.rowcount != 1:
                return False
            self._emit(c, mission_id, "step.done", {"idx": idx, "name": name})
            return True

    # -- tool calls --------------------------------------------------------------------------------------------

    def tool_get(self, key: str) -> ToolRecord | None:
        with self.lock:
            row = self.conn.execute("SELECT idem_key, status, result_json, error FROM tool_calls WHERE idem_key=?",
                                    (key,)).fetchone()
        if row is None:
            return None
        return ToolRecord(row["idem_key"], row["status"], json.loads(row["result_json"] or "null"), row["error"])

    def tool_start(self, mission_id: str, step_idx: int, tool: str, args: Any, key: str, lane: str = "lead") -> None:
        """Record that a call is starting. Never downgrades a call that already has a result (DO NOTHING on conflict).
        ``lane`` is "lead" for the lead agent's call and "delegated" for a sub-agent's (each has its own allowance)."""
        with self._tx() as c:
            self._fence(c, mission_id)
            c.execute("INSERT INTO tool_calls (idem_key, mission_id, step_idx, tool, args_json, status, updated, lane) "
                      "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(idem_key) DO NOTHING",
                      (key, mission_id, step_idx, tool, canonical(args), "started", time.time(), lane))

    def tool_set(self, key: str, status: str, result: Any = None, error: str | None = None) -> None:
        """Record a call's outcome. A call marked done is final: a later write cannot downgrade it to failed or replace its
        result (round 6m), so a replay always returns the first result the tool gave."""
        with self._tx() as c:
            self._fence_key(c, key)
            c.execute("UPDATE tool_calls SET status=?, result_json=?, error=?, updated=? "
                      "WHERE idem_key=? AND status!='done'",
                      (status, canonical(result) if result is not None else None, error, time.time(), key))

    def wipe_ledger(self) -> dict[str, int]:
        """Delete every row of the ledger, children before parents, in one transaction. Returns the rows removed per table.
        The caller must have checked that no mission is running; the schema itself is kept."""
        order = ("events", "steps", "tool_calls", "questions", "agent_notes", "lane_board", "sources", "derived_versions", "derived_tables",
                 "rewinds", "missions", "notebooks")
        with self._tx() as c:
            # Checked again here, inside the wipe's transaction: a mission that took a lease after the caller's check must
            # stop the wipe, not lose its ledger under it (round 6m).
            live = c.execute("SELECT COUNT(*) AS n FROM missions WHERE lease_owner IS NOT NULL AND lease_until>?",
                             (time.time(),)).fetchone()["n"]
            if live:
                raise RuntimeError("a mission is still running; stop it before clearing data")
            present = {r["name"] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()}
            tables = [t for t in order if t in present] + sorted(present - set(order))
            removed = {}
            for name in tables:
                removed[name] = c.execute(f"DELETE FROM {name}").rowcount
        return removed

    # -- agent notes: working memory that survives compaction, one scope per agent --------------------------------

    def note_put(self, mission_id: str, scope: str, name: str, text: str, append: bool = False) -> int:
        """Write a note (or append to it). Returns the note's length in characters."""
        with self._tx() as c:
            self._fence(c, mission_id)
            row = c.execute("SELECT text FROM agent_notes WHERE mission_id=? AND scope=? AND name=?",
                            (mission_id, scope, name)).fetchone()
            body = (row["text"] + "\n" + text) if (append and row) else text
            c.execute("INSERT INTO agent_notes (mission_id, scope, name, text, updated) VALUES (?,?,?,?,?) "
                      "ON CONFLICT(mission_id, scope, name) DO UPDATE SET text=excluded.text, updated=excluded.updated",
                      (mission_id, scope, name, body, time.time()))
        return len(body)

    def note_get(self, mission_id: str, scope: str, name: str) -> str | None:
        with self.lock:
            row = self.conn.execute("SELECT text FROM agent_notes WHERE mission_id=? AND scope=? AND name=?",
                                    (mission_id, scope, name)).fetchone()
        return None if row is None else row["text"]

    def note_list(self, mission_id: str, scope: str | None = None) -> list[dict]:
        """Notes as {scope, name, chars}, newest first. All scopes when ``scope`` is None."""
        q = "SELECT scope, name, LENGTH(text) AS chars FROM agent_notes WHERE mission_id=?"
        args: list = [mission_id]
        if scope is not None:
            q += " AND scope=?"
            args.append(scope)
        with self.lock:
            rows = self.conn.execute(q + " ORDER BY updated DESC", args).fetchall()
        return [{"scope": r["scope"], "name": r["name"], "chars": r["chars"]} for r in rows]

    # -- lane board: messages between research lanes and the lead (runtime/lanes.py) ----------------------------------

    def board_post(self, mission_id: str, sender: str, recipient: str, kind: str, text: str, topic: str = "",
                   key: str | None = None) -> int:
        """Post a message; returns its sequence number. A post with a key already used returns the first post's number, so a
        retried tool call never posts twice."""
        with self._tx() as c:
            self._fence(c, mission_id)
            if key is not None:
                row = c.execute("SELECT seq FROM lane_board WHERE idem_key=?", (key,)).fetchone()
                if row is not None:
                    return int(row["seq"])
            cur = c.execute("INSERT INTO lane_board (mission_id, sender, recipient, kind, topic, text, idem_key, created) "
                            "VALUES (?,?,?,?,?,?,?,?)", (mission_id, sender, recipient, kind, topic, text, key, time.time()))
            return int(cur.lastrowid)

    def board_read(self, mission_id: str, reader: str | None = None, after: int = 0, limit: int = 50,
                   kinds: tuple[str, ...] | None = None) -> list[dict]:
        """Messages after ``after``, oldest first. With a reader: those sent to it or to all, not its own. Without: all."""
        q = "SELECT seq, sender, recipient, kind, topic, text FROM lane_board WHERE mission_id=? AND seq>?"
        args: list = [mission_id, int(after)]
        if reader is not None:
            q += " AND sender<>? AND recipient IN ('all', ?)"
            args += [reader, reader]
        if kinds:
            q += f" AND kind IN ({','.join('?' * len(kinds))})"
            args += list(kinds)
        with self.lock:
            rows = self.conn.execute(q + " ORDER BY seq LIMIT ?", [*args, int(limit)]).fetchall()
        return [dict(r) for r in rows]

    def tool_done_elsewhere(self, mission_id: str, tool: str, args: Any) -> tuple[int, Any] | None:
        """(step index, result) of a finished call of ``tool`` with exactly these arguments anywhere in the mission, or None."""
        with self.lock:
            row = self.conn.execute("SELECT step_idx, result_json FROM tool_calls WHERE mission_id=? AND tool=? AND "
                                    "args_json=? AND status='done' ORDER BY step_idx LIMIT 1",
                                    (mission_id, tool, canonical(args))).fetchone()
        if row is None:
            return None
        try:
            return int(row["step_idx"]), json.loads(row["result_json"] or "null")
        except (TypeError, ValueError):
            return None

    def tool_args(self, mission_id: str, tool: str) -> list[dict]:
        """The arguments of every call of ``tool`` in this mission, in step order (the ledger is the complete record)."""
        with self.lock:
            rows = self.conn.execute("SELECT args_json FROM tool_calls WHERE mission_id=? AND tool=? ORDER BY step_idx",
                                     (mission_id, tool)).fetchall()
        out = []
        for r in rows:
            try:
                out.append(json.loads(r["args_json"]))
            except (TypeError, ValueError):
                continue
        return out

    def tool_results(self, mission_id: str, tool: str) -> list[Any]:
        """The full stored result of every finished call of ``tool`` in this mission, in step order."""
        with self.lock:
            rows = self.conn.execute("SELECT result_json FROM tool_calls WHERE mission_id=? AND tool=? AND status='done' ORDER BY step_idx",
                                     (mission_id, tool)).fetchall()
        out = []
        for r in rows:
            try:
                out.append(json.loads(r["result_json"] or "null"))
            except (TypeError, ValueError):
                continue
        return out

    def count_tool_calls(self, mission_id: str, exclude: tuple[str, ...] = (), lane: str | None = None) -> int:
        """Tool calls made in this mission: every lane, or one lane ("lead" or "delegated"). A call whose arguments were
        rejected before it ran (status "rejected", round 3 A5-12) is not work and is not counted."""
        marks = ",".join("?" for _ in exclude)
        sql = "SELECT COUNT(*) AS n FROM tool_calls WHERE mission_id=? AND status!='rejected'"
        args: list = [mission_id]
        if exclude:
            sql += f" AND tool NOT IN ({marks})"
            args += list(exclude)
        if lane is not None:
            sql += " AND lane=?"
            args.append(lane)
        with self.lock:
            row = self.conn.execute(sql, args).fetchone()
        return int(row["n"])

    def tool_keys_waiting(self, mission_id: str) -> list[str]:
        with self.lock:
            rows = self.conn.execute("SELECT idem_key FROM tool_calls WHERE mission_id=? AND status='waiting'",
                                     (mission_id,)).fetchall()
        return [r["idem_key"] for r in rows]

    # -- questions ---------------------------------------------------------------------------------------------

    def open_question(self, mission_id: str, tool_key: str, payload: dict) -> str | None:
        """Open the question for a waiting call, or find the one already open for it. Returns None when the call was
        already answered, in which case the caller reads the stored answers. The check and the open are one transaction, so
        an answer cannot land between them and be lost or asked again (round 6m)."""
        qid = uuid.uuid4().hex
        with self._tx() as c:
            self._fence(c, mission_id)
            call = c.execute("SELECT status FROM tool_calls WHERE idem_key=?", (tool_key,)).fetchone()
            if call is not None and call["status"] == "done":
                return None
            c.execute("UPDATE tool_calls SET status='waiting', updated=? WHERE idem_key=? AND status!='done'",
                      (time.time(), tool_key))
            open_row = c.execute("SELECT id FROM questions WHERE mission_id=? AND tool_key=? AND status='open'",
                                 (mission_id, tool_key)).fetchone()
            if open_row is not None:
                return open_row["id"]
            c.execute("INSERT INTO questions (id, mission_id, tool_key, payload_json, status, created) "
                      "VALUES (?,?,?,?,?,?)", (qid, mission_id, tool_key, canonical(payload), "open", time.time()))
            self._emit(c, mission_id, "question.opened", {"question": qid})
        return qid

    def mark_waiting(self, mission_id: str, question_id: str) -> str:
        """A run that asked a question is about to pause. In one transaction: if the question is still open, the mission
        becomes waiting_user and its lease is released; if it was answered meanwhile, nothing changes and the run must go
        on; otherwise it was cancelled. Returns "waiting", "answered" or "cancelled" (round 6m: an answer that landed
        during the pause used to leave a waiting mission with nothing open to resume it)."""
        with self._tx() as c:
            self._fence(c, mission_id)
            row = c.execute("SELECT status FROM questions WHERE id=? AND mission_id=?",
                            (question_id, mission_id)).fetchone()
            if row is None:
                raise KeyError(f"no question {question_id} in mission {mission_id}")
            if row["status"] == "answered":
                return "answered"
            if row["status"] != "open":
                return "cancelled"
            c.execute("UPDATE missions SET status='waiting_user', updated=?, lease_owner=NULL, lease_until=0 WHERE id=?",
                      (time.time(), mission_id))
            self._emit(c, mission_id, "mission.status", {"status": "waiting_user"})
            return "waiting"

    def open_questions(self, mission_id: str) -> list[dict]:
        with self.lock:
            rows = self.conn.execute("SELECT id, tool_key, payload_json FROM questions "
                                     "WHERE mission_id=? AND status='open' ORDER BY created", (mission_id,)).fetchall()
        return [{"id": r["id"], "tool_key": r["tool_key"], "payload": json.loads(r["payload_json"])} for r in rows]

    def answered_questions(self, mission_id: str) -> list[dict]:
        """Questions the person has answered in this mission, in the order they were answered. The chat shows these as
        a record of each decision, so they outlive the question card."""
        with self.lock:
            rows = self.conn.execute("SELECT id, payload_json, answers_json, answered FROM questions "
                                     "WHERE mission_id=? AND status='answered' ORDER BY answered", (mission_id,)).fetchall()
        return [{"id": r["id"], "payload": json.loads(r["payload_json"]),
                 "answers": json.loads(r["answers_json"] or "{}"), "answered": r["answered"]} for r in rows]

    def answer_question(self, question_id: str, answers: dict) -> dict:
        """Record the answer and resolve the waiting tool call, so the runtime can resume without asking again."""
        with self._tx() as c:
            row = c.execute("SELECT mission_id, tool_key, status FROM questions WHERE id=?",
                            (question_id,)).fetchone()
            if row is None:
                raise KeyError(f"no question {question_id}")
            if row["status"] == "answered":
                raise ValueError("question already answered")
            if row["status"] != "open":
                raise ValueError("question is no longer open")
            self._fence(c, row["mission_id"])
            now = time.time()
            c.execute("UPDATE questions SET status='answered', answers_json=?, answered=? WHERE id=?",
                      (canonical(answers), now, question_id))
            c.execute("UPDATE tool_calls SET status='done', result_json=?, updated=? WHERE idem_key=?",
                      (canonical({"answers": answers}), now, row["tool_key"]))
            self._emit(c, row["mission_id"], "question.answered", {"question": question_id})
        return {"mission_id": row["mission_id"]}


class LeaseBound:
    """One runtime's view of a Store. Every call made through it runs with that runtime as the caller, so the write fence
    checks that this runtime, and not another that has used the same Store, still holds the lease (round 6m, Store._fence).
    Methods are called as on the Store; only the caller differs. The lock is held for the whole call, so a call from
    another thread never sees this caller."""

    def __init__(self, store: Store, owner: str):
        self._store = store
        self._owner = owner

    def __getattr__(self, name: str):
        if name in ("_store", "_owner"):            # not yet set (copy or pickle): never look these up on the store
            raise AttributeError(name)
        attr = getattr(self._store, name)
        if not callable(attr):
            return attr
        store, owner = self._store, self._owner

        def bound(*args, **kwargs):
            with store.lock:
                previous = store._caller
                store._caller = owner
                try:
                    return attr(*args, **kwargs)
                finally:
                    store._caller = previous
        return bound
