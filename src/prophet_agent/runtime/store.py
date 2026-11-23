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
    "missions": {"title": "TEXT", "pinned": "INTEGER NOT NULL DEFAULT 0"},
    "notebooks": {"pinned": "INTEGER NOT NULL DEFAULT 0"},
    # The profiling run's id, so deleting the notebook removes that run folder with the rest.
    "sources": {"run_id": "TEXT"},
}

_MISSION_COLUMNS_V2 = {
    "notebook_id": "TEXT",
    "seq": "INTEGER",
    "parent_turn_id": "TEXT",
    "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
    "archived": "INTEGER NOT NULL DEFAULT 0",
}
ACTIVE = ("created", "running", "waiting_user")


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


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
        # Leases this process holds, by mission: the owner each mission's writes must still match (see _fence).
        self._owned: dict[str, str] = {}
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
                busy = c.execute("SELECT 1 FROM missions WHERE notebook_id=? AND status IN (?,?,?)",
                                 (notebook_id, *ACTIVE)).fetchone()
                if busy:
                    raise NotebookBusy("this notebook is still working on the previous message")
                seq = int(c.execute("SELECT COALESCE(MAX(seq),0)+1 FROM missions WHERE notebook_id=?",
                                    (notebook_id,)).fetchone()[0])
            c.execute("INSERT INTO missions (id, goal, status, budget_json, created, updated, notebook_id, seq, "
                      "parent_turn_id) VALUES (?,?,?,?,?,?,?,?,?)",
                      (mission_id, goal, "created", canonical(budget), now, now, notebook_id, seq, parent_turn_id))
        self.append_event(mission_id, "mission.created", {"goal": goal, "budget": budget, "notebook_id": notebook_id})
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
        self.append_event(mission_id, "mission.status", {"status": status})

    def set_budget(self, mission_id: str, budget: dict) -> None:
        with self._tx() as c:
            self._fence(c, mission_id)
            c.execute("UPDATE missions SET budget_json=?, updated=? WHERE id=?",
                      (canonical(budget), time.time(), mission_id))

    def set_answer(self, mission_id: str, answer: Any) -> None:
        with self._tx() as c:
            self._fence(c, mission_id)
            c.execute("UPDATE missions SET answer_json=?, updated=? WHERE id=?",
                      (canonical(answer), time.time(), mission_id))

    def _fence(self, c: sqlite3.Connection, mission_id: str) -> None:
        """Refuse a write to a mission this process leased when the lease now belongs to someone else.

        Motivation (audit: runtime store.py:409-437, HIGH): writes did not check the lease owner, so a runtime whose lease
        had been taken over kept writing events, steps and tool results beside the new owner.
        Logic: only for a mission this process holds (``_owned``). The owner is read in the same transaction as the write,
        so a takeover cannot slip between the check and the write. Writes from elsewhere (the service, a cancel) are unchanged.
        """
        owner = self._owned.get(mission_id)
        if owner is None:
            return
        row = c.execute("SELECT lease_owner FROM missions WHERE id=?", (mission_id,)).fetchone()
        if row is None or row["lease_owner"] != owner:
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
            ids = [r["id"] for r in c.execute(
                "SELECT id FROM missions WHERE status IN ('created','running') "
                "AND (lease_owner IS NULL OR lease_until<?)", (now,)).fetchall() if r["id"] not in exclude]
            for mid in ids:
                c.execute("UPDATE missions SET status='interrupted', lease_owner=NULL, lease_until=0, updated=? "
                          "WHERE id=?", (now, mid))
        for mid in ids:
            self._owned.pop(mid, None)      # the lease lapsed, so this process no longer holds it (see _fence)
        for mid in ids:
            self.append_event(mid, "mission.status", {"status": "interrupted",
                                                      "reason": "the runtime stopped before this mission finished"})
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

    def finish_step(self, mission_id: str, idx: int, name: str, output: Any) -> None:
        with self._tx() as c:
            self._fence(c, mission_id)
            c.execute("INSERT INTO steps (mission_id, idx, name, status, output_json, attempts, updated) "
                      "VALUES (?,?,?,?,?,?,?) ON CONFLICT(mission_id, idx) DO UPDATE SET "
                      "status='done', output_json=excluded.output_json, updated=excluded.updated",
                      (mission_id, idx, name, "done", canonical(output), 1, time.time()))

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
        with self._tx() as c:
            self._fence_key(c, key)
            c.execute("UPDATE tool_calls SET status=?, result_json=?, error=?, updated=? WHERE idem_key=?",
                      (status, canonical(result) if result is not None else None, error, time.time(), key))

    def wipe_ledger(self) -> dict[str, int]:
        """Delete every row of the ledger, children before parents, in one transaction. Returns the rows removed per table.
        The caller must have checked that no mission is running; the schema itself is kept."""
        order = ("events", "steps", "tool_calls", "questions", "agent_notes", "sources", "derived_versions", "derived_tables",
                 "missions", "notebooks")
        with self._tx() as c:
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

    def open_question(self, mission_id: str, tool_key: str, payload: dict) -> str:
        qid = uuid.uuid4().hex
        with self._tx() as c:
            self._fence(c, mission_id)
            c.execute("INSERT INTO questions (id, mission_id, tool_key, payload_json, status, created) "
                      "VALUES (?,?,?,?,?,?)", (qid, mission_id, tool_key, canonical(payload), "open", time.time()))
        return qid

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
            if row["status"] != "open":
                raise ValueError("question already answered")
            self._fence(c, row["mission_id"])
            c.execute("UPDATE questions SET status='answered', answers_json=?, answered=? WHERE id=?",
                      (canonical(answers), time.time(), question_id))
            c.execute("UPDATE tool_calls SET status='done', result_json=?, updated=? WHERE idem_key=?",
                      (canonical({"answers": answers}), time.time(), row["tool_key"]))
        self.append_event(row["mission_id"], "question.answered", {"question": question_id})
        return {"mission_id": row["mission_id"]}
