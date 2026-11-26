"""Checkpoints: Fork and Rewind a conversation at a user message.

Motivation vs Logic
Motivation: a person wants to ask again from an earlier question without losing the work after it (Rewind), or to try a
different follow-up beside the original (Fork). Both must keep the runtime's promises: a finished tool call is never run
again, a call a crash left "started" is never retried, model decisions and tool results are replayed from the ledger,
and the ledger is never edited in a way that loses what was there. The mechanism and its evidence are in
docs/prophet/fork_rewind_copy.md.
Logic: the checkpoint is "before turn N" (a turn is a mission; its message is the mission's goal).
  * Rewind flags the turns from N on, and the table versions they saved, with the rewind's id. Nothing is deleted: the
    rows stay in the ledger and ``rewinds`` lists them. Every reader that builds the conversation skips flagged rows. Undo
    clears the flag, and is allowed only until the next question is sent (as Claude's rewind is).
  * Fork writes a new notebook holding a copy of each earlier turn's ledger under new ids. A tool-call key hashes the
    mission id, so each copied call is re-keyed and the keys named inside copied rows are rewritten. A call keeps its
    status and stored result exactly: done stays done (never run again), started stays started (never retried). The
    original is only read. Run folders are shared by reference; see cleanup.runs_still_named.
Nothing here runs a model or a tool.
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from prophet_agent.runtime.store import ACTIVE, Store, idem_key


class Busy(RuntimeError):
    """A question in this notebook is still running; stop it first."""


class Refused(ValueError):
    """The checkpoint cannot be used (already removed, or no longer undoable)."""


def _copy_row(c: sqlite3.Connection, table: str, row: sqlite3.Row, **changes) -> None:
    data = {**dict(row), **changes}
    cols = list(data)
    c.execute(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", [data[k] for k in cols])


class Checkpoints:
    def __init__(self, store: Store, notebooks):
        self.store = store
        self.notebooks = notebooks

    # -- shared -----------------------------------------------------------------------------------------------

    def _turn(self, c: sqlite3.Connection, nid: str, turn_id: str) -> sqlite3.Row:
        row = c.execute("SELECT * FROM missions WHERE id=?", (turn_id,)).fetchone()
        if row is None or row["notebook_id"] != nid:
            raise KeyError(turn_id)
        if row["dropped_by"] is not None:
            raise Refused("that question was already removed by a rewind")
        return row

    @staticmethod
    def _live(row: sqlite3.Row) -> bool:
        held = row["lease_owner"] is not None and row["lease_until"] > time.time()
        return row["status"] in ACTIVE or held

    def _refuse_if_busy(self, c: sqlite3.Connection, nid: str) -> None:
        for row in c.execute("SELECT status, lease_owner, lease_until FROM missions "
                             "WHERE notebook_id=? AND dropped_by IS NULL", (nid,)):
            if self._live(row):
                raise Busy("Prophet is still working in this notebook; stop it first")

    # -- rewind -----------------------------------------------------------------------------------------------

    def rewind_notebook(self, nid: str, turn_id: str) -> dict:
        """Return the notebook to the point before this question. KeyError for no such question, Busy while a question runs,
        Refused for one already removed. Returns the draft text to offer back and what was hidden."""
        self.notebooks.get(nid)
        rid, now = uuid.uuid4().hex, time.time()
        with self.store._tx() as c:
            target = self._turn(c, nid, turn_id)
            self._refuse_if_busy(c, nid)
            dropped = [r["id"] for r in c.execute(
                "SELECT id FROM missions WHERE notebook_id=? AND dropped_by IS NULL AND seq>=? ORDER BY seq",
                (nid, target["seq"]))]
            hidden = []
            for mid in dropped:
                for v in c.execute("SELECT v.id, t.name, v.version FROM derived_versions v JOIN derived_tables t ON t.id=v.table_id "
                                   "WHERE v.turn_id=? AND v.dropped_by IS NULL ORDER BY t.name, v.version", (mid,)).fetchall():
                    hidden.append({"name": v["name"], "version": v["version"]})
                    c.execute("UPDATE derived_versions SET dropped_by=? WHERE id=?", (rid, v["id"]))
                c.execute("UPDATE missions SET dropped_by=? WHERE id=?", (rid, mid))
                self.store._emit(c, mid, "mission.dropped", {"rewind": rid, "to_turn": turn_id})
            c.execute("INSERT INTO rewinds (id, notebook_id, to_turn_id, to_seq, goal, dropped_json, tables_json, created) "
                      "VALUES (?,?,?,?,?,?,?,?)",
                      (rid, nid, turn_id, target["seq"], target["goal"], json.dumps(dropped), json.dumps(hidden), now))
        self.notebooks.touch(nid)
        return {"rewind_id": rid, "notebook_id": nid, "to_turn_id": turn_id, "to_seq": target["seq"],
                "prefill": target["goal"], "dropped": dropped, "tables_hidden": hidden}

    def rewind_forecast(self, mission_id: str) -> dict:
        """Rewind a quick forecast: hide it the way Hide does (reversible) and record the rewind. KeyError, Busy, Refused."""
        rid, now = uuid.uuid4().hex, time.time()
        with self.store._tx() as c:
            row = c.execute("SELECT * FROM missions WHERE id=? AND notebook_id IS NULL", (mission_id,)).fetchone()
            if row is None:
                raise KeyError(mission_id)
            if self._live(row):
                raise Busy("this forecast is still running; stop it first")
            if row["archived"]:
                raise Refused("that forecast is already hidden")
            c.execute("UPDATE missions SET archived=1 WHERE id=?", (mission_id,))
            self.store._emit(c, mission_id, "mission.dropped", {"rewind": rid, "to_turn": mission_id})
            c.execute("INSERT INTO rewinds (id, notebook_id, to_turn_id, to_seq, goal, dropped_json, tables_json, created) "
                      "VALUES (?,?,?,?,?,?,?,?)", (rid, None, mission_id, None, row["goal"], json.dumps([mission_id]), "[]", now))
        return {"rewind_id": rid, "notebook_id": None, "to_turn_id": mission_id, "prefill": row["goal"],
                "dropped": [mission_id], "tables_hidden": []}

    def undo_rewind(self, rewind_id: str) -> dict:
        """Put back what a rewind hid. Allowed until a question is sent after it, and only for the latest rewind of that
        notebook. KeyError for no such rewind, Refused otherwise."""
        with self.store._tx() as c:
            rw = c.execute("SELECT * FROM rewinds WHERE id=?", (rewind_id,)).fetchone()
            if rw is None:
                raise KeyError(rewind_id)
            if rw["undone"] is not None:
                raise Refused("that rewind was already undone")
            dropped = json.loads(rw["dropped_json"])
            if rw["notebook_id"] is None:
                c.execute("UPDATE missions SET archived=0 WHERE id=? AND notebook_id IS NULL", (rw["to_turn_id"],))
            else:
                later = c.execute("SELECT 1 FROM rewinds WHERE notebook_id=? AND undone IS NULL AND created>? LIMIT 1",
                                  (rw["notebook_id"], rw["created"])).fetchone()
                if later:
                    raise Refused("a later rewind has to be undone first")
                sent = c.execute("SELECT 1 FROM missions WHERE notebook_id=? AND dropped_by IS NULL AND seq>=? LIMIT 1",
                                 (rw["notebook_id"], rw["to_seq"])).fetchone()
                if sent:
                    raise Refused("a question was sent after the rewind, so it can no longer be undone")
                for mid in dropped:
                    c.execute("UPDATE missions SET dropped_by=NULL WHERE id=? AND dropped_by=?", (mid, rewind_id))
                c.execute("UPDATE derived_versions SET dropped_by=NULL WHERE dropped_by=?", (rewind_id,))
            for mid in dropped:
                self.store._emit(c, mid, "mission.restored", {"rewind": rewind_id})
            c.execute("UPDATE rewinds SET undone=? WHERE id=?", (time.time(), rewind_id))
        if rw["notebook_id"] is not None:
            self.notebooks.touch(rw["notebook_id"])
        return {"rewind_id": rewind_id, "notebook_id": rw["notebook_id"], "restored": dropped}

    def rewinds(self, nid: str) -> list[dict]:
        """Every rewind of a notebook, newest first, with the questions each one dropped (still in the ledger)."""
        self.notebooks.get(nid)
        with self.store.lock:
            rows = self.store.conn.execute("SELECT * FROM rewinds WHERE notebook_id=? ORDER BY created DESC", (nid,)).fetchall()
            out = []
            for r in rows:
                dropped = json.loads(r["dropped_json"])
                goals = {m["id"]: m["goal"] for m in (
                    self.store.conn.execute("SELECT id, goal FROM missions WHERE id=?", (mid,)).fetchone() for mid in dropped) if m}
                out.append({"id": r["id"], "to_turn_id": r["to_turn_id"], "to_seq": r["to_seq"], "created": r["created"],
                            "undone": r["undone"], "tables_hidden": json.loads(r["tables_json"]),
                            "dropped": [{"id": mid, "goal": goals.get(mid)} for mid in dropped]})
        return out

    # -- fork -------------------------------------------------------------------------------------------------

    def fork_notebook(self, nid: str, turn_id: str, title: str | None = None) -> dict:
        """A new notebook from the point before this question. The original is only read. KeyError, Refused."""
        nb = self.notebooks.get(nid)
        new_nid, now = uuid.uuid4().hex, time.time()
        root = Path(self.notebooks.root) / new_nid
        profile: list[str] = []
        try:
            with self.store._tx() as c:
                target = self._turn(c, nid, turn_id)
                name = (title or "").strip()[:120] or self._fork_title(c, nb["title"])
                src_nb = c.execute("SELECT * FROM notebooks WHERE id=?", (nid,)).fetchone()
                _copy_row(c, "notebooks", src_nb, id=new_nid, title=name, pinned=0, archived=0, created=now, updated=now,
                          forked_from=nid, forked_at_seq=target["seq"])
                share = c.execute("SELECT web_research, providers FROM notebook_sharing WHERE notebook_id=?", (nid,)).fetchone()
                if share is not None:
                    c.execute("INSERT INTO notebook_sharing (notebook_id, web_research, providers) VALUES (?,?,?)",
                              (new_nid, share["web_research"], share["providers"]))
                kept = c.execute("SELECT * FROM missions WHERE notebook_id=? AND dropped_by IS NULL AND seq<? ORDER BY seq",
                                 (nid, target["seq"])).fetchall()
                mapping: dict[str, str] = {}
                keys: dict[str, str] = {}
                prev: str | None = None
                for old in kept:
                    mapping[old["id"]] = prev = self._copy_turn(c, old, new_nid, prev, keys)
                for s in c.execute("SELECT * FROM sources WHERE notebook_id=? AND created<? ORDER BY created",
                                   (nid, target["created"])).fetchall():
                    sid = uuid.uuid4().hex
                    _copy_row(c, "sources", s, id=sid, notebook_id=new_nid)
                    if s["status"] == "profiling":
                        profile.append(sid)
                self._copy_tables(c, nid, new_nid, mapping, target["created"], root)
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)     # the transaction rolled back; so do the files it had copied
            raise
        for sid in profile:                             # a file still being read when copied is read again for the copy
            t = threading.Thread(target=self.notebooks.profile_source, args=(sid,), daemon=True)
            self.notebooks._profilers[sid] = t
            t.start()
        return {"notebook_id": new_nid, "title": name, "forked_from": nid, "to_turn_id": turn_id, "to_seq": target["seq"],
                "prefill": target["goal"], "copied_turns": len(kept), "turns": mapping}

    @staticmethod
    def _fork_title(c: sqlite3.Connection, title: str) -> str:
        base = re.sub(r" \(Fork(?: \d+)?\)$", "", title)[:100]
        first = f"{base} (Fork)"
        taken = {r["title"] for r in c.execute("SELECT title FROM notebooks WHERE title LIKE ?", (f"{base} (Fork%",))}
        if first not in taken:
            return first
        n = 2
        while f"{base} (Fork {n})" in taken:
            n += 1
        return f"{base} (Fork {n})"

    def _copy_turn(self, c: sqlite3.Connection, old: sqlite3.Row, new_nid: str, parent: str | None, keys: dict) -> str:
        """Copy one turn's ledger under a new mission id. See the module note for what is kept exactly."""
        mid = uuid.uuid4().hex
        status = "interrupted" if old["status"] in ACTIVE else old["status"]
        _copy_row(c, "missions", old, id=mid, notebook_id=new_nid, parent_turn_id=parent, status=status, lease_owner=None,
                  lease_until=0, cancel_requested=0, archived=0, pinned=0, draft_text=None, dropped_by=None, forked_from=old["id"])
        oid = old["id"]
        # New keys first, so every copied row that names a key can be rewritten.
        calls = c.execute("SELECT * FROM tool_calls WHERE mission_id=?", (oid,)).fetchall()
        local: dict[str, str] = {}
        for call in calls:
            try:
                args = json.loads(call["args_json"])
                new_key = idem_key(mid, call["step_idx"], call["tool"], args)
                if idem_key(oid, call["step_idx"], call["tool"], args) != call["idem_key"]:
                    raise ValueError
            except (ValueError, TypeError):
                new_key = idem_key(mid, call["step_idx"], call["tool"], f"fork:{call['idem_key']}")
            local[call["idem_key"]] = new_key
        keys.update(local)

        def fix(text: str | None) -> str | None:
            if text is None:
                return None
            for k, v in local.items():
                if k in text:
                    text = text.replace(k, v)
            return text

        for call in calls:
            _copy_row(c, "tool_calls", call, idem_key=local[call["idem_key"]], mission_id=mid, result_json=fix(call["result_json"]))
        for st in c.execute("SELECT * FROM steps WHERE mission_id=?", (oid,)).fetchall():
            _copy_row(c, "steps", st, mission_id=mid, output_json=fix(st["output_json"]))
        for ev in c.execute("SELECT kind, payload_json, ts FROM events WHERE mission_id=? ORDER BY seq", (oid,)).fetchall():
            c.execute("INSERT INTO events (mission_id, kind, payload_json, ts) VALUES (?,?,?,?)",
                      (mid, ev["kind"], fix(ev["payload_json"]), ev["ts"]))
        if status != old["status"]:
            self.store._emit(c, mid, "mission.status", {"status": status})
        for q in c.execute("SELECT * FROM questions WHERE mission_id=?", (oid,)).fetchall():
            _copy_row(c, "questions", q, id=uuid.uuid4().hex, mission_id=mid, tool_key=local.get(q["tool_key"], q["tool_key"]),
                      status="cancelled" if q["status"] == "open" else q["status"], payload_json=fix(q["payload_json"]))
        for note in c.execute("SELECT * FROM agent_notes WHERE mission_id=?", (oid,)).fetchall():
            _copy_row(c, "agent_notes", note, mission_id=mid)
        self.store._emit(c, mid, "mission.forked", {"from": oid})
        return mid

    def _copy_tables(self, c: sqlite3.Connection, nid: str, new_nid: str, mapping: dict[str, str], cut_created: float, root: Path):
        """Copy the table versions the kept turns saved (and any saved outside a turn before the cut), with their files."""
        for t in c.execute("SELECT * FROM derived_tables WHERE notebook_id=?", (nid,)).fetchall():
            copies = []
            for v in c.execute("SELECT * FROM derived_versions WHERE table_id=? AND dropped_by IS NULL ORDER BY version",
                               (t["id"],)).fetchall():
                mine = v["turn_id"] in mapping or (v["turn_id"] is None and v["created"] < cut_created)
                if mine and Path(v["file_path"]).is_file():
                    copies.append(v)
            if not copies:
                continue
            tid = uuid.uuid4().hex
            _copy_row(c, "derived_tables", t, id=tid, notebook_id=new_nid)
            for v in copies:
                dest = root / "tables" / tid / f"v{v['version']}.csv"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(v["file_path"], dest)
                _copy_row(c, "derived_versions", v, id=uuid.uuid4().hex, table_id=tid, file_path=str(dest),
                          turn_id=mapping.get(v["turn_id"]) if v["turn_id"] else None)
