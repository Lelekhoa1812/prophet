"""Notebooks: persistent workspaces of source files, derived tables and a conversation of turns.

Motivation vs Logic
Motivation: forecasting rarely ends with one question. People upload several spreadsheets, ask follow-ups, clean and
reshape the data, and forecast from the result. The work must be durable, every number must trace back to a file the
person gave, and each turn must start from what the notebook already knows.
Logic: a turn is a mission (see runtime/store.py); the notebook owns what persists between turns.
  * Sources are uploaded files (kept once by the dataset store, hashed). On upload a background run profiles every
    visible sheet in the sandbox and stores a small card; the planner reads cards, never the files themselves.
  * Derived tables are CSV versions written by the ``save_table`` tool from an analysis run. A version is immutable and
    records where it came from (run, turn, source names). The next turn mounts the latest version of each table.
  * Context for a turn is: the notebook's instructions, the source and table cards, and the recent question-and-answer
    pairs, all within a character budget. Older turns are summarised by omission with a visible note, never by guess.
Nothing here parses a file or runs code outside the sandbox.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from prophet_agent.datasets import DatasetStore, MAX_BYTES
from prophet_agent.runtime.store import ACTIVE, Store
from prophet_agent.sandbox.runner import OUT_DIR, SANDBOX_ROOT, run_python

TABLE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
RUN_ID = re.compile(r"^[0-9a-f]{32}$")
CARD_CHARS = 3200            # per source
CONTEXT_CHARS = 14000        # whole notebook context
RECENT_TURNS = 6
MAX_COLUMNS = 200

PROFILE_CODE = """
import json, prophet_lib as pl
print(json.dumps(pl.profile_workbook('in/' + {name!r})))
"""


def _col_line(c: dict) -> str:
    bits = [c["kind"]]
    for k in ("first", "last", "typical_step", "missing_periods", "min", "max", "mean", "zero_share", "missing", "unique"):
        if k in c and c[k] not in (None, 0, 0.0):
            v = c[k]
            bits.append(f"{k}={str(v)[:10] if k in ('first', 'last') else v}")
    for k in ("sentinel_candidate", "decimal_comma_values", "unparsed_values", "ambiguous_day_month", "looks_like_year"):
        if k in c:
            bits.append(f"{k}={c[k]}")
    return f"{c['name']}: " + ", ".join(bits)


def _read_as(read: dict) -> str | None:
    """How the file was parsed, in one phrase the planner can copy: e.g. a semicolon-separated text file. Plain
    comma CSV and workbook sheets need no note."""
    delim = read.get("delimiter")
    if read.get("format") == "delimited text" and delim not in (None, "','"):
        return f"text read with separator {delim}; use pl.read_table(path) rather than pandas defaults"
    return None


def make_card(profile: dict) -> dict:
    """A small, stable summary of a profiled file: per sheet, the shape, columns, issues and layout hints."""
    sheets = []
    for sh in profile.get("sheets", []):
        read = sh.get("read", {})
        sheets.append({
            "sheet": sh.get("name"), "rows": sh.get("rows", 0), "read_as": _read_as(read),
            "header_row": read.get("header_row"), "columns": [_col_line(c) for c in sh.get("columns", [])[:40]],
            "columns_omitted": sh.get("columns_omitted", 0), "issues": sh.get("issues", [])[:6],
            "layout_hints": sh.get("layout_hints", [])[:3],
        })
    card: dict[str, Any] = {"sheets": sheets}
    for k in ("hidden_sheets", "sheets_not_profiled"):
        if profile.get(k):
            card[k] = profile[k]
    while len(json.dumps(card)) > CARD_CHARS and any(s["columns"] for s in sheets):
        widest = max(sheets, key=lambda s: len(s["columns"]))
        cut = max(1, len(widest["columns"]) // 2)
        widest["columns_omitted"] += len(widest["columns"]) - cut
        widest["columns"] = widest["columns"][:cut]
    return card


class Notebooks:
    def __init__(self, store: Store, datasets: DatasetStore, root: str | Path):
        self.store = store
        self.datasets = datasets
        self.root = Path(root)
        self._profilers: dict[str, threading.Thread] = {}

    # -- notebooks -----------------------------------------------------------------------------------------------

    def create(self, title: str = "New notebook") -> dict:
        nid, now = uuid.uuid4().hex, time.time()
        with self.store._tx() as c:
            c.execute("INSERT INTO notebooks (id, title, created, updated) VALUES (?,?,?,?)",
                      (nid, (title or "New notebook").strip()[:120], now, now))
        return self.get(nid)

    def _row(self, nid: str):
        with self.store.lock:
            row = self.store.conn.execute("SELECT * FROM notebooks WHERE id=?", (nid,)).fetchone()
        if row is None:
            raise KeyError(nid)
        return row

    def get(self, nid: str) -> dict:
        row = self._row(nid)
        return {"id": row["id"], "title": row["title"], "instructions": row["instructions"],
                "archived": bool(row["archived"]), "created": row["created"], "updated": row["updated"]}

    def list(self, limit: int = 50) -> list[dict]:
        with self.store.lock:
            rows = self.store.conn.execute(
                "SELECT n.id, n.title, n.created, n.updated, "
                "(SELECT COUNT(*) FROM missions m WHERE m.notebook_id=n.id) AS turns, "
                "(SELECT COUNT(*) FROM sources s WHERE s.notebook_id=n.id AND s.removed=0) AS sources, "
                "(SELECT status FROM missions m WHERE m.notebook_id=n.id ORDER BY seq DESC LIMIT 1) AS status "
                "FROM notebooks n WHERE n.archived=0 ORDER BY n.updated DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def update(self, nid: str, *, title: str | None = None, instructions: str | None = None,
               archived: bool | None = None) -> dict:
        self._row(nid)
        with self.store._tx() as c:
            if title is not None:
                c.execute("UPDATE notebooks SET title=?, updated=? WHERE id=?", (title.strip()[:120] or "New notebook", time.time(), nid))
            if instructions is not None:
                c.execute("UPDATE notebooks SET instructions=?, updated=? WHERE id=?", (instructions[:2000], time.time(), nid))
            if archived is not None:
                c.execute("UPDATE notebooks SET archived=?, updated=? WHERE id=?", (int(archived), time.time(), nid))
        return self.get(nid)

    def touch(self, nid: str) -> None:
        with self.store._tx() as c:
            c.execute("UPDATE notebooks SET updated=? WHERE id=?", (time.time(), nid))

    # -- sources -------------------------------------------------------------------------------------------------

    def add_source(self, nid: str, name: str, data: bytes, *, profile_async: bool = True) -> dict:
        """Store a file in the notebook. The same bytes added twice are the same source."""
        self._row(nid)
        digest = hashlib.sha256(data).hexdigest()
        with self.store.lock:
            dup = self.store.conn.execute("SELECT id FROM sources WHERE notebook_id=? AND sha256=? AND removed=0",
                                          (nid, digest)).fetchone()
        if dup:
            return {**self.source(dup["id"]), "duplicate": True}
        meta = self.datasets.add(name, data)
        sid = uuid.uuid4().hex
        with self.store.lock:
            first = self.store.conn.execute("SELECT COUNT(*) FROM sources WHERE notebook_id=? AND removed=0", (nid,)).fetchone()[0] == 0
            title_now = self._row(nid)["title"]
        if first and title_now == "New notebook":
            # A notebook is named after its first file, so the sidebar says what it holds.
            stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", Path(name).name).replace("_", " ").strip()[:120]
            if stem:
                with self.store._tx() as c:
                    c.execute("UPDATE notebooks SET title=? WHERE id=?", (stem, nid))
        with self.store._tx() as c:
            c.execute("INSERT INTO sources (id, notebook_id, dataset_id, name, sha256, bytes, status, created) "
                      "VALUES (?,?,?,?,?,?,?,?)", (sid, nid, meta["id"], meta["name"], digest, meta["bytes"],
                                                    "profiling", time.time()))
        self.touch(nid)
        if profile_async:
            t = threading.Thread(target=self.profile_source, args=(sid,), daemon=True)
            self._profilers[sid] = t
            t.start()
        return self.source(sid)

    def profile_source(self, sid: str) -> None:
        src = self.source(sid)
        path = self.datasets.path(src["dataset_id"])
        try:
            res = run_python(PROFILE_CODE.format(name=path.name), timeout_s=90, mem_mb=2048, inputs={path.name: path})
            if not res.ok:
                raise RuntimeError((res.stderr.strip().splitlines() or ["the profiler stopped"])[-1][:300])
            card, status, error = make_card(json.loads(res.stdout)), "ready", None
            if all(not s["rows"] for s in card["sheets"]):
                status = "needs_attention"
                error = next((i for s in card["sheets"] for i in s["issues"]), "no table could be read from this file")
        except Exception as exc:
            card, status, error = None, "failed", str(exc)[:300]
        with self.store._tx() as c:
            c.execute("UPDATE sources SET status=?, card_json=?, error=? WHERE id=?",
                      (status, json.dumps(card) if card else None, error, sid))

    def wait_profiled(self, timeout: float = 120.0) -> None:
        for t in list(self._profilers.values()):
            t.join(timeout)

    def source(self, sid: str) -> dict:
        with self.store.lock:
            row = self.store.conn.execute("SELECT * FROM sources WHERE id=?", (sid,)).fetchone()
        if row is None:
            raise KeyError(sid)
        d = dict(row)
        d["card"] = json.loads(d.pop("card_json") or "null")
        d["included"], d["removed"] = bool(d["included"]), bool(d["removed"])
        return d

    def sources(self, nid: str, include_removed: bool = False) -> list[dict]:
        with self.store.lock:
            rows = self.store.conn.execute(
                "SELECT id FROM sources WHERE notebook_id=? " + ("" if include_removed else "AND removed=0 ") +
                "ORDER BY created", (nid,)).fetchall()
        return [self.source(r["id"]) for r in rows]

    def set_source(self, nid: str, sid: str, *, included: bool | None = None, removed: bool | None = None) -> dict:
        src = self.source(sid)
        if src["notebook_id"] != nid:
            raise KeyError(sid)
        with self.store._tx() as c:
            if included is not None:
                c.execute("UPDATE sources SET included=? WHERE id=?", (int(included), sid))
            if removed is not None:
                c.execute("UPDATE sources SET removed=? WHERE id=?", (int(removed), sid))
        self.touch(nid)
        return self.source(sid)

    def mounts(self, nid: str) -> dict[str, Path]:
        """Files for a turn: the included, readable sources, by the names the analysis code sees in in/."""
        ready = [s for s in self.sources(nid) if s["included"] and s["status"] in ("ready", "needs_attention", "profiling")]
        return self.datasets.mount([s["dataset_id"] for s in ready])

    # -- derived tables ------------------------------------------------------------------------------------------

    def tables(self, nid: str) -> list[dict]:
        with self.store.lock:
            rows = self.store.conn.execute(
                "SELECT t.id, t.name, t.description, v.version, v.rows, v.columns_json, v.sha256, v.created, v.turn_id "
                "FROM derived_tables t JOIN derived_versions v ON v.table_id=t.id "
                "WHERE t.notebook_id=? AND v.version=(SELECT MAX(version) FROM derived_versions WHERE table_id=t.id) "
                "ORDER BY t.created", (nid,)).fetchall()
        return [{**{k: r[k] for k in ("id", "name", "description", "version", "rows", "sha256", "created", "turn_id")},
                 "columns": json.loads(r["columns_json"])} for r in rows]

    def table_versions(self, nid: str, name: str) -> list[dict]:
        with self.store.lock:
            rows = self.store.conn.execute(
                "SELECT v.version, v.rows, v.columns_json, v.sha256, v.created, v.lineage_json, v.turn_id FROM derived_versions v "
                "JOIN derived_tables t ON t.id=v.table_id WHERE t.notebook_id=? AND t.name=? ORDER BY v.version",
                (nid, name)).fetchall()
        return [{"version": r["version"], "rows": r["rows"], "columns": json.loads(r["columns_json"]),
                 "sha256": r["sha256"], "created": r["created"], "lineage": json.loads(r["lineage_json"]),
                 "turn_id": r["turn_id"]} for r in rows]

    def table_path(self, nid: str, name: str, version: int | None = None) -> Path:
        with self.store.lock:
            row = self.store.conn.execute(
                "SELECT v.file_path FROM derived_versions v JOIN derived_tables t ON t.id=v.table_id "
                "WHERE t.notebook_id=? AND t.name=? " + ("AND v.version=?" if version else
                "AND v.version=(SELECT MAX(version) FROM derived_versions WHERE table_id=t.id)"),
                (nid, name, version) if version else (nid, name)).fetchone()
        if row is None:
            raise KeyError(name)
        return Path(row["file_path"])

    def derived_mounts(self, nid: str) -> dict[str, Path]:
        return {f"derived/{t['name']}.csv": self.table_path(nid, t["name"]) for t in self.tables(nid)}

    def promote(self, nid: str, run_id: str, file: str, name: str, description: str = "", turn_id: str | None = None) -> dict:
        """Turn a file an analysis run wrote into a new, immutable version of a named table."""
        if not TABLE_NAME.match(name or ""):
            raise ValueError("table name must be lowercase letters, digits and underscores, starting with a letter (max 40)")
        if not RUN_ID.match(run_id or ""):
            raise ValueError("run_id must be the id of an analysis run")
        base = (SANDBOX_ROOT / run_id / OUT_DIR).resolve()
        rel = file or f"{OUT_DIR}/table.csv"
        if not rel.startswith(OUT_DIR + "/"):
            rel = f"{OUT_DIR}/{rel}"
        src = (SANDBOX_ROOT / run_id / rel).resolve()
        if not str(src).startswith(str(base) + os.sep) or not src.is_file():
            raise ValueError(f"no file {file!r} in the outputs of run {run_id}")
        if src.suffix.lower() != ".csv":
            raise ValueError("save a CSV file (for example df.to_csv('out/table.csv', index=False))")
        if src.stat().st_size > MAX_BYTES:
            raise ValueError("the table is larger than 50 MB")
        with src.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            try:
                columns = next(reader)
            except StopIteration:
                raise ValueError("the CSV file is empty")
            rows = sum(1 for _ in reader)
        if not columns or len(columns) > MAX_COLUMNS or len(set(columns)) != len(columns):
            raise ValueError(f"the CSV needs 1 to {MAX_COLUMNS} uniquely named columns")
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        srcs = self.sources(nid)
        with self.store._tx() as c:
            t = c.execute("SELECT id FROM derived_tables WHERE notebook_id=? AND name=?", (nid, name)).fetchone()
            if t is None:
                tid = uuid.uuid4().hex
                c.execute("INSERT INTO derived_tables (id, notebook_id, name, description, created) VALUES (?,?,?,?,?)",
                          (tid, nid, name, description[:300], time.time()))
            else:
                tid = t["id"]
                if description:
                    c.execute("UPDATE derived_tables SET description=? WHERE id=?", (description[:300], tid))
            last = c.execute("SELECT version, sha256 FROM derived_versions WHERE table_id=? ORDER BY version DESC LIMIT 1",
                             (tid,)).fetchone()
            if last and last["sha256"] == digest:
                return {"name": name, "version": last["version"], "rows": rows, "columns": columns, "unchanged": True}
            version = (last["version"] + 1) if last else 1
            dest = self.root / nid / "tables" / tid / f"v{version}.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dest)
            lineage = {"run_id": run_id, "file": rel, "sources": sorted(s["name"] for s in srcs if s["included"]),
                       "parent_version": last["version"] if last else None}
            c.execute("INSERT INTO derived_versions (id, table_id, version, rows, columns_json, sha256, file_path, "
                      "lineage_json, turn_id, created) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (uuid.uuid4().hex, tid, version, rows, json.dumps(columns), digest, str(dest), json.dumps(lineage),
                       turn_id, time.time()))
        self.touch(nid)
        return {"name": name, "version": version, "rows": rows, "columns": columns}

    # -- turns ---------------------------------------------------------------------------------------------------

    def turns(self, nid: str) -> list[dict]:
        with self.store.lock:
            rows = self.store.conn.execute(
                "SELECT id, seq, goal, status, created, updated, answer_json FROM missions WHERE notebook_id=? ORDER BY seq",
                (nid,)).fetchall()
        return [{"id": r["id"], "seq": r["seq"], "goal": r["goal"], "status": r["status"], "created": r["created"],
                 "updated": r["updated"], "answer": json.loads(r["answer_json"] or "null")} for r in rows]

    def context(self, nid: str, before_seq: int | None = None) -> str:
        """What the planner should know about this notebook before it reads the question."""
        nb = self.get(nid)
        parts: list[str] = []
        if nb["instructions"]:
            parts.append("Notebook instructions from the person:\n" + nb["instructions"])
        srcs = [s for s in self.sources(nid) if s["included"]]
        if srcs:
            names = self.datasets.mount([s["dataset_id"] for s in srcs])
            lines = ["Sources in this notebook (analysis code sees them in in/ under these names; "
                     "pl.read_table(path, sheet) reads one sheet):"]
            for (fname, _), s in zip(names.items(), srcs):
                head = f"- {fname} ({s['bytes']:,} bytes) status: {s['status']}"
                if s["status"] == "profiling":
                    head += " (still being checked; call inspect_data for it)"
                elif s["error"]:
                    head += f". Problem: {s['error']}"
                lines.append(head)
                for sh in (s["card"] or {}).get("sheets", []):
                    label = f"  sheet {sh['sheet']!r}" if sh["sheet"] else "  table"
                    lines.append(f"{label}: {sh['rows']} rows" + (f", header on row {sh['header_row']}" if sh["header_row"] else ", no header row")
                                 + (f"; {sh['read_as']}" if sh.get("read_as") else ""))
                    lines += [f"    {c}" for c in sh["columns"]]
                    if sh["columns_omitted"]:
                        lines.append(f"    ... {sh['columns_omitted']} more columns")
                    lines += [f"    ISSUE: {i}" for i in sh["issues"]] + [f"    HINT: {h}" for h in sh["layout_hints"]]
                if (s["card"] or {}).get("hidden_sheets"):
                    lines.append(f"  hidden sheets not read: {s['card']['hidden_sheets']}")
            parts.append("\n".join(lines))
        tbl = self.tables(nid)
        if tbl:
            parts.append("Tables created earlier in this notebook (latest version, in in/derived/<name>.csv):\n" + "\n".join(
                f"- {t['name']} v{t['version']}: {t['rows']} rows; columns {', '.join(t['columns'][:12])}"
                + (f"; {t['description']}" if t["description"] else "") for t in tbl))
        past = [t for t in self.turns(nid) if before_seq is None or t["seq"] < before_seq]
        done = [t for t in past if t["status"] in ("completed", "impossible", "stalled", "iteration_budget")]
        if done:
            shown = done[-RECENT_TURNS:]
            lines = ["Earlier in this conversation:"]
            if len(done) > len(shown):
                lines.append(f"(the {len(done) - len(shown)} earlier exchanges are not shown)")
            for t in shown:
                ans = t["answer"] if isinstance(t["answer"], dict) else {}
                text = ans.get("answer") or ans.get("reason") or f"({t['status']})"
                lines.append(f"Q{t['seq']}: {t['goal'][:300]}\nA{t['seq']}: {text[:700]}")
            parts.append("\n".join(lines))
        out = "\n\n".join(parts)
        return out if len(out) <= CONTEXT_CHARS else out[:CONTEXT_CHARS] + "\n...[notebook context cut]"
