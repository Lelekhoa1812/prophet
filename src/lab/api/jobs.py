"""Background job runner with progress. State is written to data/jobs so a restart does not lose results.

Motivation vs Logic
Motivation: backtests and replays take minutes to tens of minutes. Job state lived only in process memory, so a restart or
crash erased the progress and the finished result, and the console showed nothing, as if the run never happened.
Logic: every state change is written atomically as one JSON file per job. On start, saved jobs are loaded; any still marked
running died with the old process and are marked interrupted (they are not resumed: a replay is cheap to re-run from cache).
"""
from __future__ import annotations

import json
import os
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from ..config import DATA

JOB_DIR = DATA / "jobs"
KEEP = 200                                   # newest jobs kept on disk

_pool = ThreadPoolExecutor(max_workers=2)
_jobs: dict[str, dict] = {}
_io = threading.Lock()


def _save(job: dict) -> None:
    with _io:
        JOB_DIR.mkdir(parents=True, exist_ok=True)
        tmp = JOB_DIR / f".{job['id']}.tmp"
        tmp.write_text(json.dumps(job, default=str))
        os.replace(tmp, JOB_DIR / f"{job['id']}.json")


def load_saved() -> None:
    """Load saved jobs; those left running by a previous process are marked interrupted."""
    if not JOB_DIR.exists():
        return
    files = sorted(JOB_DIR.glob("J-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files[KEEP:]:
        f.unlink(missing_ok=True)
    for f in files[:KEEP]:
        try:
            j = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if j.get("status") == "running":
            j.update(status="interrupted", error="The server restarted while this was running. Start it again.")
            _save(j)
        _jobs.setdefault(j["id"], j)


def submit(kind: str, fn, **kwargs) -> str:
    jid = f"J-{uuid.uuid4().hex[:8]}"
    _jobs[jid] = {"id": jid, "kind": kind, "status": "running", "progress": 0.0, "message": "", "started": datetime.now(timezone.utc).isoformat(),
                  "result": None, "error": None}
    _save(_jobs[jid])
    last = {"p": -1.0}

    def progress(done, total):
        p = round(done / max(total, 1), 3)
        _jobs[jid]["progress"] = p
        if p - last["p"] >= 0.05 or p >= 1.0:           # persist about every 5%, not every call
            last["p"] = p
            _save(_jobs[jid])

    def run():
        try:
            final = {**_jobs[jid], "result": fn(progress=progress, **kwargs), "status": "done", "progress": 1.0}
        except Exception as e:  # noqa: BLE001
            final = {**_jobs[jid], "status": "error", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
        _save(final)                                     # on disk first, so a reader that sees 'done' can rely on the file
        _jobs[jid].update(final)

    _pool.submit(run)
    return jid


def get(jid: str) -> dict | None:
    j = _jobs.get(jid)
    if j is None:
        return None
    out = {k: v for k, v in j.items() if k != "result"}
    if j["status"] == "done":
        out["result"] = j["result"]
    return out


def list_jobs() -> list[dict]:
    return [{k: v for k, v in j.items() if k not in ("result", "trace")} for j in sorted(_jobs.values(), key=lambda j: j["started"], reverse=True)][:30]


load_saved()
