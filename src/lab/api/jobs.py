"""Tiny background job runner with progress, so long backtests and replays do not block the console."""
from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

_pool = ThreadPoolExecutor(max_workers=2)
_jobs: dict[str, dict] = {}


def submit(kind: str, fn, **kwargs) -> str:
    jid = f"J-{uuid.uuid4().hex[:8]}"
    _jobs[jid] = {"id": jid, "kind": kind, "status": "running", "progress": 0.0, "message": "", "started": datetime.now(timezone.utc).isoformat(),
                  "result": None, "error": None}

    def progress(done, total):
        _jobs[jid]["progress"] = round(done / max(total, 1), 3)

    def run():
        try:
            _jobs[jid]["result"] = fn(progress=progress, **kwargs)
            _jobs[jid].update(status="done", progress=1.0)
        except Exception as e:  # noqa: BLE001
            _jobs[jid].update(status="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])

    _pool.submit(run)
    return jid


def get(jid: str) -> dict | None:
    j = _jobs.get(jid)
    return None if j is None else {k: v for k, v in j.items() if k != "result"} | ({"result": j["result"]} if j["status"] == "done" else {})


def list_jobs() -> list[dict]:
    return [{k: v for k, v in j.items() if k not in ("result", "trace")} for j in sorted(_jobs.values(), key=lambda j: j["started"], reverse=True)][:30]
