"""Four named Prophet workflows. Each one is a session (a notebook) with its title, research mode, an optional spreadsheet and
one long-running turn, followed to its final status over the local HTTP API.

Run one workflow by key, once:
    PYTHONPATH=src .venv/bin/python scripts/workflows/prophet_workflows.py SciFore

Keys and what they do:
    SciFore  spreadsheet upload, scientific forecast, research mode OFF
    FinFore  spreadsheet upload, financial (price) forecast, research mode ON
    FinRes   no upload, research mode ON: find finance data, strategise, analyse
    SciRes   no upload, research mode ON: find scientific data, strategise, analyse

Motivation vs Logic
Motivation: the same four runs should be repeatable and their records kept in one place, so a review can compare them.
Logic: the workflow only uses the public routes in API.md. It creates the notebook, sets sharing (web research on or off),
uploads the file, waits for profiling to finish, sends one turn and polls it. At a checkpoint it continues, up to a limit.
A question for the user (waiting_user) is recorded and the run stops, because an answer is the user's to give. Each run
writes its record and the full mission into data/workflow_runs/, which is git-ignored.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
BASE = os.environ.get("PROPHET_API", "http://127.0.0.1:8010")
POLL_SECONDS = 2
MAX_CONTINUES = 6                 # checkpoints continued before the run is recorded as still at a checkpoint
MAX_MINUTES = 150                 # wall-clock limit for one workflow; the record says so if it is reached
CHECKPOINT_STATUSES = ("checkpoint", "budget_exhausted", "iteration_budget")
FINAL_STATUSES = ("met", "completed", "failed", "stalled", "impossible", "outcome_unknown", "cancelled")

SCI_FILE = "examples/science/mauna_loa_co2_daily_2014_2023.csv"
FIN_FILE = "examples/finance/eia_wti_spot_daily_2016_2026.csv"

WORKFLOWS: dict[str, dict] = {
    "SciFore": {
        "title": "SciFore {name} Research OFF", "file": SCI_FILE, "research": False,
        "goal": ("Forecast the next 30 days of the daily CO2 series in the uploaded file, with a range. Name the season used and "
                 "why, compare the forecast with the simple rules, and say plainly whether it beats them. Use only the uploaded "
                 "file; web research is off for this session."),
    },
    "FinFore": {
        "title": "FinFore {name} Research ON", "file": FIN_FILE, "research": True,
        "goal": ("Forecast the next 30 days of the WTI spot price in the uploaded file, with a range. Name the season used and "
                 "compare the forecast with the simple rules. Use web research to check recent market context and any events "
                 "that should widen the range, and cite each source you use."),
    },
    "FinRes": {
        "title": "FinRes {name}", "file": None, "research": True, "name": "S&P 500 daily closes",
        "goal": ("Find public daily closing prices for the S&P 500 index from reputable open sources. Download what can be verified "
                 "and record where each figure came from. Then choose a forecast approach, backtest it against the simple rules, "
                 "and forecast the next 20 trading days with a range. Say what is measured, what is a source claim, and what "
                 "could not be verified."),
    },
    "SciRes": {
        "title": "SciRes {name}", "file": None, "research": True, "name": "Mauna Loa daily CO2",
        "goal": ("Find public daily CO2 measurements from Mauna Loa in open scientific sources. Download what can be verified and "
                 "record the source for each figure. Then choose a forecast approach, backtest it against the simple rules, and "
                 "forecast the next 30 days with a range. Say what is measured, what is a source claim, and what could not be "
                 "verified."),
    },
}


def call(method: str, path: str, body: dict | None = None, raw: bytes | None = None, headers: dict | None = None):
    data = raw if raw is not None else (json.dumps(body).encode("utf-8") if body is not None else None)
    head = {"content-type": "application/json"} if body is not None else {}
    head.update(headers or {})
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=head)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = resp.read()
            return json.loads(payload) if payload else None
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{method} {path} returned {exc.code}: {exc.read()[:300]!r}") from None


def wait_for_profiling(nid: str, limit_s: int = 600) -> list[dict]:
    """Sources are profiled in the background; the planner needs their cards, so wait until none is still profiling."""
    deadline = time.time() + limit_s
    while True:
        nb = call("GET", f"/api/notebooks/{nid}")
        sources = nb.get("sources", [])
        if all(s.get("status") != "profiling" for s in sources) or time.time() > deadline:
            return sources
        time.sleep(POLL_SECONDS)


def run(key: str) -> dict:
    wf = WORKFLOWS[key]
    name = wf.get("name") or Path(wf["file"]).stem
    title = wf["title"].format(name=name)
    started = datetime.now()
    out_dir = ROOT / "data" / "workflow_runs" / f"{key}-{started:%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {"key": key, "title": title, "research": wf["research"], "file": wf["file"], "started": started.isoformat()}

    nb = call("POST", "/api/notebooks", {"title": title})
    nid = nb["id"]
    record["notebook_id"] = nid
    call("PUT", f"/api/notebooks/{nid}/sharing", {"web_research": wf["research"]})
    if wf["file"]:
        path = ROOT / wf["file"]
        src = call("POST", f"/api/notebooks/{nid}/sources", raw=path.read_bytes(),
                   headers={"X-File-Name": quote(path.name), "content-type": "application/octet-stream"})
        record["source"] = {"id": src.get("id"), "name": path.name, "bytes": path.stat().st_size, "duplicate": src.get("duplicate")}
        record["sources_after_profiling"] = [{"name": s.get("name"), "status": s.get("status")} for s in wait_for_profiling(nid)]

    turn = call("POST", f"/api/notebooks/{nid}/turns", {"goal": wf["goal"]})
    tid = turn["id"]
    record["turn_id"] = tid
    deadline = time.time() + MAX_MINUTES * 60
    continues = 0
    mission: dict = {}
    while True:
        mission = call("GET", f"/api/missions/{tid}?after=0")
        status = mission.get("status")
        if status in ("created", "running"):
            if time.time() > deadline:
                record["stopped"] = f"wall-clock limit of {MAX_MINUTES} minutes reached while {status}"
                break
            time.sleep(POLL_SECONDS)
            continue
        if status == "waiting_user":
            record["stopped"] = "waiting_user: a question needs an answer from the user"
            record["questions"] = mission.get("questions")
            break
        if status in CHECKPOINT_STATUSES and continues < MAX_CONTINUES:
            try:
                call("POST", f"/api/missions/{tid}/continue")
                continues += 1
                time.sleep(POLL_SECONDS)
                continue
            except RuntimeError:
                pass                                  # not at a checkpoint after all: fall through and record the status
        break

    record["final_status"] = mission.get("status")
    record["continues"] = continues
    record["finished"] = datetime.now().isoformat()
    record["answer_status"] = (mission.get("answer") or {}).get("status") if isinstance(mission.get("answer"), dict) else None
    record["outcome"] = mission.get("outcome")
    (out_dir / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "mission.json").write_text(json.dumps(mission, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    record["out_dir"] = str(out_dir)
    return record


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in WORKFLOWS:
        print(__doc__)
        print("Keys:", ", ".join(WORKFLOWS))
        return 2
    result = run(argv[1])
    print(json.dumps({k: result[k] for k in ("title", "final_status", "continues", "out_dir") if k in result}, indent=2,
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
