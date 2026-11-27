"""Open research panel for Mission 9 (docs/prophet/GOAL.md, criterion M9-8): open research missions across finance, sport,
science, economics and energy, run through the local HTTP API with web research on and no uploaded file.

Run the whole panel, or chosen keys, a few at a time:
    PYTHONPATH=src .venv/bin/python scripts/workflows/research_panel.py               # every mission
    PYTHONPATH=src .venv/bin/python scripts/workflows/research_panel.py NVDA EPL      # chosen keys
    PYTHONPATH=src .venv/bin/python scripts/workflows/research_panel.py --parallel 2

Motivation vs Logic
Motivation: Mission 9 is judged on open research missions that must find their own data, not on uploaded files. The same
goals must be re-run after each batch of fixes so rounds can be compared; NVDA and EPL keep the exact goals of the two
missions that failed on 2026-11-26.
Logic: each mission gets its own notebook with web research on and one turn. Checkpoints are continued up to a limit. A
question for the person is answered as an unattended run must: the first option that neither waits for an upload nor
swaps in invented data, or else a free-text reply to keep researching. The record keeps every answer given. Each record keeps the final status, the outcome reasons, tool-call counts, sandbox cells, blocked
fetches and wall time; the run writes record.json per mission, the full mission JSON, and summary.json under
data/research_panel/ (git-ignored).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prophet_workflows import CHECKPOINT_STATUSES, POLL_SECONDS, ROOT, call  # noqa: E402

MAX_CONTINUES = 4
MAX_ANSWERS = 3
MAX_REFUSALS = 5     # refused continues (lease still held) retried before the panel stops watching
MAX_MINUTES = 120

PANEL: dict[str, dict] = {
    "NVDA": {"domain": "finance", "goal": (
        "Research on Nvidia stock today and forecast their stock and best buy in time over the next 2 months. \n\n"
        "Ask me questions if you may need to. Search extensively for all possible insights and signals that can possibly "
        "influence this.")},
    "EPL": {"domain": "sport", "goal": (
        "Research who can win the English Premier League in the 2026/27 season and forecast the points path for the main "
        "contenders.\nCollect public results or current standings as a table. If the file download returns a page or an "
        "error, diagnose that failure and extract the table from the page. Do not invent a results address.\nState the "
        "competition, the season, and the measure (points or position) as a claim. Bind the series name only when the "
        "table is that competition. A table for another league or another season stays unbound.\nThen compare the "
        "contenders and forecast the rest of the season. Separate observed results from any derived ranking. Ask me which "
        "clubs to keep if the table has many rows.")},
    "SPX": {"domain": "finance", "goal": (
        "Find daily closing values of the S&P 500 index from public sources, check them, and forecast the next 20 trading "
        "days with a range. Compare against the simple rules and say what is measured and what is a source claim.")},
    "AUUNEMP": {"domain": "economics", "goal": (
        "Find Australia's monthly unemployment rate (seasonally adjusted) from official open data and forecast the next "
        "6 months with a range. Cite the source of every figure.")},
    # The owner's VN Unemployment notebook (2026-11-27): a monthly seasonally adjusted series is not published for Vietnam,
    # so the mission has to find what is published and ask.
    "VNUNEMP": {"domain": "economics", "goal": (
        "Find Vietnam monthly unemployment rate (seasonally adjusted) from the internet, and forecast the next 6 months "
        "with a range.")},
    "NEM": {"domain": "energy", "goal": (
        "Find recent electricity demand for New South Wales in Australia's National Electricity Market from public data "
        "and forecast daily demand for the next 14 days with a range. Say which data are measured and which are derived.")},
    "GISTEMP": {"domain": "science", "goal": (
        "Find the monthly global surface temperature anomaly from an open scientific source (for example NASA GISTEMP or "
        "NOAA) and forecast the next 12 months with a range. Name the baseline period of the anomaly.")},
    "NGSTORAGE": {"domain": "energy", "goal": (
        "Find the weekly US natural gas working storage (Lower 48) from the EIA or another official source and forecast "
        "the next 8 weeks with a range, using the seasonal pattern. Cite each source.")},
}


UNATTENDED_REPLY = ("Nobody can upload a file in this run. Keep researching public sources and use the data you can find. "
                    "Do not use synthetic or approximate data in place of the real series.")
AVOID = ("upload", "synthetic", "approximat", "provide your own", "i'll provide", "i will provide", "api key")


def _pick(options: list[dict]) -> str:
    """The first option that neither waits for a file nor swaps in invented data; otherwise a free-text reply."""
    for option in options:
        text = f"{option.get('label', '')} {option.get('description', '')}".lower()
        if not any(word in text for word in AVOID):
            return option.get("label")
    return UNATTENDED_REPLY


def _answer_questions(mission: dict, record: dict) -> bool:
    """Answer every open question as an unattended run must. Returns False when nothing could be answered."""
    answered = False
    for q in mission.get("questions") or []:
        answers = {}
        items = q.get("questions") or (q.get("payload") or {}).get("questions") or []
        for item in items:
            options = item.get("options") or []
            label = _pick(options) if options else UNATTENDED_REPLY
            answers[item.get("header") or item.get("question", "")[:12]] = [label] if item.get("multiSelect") else label
        if not answers:
            continue
        call("POST", f"/api/questions/{q['id']}/answer", {"answers": answers})
        record.setdefault("auto_answers", []).append({"question_id": q["id"], "answers": answers})
        answered = True
    return answered


def _score(mission: dict) -> dict:
    """Counts that show where a run spent its effort and where it was blocked."""
    tools = collections.Counter()
    blocked = 0
    errors = 0
    for step in mission.get("steps") or []:
        out = step.get("output")
        if not (step.get("name", "").startswith("act:") and isinstance(out, dict)):
            continue
        args = out.get("args") or {}
        tool = out.get("tool") or ("sandbox_python" if isinstance(args, dict) and "code" in args else "other")
        tools[tool] += 1
        if out.get("error"):
            errors += 1
        result = out.get("result") or {}
        note = (result.get("note") or "") if isinstance(result, dict) else ""
        if isinstance(result, dict) and result.get("ok") is False and any(
                word in note.lower() for word in ("interstitial", "captcha", "bot check", "adblock", "consent", "login")):
            blocked += 1
    outcome = mission.get("outcome") or {}
    return {
        "tool_calls": dict(tools),
        "sandbox_cells": tools.get("sandbox_python", 0),
        "tool_errors": errors,
        "blocked_pages": blocked,
        "reason_codes": [r.get("code") for r in outcome.get("reasons") or []],
        "unverified_figures": [f for r in outcome.get("reasons") or [] for f in (r.get("figures") or [])],
        "stop_reason": (outcome.get("developer") or {}).get("stop_reason"),
        "steps": len(mission.get("steps") or []),
    }


def run_one(key: str, out_root: Path) -> dict:
    spec = PANEL[key]
    started = datetime.now()
    out_dir = out_root / key
    out_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {"key": key, "domain": spec["domain"], "goal": spec["goal"], "started": started.isoformat()}
    try:
        nb = call("POST", "/api/notebooks", {"title": f"Panel M9 {key} {started:%m%d-%H%M}"})
        nid = nb["id"]
        record["notebook_id"] = nid
        call("PUT", f"/api/notebooks/{nid}/sharing", {"web_research": True})
        turn = call("POST", f"/api/notebooks/{nid}/turns", {"goal": spec["goal"]})
        tid = turn["id"]
        record["turn_id"] = tid
        deadline = time.time() + MAX_MINUTES * 60
        continues = answers = refused = 0
        mission: dict = {}
        while True:
            mission = call("GET", f"/api/missions/{tid}?after=0")
            status = mission.get("status")
            if status in ("created", "running"):
                if time.time() > deadline:
                    record["stopped"] = f"wall-clock limit of {MAX_MINUTES} minutes reached while {status}"
                    call("POST", f"/api/missions/{tid}/cancel")
                    mission = call("GET", f"/api/missions/{tid}?after=0")
                    break
                time.sleep(POLL_SECONDS)
                continue
            if status == "waiting_user" and answers < MAX_ANSWERS and _answer_questions(mission, record):
                answers += 1
                time.sleep(POLL_SECONDS)
                continue
            if status in CHECKPOINT_STATUSES and continues < MAX_CONTINUES:
                try:
                    call("POST", f"/api/missions/{tid}/continue")
                    continues += 1
                    time.sleep(POLL_SECONDS)
                    continue
                except RuntimeError as exc:
                    # Round 2 AUUNEMP (2026-11-26): the checkpoint was read while the worker still held its lease, the
                    # continue was refused once and the panel stopped; the mission then ran on unwatched. Retry a few times.
                    refused += 1
                    record.setdefault("continue_refusals", []).append(str(exc)[:200])
                    if refused <= MAX_REFUSALS:
                        time.sleep(POLL_SECONDS * 2)
                        continue
            break
        record.update({"final_status": mission.get("status"), "continues": continues, "auto_answer_rounds": answers})
        record.update(_score(mission))
        (out_dir / "mission.json").write_text(json.dumps(mission, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    except Exception as exc:  # one mission's failure is recorded and the panel goes on
        record["error"] = f"{type(exc).__name__}: {exc}"[:500]
    record["finished"] = datetime.now().isoformat()
    record["minutes"] = round((datetime.now() - started).total_seconds() / 60, 1)
    (out_dir / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return record


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the Mission 9 open research panel over the local API.")
    parser.add_argument("keys", nargs="*", help=f"mission keys (default all): {', '.join(PANEL)}")
    parser.add_argument("--parallel", type=int, default=3, help="missions run at the same time (default 3)")
    args = parser.parse_args(argv[1:])
    keys = args.keys or list(PANEL)
    unknown = [k for k in keys if k not in PANEL]
    if unknown:
        parser.error(f"unknown keys: {', '.join(unknown)}")
    out_root = ROOT / "data" / "research_panel" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        records = list(pool.map(lambda k: run_one(k, out_root), keys))
    met = [r["key"] for r in records if r.get("final_status") in ("met", "completed")]
    summary = {"run": out_root.name, "missions": len(records), "met_or_completed": met,
               "by_status": dict(collections.Counter(r.get("final_status") or "error" for r in records)),
               "records": [{k: r.get(k) for k in ("key", "final_status", "reason_codes", "sandbox_cells", "blocked_pages",
                                                   "tool_errors", "unverified_figures", "minutes", "error")}
                           for r in records]}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
