"""Run the research panel round after round, all missions in parallel, until a STOP file appears.

    PYTHONPATH=src .venv/bin/python scripts/workflows/panel_loop.py              # all six topics, in parallel, forever
    PYTHONPATH=src .venv/bin/python scripts/workflows/panel_loop.py --rounds 2   # two rounds
    touch data/panel_loop/STOP                                                   # stop after the current round

Motivation vs Logic
Motivation: the owner asked (2026-11-27) for the six panel topics to run in parallel and repeat until they score at their
best, with each old round cleaned up once the next one is done. Restarting the API by hand between rounds raced with running
missions, and an API serving the working tree could load a file in the middle of an edit.
Logic: the API is served from its own checkout (data/apiwt) pinned to the last commit; its data stays in the main data folder
through PROPHET_HOME. Before each round, when HEAD has moved, the checkout moves to HEAD and the API restarts; it is never
restarted during a round. A round runs research_panel.py with one worker per mission. When a round finishes, the notebooks and
run folder of the round before it are deleted, so the newest finished round is always kept for triage. Every round appends one
line to data/panel_loop/rounds.jsonl: its folder, the commit it ran on, and each mission's status, stop reason and minutes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
STATE_DIR = DATA / "panel_loop"
STOP = STATE_DIR / "STOP"
STATE = STATE_DIR / "state.json"
ROUNDS = STATE_DIR / "rounds.jsonl"
API_TREE = DATA / "apiwt"
PANELS = DATA / "research_panel"
PORT = int(os.environ.get("PROPHET_API_PORT", "8010"))
BASE = f"http://127.0.0.1:{PORT}"
PYTHON = ROOT / ".venv" / "bin" / "python"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def healthy() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE}/api/health", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def api(method: str, path: str) -> None:
    req = urllib.request.Request(f"{BASE}{path}", method=method)
    try:
        urllib.request.urlopen(req, timeout=30).read()
    except OSError as exc:
        log(f"{method} {path} failed: {exc}")


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def panel_running() -> bool:
    out = subprocess.run(["pgrep", "-f", "scripts/workflows/research_panel.py"], capture_output=True, text=True).stdout
    return bool(out.strip())


def stop_api() -> None:
    pids = subprocess.run(["lsof", "-ti", f"tcp:{PORT}"], capture_output=True, text=True).stdout.split()
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass
    for _ in range(30):
        if not healthy():
            return
        time.sleep(1)


def sync_api(state: dict) -> None:
    """Serve the API from data/apiwt at HEAD; restart only when HEAD moved or the API is down."""
    head = git("rev-parse", "HEAD")
    if not (API_TREE / ".git").exists():
        git("worktree", "add", "--detach", str(API_TREE), head)
    if state.get("api_sha") == head and healthy():
        return
    git("checkout", "-q", "--detach", head, cwd=API_TREE)
    stop_api()
    env = {**os.environ, "PYTHONPATH": str(API_TREE / "src"), "PROPHET_HOME": str(ROOT)}
    logf = open(DATA / f"api.loop.{datetime.now():%m%d-%H%M}.log", "w")
    subprocess.Popen([str(PYTHON), "-m", "prophet_agent.api"], cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT,
                     start_new_session=True)
    for _ in range(60):
        if healthy():
            break
        time.sleep(1)
    else:
        raise SystemExit("the API did not come up; see data/api.loop.*.log")
    state["api_sha"] = head
    save_state(state)
    log(f"API serving {head[:7]}")


def newest_run(after: set[str]) -> Path | None:
    runs = sorted(p for p in PANELS.iterdir() if p.is_dir() and p.name not in after) if PANELS.is_dir() else []
    return runs[-1] if runs else None


def records(run: Path) -> list[dict]:
    out = []
    for f in sorted(run.glob("*/record.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def clean(run_name: str | None) -> None:
    """Delete a finished round's notebooks and its run folder."""
    if not run_name:
        return
    run = PANELS / run_name
    if not run.is_dir():
        return
    for r in records(run):
        if r.get("notebook_id"):
            api("DELETE", f"/api/notebooks/{r['notebook_id']}")
    shutil.rmtree(run, ignore_errors=True)
    log(f"cleaned round {run_name}")


def run_round(parallel: int, keys: list[str]) -> Path | None:
    before = {p.name for p in PANELS.iterdir()} if PANELS.is_dir() else set()
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    with open(DATA / "panel.loop.log", "a") as logf:
        subprocess.run([str(PYTHON), str(ROOT / "scripts" / "workflows" / "research_panel.py"), "--parallel", str(parallel),
                        *keys], cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return newest_run(before)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Repeat the research panel, all missions in parallel.")
    parser.add_argument("keys", nargs="*", help="mission keys (default all)")
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=0, help="rounds to run (0: until data/panel_loop/STOP exists)")
    args = parser.parse_args(argv[1:])
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    done = 0
    while not STOP.exists() and (args.rounds == 0 or done < args.rounds):
        while panel_running():          # a round started by hand finishes first
            time.sleep(30)
        sync_api(state)
        started = datetime.now()
        log(f"round starts on {state['api_sha'][:7]}")
        run = run_round(args.parallel, args.keys)
        if run is None:
            log("the panel wrote no run folder; waiting a minute before the next round")
            time.sleep(60)
            continue
        recs = records(run)
        line = {"run": run.name, "sha": state["api_sha"], "started": started.isoformat(timespec="seconds"),
                "minutes": round((datetime.now() - started).total_seconds() / 60, 1),
                "missions": {r["key"]: {"status": r.get("final_status"), "stop": r.get("stop_reason"),
                                        "reasons": r.get("reason_codes"), "minutes": r.get("minutes")} for r in recs}}
        with open(ROUNDS, "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        log(f"round {run.name} done: " + ", ".join(f"{k}={v['stop']}" for k, v in line["missions"].items()))
        clean(state.get("last_run"))
        state["last_run"] = run.name
        save_state(state)
        done += 1
    log("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
