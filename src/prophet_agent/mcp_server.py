"""Prophet as an MCP server (stdio), so Claude, Cursor, ChatGPT-style clients and other agents can use it as a tool.

It is a thin bridge over the local HTTP API in API.md: it holds no state and no keys. Start Prophet first
(`python -m prophet_agent.api`), then point the client at `python -m prophet_agent.mcp_server`.

Standard library only: newline-delimited JSON-RPC 2.0 on stdin and stdout, MCP protocol 2024-11-05.
"""
from __future__ import annotations

import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

BASE = os.environ.get("PROPHET_URL", "http://127.0.0.1:8010").rstrip("/")
PROTOCOL = "2024-11-05"
FINAL = {"completed", "met", "partial", "failed", "stalled", "impossible", "outcome_unknown", "cancelled"}
CHECKPOINT = "checkpoint"                # not final: the person continues from here
WORKING = {"created", "running"}


class ProphetError(Exception):
    pass


def _call(method: str, path: str, body: Any = None, raw: bytes | None = None, headers: dict | None = None,
          timeout: float = 60.0) -> Any:
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    hdrs = dict(headers or {})
    if body is not None:
        hdrs["content-type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            return json.loads(payload) if payload and "json" in resp.headers.get("content-type", "") else payload
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("detail", "")
        except Exception:
            detail = ""
        raise ProphetError(f"Prophet said {exc.code}: {detail or exc.reason}")
    except urllib.error.URLError:
        raise ProphetError(f"Prophet is not running at {BASE}. Start it with: python -m prophet_agent.api")


def _reasons(mission: dict) -> list[dict]:
    """The named reasons from the API's outcome: plain words and a next step. The planner's own text is not sent."""
    rows = []
    for r in (mission.get("outcome") or {}).get("reasons", []):
        row: dict[str, Any] = {"code": r["code"], "reason": r["user_reason"], "next": r["next_step"]}
        if r.get("checks"):
            row["checks"] = [c["plain"] for c in r["checks"]]
        if r.get("figures"):
            row["figures"] = r["figures"]
        rows.append(row)
    return rows


def _answer_text(mission: dict) -> dict:
    ans = mission.get("answer")
    out: dict[str, Any] = {"turn_id": mission["id"], "status": mission["status"]}
    if mission["status"] == "waiting_user":
        out["questions"] = [{"question_id": q["id"], "questions": q["payload"]["questions"]} for q in mission.get("questions", [])]
        out["next"] = "Call prophet_answer with the question_id and an answers object keyed by each question's header."
    elif mission["status"] == CHECKPOINT:
        out["reasons"] = _reasons(mission)
        out["next"] = ("Paused at a checkpoint. Call prophet_continue to keep going with a fresh allowance, "
                       "or prophet_answer_now to write the answer from what has been found.")
    elif isinstance(ans, dict) and ans.get("status") == "met":
        out["answer"] = ans.get("answer")
    elif isinstance(ans, dict) and ans.get("status") == "partial":
        out["answer"] = ans.get("answer")
        out["reasons"] = _reasons(mission)
        out["partial"] = " ".join(r["reason"] for r in out["reasons"]) or "This answer is partial."
        figures = [f for r in out["reasons"] for f in r.get("figures", [])]
        if figures:
            out["unverified_figures"] = figures
    elif ans:
        # Stopped without a written answer (impossible, failed, stalled, cancelled): the plain reasons, never the raw answer.
        out["reasons"] = _reasons(mission)
    return out


def _wait(mission_id: str, seconds: float) -> dict:
    deadline = time.time() + max(1.0, seconds)
    while True:
        m = _call("GET", f"/api/missions/{mission_id}")
        if m["status"] in FINAL or m["status"] in (CHECKPOINT, "waiting_user") or time.time() >= deadline:
            return m
        time.sleep(2.0)


# -- tools ------------------------------------------------------------------------------------------------------------

def t_list_notebooks(_: dict) -> Any:
    return _call("GET", "/api/notebooks")


def t_create_notebook(a: dict) -> Any:
    nb = _call("POST", "/api/notebooks", {"title": a.get("title") or "New notebook"})
    if a.get("instructions"):
        _call("PATCH", f"/api/notebooks/{nb['id']}", {"instructions": a["instructions"]})
    return nb


def t_add_file(a: dict) -> Any:
    path = Path(str(a["path"])).expanduser()
    if not path.is_file():
        raise ProphetError(f"No file at {path}")
    src = _call("POST", f"/api/notebooks/{a['notebook_id']}/sources", raw=path.read_bytes(),
                headers={"X-File-Name": urllib.parse.quote(path.name),
                         "content-type": mimetypes.guess_type(path.name)[0] or "application/octet-stream"}, timeout=120)
    deadline = time.time() + float(a.get("wait_seconds", 60))
    while src.get("status") == "profiling" and time.time() < deadline:
        time.sleep(1.5)
        nb = _call("GET", f"/api/notebooks/{a['notebook_id']}")
        src = next((s for s in nb["sources"] if s["id"] == src["id"]), src)
    return src


def t_notebook(a: dict) -> Any:
    return _call("GET", f"/api/notebooks/{a['notebook_id']}")


def t_ask(a: dict) -> Any:
    turn = _call("POST", f"/api/notebooks/{a['notebook_id']}/turns", {"goal": a["question"]})
    m = _wait(turn["id"], float(a.get("wait_seconds", 240)))
    out = _answer_text(m)
    if m["status"] in WORKING:
        out["next"] = "Still working. Call prophet_status with this turn_id to continue waiting."
    return out


def t_status(a: dict) -> Any:
    m = _wait(a["turn_id"], float(a.get("wait_seconds", 60)))
    out = _answer_text(m)
    if m["status"] in WORKING:
        out["next"] = "Still working. Call prophet_status again."
    return out


def _resume(a: dict, action: str) -> Any:
    """Continue a checkpointed mission, or write its answer now, as the console does; then wait for the next stop."""
    turn_id = a["turn_id"]
    _call("POST", f"/api/missions/{turn_id}/{action}")
    wait = float(a.get("wait_seconds", 240))
    m = _wait(turn_id, wait) if wait > 0 else _call("GET", f"/api/missions/{turn_id}")
    out = _answer_text(m)
    if m["status"] in WORKING:
        out["next"] = "Still working. Call prophet_status with this turn_id to continue waiting."
    return out


def t_continue(a: dict) -> Any:
    return _resume(a, "continue")


def t_answer_now(a: dict) -> Any:
    return _resume(a, "answer-now")


def t_answer(a: dict) -> Any:
    r = _call("POST", f"/api/questions/{a['question_id']}/answer", {"answers": a["answers"]})
    m = _wait(r["mission_id"], float(a.get("wait_seconds", 240)))
    return _answer_text(m)


def t_cancel(a: dict) -> Any:
    return _call("POST", f"/api/missions/{a['turn_id']}/cancel")


def t_files(a: dict) -> Any:
    runs = _call("GET", f"/api/missions/{a['turn_id']}/artifacts")
    return [{"run_id": r["run_id"], "files": [{"name": f["path"], "bytes": f["bytes"], "sha256": f["sha256"],
                                               "url": f"{BASE}/api/artifacts/{r['run_id']}/{f['path']}"} for f in r["files"]]}
            for r in runs]


def t_quick_forecast(a: dict) -> Any:
    ids = []
    if a.get("path"):
        path = Path(str(a["path"])).expanduser()
        if not path.is_file():
            raise ProphetError(f"No file at {path}")
        ds = _call("POST", "/api/datasets?name=" + urllib.parse.quote(path.name), raw=path.read_bytes(), timeout=120)
        ids.append(ds["id"])
    m = _call("POST", "/api/missions", {"goal": a["goal"], "dataset_ids": ids})
    return _answer_text(_wait(m["id"], float(a.get("wait_seconds", 240))))


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


S = {"type": "string"}
N = {"type": "number"}
TOOLS: dict[str, tuple[str, dict, Callable[[dict], Any]]] = {
    "prophet_list_notebooks": ("List notebooks (workspaces that keep files and questions).", _schema({}, []), t_list_notebooks),
    "prophet_create_notebook": (
        "Create a notebook. Optional standing instructions (units, currency, fiscal calendar, known events) are given to the analyst on every question.",
        _schema({"title": S, "instructions": S}, []), t_create_notebook),
    "prophet_add_file": (
        "Upload a local spreadsheet or data file (CSV, Excel, ODS, Parquet and more) into a notebook and wait for it to be checked. Returns its status, row counts and any issues found.",
        _schema({"notebook_id": S, "path": S, "wait_seconds": N}, ["notebook_id", "path"]), t_add_file),
    "prophet_notebook": ("Show a notebook: its files and their status, kept tables and past questions.",
                         _schema({"notebook_id": S}, ["notebook_id"]), t_notebook),
    "prophet_ask": (
        "Ask a question about the notebook's files in plain language: totals, comparisons, anomalies, joins, cleaning, or a forecast with a range. Waits for the answer. May return status waiting_user with questions to answer via prophet_answer. One question per call gives the best results.",
        _schema({"notebook_id": S, "question": S, "wait_seconds": N}, ["notebook_id", "question"]), t_ask),
    "prophet_status": ("Keep waiting for a turn that was still working, or re-read its answer.",
                       _schema({"turn_id": S, "wait_seconds": N}, ["turn_id"]), t_status),
    "prophet_answer": (
        "Answer questions Prophet asked (status waiting_user). 'answers' is an object keyed by each question's header; the value is an offered label, a list of labels for multi-select, or your own text.",
        _schema({"question_id": S, "answers": {"type": "object"}, "wait_seconds": N}, ["question_id", "answers"]), t_answer),
    "prophet_cancel": ("Stop a turn that is working or waiting.", _schema({"turn_id": S}, ["turn_id"]), t_cancel),
    "prophet_continue": (
        "Continue a turn that paused at a checkpoint (its allowance of time or tool calls ran out). Every step so far is kept; waits for the next stop.",
        _schema({"turn_id": S, "wait_seconds": N}, ["turn_id"]), t_continue),
    "prophet_answer_now": (
        "Write the answer for a turn paused at a checkpoint from the evidence found so far. It is labelled partial and says what is missing.",
        _schema({"turn_id": S, "wait_seconds": N}, ["turn_id"]), t_answer_now),
    "prophet_files": ("List the charts, workbooks and reports a turn produced, with download URLs and sha256.",
                      _schema({"turn_id": S}, ["turn_id"]), t_files),
    "prophet_quick_forecast": (
        "One-off forecast without a notebook. Optionally attach one local data file by path.",
        _schema({"goal": S, "path": S, "wait_seconds": N}, ["goal"]), t_quick_forecast),
}


# -- JSON-RPC loop ----------------------------------------------------------------------------------------------------

def handle(msg: dict) -> dict | None:
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if method is None or (mid is None and method.startswith("notifications/")):
        return None
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": PROTOCOL, "capabilities": {"tools": {}},
            "serverInfo": {"name": "prophet", "version": "1.0"},
            "instructions": "Prophet analyses and forecasts spreadsheets. Create a notebook, add files, then ask questions with prophet_ask."}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": [
            {"name": n, "description": d, "inputSchema": s} for n, (d, s, _) in TOOLS.items()]}}
    if method == "tools/call":
        name = params.get("name")
        if name not in TOOLS:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"unknown tool {name}"}}
        try:
            result = TOOLS[name][2](params.get("arguments") or {})
            text, err = json.dumps(result, indent=2, default=str), False
        except ProphetError as exc:
            text, err = str(exc), True
        except KeyError as exc:
            text, err = f"missing argument {exc}", True
        return {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": text}], "isError": err}}
    if mid is None:
        return None
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown method {method}"}}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            reply = handle(json.loads(line))
        except Exception as exc:  # a bad message must not end the session
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(exc)}}
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
