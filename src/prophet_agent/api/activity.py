"""What a mission is doing now, for the console's right-hand panel: the main thread, code runs, sub-agents and lanes, the
files it wrote (deduplicated and grouped by question) and the tools and skills this round used. Also one sub-agent's full
session (its brief, each plan, each tool result and each evaluation) for the panel's drill-down.

Motivation vs Logic
Motivation: the owner asked (2026-11-27) for a right-hand panel like Claude Desktop's: background tasks (the analysis running
now and the sub-agents running now), the output files kept there without duplicates and grouped by topic, the tools and skills
loaded this round, and a click on any sub-agent that shows its tasks and its whole session.
Logic: everything is read from the ledger, never from model calls. A step name carries its loop's tag (``plan:d0:3`` is the
lead, ``act:d1.a4:2`` is sub-agent a4), so steps and tool calls are attributed by tag. A sub-agent's brief comes from its
``agent.started`` event (written when it starts, so a running lane already has one) or, for runs before that event existed,
from its finished report. A file seen twice (same digest, or same name when no digest was stored) is listed once, under the
newest question that wrote it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from prophet_agent import mentions
from prophet_agent.runtime import names

STEP_TAG = re.compile(r"^(?P<kind>[a-z_]+):(?P<tag>d\d+(?:\.a\d+)*)(?::(?P<i>\d+))?")
AGENT_IN_TAG = re.compile(r"a(\d+)$")
CODE_TOOLS = {"sandbox_python", "forecast_report", "backtest", "panel_backtest", "run_python"}
FILE_KINDS = (("chart", (".png", ".svg", ".jpg", ".jpeg", ".gif", ".webp")), ("table", (".csv", ".xlsx", ".parquet", ".tsv")),
              ("document", (".docx", ".pdf", ".md", ".html", ".txt")), ("data", (".json",)))
TEXT_CHARS = 4000          # one observation in the drill-down; the full record stays in the ledger and the bundle


def _agent_of(step_name: str) -> str:
    """'lead' for the main loop, else the sub-agent id (the last aN in the tag)."""
    m = STEP_TAG.match(step_name or "")
    if not m:
        return "lead"
    a = AGENT_IN_TAG.search(m.group("tag"))
    return f"a{a.group(1)}" if a else "lead"


def _file_kind(path: str) -> str:
    low = path.lower()
    for kind, exts in FILE_KINDS:
        if low.endswith(exts):
            return kind
    return "other"


def _short(value: Any, n: int = TEXT_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= n else text[:n] + f" … [{len(text) - n:,} more characters]"


def _code_title(args: dict, tool: str) -> str:
    """The run's 1 to 4 word name: the planner's ``name``, else derived from the code; a tool without code goes by its name."""
    if args.get("code"):
        return names.for_code(args.get("code"), args.get("name"))
    return names.clean(args.get("name")) or ("Forecast report" if tool == "forecast_report"
                                             else tool.replace("_", " ").capitalize())


def _agent_name(agent_id: str, info: dict, report: dict | None) -> str:
    """The sub-agent's 1 to 4 word name: the one it started with, else the planner's, else derived from its brief."""
    rep = report or {}
    number = agent_id[1:] if agent_id[1:].isdigit() else agent_id
    fallback = f"{'Lane' if info.get('lane') else 'Sub-agent'} {number}"
    return names.for_task(info.get("task") or rep.get("task"), info.get("name") or rep.get("name"), fallback=fallback)


def _started_events(store, mission_id: str) -> dict[str, dict]:
    with store.lock:
        rows = store.conn.execute("SELECT payload_json, ts FROM events WHERE mission_id=? AND kind='agent.started' ORDER BY seq",
                                  (mission_id,)).fetchall()
    out: dict[str, dict] = {}
    for r in rows:
        p = json.loads(r["payload_json"] or "{}")
        if p.get("agent_id") and p["agent_id"] not in out:
            out[p["agent_id"]] = {**p, "started": r["ts"]}
    return out


def _steps(store, mission_id: str) -> list[dict]:
    with store.lock:
        rows = store.conn.execute("SELECT idx, name, status, output_json FROM steps WHERE mission_id=? ORDER BY idx",
                                  (mission_id,)).fetchall()
    return [{"idx": r["idx"], "name": r["name"], "status": r["status"],
             "output": (json.loads(r["output_json"]) or {}).get("v") if r["output_json"] else None} for r in rows]


def _calls(store, mission_id: str) -> list[dict]:
    with store.lock:
        rows = store.conn.execute("SELECT step_idx, tool, args_json, status, result_json, error, updated FROM tool_calls "
                                  "WHERE mission_id=? ORDER BY step_idx", (mission_id,)).fetchall()
    return [{"step": r["step_idx"], "tool": r["tool"], "args": json.loads(r["args_json"] or "null") or {},
             "status": r["status"], "result": json.loads(r["result_json"] or "null"), "error": r["error"],
             "updated": r["updated"]} for r in rows]


def _sub_reports(steps: list[dict]) -> dict[str, dict]:
    """Finished sub-agent reports by id, from the lead's act steps and the lanes' end steps."""
    out: dict[str, dict] = {}

    def take(obs: Any) -> None:
        if isinstance(obs, dict) and obs.get("kind") == "subagent" and obs.get("agent_id"):
            out[obs["agent_id"]] = obs

    for s in steps:
        v = s["output"]
        if s["name"].startswith("lane:"):
            take(v)
        elif isinstance(v, dict):
            take(v)
            for item in v.get("results") or v.get("reports") or []:
                take(item)
        elif isinstance(v, list):
            for item in v:
                take(item)
    return out


def _planned(steps: list[dict]) -> dict[str, dict]:
    """Briefs of sub-agents from the lead's delegate and lanes plans, for runs whose agents wrote no start event. Agents are
    numbered in the order they start, across the mission, one number per brief."""
    from prophet_agent.runtime.goal import _specs

    out: dict[str, dict] = {}
    n = 0
    for s in steps:
        v = s["output"]
        if _agent_of(s["name"]) != "lead" or not s["name"].startswith(("plan:", "auto:")) or not isinstance(v, dict):
            continue
        if v.get("action") == "lanes":
            specs, lane = [x for x in v.get("lanes") or [] if isinstance(x, dict)], True
        elif v.get("action") == "delegate":
            specs, lane = _specs(v), False
        else:
            continue
        for spec in specs:
            n += 1
            out[f"a{n}"] = {"task": spec.get("task"), "agent_type": spec.get("agent_type"),
                            "deliverable": spec.get("deliverable"), "lane": lane, "name": spec.get("name")}
    return out


def _verdicts(steps: list[dict]) -> dict[str, str]:
    """Each sub-agent's last evaluation verdict, from its own eval steps."""
    out: dict[str, str] = {}
    for s in steps:
        if s["name"].startswith(("eval:", "evalstall:")) and isinstance(s["output"], dict) and s["output"].get("verdict"):
            agent = _agent_of(s["name"])
            if agent != "lead":
                out[agent] = str(s["output"]["verdict"])
    return out


def _status(agent: str, report: dict | None, verdicts: dict[str, str], steps: list[dict], mission_status: str) -> str:
    if report:
        return str(report.get("status") or "done")
    mine = [s["idx"] for s in steps if _agent_of(s["name"]) == agent]
    after = mine and any(_agent_of(s["name"]) == "lead" and s["idx"] > max(mine) for s in steps)
    if mission_status == "running" and not after:
        return "running"
    if agent in verdicts:
        return verdicts[agent]
    return "done" if after else "stopped"


def _agents(steps: list[dict], started: dict[str, dict], mission_status: str) -> list[dict]:
    reports = _sub_reports(steps)
    planned, verdicts = _planned(steps), _verdicts(steps)
    started = {**planned, **started}
    seen: dict[str, dict] = {}
    for s in steps:
        agent = _agent_of(s["name"])
        if agent == "lead":
            continue
        row = seen.setdefault(agent, {"agent_id": agent, "steps": 0, "tool_calls": 0, "last": None})
        row["steps"] += 1
        v = s["output"]
        if s["name"].startswith("act:") and isinstance(v, dict) and v.get("kind") == "tool":
            row["tool_calls"] += 1
            row["last"] = v.get("tool")
    ids = sorted(set(seen) | set(started) | set(reports), key=lambda a: int(a[1:]) if a[1:].isdigit() else 0)
    out = []
    for a in ids:
        info, rep, row = started.get(a, {}), reports.get(a), seen.get(a, {})
        status = _status(a, rep, verdicts, steps, mission_status)
        out.append({"agent_id": a, "name": _agent_name(a, info, rep), "agent_type": info.get("agent_type") or (rep or {}).get("agent_type") or "general",
                    "task": str(info.get("task") or (rep or {}).get("task") or "")[:600], "lane": bool(info.get("lane")),
                    "status": status, "steps": row.get("steps", 0), "tool_calls": row.get("tool_calls", 0),
                    "last_tool": row.get("last"), "started": info.get("started")})
    return out


def _code_runs(calls: list[dict], by_idx: dict[int, str], mission_status: str) -> list[dict]:
    out = []
    for c in calls:
        if c["tool"] not in CODE_TOOLS:
            continue
        res = c["result"] if isinstance(c["result"], dict) else {}
        status = c["status"]
        if status == "started" and mission_status != "running":
            status = "stopped"
        elif status == "done" and res.get("ok") is False:
            status = "failed"
        out.append({"step": c["step"], "tool": c["tool"], "title": _code_title(c["args"], c["tool"]),
                    "agent": _agent_of(by_idx.get(c["step"], "")), "status": status, "run_id": res.get("run_id"),
                    "updated": c["updated"]})
    return out


def _files(calls: list[dict], sandbox_root: Path, merge) -> list[dict]:
    out = []
    for c in calls:
        res = c["result"] if isinstance(c["result"], dict) else {}
        run_id = res.get("run_id")
        if c["status"] != "done" or not run_id:
            continue
        listed = res.get("outputs") or res.get("files") or []
        for f in merge(listed, sandbox_root, run_id):
            path = str(f.get("path") or "")
            if not path.startswith("out/"):
                continue
            name = path[len("out/"):]
            out.append({"name": name, "path": path, "run_id": run_id, "sha256": f.get("sha256") or "",
                        "bytes": f.get("bytes"), "kind": _file_kind(name), "url": f"/api/artifacts/{run_id}/{name}",
                        "step": c["step"]})
    return out


def _tools_and_skills(mission: dict, calls: list[dict], started: dict[str, dict], titles: dict[str, str]) -> tuple[list, list]:
    from prophet_agent.runtime import agents as agent_types

    counts: dict[str, int] = {}
    skills: dict[str, str] = {}
    for c in calls:
        counts[c["tool"]] = counts.get(c["tool"], 0) + 1
        if c["tool"] == "analyst_skill" and c["status"] == "done" and isinstance(c["result"], dict) and c["result"].get("name"):
            skills.setdefault(c["result"]["name"], "read by the planner")
    for m in mentions.parse(mission.get("goal") or ""):
        if m.kind == "skill":
            skills.setdefault(m.id, "asked for in the question")
    for info in started.values():
        kind = agent_types.get(info.get("agent_type"))
        for name in (kind.skills if kind else ()):
            skills.setdefault(name, f"playbook of the {kind.name} sub-agents")
    tools = [{"name": n, "title": titles.get(n) or mentions.humanise(n), "calls": k}
             for n, k in sorted(counts.items(), key=lambda kv: -kv[1])]
    return tools, [{"name": n, "title": mentions.humanise(n), "source": why} for n, why in skills.items()]


def mission_activity(service, mission_ids: list[str], sandbox_root: Path, merge) -> dict:
    """The panel for a list of missions in order (a notebook's turns, or one quick forecast). Tasks, sub-agents, tools and
    skills come from the newest mission (this round); files come from all of them, grouped by the question that wrote them."""
    store = service.store
    current_id = mission_ids[-1]
    mission = store.mission(current_id)
    status = mission["status"]
    steps = _steps(store, current_id)
    by_idx = {s["idx"]: s["name"] for s in steps}
    calls = _calls(store, current_id)
    started = {**_planned(steps), **_started_events(store, current_id)}
    titles = {t.name: (t.title or t.name) for t in service.tools.values()}
    progress = None
    with store.lock:
        row = store.conn.execute("SELECT payload_json, ts FROM events WHERE mission_id=? AND kind='mission.progress' "
                                 "ORDER BY seq DESC LIMIT 1", (current_id,)).fetchone()
    if row:
        progress = {**json.loads(row["payload_json"] or "{}"), "ts": row["ts"]}
    lead_steps = [s for s in steps if _agent_of(s["name"]) == "lead" and s["name"].startswith(("act:", "plan:"))]
    last = lead_steps[-1] if lead_steps else None
    doing = None
    if last and isinstance(last["output"], dict):
        v = last["output"]
        doing = v.get("tool") or v.get("action")
    main = {"mission_id": current_id, "title": mentions.plain(mission["goal"])[:160], "status": status,
            "step": (progress or {}).get("step"), "tool_calls": (progress or {}).get("tool_calls"),
            "elapsed_s": (progress or {}).get("elapsed_s"), "next": (progress or {}).get("next"), "doing": doing}

    groups: list[dict] = []
    seen: set[str] = set()
    for mid in reversed(mission_ids):        # newest first, so a file written again is kept under the newest question
        try:
            m = store.mission(mid)
        except KeyError:
            continue
        files = []
        names: set[str] = set()              # a file rebuilt later in the same question replaces the earlier one
        listed = _files(_calls(store, mid), sandbox_root, merge)
        raster = {(f["run_id"], f["name"].rsplit(".", 1)[0].lower()) for f in listed if f["name"].lower().endswith(".png")}
        for f in reversed(listed):
            if f["name"].lower().endswith(".svg") and (f["run_id"], f["name"].rsplit(".", 1)[0].lower()) in raster:
                continue                     # one chart saved as PNG and SVG is one picture; the PNG stands for both
            key = f["sha256"] or f["name"]
            if key in seen or f["name"].lower() in names:
                continue
            seen.add(key)
            names.add(f["name"].lower())
            files.append(f)
        if files:
            files.sort(key=lambda f: (f["kind"], f["name"]))
            groups.append({"mission_id": mid, "topic": mentions.plain(m["goal"])[:120], "files": files})
    tools, skills = _tools_and_skills(mission, calls, started, titles)
    return {"main": main, "code": _code_runs(calls, by_idx, status)[-40:], "agents": _agents(steps, started, status),
            "files": groups, "tools": tools, "skills": skills}


def agent_session(service, mission_id: str, agent_id: str) -> dict:
    """One sub-agent's whole session in order: its brief, then each plan, tool result, board read and evaluation."""
    store = service.store
    mission = store.mission(mission_id)
    steps = _steps(store, mission_id)
    started = {**_planned(steps), **_started_events(store, mission_id)}
    report = _sub_reports(steps).get(agent_id)
    mine = [s for s in steps if _agent_of(s["name"]) == agent_id]
    if not mine and agent_id not in started and report is None:
        raise KeyError(agent_id)
    info = started.get(agent_id, {})
    entries: list[dict] = []
    for s in mine:
        kind = (STEP_TAG.match(s["name"]) or re.match(r"(?P<kind>[a-z_]+)", s["name"])).group("kind")
        v = s["output"]
        if kind == "plan" and isinstance(v, dict):
            entries.append({"role": "plan", "step": s["idx"], "action": v.get("action"), "tool": v.get("tool"),
                            "rationale": str(v.get("rationale") or "")[:1500],
                            "args": _short(v.get("args"), 1500) if v.get("args") is not None else None})
        elif kind == "act" and isinstance(v, dict):
            entries.append({"role": "result", "step": s["idx"], "tool": v.get("tool"),
                            "ok": "error" not in v, "text": _short(v.get("error") or v.get("result"))})
        elif kind in ("eval", "evalstall") and isinstance(v, dict):
            entries.append({"role": "evaluation", "step": s["idx"], "verdict": v.get("verdict"),
                            "text": str(v.get("reason") or "")[:1500]})
        elif kind == "inbox" and v:
            entries.append({"role": "board", "step": s["idx"], "text": _short(v, 1500)})
    return {"mission_id": mission_id, "agent_id": agent_id, "name": _agent_name(agent_id, info, report),
            "agent_type": info.get("agent_type") or (report or {}).get("agent_type") or "general",
            "task": info.get("task") or (report or {}).get("task") or "", "deliverable": info.get("deliverable") or "",
            "lane": bool(info.get("lane")),
            "status": _status(agent_id, report, _verdicts(steps), steps, mission["status"]),
            "answer": str((report or {}).get("answer") or "")[:8000],
            "findings": (report or {}).get("findings") or [], "entries": entries}
