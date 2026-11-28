"""Research lanes: up to 50 sub-agents working one mission at once, each with its own brief, loop and context, talking through a
shared board.

Motivation vs Logic
Motivation: the owner asked (2026-11-27) for a research mission to run as many lanes as a hard problem needs, not one
orchestration lane with at most four helpers per step. Published practice (Anthropic's multi-agent research system, 2025)
scales effort with the question: one agent and 3 to 10 calls for a fact, 2 to 4 sub-agents for a comparison, more than 10 for
broad research; its reported failures were too many sub-agents for simple questions, lanes duplicating each other, and vague
briefs. Its lead also waits for its workers, which this module keeps: the lead plans the fan-out, the lanes work, and the
lead reads a digest and the board when they finish.
Logic:
- A plan with action ``lanes`` lists up to MAX_LANES briefs. Each lane is a depth-1 loop (``goal._one_agent``) with its own
  history, memo, notes and tool list, on its own block of ledger steps (LANE_STRIDE numbers each), so concurrent lanes keep a
  replay-stable order and a resumed run reads every finished step back.
- Lanes run on a bounded pool (``workers``, default DEFAULT_WORKERS). The model client paces calls per key, so the pool bounds
  how many loops compete for it, not how many lanes there are; queued lanes start as others finish.
- The board (store table ``lane_board``) carries messages: a lane posts a finding, a question, an answer or a claim to all
  lanes, to one lane or to the lead. Each lane's loop reads its new messages through a ledger step before it plans
  (``inbox`` in goal._run), so what a lane saw is stored, and a replay sees the same messages. ``claim`` marks a topic as
  taken, the first claim wins, and a lane told a topic is taken works on another (overlap control).
- An identical read-only web call already finished by any lane is answered from that call (``shared_tools``), so twenty
  lanes do not fetch the same page twenty times.
- A lane's end is stored once, at the last number of its block. A lane that failed or ran out of its allowance reports that
  as its result and does not sink the others; only a cancel, a lost lease, an unknown outcome or the mission's own deadline
  stop the fan-out, after every lane has settled.
- The lead gets one ``lanes`` digest (status, findings, figures and files per lane, and the board messages sent to it) and
  each lane's report, which read_observation pages.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Callable

from prophet_agent.runtime.runner import BudgetExceeded, Cancelled, Ctx, OutcomeUnknown, Paused, Tool
from prophet_agent.runtime import names
from prophet_agent.runtime.store import LeaseLost

MAX_LANES = 50              # lanes one plan may start
MAX_LANE_AGENTS = 150       # sub-agents of every kind one mission may start, lanes included
DEFAULT_WORKERS = 12        # lanes running at once; the rest wait in order
LANE_STRIDE = 1_900         # ledger step numbers per lane: MAX_LANES * LANE_STRIDE stays inside one BRANCH_BLOCK
LANE_CALLS = 24             # delegated tool calls each lane adds to the part's allowance
MAX_POST = 1_200
MAX_TOPIC = 80
KINDS = ("finding", "question", "answer", "claim", "note")
INBOX_LIMIT = 20            # board messages one inbox step delivers
SHARED = frozenset({"web_search", "web_fetch", "web_research", "find_in_page", "search_open_data"})
_TOPIC_SPACE = re.compile(r"[^a-z0-9]+")


def topic_key(topic: str) -> str:
    return _TOPIC_SPACE.sub(" ", str(topic or "").lower()).strip()[:MAX_TOPIC]


BOARD_DOC = ("The research board shared by the lanes of this mission. action=post with to (all, lead or a lane id), kind "
             "(finding, question, answer, note) and text sends a message; action=claim with a topic marks it as yours (the "
             "first claim wins; when it is taken, cover another angle); action=read with after=<number> lists messages sent to "
             "you and every claimed topic. New messages for you also arrive before each of your steps.")


def board_tool(ctx: Ctx, sender: str, roster: list[str]) -> Tool:
    """The board for one lane (or the lead, sender "lead"): post, read and claim."""
    store, mission = ctx.store, ctx.mission_id
    targets = set(roster) | {"all", "lead"}

    def run(args: dict, key: str) -> dict:
        extra = sorted(set(args) - {"action", "to", "kind", "text", "topic", "after"})
        if extra:
            raise ValueError(f"unknown arguments {extra}; allowed: action, to, kind, text, topic, after")
        action = args.get("action") or "read"
        if action == "read":
            after = args.get("after", 0)
            if isinstance(after, bool) or not isinstance(after, int) or after < 0:
                raise ValueError("after must be a message number (0 or more)")
            msgs = store.board_read(mission, sender, after=after, limit=INBOX_LIMIT)
            claims = store.board_read(mission, None, kinds=("claim",), limit=200)
            return {"messages": msgs, "claimed_topics": [{"topic": c["topic"], "by": c["sender"]} for c in claims]}
        if action == "claim":
            topic = topic_key(args.get("topic", ""))
            if not topic:
                raise ValueError("topic must name what you will cover, for example 'rba minutes 2026'")
            taken = next((c for c in store.board_read(mission, None, kinds=("claim",), limit=500)
                          if c["topic"] == topic), None)
            if taken and taken["sender"] != sender:
                return {"claimed": False, "topic": topic, "by": taken["sender"],
                        "note": f"{taken['sender']} already covers this topic; take another angle or read its findings"}
            seq = store.board_post(mission, sender, "all", "claim", f"{sender} covers {topic}", topic=topic, key=key)
            return {"claimed": True, "topic": topic, "seq": seq}
        if action != "post":
            raise ValueError("action must be one of: post, read, claim")
        to = str(args.get("to") or "all").strip()
        if to not in targets:
            raise ValueError(f"to must be all, lead or one of the lanes: {', '.join(sorted(roster))}")
        kind = str(args.get("kind") or "finding").strip()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of: {', '.join(KINDS)}")
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must say what you found, ask or answer")
        if len(text) > MAX_POST:
            raise ValueError(f"a post is at most {MAX_POST} characters; keep figures with their source ids")
        seq = store.board_post(mission, sender, to, kind, text.strip(), topic=topic_key(args.get("topic", "")), key=key)
        return {"posted": True, "seq": seq, "to": to, "kind": kind}

    return Tool("board", run, idempotent=True, doc=BOARD_DOC, title="Research board", activity="Reading the research board")


def shared_tools(ctx: Ctx, tools: dict[str, Tool]) -> dict[str, Tool]:
    """Read-only web tools that answer an identical call another lane already finished from that call's stored result."""
    out = dict(tools)
    for name in SHARED & set(tools):
        tool = tools[name]

        def run(args: dict, key: str, _tool: Tool = tool) -> Any:
            found = ctx.store.tool_done_elsewhere(ctx.mission_id, _tool.name, args)
            if found is not None and isinstance(found[1], dict):
                return {**found[1], "shared_from_step": found[0],
                        "shared_note": "another lane made this exact call; its result is reused, not fetched again"}
            return _tool.fn(args, key)

        out[name] = replace(tool, fn=run)
    return out


def specs(plan: dict) -> list[dict]:
    keys = ("task", "agent_type", "deliverable", "constraints", "tools", "name")
    raw = plan.get("lanes")
    if not isinstance(raw, list):
        return []
    return [{k: t.get(k) for k in keys} for t in raw if isinstance(t, dict) and str(t.get("task") or "").strip()][:MAX_LANES]


def _lane_keys(spec: dict) -> set[str]:
    task = " ".join(str(spec.get("task") or "").lower().split())[:200]
    name = " ".join(str(spec.get("name") or "").lower().split())
    return {k for k in (f"task:{task}" if task else "", f"name:{name}" if name else "") if k}


# Words that every brief carries ("find the latest ... return each finding with its date and URL"), years and quarters.
_GENERIC = frozenset("""find search return each finding findings with its for and the of to in on or any from latest newest new
whether check now state that still more than over this these those their there what which who when where how dated date dates
quote quotes quoted exact url urls source sources value values unit units period periods figure figures list give report
reports only not use do does did be is are was were has have had an by at as it if no""".split())
MIN_TASK_WORDS = 5
SIMILAR_TASK = 0.5   # Jaccard of content words at which a reworded brief counts as one that already ran


def _task_words(task: str) -> frozenset[str]:
    ws = re.findall(r"[a-z][a-z0-9_.-]+", str(task or "").lower())
    return frozenset(w for w in ws if len(w) > 2 and w not in _GENERIC
                     and not re.fullmatch(r"(?:19|20)\d\d(?:-q\d)?|q\d", w))


def _similar(a: frozenset[str], b: frozenset[str]) -> float:
    # Briefs of a few words ("angle 2") say too little to call two of them the same task.
    return len(a & b) / len(a | b) if len(a) >= MIN_TASK_WORDS and len(b) >= MIN_TASK_WORDS else 0.0


def _not_run_before(lane_specs: list[dict], history: list[dict]) -> tuple[list[dict], list[str]]:
    """The briefs that did not run before in this mission, and a line for each one left out.

    Motivation: live AURATE (round 20261128-063203) planned lanes fifteen times in a row and re-sent the same six briefs
    ("RBA policy path" five times, "Risk scenarios" six times); VNUNEMP repeated briefs up to 140 times. Each re-run read the
    same pages and reported the same gaps. Logic: a brief whose task or name matches an earlier lane is left out when that
    lane wrote a report (met, partial or not met), or after two earlier attempts of any status. Read from the stored history,
    so a replay leaves out the same briefs."""
    earlier: dict[str, list[str]] = {}
    for h in history:
        if isinstance(h, dict) and h.get("kind") == "subagent" and h.get("lane"):
            for key in _lane_keys({"task": h.get("task"), "name": h.get("name")}):
                earlier.setdefault(key, []).append(str(h.get("status") or ""))
    reported = [(_task_words(h.get("task")), str(h.get("name") or h.get("task") or "")[:40], str(h.get("status") or ""))
                for h in history if isinstance(h, dict) and h.get("kind") == "subagent" and h.get("lane")
                and h.get("status") in ("met", "partial", "not_met")]
    keep, skipped = [], []
    for sp in lane_specs:
        seen = [st for key in _lane_keys(sp) for st in earlier.get(key, [])]
        if any(st in ("met", "partial", "not_met") for st in seen) or len(seen) >= 2:
            skipped.append(f"{str(sp.get('name') or sp.get('task'))[:60]}: {seen[-1] or 'ran'}")
            continue
        # Live VNUNEMP (round 20261128-173823): the same statistics-office search ran as lanes a2, a16 and a20 under new
        # names and new year ranges, each ending "DNS error"; its briefs shared 54 to 100 per cent of their content words,
        # while distinct topics shared at most 44 per cent.
        words = _task_words(sp.get("task"))
        twin = next(((nm, st) for w, nm, st in reported if _similar(words, w) >= SIMILAR_TASK), None)
        if twin:
            skipped.append(f"{str(sp.get('name') or sp.get('task'))[:60]}: the same task as the earlier lane '{twin[0]}' "
                           f"({twin[1]})")
            continue
        keep.append(sp)
    return keep, skipped


def roster_text(ids: list[str], lane_specs: list[dict]) -> str:
    lines = [f"- {i} ({sp.get('agent_type') or 'researcher'}): {str(sp.get('name') or '').strip() + ': ' if sp.get('name') else ''}"
             f"{' '.join(str(sp['task']).split())[:160]}" for i, sp in zip(ids, lane_specs)]
    return "The lanes working beside you, and what each covers:\n" + "\n".join(lines)


def lane_brief(lane_id: str, roster: str) -> str:
    return (f"You are research lane {lane_id}, one of several lanes working at the same time for a lead agent. Stay in your "
            "own lane: do not repeat another lane's task. Before a search angle that another lane could also take, claim it on "
            "the board (board action=claim); if it is taken, choose another. Post each solid finding to the board (kind "
            "finding, with the figure, unit, date and source id) as soon as you have it, so other lanes can build on it. Ask "
            "another lane or the lead with kind question; answer questions sent to you with kind answer. You cannot ask the "
            "person: post the question to the lead instead and carry on with what you can do.\n\n" + roster)


def _digest(results: list[dict], board_to_lead: list[dict]) -> dict:
    rows = []
    for r in results:
        rows.append({"lane": r.get("agent_id"), "name": r.get("name"), "type": r.get("agent_type"),
                     "status": r.get("status"), "task": str(r.get("task") or "")[:140],
                     "findings": [str(f)[:200] for f in (r.get("findings") or [])][:4],
                     "figures": [f for f in (r.get("figures") or [])][:6],
                     "files": (r.get("files") or [])[:6],
                     **({"unverified": r["unverified"][:6]} if r.get("unverified") else {}),
                     **({"error": str(r.get("reason") or "")[:200]} if r.get("status") in ("failed", "budget") else {})})
    done = sum(1 for r in results if r.get("status") == "met")
    return {"kind": "lanes", "lanes": rows, "summary": f"{len(results)} lanes: {done} met their brief",
            "to_lead": [{k: m[k] for k in ("seq", "sender", "kind", "text")} for m in board_to_lead][-30:],
            "note": "each lane's full report follows this digest; read_observation pages a long one"}


def run_lanes(ctx: Ctx, brain: Any, plan: dict, tools: dict[str, Tool], cfg: Any, context: str, history: list[dict],
              one_agent: Callable[..., dict], branch_block: int, workers: int = DEFAULT_WORKERS) -> list[dict]:
    """Runs the plan's lanes and returns their reports followed by one ``lanes`` digest."""
    if cfg.depth > 0:
        return [{"kind": "lanes", "error": "a lane cannot start lanes; do the work yourself or post to the lead"}]
    lane_specs = specs(plan)
    if not lane_specs:
        return [{"kind": "lanes", "error": "lanes needs a list of 1 to 50 briefs, each with a task"}]
    from prophet_agent.runtime import agents
    unknown = [sp["agent_type"] for sp in lane_specs if sp.get("agent_type") and agents.get(sp["agent_type"]) is None]
    if unknown:
        return [{"kind": "lanes", "error": f"unknown agent_type {unknown}; types: {', '.join(agents.TYPES)}"}]
    started = sum(1 for h in history if h.get("kind") == "subagent")
    if started + len(lane_specs) > MAX_LANE_AGENTS:
        return [{"kind": "lanes", "error": f"this mission already started {started} sub-agents; at most {MAX_LANE_AGENTS}. "
                 "Start fewer lanes or do the step yourself."}]
    lane_specs, skipped = _not_run_before(lane_specs, history)
    if not lane_specs:
        return [{"kind": "lanes", "error": "every lane in this plan already ran in this mission ("
                 + "; ".join(skipped[:8]) + "). Do not start them again: read their reports (kind subagent) and take the "
                 "next step yourself (download and test the drivers they found, forecast, record the driver map), or start "
                 "lanes with new tasks for what is still missing"}]
    for sp in lane_specs:
        sp["agent_type"] = sp.get("agent_type") or "researcher"
    ids = [f"a{started + 1 + k}" for k in range(len(lane_specs))]
    roster = roster_text(ids, lane_specs)
    # Each lane draws on the delegated allowance; a fan-out adds its share for this part, the same on every replay.
    budget = ctx.rt.budget
    if budget.get("max_delegated_calls") is not None:
        budget["max_delegated_calls"] = int(budget["max_delegated_calls"]) + LANE_CALLS * len(lane_specs)
    base = (ctx.step_idx + 2) * branch_block
    results: list[Any] = [None] * len(lane_specs)
    stops: list[BaseException | None] = [None] * len(lane_specs)

    def work(k: int) -> None:
        end_idx = base + (k + 1) * LANE_STRIDE - 1
        stored = ctx.store.step_record(ctx.mission_id, end_idx)
        if stored is not None and stored.get("status") == "done":
            results[k] = stored["v"]
            return
        lane_ctx = ctx.branch(base + k * LANE_STRIDE)
        lane = {"id": ids[k], "roster": [i for i in ids if i != ids[k]], "brief": lane_brief(ids[k], roster)}
        try:
            obs = one_agent(lane_ctx, brain, lane_specs[k], started + 1 + k, tools, cfg, context, history, lane=lane)
        except (Cancelled, LeaseLost, OutcomeUnknown, Paused) as exc:
            stops[k] = exc
            return
        except BudgetExceeded as exc:
            if exc.which == "max_seconds":
                stops[k] = exc          # the mission's own time is up: checkpoint, and the lane resumes on Continue
                return
            obs = {"kind": "subagent", "agent_id": ids[k],
                   "name": names.for_task(lane_specs[k]["task"], lane_specs[k].get("name")),
                   "agent_type": lane_specs[k]["agent_type"],
                   "task": str(lane_specs[k]["task"]).strip(), "status": "budget", "answer": "",
                   "reason": f"the lane reached its {exc.which} allowance", "evidence": []}
        except Exception as exc:        # noqa: BLE001 - one lane's failure is its result, not the mission's
            obs = {"kind": "subagent", "agent_id": ids[k],
                   "name": names.for_task(lane_specs[k]["task"], lane_specs[k].get("name")),
                   "agent_type": lane_specs[k]["agent_type"],
                   "task": str(lane_specs[k]["task"]).strip(), "status": "failed", "answer": "",
                   "reason": f"{type(exc).__name__}: {str(exc)[:300]}", "evidence": []}
        obs = {**obs, "lane": True}
        results[k] = ctx.branch(end_idx).step(f"lane:{ids[k]}:end", lambda _c: obs)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(lane_specs)))) as pool:
        list(pool.map(work, range(len(lane_specs))))
    for kind in (Cancelled, LeaseLost, OutcomeUnknown, Paused, BudgetExceeded):
        for exc in stops:
            if isinstance(exc, kind):
                raise exc
    to_lead = ctx.step("lanes:board", lambda c: c.store.board_read(c.mission_id, "lead", limit=200))
    reports = [r for r in results if isinstance(r, dict)]
    digest = _digest(reports, to_lead if isinstance(to_lead, list) else [])
    if skipped:
        digest["not_started_again"] = skipped
    return reports + [digest]
