"""Goal loop: plan, act with a tool, ask the user or delegate, then evaluate against the goal.

Motivation vs Logic
Motivation: a forecasting request is open-ended. The agent must decide the next action, use tools (including TimesFM),
know when the goal is met or cannot be met, and stop rather than spin when it is stuck. A fixed step cap that ends the
run with "reached its limit" is not a stop condition the person chose; it throws away a session that was still making
progress. So the loop never ends at a cap: it checkpoints, and the person continues or asks for the answer now.

Logic: each step has durable parts: plan (model chooses an action), act (tool call, question, or a delegated sub-agent),
then evaluate when the plan is finish. Every model call lives inside ``ctx.step`` so a resumed run reads the stored
decision. Observations are rebuilt from those stored outputs, so the planner sees the same history after a crash.

Context. The planner sees a memo of earlier steps plus the observations since the memo. When the uncompacted observations
outgrow ``VIEW_CHARS``, the older ones are folded into the memo by a durable ``compact:*`` step (the writer's free-text
call), and the newest ``KEEP_RECENT`` stay verbatim. The memo's text is stable between compactions, so the head of every
prompt (instructions, tool list, goal, memo) is byte-identical from one step to the next, which lets the provider's prefix
cache reuse it. The full history is kept for grounding and stall detection; only the view is compacted.

Sub-agents. ``delegate`` runs a nested loop with its own history, its own memo and a task-scoped tool list. The parent
gets back only the sub-agent's report (status, answer, reason). The sub-agent's evidence is kept on the parent
observation so the parent's figures can be grounded against it. A sub-agent cannot delegate again (depth 1).

Stall detection looks at the last ``stall_window`` tool actions: if they are identical, further iterations will not
change the outcome, so the loop stops with the best answer it has.
"""
from __future__ import annotations

import contextlib

import functools
import hashlib
import io
import json
import re
import time
import tokenize
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from prophet_agent import prompts
from prophet_agent.nim.errors import Exhausted, NimError, NotConfigured, redact
from prophet_agent.nim.transport import BUDGET_S, DEFAULT_CHAIN, preview_text

from . import contracts
from .grounding import snap, snap_note, unverified
from . import agents
from .questions import QUESTIONS_SCHEMA, validate_questions
from .runner import READ_ONLY, UNCHARGED, ArgumentRejected, BudgetExceeded, Cancelled, Ctx, Paused, Tool
from .store import LeaseLost, OutcomeUnknown, canonical

from prophet_agent.sandbox.runner import OUT_DIR, SANDBOX_ROOT

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["action", "rationale"],
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["call_tool", "ask_user", "delegate", "finish"]},
        "tool": {"type": "string", "minLength": 1},
        "args": {"type": "object"},
        "questions": QUESTIONS_SCHEMA["properties"]["questions"],
        "task": {"type": "string", "maxLength": 1200},
        "tools": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        "agent_type": {"type": "string", "enum": sorted(agents.TYPES)},
        "deliverable": {"type": "string", "maxLength": 600},
        "constraints": {"type": "string", "maxLength": 600},
        "resume": {"type": "string", "maxLength": 12},
        "tasks": {"type": "array", "maxItems": agents.MAX_PARALLEL, "items": {
            "type": "object", "required": ["task"], "additionalProperties": False,
            "properties": {"task": {"type": "string", "maxLength": 1200},
                           "agent_type": {"type": "string", "enum": sorted(agents.TYPES)},
                           "deliverable": {"type": "string", "maxLength": 600},
                           "constraints": {"type": "string", "maxLength": 600},
                           "tools": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                           "resume": {"type": "string", "maxLength": 12}}}},
        "rationale": {"type": "string", "maxLength": 400},
    },
    # A call_tool plan names the tool and its arguments. Without this a plan with no tool passed the check, ran nothing and
    # stalled the mission (seen live: Matilda sent three such plans in a row on the electricity scenario).
    "allOf": [{"if": {"properties": {"action": {"const": "call_tool"}}, "required": ["action"]},
               "then": {"required": ["tool", "args"]}}],
}

EVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["verdict", "reason"],
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["met", "not_met", "impossible"]},
        "reason": {"type": "string", "maxLength": 400},
        "answer": {"type": "string", "maxLength": 4000},
        "questions": QUESTIONS_SCHEMA["properties"]["questions"],
    },
}

# Round 3, A5-05: a sub-agent's final verdict also carries its findings and the figures it printed. Optional, so a model that
# leaves them out still gets a valid verdict; the sourcing of each figure is checked in code, not taken from the model.
SUB_EVAL_SCHEMA: dict[str, Any] = {
    **EVAL_SCHEMA,
    "properties": {
        **EVAL_SCHEMA["properties"],
        "findings": {"type": "array", "maxItems": 12, "items": {"type": "string", "maxLength": 300}},
        "figures": {"type": "array", "maxItems": 20, "items": {
            "type": "object", "required": ["name", "value"], "additionalProperties": False,
            "properties": {"name": {"type": "string", "maxLength": 120}, "value": {"type": "string", "maxLength": 60}}}},
    },
}

OBS_CHARS = 4000          # one observation, as the planner sees it
VIEW_CHARS = 16000        # uncompacted observations shown before the older ones fold into the memo
KEEP_RECENT = 4           # newest observations never compacted
MIN_FOLD = 3              # older observations folded per compaction, unless the view is far over its limit
HARD_VIEW_FACTOR = 2.5    # the view may grow to this many times VIEW_CHARS before a smaller fold is forced
MEMORY_CHARS = 8000       # hard cap on the memo; a memo over this is cut and says so
ASKS_KEEP_CHARS = 2000    # head of the memo (the ASKS section) that a cut to fit never removes
EXTERNAL_GOAL = re.compile(r"\b(?:wti|brent|crude|oil|gas|gold|silver|copper|commodit\w*|fx|forex|exchange rate|usd|eur|gbp|jpy|aud|"
                           r"stock|equit\w*|share price|s&p|nasdaq|index|yield|bond|treasur\w*|bitcoin|btc|crypto\w*|spot price|"
                           r"interest rate|inflation|cpi|unemployment|gdp|retail|prices?|electricity|energy|weather|"
                           r"rainfall|temperature|tariffs?|housing|tourism|passengers|traffic|footfall|"
                           r"pm2\.?5|pm10|particulates?|pollution|smog|air quality)\b", re.I)
# Method words, and the plain ways a person asks about the future (round 5, live business users: "what should I expect for
# next month?", "what will crude cost on average next month?" matched nothing, so those missions ran with no forecast
# checklist, no horizon check and no baseline gate, and one finished "verified" without a backtest).
FORECAST_GOAL = re.compile(
    r"forecast|predict|projection|project(?:ed)? (?:the|next)|next \d+|demand|outlook"
    r"|\bnext (?:few |couple of )?(?:days?|weeks?|months?|quarters?|years?|season|period)\b|\bnext month'?s\b"
    r"|\b(?:coming|upcoming|following) (?:days?|weeks?|months?|quarters?|year)\b"
    r"|\bwhat (?:will|would|should (?:i|we) expect|can (?:i|we) expect)\b|\bhow (?:much|many) will\b"
    r"|\bexpect (?:for|in|over|by|during) (?:the )?(?:next|coming|this)\b|\bgoing forward\b|\bbudget for\b", re.I)
FILES_GOAL = re.compile(r"\b(?:chart|graph|plot|workbook|spreadsheet|xlsx|docx|pdf|png|svg|report|deliverable)s?\b", re.I)
MAX_CORRECTIONS = 2       # times an answer with unmatched figures goes back to gather before it is repaired
DONE_WHEN = ("the evaluator judges the goal met", "the deterministic checks pass on the evidence "
             "(data inspected, forecast compared with baselines, forecast health repaired, deliverables present)",
             "every figure in the answer traces to a tool result")
MUST_NOT_CHANGE = ("synthetic or expected numbers are never presented as observed", "the 15 percent improvement bar is not moved")
STOP_REASONS = {"met": "done_verified", "partial": "partial_unverified", "impossible": "impossible", "stalled": "stalled",
                "checkpoint": "budget_part_spent"}
MAX_CONTRACT_BLOCKS = 2    # times a failed deterministic check may send a finished answer back before it leaves as partial
REPAIR_CALLS = 8          # one extra allowance, granted once, when a blocked answer meets a spent tool budget (2026-11-22)
MAX_PREMATURE_FINISHES = 2  # round 3, A5-04: finishes refused for checklist items still to do, per run, before the evaluator judges
MAX_REJECTED_FINISHES = 3  # finishes the evaluator may reject in a row before the work is written up as it stands
_LEVEL_SENTENCE = re.compile(r"centre line is a level|center line is a level", re.I)
_RATIONALE_JUDGED = re.compile(r"finish rationale|required language", re.I)
# A refusal that also names a missing result is not only the level sentence. Do not turn that into met.
_OTHER_GAP = re.compile(r"\b(total|workbook|backtest|chart|rows|table)\b", re.I)
# Decision R1 (the person, 2026-11-22): no fixed number of questions. A question reaches the person only when the planner
# declares it high-risk and says how a wrong guess changes the numbers; anything else the agent decides and states as an
# assumption. Replaces MAX_ASKS = 8, which was set after a mission asked 109 low-value questions (mission 0c7432c7).
HIGH_RISK = re.compile(r"\bhigh[- ]risk\b\s*[:,-]?\s*(.{20,})", re.I | re.S)
SUB_STEPS = 14            # default step budget of a delegated sub-agent
BRANCH_BLOCK = 100_000    # step numbers reserved per plan step for parallel sub-agents, and per agent within it
BRANCH_STRIDE = 20_000
# The horizon a goal asks for: "next 20 trading days", "60 days ahead", "horizon of 12 periods" (digits only). Units are
# read as written; converting them to the series' own periods needs the series step (see horizon_checks).
HORIZON_REQUEST = re.compile(
    r"\b(?:next|horizon of|ahead)\s+(\d{1,4})[\s-]*(trading\s+days?|business\s+days?|days?|weeks?|months?|quarters?|years?"
    r"|periods?|steps?)\b"
    r"|\b(\d{1,4})[\s-]+(trading\s+days?|business\s+days?|days?|weeks?|months?|quarters?|years?|periods?|steps?)\s+"
    r"(?:ahead|horizon)\b", re.I)
HORIZON_LABELS = {"period": "periods", "trading_day": "trading days", "day": "days", "week": "weeks", "month": "months",
                  "quarter": "quarters", "year": "years"}
UNIT_DAYS = {"day": 1.0, "trading_day": 7.0 / 5, "week": 7.0, "month": 365.25 / 12, "quarter": 365.25 / 4, "year": 365.25}
SERIES_STEPS = {"daily": 1.0, "day": 1.0, "weekly": 7.0, "week": 7.0, "monthly": 365.25 / 12, "quarterly": 365.25 / 4,
                "yearly": 365.25, "annual": 365.25}
CALENDAR_STEPS = frozenset({"monthly", "quarterly", "yearly", "annual"})   # periods that follow the calendar, not a fixed length
STEP_TEXT = re.compile(r"(\d+(?:\.\d+)?)\s*days?(?:\s+00:00:00)?")         # the profile's text for a regular step: "7 days 00:00:00"


@dataclass
class GoalConfig:
    max_iterations: int = 200         # steps in one part of the work; the part ends in a checkpoint
    stall_window: int = 3
    tool_budget: int | None = None    # tool calls allowed in total; the planner is told how many are left
    finish_now: bool = False          # the person asked for the best answer so far, not more work
    depth: int = 0                    # 0 for the goal, 1 for a delegated sub-agent (which cannot delegate)
    sub_steps: int = SUB_STEPS
    segment: int = 0                  # which part of the work this run is: 0 first, 1 after one Continue, ...
    view_chars: int = VIEW_CHARS      # uncompacted observations the planner sees before older ones fold into memory
    keep_recent: int = KEEP_RECENT    # newest observations that are never folded
    memory_chars: int = MEMORY_CHARS  # cap on the folded memory
    tag: str = ""                     # names this loop's steps: "d0" for the goal, "d1.a2" for sub-agent a2
    auto_delegate: bool = True        # the harness starts a researcher, panel analysts or a repair analyst itself (see _harness_plan)

    @property
    def end(self) -> int:
        """The step index at which this part ends (exclusive). Earlier parts replay from the ledger."""
        return self.max_iterations * (self.segment + 1)


@dataclass
class Memo:
    """What the planner remembers of the steps it no longer sees verbatim. ``covered`` counts history items folded in."""
    text: str = ""
    covered: int = 0


class Brain(Protocol):
    def decide(self, messages: list[dict], schema: dict, name: str) -> dict: ...

    # Optional: ``write(messages) -> str`` returns free text. The final answer is written through it when present,
    # because a schema-bound reply loses its line breaks. Compaction also uses it; without it, older steps are cut.


@dataclass
class Trace:
    observations: list[dict] = field(default_factory=list)


def _view(obs: dict) -> dict:
    """What the planner sees of an observation. A sub-agent shows its report, not its own working. A tool's full result
    ("full") is never shown: read_observation pages it and grounding reads it."""
    if obs.get("kind") == "subagent":
        return {k: obs[k] for k in ("kind", "agent_id", "agent_type", "task", "tools", "status", "answer", "reason",
                                    "steps", "files", "unverified") if k in obs}
    if "full" in obs:
        return {k: v for k, v in obs.items() if k != "full"}
    return obs


def _clip(obj: Any, step: int | None = None) -> str:
    s = canonical(obj)
    if len(s) <= OBS_CHARS:
        return s
    # Round 3, A5-02: the rest is not lost. The planner is told the step to page it back with read_observation.
    where = f"read_observation with step {step} pages the rest" if step else "clipped"
    return s[:OBS_CHARS] + f"...[clipped; {where}]"


PAGE_CHARS = 3000   # one read_observation page; with its framing it stays under OBS_CHARS, so a page is not clipped again
FULL_KEEP_CHARS = 200_000   # a full result longer than this is not kept beside its summary (the ledger still holds it)


def _observation_tool(history: list[dict]) -> Tool:
    """Pages a stored tool result back to the planner (round 3, A5-02). Reads the live history, which a resumed run rebuilds
    from the ledger, so the page is the same text. The result is a ledger entry like any tool call, but it is not charged."""
    def read(args: dict, key: str) -> dict:
        n = args.get("step")
        page = args.get("page", 1) or 1
        if not isinstance(n, int) or not 1 <= n <= len(history) or history[n - 1].get("kind") != "tool":
            raise ValueError(f"no observation at step {n}; read a tool result whose step the evidence shows")
        if not isinstance(page, int):
            raise ValueError("page must be a whole number")
        obs = history[n - 1]
        # Round 4 (live panels): the tool's own summary may already have cut the middle (sandbox stdout), so the full
        # result kept beside it is paged when there is one.
        whole = canonical(obs.get("full", obs["result"]) if "result" in obs else {"error": obs.get("error")})
        pages = max(1, -(-len(whole) // PAGE_CHARS))
        if not 1 <= page <= pages:
            raise ValueError(f"page {page} is outside 1..{pages} for step {n}")
        out = {"step": n, "page": page, "pages": pages, "text": whole[(page - 1) * PAGE_CHARS:page * PAGE_CHARS]}
        if page < pages:
            out["next"] = page + 1
        return out
    return Tool("read_observation", read, idempotent=True,
                doc="read_observation(step, page): the rest of a tool result the evidence clipped; step is the number the "
                    "clipped line names, page starts at 1 and the result says next when more pages follow")


def _sub_observation_tools(tools: dict[str, Tool], sub_history: list[dict]) -> dict[str, Tool]:
    """A sub-agent's tools with read_observation bound to its own history. Live (round 4 panels): sub-agents inherited the
    lead's pager, so step 5 of a sub-agent paged the lead's step 5."""
    return {**tools, "read_observation": _observation_tool(sub_history)}


def _tool_lines(tools: dict[str, Tool]) -> str:
    if not tools:
        return "- (none)"
    return "\n".join(f"- {name}: {tools[name].doc}" if tools[name].doc else f"- {name}" for name in sorted(tools))


def _evidence_lines(history: list[dict], start: int = 1) -> str:
    # Each observation is clipped on its own, so the goal and the tool list are never cut off by a long result.
    if not history:
        return "(none yet)"
    return "\n".join(f"{n}. {_clip(_view(obs), obs.get('step'))}" for n, obs in enumerate(history, start=start))


def _question_texts(questions: Any) -> list[tuple[str, str]]:
    """(header, question) pairs from a stored question set. Old ledger entries may lack the questions; they give none."""
    out = []
    for q in questions if isinstance(questions, list) else []:
        if isinstance(q, dict) and isinstance(q.get("question"), str):
            out.append((str(q.get("header", "")).strip().lower(), " ".join(q["question"].split()).lower()))
    return out


def _decisions(history: list[dict]) -> list[str]:
    """Every answer the person has given in this mission, verbatim and in order. Kept out of the folded memory on purpose:
    a decision that compaction drops would be asked again, which is the defect this guards against."""
    lines = []
    for h in history:
        if h.get("kind") not in ("ask_user", "evaluator_question") or not h.get("answers"):
            continue
        asked = [q for q in (h.get("questions") or []) if isinstance(q, dict) and q.get("question")]
        answers = canonical(h["answers"])
        if len(answers) > 300:
            answers = answers[:300] + "...[clipped]"
        topic = " / ".join(" ".join(q["question"].split())[:160] for q in asked) or "(question not stored)"
        lines.append(f"- {topic} -> {answers}")
    return lines


def _evidence_block(history: list[dict], memo: Memo) -> str:
    parts = []
    decided = _decisions(history)
    if decided:
        parts.append("Decisions the person has already made (verbatim; act on them, and do not ask about these again):\n"
                     + "\n".join(decided))
    if memo.text:
        parts.append("Memory of earlier steps (compacted; every figure is as a tool printed it):\n" + memo.text)
    parts.append("Evidence so far (oldest first):\n" + _evidence_lines(history[memo.covered:], start=memo.covered + 1))
    return "\n\n".join(parts)


# A new question is the same question as an earlier one when its header matches or its wording is nearly identical. Two
# questions about one decision are one decision: the second is dropped and the earlier answer stands.
SAME_QUESTION_RATIO = 0.75


# Values a question is about: dates, and numbers of three or more digits. Two questions that name the same dates and
# figures are about the same decision even when the wording differs (a re-phrased promo-window question is one decision).
_ANCHOR = re.compile(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{3,}(?:[.,]\d+)?")
ANCHOR_SHARE = 0.5


def _same_anchors(a: str, b: str) -> bool:
    """True when two questions name at least two of the same values and most of the values either one names."""
    xa, xb = set(_ANCHOR.findall(a)), set(_ANCHOR.findall(b))
    shared = xa & xb
    return len(shared) >= 2 and len(shared) / len(xa | xb) >= ANCHOR_SHARE


# Percentages (interval levels and bounds) and time units (horizons, periods) set what a question asks. Dates and plain figures are
# not included: live run 04 (test_goal_decisions) re-worded a promo question whose end date moved by one day and was the same decision.
_QUALIFIER = re.compile(r"\d+(?:[.,]\d+)*\s?%|\b(?:day|week|month|quarter|year|hour|minute)s?\b", re.I)


def _qualifiers(text: str) -> list[str]:
    """The percentages and time units a question names, in order-free form.

    Audit round 6m, item 2: "the 80% interval?" and "the 90% interval?" are alike in wording, so the stored answer to the first
    was replayed for the second ("not asked again"), and "horizon in weeks?" was replayed for "horizon in days?". A question that
    names other levels or units is a different question, whatever its wording or header."""
    return sorted(re.sub(r"\s", "", m.group()).lower() for m in _QUALIFIER.finditer(text))


def _already_asked(questions: list[dict], history: list[dict]) -> tuple[list[dict], list[str]]:
    """Split a new question set into the questions still open and the ones already asked and answered before."""
    earlier = [pair for h in history if h.get("kind") in ("ask_user", "evaluator_question") and h.get("answers")
               for pair in _question_texts(h.get("questions"))]
    if not earlier:
        return questions, []
    import difflib
    open_qs, repeated = [], []
    for q in questions:
        head = str(q.get("header", "")).strip().lower()
        text = " ".join(str(q.get("question", "")).split()).lower()
        qual = _qualifiers(text)
        same = any(_qualifiers(et) == qual
                   and ((head and head == eh) or difflib.SequenceMatcher(None, text, et).ratio() >= SAME_QUESTION_RATIO
                        or _same_anchors(text, et))
                   for eh, et in earlier)
        (repeated if same else open_qs).append(q)
    return open_qs, [str(q.get("question", ""))[:160] for q in repeated]


# Round 3 (A5-09, A3-10): live electricity mission, 2026-11-22, identical (tool, arguments) calls recurred 21 times (panel_backtest
# 6, sandbox_python 9). Web calls are re-run because the web changes; notes and the read-only tools change with the ledger.
REPEAT_EXEMPT = frozenset({"web_research", "web_search", "web_fetch", *READ_ONLY})
# A call whose result depends only on its arguments, so an identical call after an unrelated cell is still the same result.
PURE_TOOLS = frozenset({"sandbox_python"})


# Round 6m, goal audit item 1. A sandbox cell is the same call when its code, read as Python (comments and layout dropped), is the
# same and each carried file the code can read holds the same bytes. Live stored missions (data/prophet.sqlite, 2026-11-23): 553 of
# 4,917 sandbox cells had the code of an earlier cell. Auto-carry mounts every newest data file of the mission, so a cell that reads
# one file gained a carry entry each time another output appeared, and a re-run of an upstream cell wrote a new run folder with the
# same bytes. A comment that named the run changed the code text. None of these can change what the cell prints or writes.
_LAYOUT = frozenset({tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER})
# Code that reads the carried folder as a whole (a listing, a glob, a walk) depends on every carried file, not on the names it spells.
CARRY_LISTING = re.compile(r"carried|listdir|glob|scandir|walk|iterdir")
RUN_FOLDER = re.compile(r"(?<![0-9a-f])[0-9a-f]{32}(?=/)")    # a run folder in a path: data/sandbox/<run id>/program.py


@functools.lru_cache(maxsize=512)
def _code_form(code: str) -> str:
    """The code as Python reads it: its tokens, with comments and layout dropped. Text that does not tokenise is kept as it is."""
    try:
        return " ".join(t.string for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type not in _LAYOUT)
    except (tokenize.TokenError, SyntaxError):
        return code


def _output_digests(history: list[dict]) -> dict[tuple[str, str], str]:
    """The sha256 of each file a sandbox run wrote, by (run id, path), from the stored results. Read from the history only, so a
    resumed run makes the same decision whatever is still on disk. A result too large to keep has no digest here."""
    digests: dict[tuple[str, str], str] = {}
    for h in history:
        if not isinstance(h, dict) or h.get("kind") != "tool" or h.get("tool") != "sandbox_python":
            continue
        res = h.get("full", h.get("result"))
        if not isinstance(res, dict) or not isinstance(res.get("run_id"), str):
            continue
        for o in res.get("outputs") or []:
            if isinstance(o, dict) and isinstance(o.get("path"), str) and isinstance(o.get("sha256"), str):
                digests[(res["run_id"], o["path"])] = o["sha256"]
    return digests


def _cell_key(args: Any, digests: dict[tuple[str, str], str]) -> str:
    """What a sandbox cell computes from: its code, and each carried file the code can read, by its bytes where they are known and by
    its run where they are not. A carried file the code cannot read is left out, so mounting it again changes nothing."""
    if not isinstance(args, dict) or not isinstance(args.get("code"), str):
        return canonical(args)
    code = args["code"]
    listing = bool(CARRY_LISTING.search(code))
    carried = []
    for c in args.get("carry") or []:
        if not isinstance(c, dict):
            continue
        run, rel = str(c.get("run_id", "")), str(c.get("file", ""))
        name = rel.rsplit("/", 1)[-1]
        if not listing and name.rsplit(".", 1)[0] not in code:
            continue          # the name (or its stem, which the name contains) is not in the code: the cell cannot read this file
        carried.append([name, digests.get((run, rel)) or [run, rel]])
    return canonical({"code": _code_form(code), "carried": carried})


def _repeat_of(name: str, args: Any, history: list[dict]) -> int | None:
    """The 1-based position of the earlier identical call, or None. Read from the stored history, so a resumed run makes the
    same decision and sends nothing to the tool."""
    if name in REPEAT_EXEMPT:
        return None
    pure = name in PURE_TOOLS
    digests = _output_digests(history) if pure else {}
    want = _cell_key(args, digests) if pure else canonical(args)
    for pos in range(len(history), 0, -1):          # the most recent identical call
        h = history[pos - 1]
        if h.get("kind") != "tool" or h.get("tool") != name or h.get("repeat_of"):
            continue
        same = _cell_key(h.get("args"), digests) == want if pure else canonical(h.get("args")) == want
        if same:
            # A cell may have written the files the tool reads: live, a forecast was blocked until it ran again after the
            # inspection. So a call made after any sandbox cell is run again, not answered with a pointer.
            # sandbox_python is not one of them: its result depends only on its code and on the carried bytes it reads, so an
            # identical cell is answered with a pointer. Live Beijing e14af4bd (2026-11-23): 12 sandbox cells in a row re-ran the
            # same code after another cell had run, and the stall rule stopped the mission at 56 calls.
            if not pure and any(x.get("kind") == "tool" and x.get("tool") == "sandbox_python" for x in history[pos:]):
                return None
            return pos
    return None


def _repeat_obs(name: str, args: Any, history: list[dict], pos: int) -> dict:
    earlier = history[pos - 1]
    obs = {"kind": "tool", "tool": name, "args": args, "repeat_of": pos, "sig": _signature(earlier),
           "outcome": _progress_key(earlier),      # the stall key of the call it points to (round 7)
           "note": (f"this call was already made at step {pos}; it gave the same result and was not run again. Read that "
                    "result there, or change the arguments.")}
    if "error" in earlier:
        obs["error"] = earlier["error"]
        obs["note"] = obs["note"].replace("gave the same result", "gave the same error")
    if earlier.get("rejected"):
        obs["rejected"] = True     # an identical refused call is refused again, and is not charged
    if earlier.get("coverage_note"):
        # Live Beijing rep4: after the finish was refused for 3 of 12 stations, the same backtest was sent three more times.
        # The repeat says what the call is short of, not only that it was made.
        obs["coverage_note"] = earlier["coverage_note"]
        obs["note"] += " It is short of entities: " + earlier["coverage_note"]
    return obs


def _calls_used(history: list[dict]) -> int:
    """The lead agent's tool calls so far: the count the planner's allowance is held to. A delegated sub-agent's calls are
    not in it; they have their own grant (see ``_delegated_calls``). Round 3, A5-01: before, they were, so a lead that
    delegated four analysts was forced to finish with its own allowance unused. Reads of playbooks, notes and stored
    observations are not work, and an unknown-tool slip never reached the runtime."""
    return sum(1 for h in history if h.get("kind") == "tool" and h.get("tool") not in UNCHARGED
               and not str(h.get("error", "")).startswith("unknown tool")
               and "repeat_of" not in h and not h.get("rejected"))


def _delegated_calls(history: list[dict]) -> int:
    """Tool calls the lead's sub-agents made, counted against their own grant (round 3, A5-01)."""
    total = 0
    for h in history:
        if h.get("kind") == "subagent":
            total += _calls_used(h.get("evidence") or []) + _delegated_calls(h.get("evidence") or [])
    return total


def _error_signature(h: dict) -> str | None:
    """The last line of a failed tool call's error, with numbers and run ids removed, or None when the call did not fail."""
    if h.get("kind") != "tool":
        return None
    res = h.get("result")
    text = h.get("error") or (res.get("stderr_tail") if isinstance(res, dict) and res.get("ok") is False else None)
    if not text:
        return None
    lines = [ln for ln in str(text).strip().splitlines() if ln.strip()]
    # A whitespace-only error still failed: it gets one fixed signature, so repeats of it still count as one failure (round 6n).
    last = lines[-1] if lines else "the tool failed without a message"
    return re.sub(r"[0-9a-f]{32}|\d+", "#", last)[:160]


def _failing_line(h: dict) -> str:
    """The program line that raised, read from the traceback's last program.py frame ('' when it is not there)."""
    res = h.get("result")
    text = str(h.get("error") or (res.get("stderr_tail") if isinstance(res, dict) else "") or "")
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if "program.py" in lines[i] and i + 1 < len(lines):
            return lines[i + 1].strip()[:200]
    return ""


def _repeat_error_note(history: list[dict]) -> str:
    """When the newest tool calls failed the same way three times running, say so, quote the failing line and say what to do. One
    live mission rewrote its whole pipeline 28 times to meet the same wrong file path, another read a flags array as a dict 30 times.
    After five identical failures the advice is a fresh context: a delegated analyst starts without the failed attempts. The note
    changes the planner's approach; it blocks nothing."""
    run = []
    for h in reversed(history):
        if h.get("kind") != "tool":
            continue
        sig = _error_signature(h)
        if not sig or (run and sig != run[0][0]):
            break
        run.append((sig, h))
        if len(run) == 8:
            break
    if len(run) < 3:
        return ""
    line = _failing_line(run[0][1])
    text = ("The last " + str(len(run)) + " tool calls failed with the same error: " + run[0][0]
            + (f" at the line `{line}`" if line else "") + ". Do not rewrite the whole program again. Read the error, list what exists "
            "(files under in/ and in/carried, what the function returns), change only the failing line, and if a helper cannot do it, "
            "do that one step in plain pandas.")
    if len(run) >= 5:
        text += (" This is not converging: delegate the step to an analyst sub-agent (agent_type analyst) with a fresh context, giving "
                 "it the goal, the file and run id it needs, and this error, instead of trying a sixth time.")
    return text


AUTO_REPEAT = 5          # identical failures in a row after which the harness hands the step to a fresh analyst


def _panel_hint(history: list[dict], full: Callable[[str], list] | None = None) -> dict | None:
    """The grouping column of a long panel (several entities sharing dates), read from inspect_data's full profile, or None.
    The loop's history holds the planner's compact view of a result, so the full one is read from the ledger when given."""
    results = list(full("inspect_data")) if full is not None else []
    results += [h.get("result") for h in history if h.get("kind") == "tool" and h.get("tool") == "inspect_data"]
    for res in results:
        prof = res.get("profile") if isinstance(res, dict) else None
        if not isinstance(prof, dict):
            continue
        cols = prof.get("columns") or []
        time_col = next((c for c in cols if c.get("kind") == "time" and (c.get("duplicate_timestamps") or 0) > 0), None)
        groups = [c for c in cols if c.get("kind") == "text" and 2 <= (c.get("unique") or 0) <= 200]
        nums = [c["name"] for c in cols if c.get("kind") == "number"]
        if time_col and groups and nums:
            return {"file": res.get("file"), "time": time_col["name"], "entity": groups[0]["name"],
                    "entities": groups[0]["unique"], "values": nums[:6]}
    return None


LAST_DATE = re.compile(r"\blast=(\d{4}-\d{2}-\d{2})")


def _data_facts(history: list[dict], full: Callable[[str], list] | None = None) -> dict:
    """The inspected file names and the last date in their time columns, for a researcher's brief (round 3, A4-08 and
    A3-08: web_research scopes a regional family by region and places dated events against data_end). Reads the full
    inspect_data profile when the ledger is given, else the planner's compact column lines."""
    files, ends = [], []
    results = list(full("inspect_data")) if full is not None else []
    results += [h.get("result") for h in history if h.get("kind") == "tool" and h.get("tool") == "inspect_data"]
    for res in results:
        if not isinstance(res, dict):
            continue
        if res.get("file") and res["file"] not in files:
            files.append(str(res["file"]))
        prof = res.get("profile")
        cols = prof.get("columns") if isinstance(prof, dict) else res.get("columns")
        for c in cols or []:
            if isinstance(c, dict) and c.get("kind") == "time" and c.get("last"):
                ends.append(str(c["last"])[:10])
            elif isinstance(c, str) and ": time" in c and (m := LAST_DATE.search(c)):
                ends.append(m.group(1))
    valid = [e for e in ends if re.fullmatch(r"\d{4}-\d{2}-\d{2}", e)]
    return {"files": files[:4], "data_end": max(valid) if valid else None}


def _last_failure(history: list[dict]) -> tuple[str, str]:
    """(code of the newest failed sandbox cell, its last error line), for a hand-over brief."""
    for h in reversed(history):
        if h.get("kind") == "tool" and _error_signature(h):
            code = str((h.get("args") or {}).get("code", ""))[:2500]
            return code, _error_signature(h) or ""
    return "", ""


def _stored_step_name(ctx: Ctx) -> str | None:
    """The name of the step the ledger already holds at the next position, or None on a fresh run."""
    store, mid = getattr(ctx, "store", None), getattr(ctx, "mission_id", None)
    if store is None or mid is None:
        return None
    rec = store.step_record(mid, ctx.step_idx + 1)
    return rec["name"] if rec is not None and rec.get("status") == "done" else None


MODEL_WORK = re.compile(r"backtest|forecast|timesfm|control_lab|baseline", re.I)


def _agent_for(item: str) -> str:
    """The sub-agent type for a stuck checklist item. Live electricity rep5: "compare with the simple baselines
    (backtest_series)" went to an analyst, whose tools have no backtest, and it reported the tools were not available."""
    return "modeler" if MODEL_WORK.search(item) else "analyst"


def _harness_plan(goal: str, history: list[dict], tools: dict[str, Tool],
                  full: Callable[[str], list] | None = None, stuck: tuple | None = None) -> dict | None:
    """Delegations the harness starts itself instead of hoping the planner will. Live, the planner delegated once in 1,983
    decisions although its rules ask for it, so three situations are routed deterministically:
    1. a market or macro series with no outside research yet, once the data has been looked at: a researcher;
    2. a long panel (several entities sharing dates), once inspected and before any forecast: analysts on slices in parallel;
    3. the same failure five times in a row: a fresh-context analyst gets the failing cell and the error;
    4. ``stuck`` = (checklist item, idle calls): no item ticked for STUCK_CALLS calls, so a fresh analyst does that item.
    Each fires at most once per mission (or once per failure signature or item), and the decision is stored as a step."""
    inspected = any(h.get("kind") == "tool" and h.get("tool") in contracts.INSPECTION_TOOLS and not h.get("error") for h in history)
    auto = [h for h in history if h.get("kind") == "subagent" and h.get("auto")]
    kinds = {h.get("auto") for h in auto}
    if stuck and stuck[1] >= STUCK_CALLS and "sandbox_python" in tools and ("stuck:" + stuck[0]) not in kinds:
        card = _state_card(history)
        return {"action": "delegate", "auto": "stuck:" + stuck[0], "agent_type": _agent_for(stuck[0]),
                "rationale": f"no checklist item was ticked in {stuck[1]} calls; a fresh context does the next one",
                "task": "A lead agent working on this goal has gone " + str(stuck[1]) + " calls without finishing one step: "
                        + goal + "\n\nThe step to do now: " + stuck[0]
                        + (f" ({stuck[2]})" if len(stuck) > 2 and stuck[2] else "") + "\n\nWhat already exists (pass files in carry and "
                        "open them at in/carried/<file name>; do not redo this work):\n" + (card or "nothing saved yet")
                        + "\n\nDo exactly this step and nothing else, in as few cells as possible, from the saved files.",
                "deliverable": "The run id and files the step wrote, and in a few lines what each holds.",
                "constraints": "Do not forecast unless the step is a forecast. Do not re-profile what the files already hold."}
    if max(_error_run(history), _error_recurrence(history)) >= AUTO_REPEAT:
        code, err = _last_failure(history)
        key = "repeat:" + err
        if key not in kinds and "sandbox_python" in tools:
            return {"action": "delegate", "auto": key, "agent_type": "analyst",
                    "rationale": "the same step failed five times in a row; a fresh context gets it working",
                    "task": "A lead agent working on this goal could not get one analysis step to run: " + goal + "\n\nIts last "
                            "attempt failed with: " + err + "\n\nThe cell was:\n```python\n" + code + "\n```\nMake this step "
                            "work. Read the error, check what the inputs and helpers really are (print df.columns, df.dtypes, the "
                            "helper's return value), change what is wrong, and use plain pandas where a helper is unclear. Save "
                            "what the step was meant to produce under out/.",
                    "deliverable": "The run id and the files the working cell wrote, what each holds, the key figures it "
                                   "printed, and one sentence on what was wrong before.",
                    "constraints": "Do not forecast. Keep printed output short. Do not rewrite unrelated analysis."}
    if "table-claim" not in kinds and "state_table_claim" in tools:
        download_failed = any(h.get("kind") == "tool" and h.get("tool") == "download_web_data" and h.get("error") for h in history)
        claimed = any(h.get("kind") == "tool" and h.get("tool") == "state_table_claim" and not h.get("error") for h in history)
        if download_failed and not claimed:
            return {"action": "delegate", "auto": "table-claim", "agent_type": "principal_analyst",
                    "rationale": "a file download failed; the series name stays unbound until a claim is checked",
                    "task": "A file download failed while working on this goal: " + goal + "\nState the table claim "
                            "(entity, series name, measure). Diagnose each failed download. Fetch the page when the signal "
                            "is html_page, then extract its tables. Do not store a different series under the requested name. "
                            "Do not invent an address that no page listed.",
                    "deliverable": "The claim, each signal, and any candidate dataset id. Say when the series is still unbound.",
                    "constraints": "Do not forecast. Do not call bind_series."}
    if not inspected:
        return None
    if "research" not in kinds and EXTERNAL_GOAL.search(goal) and "web_research" in tools and not contracts.researched(history):
        facts = _data_facts(history, full)
        scope = ("\nThe data: file " + (", ".join(facts["files"]) or "unknown")
                 + (", last observed date " + facts["data_end"] if facts["data_end"] else "") + ". Pass to every "
                 "web_research call: region (the place the data covers, from the goal, the file name or the entity names; "
                 "for electricity, retail, weather, prices or pollution a call without a region runs no search), units, "
                 + ("data_end='" + facts["data_end"] + "', " if facts["data_end"] else "")
                 + "and horizon_end (the last date the forecast covers). Its events list says which dated events fall "
                 "inside the data and which inside the horizon.")
        return {"action": "delegate", "auto": "research", "agent_type": "researcher",
                "rationale": "a market or macro series is driven by things its history cannot show",
                "task": "Research what is driving the series in this goal now, and what is scheduled inside the forecast "
                        "horizon. Goal: " + goal + scope + "\nToday is " + time.strftime("%Y-%m-%d") + ". Plan four to six angles "
                        "first (current regime and level, policy path and the next decision dates, scheduled data releases, "
                        "supply or demand shocks, consensus or market-implied levels, the case against the consensus), "
                        "search them together, read primary sources first, then follow up on gaps.",
                "deliverable": "A dated briefing: (1) the current regime and level, each with source id, site and date; "
                               "(2) scheduled events inside the horizon with dates; (3) up to three upside and three "
                               "downside risks with the mechanism; (4) published consensus or market-implied levels with "
                               "source. Every number must appear in a fetched passage and carry its source id.",
                "constraints": "Do not forecast and do not analyse the data file. Primary sources (central banks, "
                               "statistics offices, EIA, OPEC, IEA) before news; report only figures a fetched passage contains."}
    panel = _panel_hint(history, full)
    forecast_done = any(contracts._forecast_record(h) for h in history)
    if "panel" not in kinds and panel and not forecast_done and "sandbox_python" in tools:
        n = int(panel["entities"])
        groups = min(4, n)
        size = -(-n // groups)
        tasks = []
        for k in range(groups):
            lo, hi = k * size + 1, min(n, (k + 1) * size)
            if lo > hi:
                break
            tasks.append({"agent_type": "analyst",
                          "task": f"File {panel['file']} is a long panel: entity column {panel['entity']!r}, time column "
                                  f"{panel['time']!r}, value columns {panel['values']}. Take the entities ranked {lo} to {hi} in "
                                  f"sorted order of {panel['entity']!r} (print their names first). For each one separately: "
                                  "profile coverage and gaps, zeros, outliers and level shifts, seasonality and trend, and how "
                                  "it differs from the others in your slice. Prepare them with `series, report = pl.panel(df_slice, "
                                  f"{panel['entity']!r}, {panel['time']!r}, <value column>)` (gaps filled by interpolation, "
                                  "never zero), print the report table, and save them with pl.save_panel(series, "
                                  "'out/series.json').",
                          "deliverable": "A table, one row per entity: rows, first and last date, missing share, gaps filled, "
                                         "outliers, level shifts with dates, dominant period, and one line on what is unusual; "
                                         "then the run id of out/series.json.",
                          "constraints": "Do not forecast. Only your slice of entities. Print short summaries, not raw rows."})
        return {"action": "delegate", "auto": "panel", "tasks": tasks,
                "rationale": f"{n} entities share dates; slices are analysed in parallel before modelling"}
    return None


def _state_card(history: list[dict]) -> str:
    """A short card of where the work stands, rebuilt from the evidence on every step so compaction can never drop it: the
    newest file of each kind with the run that wrote it (to pass in carry), the backtest verdict, and the sub-agents' reports.
    Live, missions reloaded the raw file in 38 of 40 cells and never carried a result, and one re-ran its whole pipeline to
    find a file it had already written."""
    newest: dict[str, str] = {}
    for run, paths in _runs_with_outputs(history):
        for p in paths:
            newest[p] = run
    lines = []
    if newest:
        keep = sorted(newest.items(), key=lambda kv: kv[0])[-12:]
        lines.append("Files you can carry (carry=[{run_id, file}], then open in/carried/<file name>): "
                     + "; ".join(f"{p} from run {r}" for p, r in keep))
    facts = contracts.trust_facts(history)
    if facts and facts.get("timesfm_mase") is not None:
        lines.append(f"Backtest so far: TimesFM MASE {facts['timesfm_mase']} vs {facts['best_baseline']} {facts['baseline_mase']} "
                     f"({facts.get('verdict') or 'no verdict'}); range coverage {facts.get('backtest_coverage')}.")
    subs = [h for h in history if h.get("kind") == "subagent"]
    for h in subs[-4:]:
        lines.append(f"Sub-agent {h.get('agent_id')} ({h.get('agent_type')}, {h.get('status')}): "
                     + " ".join(str(h.get("answer") or h.get("reason") or "").split())[:300]
                     + (f" Files: {', '.join(h.get('files', [])[:4])}." if h.get("files") else ""))
    return ("State of the work (computed from the evidence):\n- " + "\n- ".join(lines)) if lines else ""


def _skills_read_note(ctx: Ctx) -> str:
    """The playbooks already read in this mission, from the ledger (compaction drops their text from the view, and a planner that
    cannot see them reads them again: ten of twenty calls in one live run). Replay-safe: it only feeds a planner step."""
    seen: list[str] = []
    for a in ctx.store.tool_args(ctx.mission_id, "analyst_skill"):
        n = a.get("name") if isinstance(a, dict) else None
        if isinstance(n, str) and n not in seen:
            seen.append(n)
    if not seen:
        return ""
    return ("Playbooks already read in this mission: " + ", ".join(seen) + ". Their rules are applied in the work above and recorded in "
            "FOUND and notes; read one again only for an exact threshold you need now, never to refresh your memory.")


def _corrections(history: list[dict]) -> int:
    return sum(1 for h in history if h.get("kind") == "evaluation" and h.get("unverified"))


def _failure_text(h: dict) -> str | None:
    """Why a tool step failed, or None when it did not. A tool that raised has ``error``; an analysis cell that crashed
    returns normally with ``ok`` false and a traceback, which is a failure all the same (live: the same KeyError was sent six
    times because only raised errors were counted)."""
    if h.get("error"):
        return str(h["error"])[:200]
    res = h.get("result")
    if isinstance(res, dict) and res.get("ok") is False:
        tail = str(res.get("stderr_tail") or res.get("error") or "the step reported ok=false")
        lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
        return (lines[-1] if lines else "the step reported ok=false")[:200]
    return None


def _error_class(why: str) -> str:
    """The exception name in a failure line ("TypeError: list indices ..." gives "TypeError"). Lines without one give the
    whole line, so two identical unnamed failures still match."""
    head = why.split(":", 1)[0].strip()
    return head if head and " " not in head else why.strip()


def _class_streak(history: list[dict]) -> int:
    """How many of the latest tool calls failed with the same error class as the latest failure (round 4, A3-11)."""
    count, cls = 0, None
    for h in reversed(history):
        if h.get("kind") not in ("tool", "ask_user"):
            continue
        why = _failure_text(h)
        if why is None:
            break
        this = _error_class(why)
        if cls is None:
            cls = this
        if this != cls:
            break
        count += 1
    return count


_MODULE_MISS = re.compile(r"No module named '([A-Za-z_][\w]*)'|module '[\w.]+' has no attribute '([A-Za-z_]\w*)'")


def tool_module_hint(result: Any, names: Any) -> str | None:
    """A sandbox cell that imports, or reads an attribute named after, a tool gets the fix in the observation (round 4,
    A3-11). Live: `import backtest_series` and `prophet_stats.panel_backtest(...)` failed as module lookups."""
    if not isinstance(result, dict):
        return None
    text = str(result.get("stderr_tail") or result.get("stderr") or "")
    for m in _MODULE_MISS.finditer(text):
        name = m.group(1) or m.group(2)
        if name in names:
            return (f"'{name}' is a tool, not a Python module. Call it with call_tool (action call_tool, tool '{name}', "
                    "args as its JSON fields); do not import or call it from sandbox_python.")
    return None


def _failure_streak(history: list[dict]) -> tuple[int, str, str]:
    """How many of the latest tool calls or questions were refused in a row, the latest error, and the kind of step.

    A repeated failure must change the plan. A question the runtime refuses is a failure too: live run 06 sent the same
    refused question 80 times because only tool failures were counted.
    """
    streak, last, kind = 0, "", ""
    for h in reversed(history):
        if h.get("kind") not in ("tool", "ask_user"):
            continue
        why = _failure_text(h)
        if why is None:
            break
        streak += 1
        last = last or why
        kind = kind or str(h["kind"])
    return streak, last, kind


IDLE_CALLS = 8     # calls without a newly ticked checklist item before the planner is told which item to do next
STUCK_CALLS = 16   # ... and before the harness hands that item to a fresh analyst
DONE_IDLE_CALLS = 4  # calls after the last item was ticked before the planner is told to finish


def _idle_note(idle: int, items: list[dict]) -> str:
    """After IDLE_CALLS calls with no new checklist item ticked, name the next item. Live (Beijing): about 35 profiling cells
    in a row, each re-checking the last, and the ledger and backtest never ran."""
    nxt = next((it["item"] for it in items if it.get("next")), None) or \
        next((it["item"] for it in items if it.get("status") != "done"), None)
    if not nxt and items and idle >= DONE_IDLE_CALLS:
        # Every item ticked and the run goes on (live retail round 9: 7 of 7, then calls until the budget ran out).
        return (f"Every checklist item is ticked and {idle} calls have passed since. Finish now with what is established; "
                "only a figure the answer needs that no tool has printed justifies one more cell (print it as RESULT).")
    if idle < IDLE_CALLS or not nxt:
        return ""
    return (f"No checklist item has been ticked in the last {idle} calls. What is established is enough to move on: do this "
            f"next, in one call: {nxt}. Re-running or re-checking an earlier analysis is not progress.")


SHORT_CALLS_SLACK = 3    # calls beyond one per open checklist item before the run is told to stop exploring


def _short_calls_note(cfg: "GoalConfig", history: list[dict], items: list[dict]) -> str:
    """When the calls left barely cover the open checklist items, say which items to spend them on, in order. Live (WTI): 26
    calls went on analysis cells, the backtest never ran, and the answer had no forecast."""
    if cfg.tool_budget is None:
        return ""
    left = cfg.tool_budget - _calls_used(history)
    open_items = [it["item"] for it in items if it.get("status") != "done"]
    if not open_items or left > len(open_items) + SHORT_CALLS_SLACK:
        return ""
    return (f"Calls are short: {max(left, 0)} left in this part and {len(open_items)} checklist item(s) open. Spend them on "
            "these, in order, one call each, with no further exploratory cells: " + "; ".join(open_items[:6]) + ".")


def _budget_note(i: int, cfg: "GoalConfig", history: list[dict]) -> str:
    left = cfg.end - i
    if cfg.tool_budget is not None:
        left = min(left, cfg.tool_budget - _calls_used(history))
    note = f"Budget: step {i + 1} of this part of the work; {max(left, 0)} steps left in it."
    streak, err, kind = _failure_streak(history)
    same = _class_streak(history)
    if streak >= 2 and kind == "ask_user":
        note += (f" The last {streak} questions were refused ({err}). Do not send that question again. Fix its shape, or "
                 "decide without asking and say what you assumed.")
    elif same >= 2 or streak >= 3:
        # Round 4, A3-11: the probe is due at two failures of one error class, not three (live retail: three TypeErrors in a
        # row before any probe). A streak of three of mixed classes keeps the probe too.
        count = same if same >= 2 else streak
        note += (f" The last {count} calls failed in a row ({err}). Stop writing the large step again. Run one small cell "
                 "that only prints the structure that failed (print(type(x)); print(list(x)[:20]) or x.keys()), read it, "
                 "then fix that one line; keep the part that worked in a separate cell so it is not re-run.")
    elif streak >= 2:
        note += (f" The last {streak} calls failed ({err}). Do not repeat them: change the arguments or the approach, "
                 "or ask the person.")
    if left <= 2:
        note += " Stop exploring. Finish with what is established and say plainly what is missing."
    return note


def _plan_messages(goal: str, tools: dict[str, Tool], history: list[dict], memo: Memo, context: str = "",
                   budget: str = "") -> list[dict]:
    # Stable head first (instructions, tools, goal, context, memo), volatile tail last: the prefix cache reuses the head.
    system = prompts.current("planner").text
    user = (f"Goal:\n{goal}\n\n{context + chr(10) + chr(10) if context else ''}Available tools:\n{_tool_lines(tools)}\n\n"
            f"{_evidence_block(history, memo)}" + (f"\n\n{budget}" if budget else ""))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


MAX_ASKED = 5


def asked_questions(goal: str) -> list[str]:
    """The questions the person wrote in the goal, in their order: each sentence that ends with a question mark.

    Live retail_biz (2026-11-23): "Sales feel like they've been slipping lately. Is that real, or just the usual ups and
    downs? And what should I expect for next month?" The answer was a forecast report and never said whether the slip was
    real. Questions a person asks are what the answer is judged by first."""
    units = re.split(r"(?<=[.!?])\s+", " ".join((goal or "").split()))
    return [u.strip()[:200] for u in units if u.strip().endswith("?")][:MAX_ASKED]


# A question about whether a movement is genuine. Live retail_biz rep4 (2026-11-23): "is that real?" was answered by comparing
# the last 60 days with the 60 before, ignoring the weekly and yearly cycle; ps.change_test exists for exactly this.
CHANGE_QUESTION = re.compile(r"\b(?:is|was|are) (?:that|this|it|the \w+) (?:real|genuine|significant|just noise)"
                             r"|\breal or (?:noise|random)|\bslipping|\bdropp(?:ing|ed)|\bdeclin(?:ing|ed)|\bris(?:ing|en)\b"
                             r"|\bgrow(?:ing|n)\b|\bsignificant", re.I)


def _asked_block(goal: str, for_whom: str) -> str:
    qs = asked_questions(goal)
    if not qs:
        return ""
    listed = "\n".join(f"{n}. {q}" for n, q in enumerate(qs, 1))
    if for_whom == "evaluator":
        return ("\n\nThe person asked these questions. The goal is met only when the evidence answers each one directly "
                "(a yes or no, a number, or a plain statement), not only the forecast around it:\n" + listed)
    if for_whom == "composer":
        return ("\n\nThe person asked these questions. Open the answer by answering each one directly, in their order and in "
                "plain words, from the evidence; say so when the evidence cannot answer one. When the evidence contradicts "
                "a question's premise (the person says sales are slipping and the measured change is a rise), say that "
                "first, in those words, before any verdict:\n" + listed)
    tests = [n for n, q in enumerate(qs, 1) if CHANGE_QUESTION.search(q)]
    hint = ("\nQuestion " + " and ".join(map(str, tests)) + " asks whether a change is real: answer it with ps.change_test in "
            "the sandbox (the latest weeks against the same weeks a cycle earlier, judged against the usual change), not by "
            "comparing two raw windows. Quote its verdict and p-value.") if tests else ""
    return "Questions the person asked (the answer must address each one with evidence):\n" + listed + hint


def _health_pin(history: list[dict]) -> str:
    """The stored health verdict, pinned so the evaluator cannot invent one from the finish note.

    SciRes-20261124-124543: the judge read the 400-character rationale and applied the flat_expected
    level sentence to a check verdict. The verdict field is the fact. The rationale is not the answer.
    """
    verdict = _stored_health_verdict(history)
    if not verdict:
        return ""
    return (f"Stored forecast health verdict: {verdict}. Judge this field. Do not read a different verdict from the "
            "causes text or from the finish rationale. The finish rationale is not the answer. The level sentence is "
            "required only when this verdict is flat_expected.\n\n")


def _answer_channel(history: list[dict]) -> str:
    """The text the evaluator judges. It is built from the stored verdict, not from the finish note."""
    verdict = _stored_health_verdict(history)
    if not verdict:
        return ""
    if verdict == "flat_expected":
        line = "The centre line is a level because the history shows no repeating pattern."
    else:
        line = "The level sentence is not required for this verdict."
    return (f"Answer under test (judge this text, not the finish rationale): The health verdict is {verdict}. "
            f"{line}\n\n")


def _eval_messages(goal: str, history: list[dict], memo: Memo, context: str = "") -> list[dict]:
    system = prompts.current("evaluator").text
    channel = "" if "Written answer to judge:" in (context or "") else _answer_channel(history)
    user = (f"Goal:\n{goal}\n\n{context + chr(10) + chr(10) if context else ''}{_health_pin(history)}"
            f"{channel}{_evidence_block(history, memo)}"
            + _asked_block(goal, "evaluator"))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _repair_messages(goal: str, history: list[dict], memo: Memo, answer: str, bad: list[str],
                     context: str = "") -> list[dict]:
    system = prompts.current("evaluator").text
    user = (f"Goal:\n{goal}\n\n{context + chr(10) + chr(10) if context else ''}{_evidence_block(history, memo)}\n\n"
            f"Your draft answer:\n{answer}\n\n"
            f"These figures in the draft do not appear in the evidence: {', '.join(bad)}. Rewrite the answer and remove "
            "each of them, row numbers included. Describe a row in words if the evidence gives no number for it. Use only "
            "figures that appear in the evidence above, and say plainly when a figure is not available. Return the "
            "verdict met with the rewritten answer.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


OUTSIDE_SOURCES = 8        # web sources shown to the composer
OUTSIDE_PASSAGE = 320      # characters of the best passage per source


def _outside_block(ctx: Ctx | None) -> str:
    """The web sources this mission read, as one pinned block for the composer, taken from the ledger so compaction cannot drop them
    (live: research ran, yet the answer said no outside drivers were considered). Newest research first, deduplicated by url."""
    if ctx is None:
        return ""
    seen: dict[str, str] = {}
    for res in reversed(ctx.store.tool_results(ctx.mission_id, "web_research")):
        for src in (res.get("sources") or []) if isinstance(res, dict) else []:
            url = str(src.get("final_url") or src.get("url") or "")
            if not url or url in seen:
                continue
            best = (src.get("passages") or [{}])[0].get("text", "")
            host = re.sub(r"^https?://(?:www\.)?", "", url).split("/")[0]
            seen[url] = (f"- {src.get('title', '')[:140]} | {host} | {src.get('date') or 'undated'} | tier {src.get('tier', '?')}: "
                         f"{' '.join(str(best).split())[:OUTSIDE_PASSAGE]}")
    if not seen:
        return ""
    return "Outside sources read by the research step (cite by site and date; context, never a forecast input):\n" + "\n".join(list(seen.values())[:OUTSIDE_SOURCES])


def _gap_draft(verdict: dict) -> str:
    """The composer's draft when the run stops short of "met". The evaluator's answer when it wrote one; otherwise an instruction
    to write up what the evidence establishes, with the open gap last. Live electricity rep9 (2026-11-23): the draft was the
    evaluator's reason alone, so the person got one paragraph about missing ledger entries and none of the panel backtest or the
    forecast pack the run had built."""
    answer, reason = verdict.get("answer", "") or "", verdict.get("reason", "") or ""
    if answer.strip():
        return answer
    # The gap is described without its figures. Live WTI rep10 (2026-11-23): the reason named the unsupported figures ("2.1%
    # outliers, drift MASE 7.12"), the composer copied it into 'Not settled', and the answer failed the figure check on them.
    reason = re.sub(r"\s*\([^()]*\d[^()]*\)", "", reason)
    reason = re.sub(r"[$\u00a3\u20ac]?-?\d[\d,]*(?:\.\d+)?\s?%?", "", reason)
    reason = re.sub(r"\s{2,}", " ", reason).strip()
    return ("(No draft was written.) Write the answer from the evidence above: every result the analysis established for the "
            "goal (data checks, backtest scores against the simple rules, the forecast and its range, files), as fully as for a "
            "finished answer. End with a short section headed 'Not settled' that says, in plain words: " + (reason or "the goal is "
            "not fully met."))


def _compose_messages(goal: str, history: list[dict], memo: Memo, context: str, draft: str,
                      bad: list[str] | None = None, ctx: Ctx | None = None, fixes: str | None = None) -> list[dict]:
    system = prompts.current("composer").text
    outside = _outside_block(ctx)
    trust = contracts.trust_statement(history)
    if trust:
        outside = ("Accuracy facts (computed; an accuracy paragraph with exactly these figures is appended after your answer, so "
                   "do not restate coverage or MASE figures that differ from it, and never call the forecast better than a simple rule "
                   "unless these facts say so. When they give the total's 80% range, that is the range to quote; any other "
                   "bound is named for what it assumes. When they say the measure is a level, give no total over days or "
                   "series, only averages and each day's or series' own values):\n" + trust + ("\n\n" + outside if outside else ""))
    user = (f"Goal:\n{goal}\n\n{context + chr(10) + chr(10) if context else ''}{_evidence_block(history, memo)}"
            f"\n\n{outside + chr(10) + chr(10) if outside else ''}Draft (the content to keep; its formatting was lost):\n{draft}"
            + _asked_block(goal, "composer"))
    if bad:
        user += (f"\n\nThese figures in the draft do not appear in the evidence: {', '.join(bad)}. Remove each of them, "
                 "row numbers included. Describe a row in words if the evidence gives no number for it.")
    if fixes:
        user += (f"\n\nThese statements in the draft disagree with the stored analysis output: {fixes}. Rewrite each one to say "
                 "what the output says, or take it out. Keep everything else.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user + "\n\nWrite the final answer in Markdown."}]


def _decide_plan(brain: Any, goal: str, tools: dict[str, Tool], history: list[dict], memo: Memo, context: str,
                 note: str, *, finish_now: bool, out_of_calls: bool) -> dict:
    """The planner's decision for one step, made inside a stored step. Forced finishes never reach a tool: when the
    person asked for the answer now, or the tool allowance is spent, the decision is a finish."""
    if finish_now:
        return {"action": "finish", "rationale": "the person asked for the answer so far"}
    if out_of_calls:
        # No tool call is left, so planning another would only end the run with nothing to show.
        return {"action": "finish", "rationale": "tool budget used"}
    return brain.decide(_plan_messages(goal, tools, history, memo, context, note), PLAN_SCHEMA, "plan")


def _compact_messages(goal: str, previous: str, chunk: list[dict]) -> list[dict]:
    system = prompts.current("compactor").text
    lines = _evidence_lines(chunk, start=1)
    user = (f"Goal:\n{goal}\n\nMemory so far:\n{previous or '(nothing yet)'}\n\n"
            f"New observations to fold in (oldest first):\n{lines}\n\nWrite the updated memory.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


STUB_CHARS = 240   # one folded observation, in the memo


def _short(value: Any) -> str:
    text = value if isinstance(value, str) else canonical(value)
    return text[:60] + ("..." if len(text) > 60 else "")


def _stub(obs: dict, n: int) -> str:
    """One observation folded into the memo with no model call (round 3, A5-03). It keeps what the planner relies on: the call,
    its key result fields and its step, so read_observation can page the full result back. A pure function of the observation."""
    if obs.get("kind") == "subagent":
        return (f"{n}. sub-agent {obs.get('agent_id')} ({obs.get('agent_type')}) {obs.get('status')}: "
                f"{str(obs.get('answer', ''))[:STUB_CHARS]}")
    if obs.get("kind") == "tool":
        step = obs.get("step", n)
        args = canonical(obs.get("args"))[:120]
        if "error" in obs:
            body = "error: " + str(obs["error"])[:160]
        else:
            res = obs.get("result")
            body = "; ".join(f"{k}={_short(v)}" for k, v in list(res.items())[:6]) if isinstance(res, dict) else _short(res)
        return f"{n}. {obs.get('tool')}({args}) -> {body} [step {step}: read_observation pages the full result]"
    return f"{n}. {obs.get('kind')}: {canonical(_view(obs))[:STUB_CHARS]}"


def _compact(ctx: Ctx, brain: Any, goal: str, history: list[dict], memo: Memo, tag: str, *,
             view_chars: int = VIEW_CHARS, keep_recent: int = KEEP_RECENT,
             memory_chars: int = MEMORY_CHARS, boundary: bool = False) -> None:
    """Folds the observations older than the newest KEEP_RECENT into the memo once they outgrow the view. Round 3, A5-03: between
    parts this is deterministic (stubs, no model call); the model writes the memo only at a part boundary. Durable: the memo
    text comes from a stored step at a boundary, and from the stored history otherwise, so a resumed run reads the same memo."""
    pending = history[memo.covered:]
    cut = len(history) - keep_recent
    total = sum(len(_clip(_view(o))) for o in pending)
    if cut <= memo.covered or total <= view_chars:
        return
    # Fold in batches. With large cells every step is over the limit, and folding one observation per step doubles the
    # model calls (live: a compaction after every plan). A single older observation waits unless the view is far over.
    if cut - memo.covered < MIN_FOLD and total <= HARD_VIEW_FACTOR * view_chars:
        return
    chunk = history[memo.covered:cut]
    if not boundary:
        stubs = "\n".join(_stub(o, memo.covered + k + 1) for k, o in enumerate(chunk))
        text = (memo.text + "\n" if memo.text else "") + stubs
    elif (write := getattr(brain, "write", None)) is None:
        text = (memo.text + f"\n[{len(chunk)} earlier observations were left out of this view; the full record is kept.]")
    else:
        text = ctx.step(f"compact:{tag}:{cut}", lambda c: write(_compact_messages(goal, memo.text, chunk)))
    text = str(text or "").strip()
    if len(text) > memory_chars:
        # The head holds the person's asks, copied unchanged from one memory to the next: it is what a cut must not lose.
        keep_head = min(ASKS_KEEP_CHARS, memory_chars // 3)
        text = text[:keep_head] + "\n[middle of the memory cut to fit]\n" + text[-(memory_chars - keep_head):]
    memo.text = text
    memo.covered = cut


def _usable_questions(verdict: dict) -> list[dict] | None:
    """The evaluator's questions, if it asked any and they pass the same validator as the planner's."""
    qs = verdict.get("questions")
    if not qs or validate_questions({"questions": qs}):
        return None
    return qs


def _missing_figures_text(bad: list[str]) -> str:
    return (f"The draft answer states figures that no tool output printed: {', '.join(bad)}. Either compute them with a "
            "tool and print the exact numbers, or remove them. Do not restate them without a source.")


def _compose(ctx: Ctx, brain: Any, step: str, goal: str, history: list[dict], memo: Memo, context: str, draft: str,
             bad: list[str] | None = None, fixes: str | None = None) -> str | None:
    """The finished answer as plain text. A schema-bound reply cannot carry line breaks (the decoder drops them), so the
    verdict's answer is only a draft; the text the person reads is written in a free-text call. None: the brain cannot."""
    write = getattr(brain, "write", None)
    if write is None or not draft.strip():
        return None
    reveal = getattr(brain, "reveal", None)

    def run(_c: Ctx) -> str:
        # Built inside the step, so a replay returns the stored answer and does not call the model again.
        messages = _compose_messages(goal, history, memo, context, draft, bad, ctx, fixes)
        # Only the lead's answer is shown as it arrives. A sub-agent's write and the memo compaction stay buffered.
        if reveal is None or ctx.lane != "lead":
            return write(messages)
        with reveal():
            return write(messages)

    return ctx.step(step, run)


VOLATILE_KEYS = ("run_id", "outputs", "manifest", "files")   # differ on every run of the same call, so a repeat is still a repeat


def _signature(obs: dict) -> str:
    if "sig" in obs:                 # a pointer carries the signature of the call it points to, so stall detection sees it
        return obs["sig"]
    res = obs.get("result")
    if isinstance(res, dict):
        res = {k: v for k, v in res.items() if k not in VOLATILE_KEYS}
    args = obs.get("args")
    if obs.get("tool") == "sandbox_python" and isinstance(args, dict):
        # The carry list names the run ids of earlier cells, and those change on every run. Live Beijing 3581d07d (2026-11-23):
        # the same ledger cell ran 57 times with new carry ids each time, so no window of identical calls was ever seen.
        # Round 6m: the code is compared as Python reads it, and a run folder in the output (a traceback names one) is not an
        # outcome. Before, an identical failure in another folder looked like progress, and the stall rule never fired.
        args = {k: (_code_form(v) if k == "code" and isinstance(v, str) else v) for k, v in args.items() if k != "carry"}
        sig = canonical({"tool": obs.get("tool"), "args": args, "result": res, "error": obs.get("error")})
        return RUN_FOLDER.sub("<run>", sig)
    return canonical({"tool": obs.get("tool"), "args": args, "result": res, "error": obs.get("error")})


def _error_run(history: list[dict]) -> int:
    """How many of the newest tool calls in a row failed with the same error signature."""
    first, n = None, 0
    for h in reversed(history):
        if h.get("kind") != "tool":
            continue
        sig = _error_signature(h)
        if not sig or (first is not None and sig != first):
            break
        first, n = sig, n + 1
    return n


RECUR_WINDOW = 10   # recent calls of one tool in which a recurring failure is counted


def _error_recurrence(history: list[dict], window: int = RECUR_WINDOW) -> int:
    """How often the newest failure's signature appears among the last ``window`` calls of the same tool, whatever ran in
    between. Live (WTI): one broken line was resent 8 times between other calls, so a count of failures strictly in a row
    never reached the hand-over threshold."""
    calls = [h for h in history if h.get("kind") == "tool"]
    if not calls or not (sig := _error_signature(calls[-1])):
        return 0
    same = [h for h in calls if h.get("tool") == calls[-1].get("tool")][-window:]
    return sum(_error_signature(h) == sig for h in same)


def _runs_with_outputs(history: list[dict]) -> list[tuple[str, set[str]]]:
    """(run id, output paths) of every analysis or report step in this mission, oldest first."""
    runs: list[tuple[str, set[str]]] = []
    for h in history:
        res = h.get("result") if h.get("kind") == "tool" else None
        if not isinstance(res, dict) or not isinstance(res.get("run_id"), str):
            continue
        paths: set[str] = set()
        for o in res.get("outputs") or res.get("files") or []:
            if isinstance(o, str):
                paths.add(o.split(" (")[0])
            elif isinstance(o, dict) and isinstance(o.get("path"), str):
                paths.add(o["path"])
        runs.append((res["run_id"], paths))
    return runs


AUTO_CARRY_KINDS = (".json", ".csv", ".parquet", ".txt")
from prophet_agent.tools.sandbox_tool import MAX_CARRY as AUTO_CARRY_MAX   # never more than the tool accepts


def _auto_carry(args: Any, history: list[dict]) -> Any:
    """Every sandbox run starts with an empty out/. The newest earlier output of each data file (json, csv, parquet, txt) is
    mounted at in/carried/<file name> automatically, beside whatever the planner carried itself: live missions kept opening
    out/scan.json or in/carried/scan.json without carrying it (about 15 failed cells in one run). Deterministic from the
    history, so a replay asks for the same call."""
    if not isinstance(args, dict):
        return args
    newest: dict[str, tuple[str, str]] = {}
    for run, paths in _runs_with_outputs(history):
        for p in paths:
            if p.lower().endswith(AUTO_CARRY_KINDS):
                newest[p.rsplit("/", 1)[-1]] = (run, p)
    given = [c for c in args.get("carry") or [] if isinstance(c, dict)]
    have = {str(c.get("file", "")).rsplit("/", 1)[-1] for c in given}
    room = max(0, AUTO_CARRY_MAX - len(given))
    extra = [{"run_id": r, "file": p} for name, (r, p) in sorted(newest.items()) if name not in have][:room]
    if not extra:
        return args
    return {**args, "carry": given + extra}


def _repair_source(args: Any, history: list[dict]) -> Any:
    """A planner sometimes writes a run id that never existed (a model copies the shape of an id, not the id). When the
    named run has no such file on disk, the newest run of THIS mission that wrote the file stands in. Runs of other missions
    are never searched, so the repair cannot read someone else's data."""
    src = args.get("source") if isinstance(args, dict) else None
    if not isinstance(src, dict):
        return args
    want = str(src.get("file") or f"{OUT_DIR}/series.json")
    rid = str(src.get("run_id", ""))
    try:
        if rid and (SANDBOX_ROOT / rid / want).is_file():
            return args
    except (OSError, ValueError):
        pass
    for run, paths in reversed(_runs_with_outputs(history)):
        if want in paths and run != rid:
            return {**args, "source": {**src, "run_id": run}}
    return args


COVERAGE_TOOLS = ("forecast_report", "backtest_series", "timesfm_forecast", "panel_backtest")


def _coverage_note(obs: dict, history: list[dict]) -> str | None:
    """A forecast or backtest that covers fewer entities than the uploaded data holds says so where the planner reads it.
    Live Beijing rep4: one analyst's file of 3 of the 12 stations fed every backtest and the pack, and nothing said so."""
    if obs.get("tool") not in COVERAGE_TOOLS or obs.get("error"):
        return None
    k = contracts.result_entities(obs.get("result"))
    src = (obs.get("args") or {}).get("source")
    if k is None and isinstance(src, dict) and isinstance(src.get("name"), str):
        k = 1     # one entity picked from a panel file (live electricity rep5: MT_001's total was answered as the panel's)
    data = contracts.data_entity_count(history)
    if k is None or data is None or k >= data[0]:
        return None
    n, col = data
    cell = contracts.panel_build_cell(history)
    how = (f"one sandbox cell builds all {n}: {cell}" if cell else
           f"build all {n} with pl.panel(df, entity='{col}', time=..., value=...), save them with pl.save_panel in one run and "
           "pass that file")
    return (f"this result covers {k} of the {n} values of '{col}' in the uploaded data. If the goal is the whole panel, {how}; "
            "otherwise say which ones the answer covers and why")


def _act(ctx: Ctx, tool: Tool, name: str, args: dict, tool_names: Any = ()) -> dict:
    try:
        result = ctx.call(tool, args)
    except (BudgetExceeded, Cancelled, OutcomeUnknown, Paused, LeaseLost):
        # Control flow, not evidence: the runtime must see these. A lost lease is included: the renewal runs inside
        # Ctx.call, so before this it became a tool error and the loop went on writing after another runtime took the mission
        # (audit: goal.py:1177-1178, runtime HIGH).
        raise
    except ArgumentRejected as exc:
        # Round 3 (A5-12): a refused argument list is not a tool call and is not charged. Live: an "unknown arguments" error
        # was counted and the same invalid call was sent again.
        return {"kind": "tool", "tool": name, "args": args, "error": redact(str(exc)), "rejected": True}
    except Exception as exc:  # a tool failure is evidence for the planner; the ledger already recorded it
        return {"kind": "tool", "tool": name, "args": args, "error": redact(str(exc))}
    full = result
    if tool.observe is not None:
        try:
            result = tool.observe(result)
        except Exception:  # a broken summariser must not hide the result from the planner
            pass
    obs = {"kind": "tool", "tool": name, "args": args, "result": result}
    # Round 4 (live Beijing and electricity panels, 2026-11-23): the summary cut the middle of long output, and
    # read_observation could only page the cut text, so analysts re-ran cells to see it. The full result is kept beside
    # the summary (hidden from every prompt by _view) for paging and grounding, unless it is very large.
    if result is not full and len(canonical(full)) <= FULL_KEEP_CHARS:
        obs["full"] = full
    # The hint is part of the observation only: the ledger keeps the tool's own result, so a replay reads the same rows.
    if (hint := tool_module_hint(result, tool_names)) is not None:
        obs["hint"] = hint
    return obs


def _status(ctx: Ctx, i: int, history: list[dict], items: list[dict], started: float) -> None:
    """One small event per step: where the mission stands (step, tool calls, time, checklist). The console and a reader
    can follow a long run from these without reading the history. Never raises: a status line must not stop the work."""
    try:
        done = sum(1 for it in items if it["status"] == "done")
        nxt = next((it["item"] for it in items if it.get("next")), None)
        ctx.store.append_event(ctx.mission_id, "mission.progress", {
            "step": i + 1, "tool_calls": _calls_used(history), "delegated_calls": _delegated_calls(history),
            "elapsed_s": round(time.time() - started, 1),
            "checklist_done": done, "checklist_total": len(items), "next": nxt})
    except Exception:
        pass


ERROR_STALL = 8      # identical failures in a row after which the loop stops repeating (live: 33 and 19 before this existed)

# Round 7 (2026-11-24, four workflow runs ended partial/stalled; GOAL.md open item "409 identical-output sandbox cells with
# different code"). The stall key used to hold a sandbox cell's code, so cells that printed and wrote the same thing were never a
# repeat. A sandbox cell is now keyed by what it produced: its ok flag, its printed text and its errors with run ids and layout
# removed, and each file it wrote by path and digest. The code still decides an exact repeat (_repeat_of). A cell that produced
# nothing (no print, no error text, no file) says nothing about progress, so it keeps the old key: its code and its arguments.
STALL_NOTICE_REPEATS = 1     # no-progress steps (repeats of the newest result) before the planner is told the result already exists.
                             # The stop comes at cfg.stall_window (3 identical calls by default: two repeats), so the notice is the
                             # one step the planner gets before the loop stops.
RUN_ID_ANY = re.compile(r"[0-9a-f]{32}")


def _outcome_key(obs: dict) -> str | None:
    """A digest of what a sandbox cell produced, or None when it produced nothing to compare or failed before it ran."""
    if obs.get("tool") != "sandbox_python" or obs.get("error") is not None:
        return None
    res = obs.get("full") if isinstance(obs.get("full"), dict) else obs.get("result")
    if not isinstance(res, dict):
        return None
    stdout = str(res.get("stdout") or "")
    stderr = str(res.get("stderr") or res.get("stderr_tail") or "")
    files: list[Any] = []
    for o in res.get("outputs") or []:
        if isinstance(o, dict) and isinstance(o.get("path"), str):
            files.append([o["path"], o.get("sha256") or o.get("bytes")])
        elif isinstance(o, str):
            files.append(o)          # a summary lists "path (N bytes)"
    if not (stdout.strip() or stderr.strip() or files):
        return None

    def norm(text: str) -> str:
        lines = [" ".join(ln.split()) for ln in text.splitlines() if ln.strip()]
        return RUN_ID_ANY.sub("<run>", "\n".join(lines))
    body = {"ok": res.get("ok") is True, "returncode": res.get("returncode"), "timed_out": bool(res.get("timed_out")),
            "stdout": norm(stdout), "stderr": norm(stderr), "files": sorted(files, key=str)}
    return "out:" + hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


def _progress_key(obs: dict) -> str:
    """The key stall detection compares. A pointer carries the key of the call it points to; a sandbox cell that produced something
    is keyed by its outcome (so the same output from a different cell is no progress); anything else by its full signature."""
    if "outcome" in obs:
        return obs["outcome"]
    return _outcome_key(obs) or _signature(obs)


def _run_of_same_result(history: list[dict]) -> list[dict]:
    """The newest tool calls in a row whose progress key equals the newest call's, newest first. One call is a run of one."""
    calls = [h for h in history if h.get("kind") == "tool"]
    if not calls:
        return []
    key, run = _progress_key(calls[-1]), []
    for h in reversed(calls):
        if _progress_key(h) != key:
            break
        run.append(h)
    return run


def _outcome_line(obs: dict) -> str:
    """One line that says what a call gave: its error, a failed cell's last stderr line, or a cell's first printed line."""
    if obs.get("error"):
        text, last = str(obs["error"]), True
    else:
        res = obs.get("full") if isinstance(obs.get("full"), dict) else obs.get("result")
        if not isinstance(res, dict):
            return ""
        if res.get("ok") is False:
            text, last = str(res.get("stderr") or res.get("stderr_tail") or ""), True
        else:
            text, last = str(res.get("stdout") or ""), False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    return RUN_ID_ANY.sub("<run>", lines[-1] if last else lines[0])[:120]


def _stall_label(run: list[dict]) -> str:
    """Names the repeated step: its tool, the step it first ran at, and what that call gave (a run is newest first)."""
    src = next((h for h in reversed(run) if "result" in h or "error" in h), run[-1])
    line = _outcome_line(src)
    return f"{src.get('tool')} at step {src.get('step', '?')}" + (f", which gave: {line}" if line else "")


def _stall_notice(history: list[dict], next_item: str | None) -> str | None:
    """The planner's note once the newest result has repeated STALL_NOTICE_REPEATS times: the result already exists, so the next
    unfinished checklist item (or the next step of the goal) is what to do now. Derived from the history, so a resumed run sees it."""
    run = _run_of_same_result(history)
    if len(run) < 1 + STALL_NOTICE_REPEATS:
        return None
    todo = (f"The next unfinished checklist item must be done now, with a different step: {next_item}." if next_item else
            "Take the next unfinished step of the goal with a different call, or finish with what is found.")
    return ("Stall notice: the last " + str(len(run)) + " calls gave the same result as " + _stall_label(run) + ". That result "
            "already exists: do not run this step again. " + todo + " If the same result comes back again, the run is stopped "
            "with what it has found.")


def _stalled(history: list[dict], window: int) -> bool:
    """Stalled when the newest ``window`` tool calls are the same call with the same outcome (run ids aside), or when the newest
    ERROR_STALL calls failed the same way even with different code: the planner is not converging. Two calls taking turns
    (A, B, A with a window of 3; A, B, A, B with 4) are a loop too when each returns the same outcome it returned before:
    a call that changed nothing it reads back is not progress. The outcome is compared by _progress_key, so the same output
    from a different sandbox cell is no progress either.
    A refused argument list ("rejected") is not a call: it is not charged, and it is not counted here either (GOAL.md
    defect 5). Before, the same invalid call sent again reached ERROR_STALL and stopped the mission as stalled."""
    history = [h for h in history if not h.get("rejected")]
    if _error_run(history) >= ERROR_STALL:
        return True
    tool_obs = [h for h in history if h.get("kind") == "tool"]
    if len(tool_obs) < window:
        return False
    last = [_progress_key(h) for h in tool_obs[-window:]]
    if len(set(last)) == 1:
        return True
    return window >= 3 and len(set(last)) == 2 and all(last[k] == last[k - 2] for k in range(2, len(last)))


def _specs(plan: dict) -> list[dict]:
    """The delegations in a plan: ``tasks`` (parallel) or the single ``task`` with its optional fields."""
    keys = ("task", "agent_type", "deliverable", "constraints", "tools", "resume")
    raw = plan.get("tasks")
    if isinstance(raw, list) and raw:
        return [{k: t.get(k) for k in keys} for t in raw if isinstance(t, dict)][: agents.MAX_PARALLEL]
    return [{k: plan.get(k) for k in keys}]


def _brief(spec: dict, agent: agents.AgentType, agent_id: str, resumed: bool) -> str:
    """What a sub-agent works from. It does not see the parent's conversation, so the brief carries the task, the shape of
    the answer, the boundaries and how to use its notes."""
    deliverable = str(spec.get("deliverable") or "").strip() or (
        "A short report: what you found with the figures your tools printed, what you recommend and why, and any file you "
        "wrote as run id and path. Say plainly what you could not establish.")
    parts = [f"You are sub-agent {agent_id} (type {agent.name}) working for a lead agent. {agent.role}",
             f"Task: {str(spec['task']).strip()}", f"Return: {deliverable}"]
    if str(spec.get("constraints") or "").strip():
        parts.append(f"Boundaries: {str(spec['constraints']).strip()}")
    parts.append("Your final verdict lists findings (each a short statement) and figures (a name and the value a tool "
                 "printed). Give only figures a tool printed; list the ones you could not establish in the answer.")
    parts.append("You do not see the lead agent's conversation: this brief and the context below are all you have. Keep "
                 "anything you must not lose (a decision, a run id and file, open questions) in your notes. Report only "
                 "figures a tool printed. The lead agent reads your final report and nothing else; it can read your notes.")
    if resumed:
        parts.append("You are continuing earlier work: your earlier steps are in the evidence. Build on them.")
    return "\n\n".join(parts)


def _files_of(evidence: list[dict]) -> list[str]:
    """Files a sub-agent's analysis cells wrote, as run_id:path, so the lead agent can carry them into its own cells."""
    out: list[str] = []
    for h in evidence:
        res = h.get("result") if h.get("kind") == "tool" else None
        if isinstance(res, dict) and res.get("run_id") and isinstance(res.get("outputs"), list):
            for o in res["outputs"]:
                out.append(f"{res['run_id']}:{str(o).split(' (')[0]}")
    return out[:30]


def _notes_for(ctx: Ctx, scope: str):
    from prophet_agent.tools.notes_tool import NOTES_SPEC, notes_tool
    from prophet_agent.tools import spec_doc
    from dataclasses import replace
    return replace(notes_tool(ctx.store, ctx.mission_id, scope), doc=spec_doc(NOTES_SPEC), title=NOTES_SPEC["title"],
                   activity=NOTES_SPEC["activity"])


SEED_RESULTS = 12    # the lead's most recent tool results a new analyst starts with


def seed_history(history: list[dict]) -> list[dict]:
    """The lead's tool results a new analyst starts with, marked seeded. Pointers, rejected calls and the lead's own
    repeats are left out. The most recent SEED_RESULTS are kept, so the analyst's prompt does not grow with the run."""
    usable = [h for h in history if h.get("kind") == "tool" and "result" in h and not h.get("repeat_of")
              and not h.get("rejected") and "error" not in h]
    return [{**h, "seeded": True} for h in usable[-SEED_RESULTS:]]


def _one_agent(ctx: Ctx, brain: Any, spec: dict, number: int, tools: dict[str, Tool], cfg: GoalConfig, context: str,
               history: list[dict]) -> dict:
    """Runs one sub-agent with its own history, memo, notes and a tool list scoped by its type."""
    agent = agents.get(spec.get("agent_type")) or agents.TYPES["general"]
    agent_id = f"a{number}"
    sub_tools = agents.allowed(agent, tools)
    asked = [n for n in (spec.get("tools") or []) if n in sub_tools]
    if asked:
        sub_tools = {n: sub_tools[n] for n in asked}
    sub_tools.pop("notes", None)
    sub_tools["notes"] = _notes_for(ctx, agent_id)
    prior = None
    if spec.get("resume"):
        prior = next((h for h in reversed(history) if h.get("kind") == "subagent" and h.get("agent_id") == spec["resume"]), None)
        if prior is None:
            return {"kind": "delegate", "error": f"no earlier sub-agent {spec['resume']!r} to resume; known: "
                    + (", ".join(sorted({h['agent_id'] for h in history if h.get('kind') == 'subagent' and h.get('agent_id')})) or "none")}
    # The step budget is the type's own (AgentType.steps). It was declared but never read (audit item 24), so every sub-agent
    # got cfg.sub_steps whatever its type said.
    sub_cfg = GoalConfig(max_iterations=agent.steps, stall_window=cfg.stall_window, depth=1, sub_steps=cfg.sub_steps,
                         tag=f"d1.{agent_id}")
    # Live (Beijing 3c379255, 2026-11-23): four analysts started with an empty history and each ran the lead's inspect_data
    # again, then read the same pages, and the step stalled. They start with the lead's tool results so far, so an identical
    # call is answered with the earlier result, as it is for the lead.
    sub_history: list[dict] = list(prior["evidence"]) if prior else seed_history(history)
    sub_tools = _sub_observation_tools(sub_tools, sub_history)
    goal = _brief(spec, agent, agent_id, prior is not None)
    play = agents.playbooks(agent)
    sub_context = (context + "\n\n" if context else "") + (play[:7000] if play else "")
    # Sub-agent steps are top-level steps of the same ledger, numbered in call order like any others, so a resumed run
    # replays them. They are not wrapped in an outer ctx.step, which would nest their numbers inside one another.
    # Round 3 (A5-01): the sub-agent's calls run on the delegated lane, so they draw on its own grant, not the lead's.
    prev_lane, ctx.lane = ctx.lane, "delegated"
    try:
        sub = _run(ctx, brain, goal, sub_tools, sub_cfg, sub_context.strip(), sub_history)
    finally:
        ctx.lane = prev_lane
    answer = sub.get("answer", "") or sub.get("draft", "")
    status, reason = sub.get("status"), sub.get("reason", "")
    if not str(answer).strip():
        # Round 3, A5-05: live, an analyst that wrote no answer was reported as "partial" with its run status, and the lead read
        # the empty report as work done. No written answer is not met, whatever the run status says.
        status, reason = "not_met", "the sub-agent gave no written answer" + (f": {reason}" if reason else "")
    task = str(spec["task"]).strip()
    # A sub-agent's figures are checked against the tool results it saw, never against its brief: the task is the lead
    # agent's own words, so a figure repeated from it would pass without any tool having printed it (audit item 3).
    obs = {"kind": "subagent", "agent_id": agent_id, "agent_type": agent.name, "task": task,
           "tools": sorted(sub_tools), "status": status, "answer": answer, "reason": reason,
           "steps": len(sub_history), "files": _files_of(sub_history), "evidence": sub_history,
           "findings": [str(f)[:300] for f in sub.get("findings") or []][:12],
           "figures": [{"name": str(f["name"])[:120], "value": str(f["value"])[:60],
                        "sourced": not unverified(str(f["value"]), sub_history, "")}
                       for f in sub.get("figures") or [] if isinstance(f, dict) and "name" in f and "value" in f][:20]}
    # A4-14: the sub-agent's own figure list is reconciled too. A figure it marked unsourced reaches the lead even when the
    # answer text leaves it out, so the lead never reads an unsourced figure as checked.
    bad = unverified(answer, sub_history, "") if answer else []
    bad = list(dict.fromkeys(bad + [f["value"] for f in obs["figures"] if not f["sourced"]]))
    if bad:
        obs["unverified"] = bad          # the lead agent is told which of the report's figures no tool printed
    return obs


def _delegate(ctx: Ctx, brain: Any, plan: dict, tools: dict[str, Tool], cfg: GoalConfig, context: str,
              history: list[dict]) -> list[dict]:
    """Runs one or several sub-agents (in parallel when several) and returns an observation for each. Only a sub-agent's
    report returns to the parent; its working stays in its own history."""
    if cfg.depth > 0:
        return [{"kind": "delegate", "error": "a sub-agent cannot start another sub-agent; do the step yourself"}]
    specs = [sp for sp in _specs(plan) if str(sp.get("task") or "").strip()]
    if not specs:
        return [{"kind": "delegate", "error": "a delegated task needs a task description"}]
    started = sum(1 for h in history if h.get("kind") == "subagent")
    if started + len(specs) > agents.MAX_AGENTS:
        return [{"kind": "delegate", "error": f"this mission already started {started} sub-agents; the limit is "
                 f"{agents.MAX_AGENTS}. Do the step yourself."}]
    unknown = [sp.get("agent_type") for sp in specs if sp.get("agent_type") and agents.get(sp["agent_type"]) is None]
    if unknown:
        return [{"kind": "delegate", "error": f"unknown agent_type {unknown}; types: {', '.join(agents.TYPES)}"}]
    if len(specs) == 1:
        return [_one_agent(ctx, brain, specs[0], started + 1, tools, cfg, context, history)]

    from concurrent.futures import ThreadPoolExecutor
    base = (ctx.step_idx + 2) * BRANCH_BLOCK
    results: list[Any] = [None] * len(specs)
    failures: list[BaseException | None] = [None] * len(specs)

    def work(k: int) -> None:
        try:
            branch = ctx.branch(base + k * BRANCH_STRIDE)
            results[k] = _one_agent(branch, brain, specs[k], started + 1 + k, tools, cfg, context, history)
        except BaseException as exc:      # re-raised below, on the calling thread, once the others have settled
            failures[k] = exc

    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        list(pool.map(work, range(len(specs))))
    # A question for the person, a cancel or a spent budget in any branch stops the whole step, the way it would inline.
    for kind in (Paused, Cancelled, OutcomeUnknown, BudgetExceeded):
        for exc in failures:
            if isinstance(exc, kind):
                raise exc
    for exc in failures:
        if exc is not None:
            raise exc
    return results


def _horizon_unit(text: str) -> str:
    """The unit a horizon phrase names, as one of the keys of HORIZON_LABELS."""
    t = text.lower()
    if t.startswith(("trading", "business")):
        return "trading_day"
    if t.startswith(("period", "step")):
        return "period"
    for unit in ("day", "week", "month", "quarter", "year"):
        if t.startswith(unit):
            return unit
    return "period"


def requested_horizon(goal: str) -> tuple[int, str] | None:
    """The horizon the goal asks for, as (count, unit), or None when it states none or states more than one.

    Motivation vs Logic
    Motivation: "next 20 trading days" is the span the person wants a forecast over. A plan that forecasts 14 steps answers
    a different question, and nothing compared the two (audit item 10, 2026-11-22).
    Logic: read the phrasings people use with digits ("next N unit", "N unit ahead", "horizon of N unit"). Word numbers are
    not read. Two different horizons in one goal are ambiguous, so none is returned and no comparison is made: that is a
    decision for the person, not a guess.
    """
    found = set()
    for m in HORIZON_REQUEST.finditer(goal or ""):
        found.add((int(m.group(1) or m.group(3)), _horizon_unit(m.group(2) or m.group(4))))
    if len(found) != 1:
        return None
    count, unit = next(iter(found))
    return (count, unit) if count >= 1 else None


def _plan_horizon(history: list[dict]) -> tuple[int, dict] | None:
    """The horizon of the forecast that stands (the newest single-series one) and its record. None when no such forecast
    has run. A panel pack is skipped: it holds several series and has no horizon of its own.

    Motivation vs Logic
    Motivation: the check must read the horizon the plan actually used, which is the one TimesFM was given, not the one the
    planner says it meant to use.
    Logic: the record's configuration holds the horizon TimesFM ran; if a record does not carry it, the call's own argument
    is the next source.
    """
    latest = None
    for h in history:
        rec = contracts._forecast_record(h) if isinstance(h, dict) else None
        if rec is not None and not rec.get("panel"):
            latest = (h, rec)
    if latest is None:
        return None
    h, rec = latest
    cfg = rec.get("config") if isinstance(rec.get("config"), dict) else {}
    horizon = cfg.get("horizon")
    if not (isinstance(horizon, int) and not isinstance(horizon, bool)):
        args = h.get("args") if isinstance(h.get("args"), dict) else {}
        horizon = args.get("horizon")
    if not (isinstance(horizon, int) and not isinstance(horizon, bool)) or horizon < 1:
        return None
    return horizon, rec


def _series_step(rec: dict) -> tuple[float, bool] | None:
    """The series step in days, and whether it is a calendar step (monthly, quarterly, yearly). None when the forecast record
    does not state a usable step.

    Motivation vs Logic
    Motivation: "6 weeks" is 6 periods of a weekly series and 42 of a daily one. Without the step the request cannot be turned
    into the periods the forecast runs in.
    Logic: accept a step name ("weekly"), the profile's own text ("7 days 00:00:00") or a number of days. Anything else is
    no step at all, so the caller reports the comparison as not made rather than guessing one.
    """
    cfg = rec.get("config") if isinstance(rec.get("config"), dict) else {}
    step = cfg.get("series_step", rec.get("series_step"))
    if isinstance(step, bool):
        return None
    if isinstance(step, (int, float)):
        return (float(step), False) if 0 < step < 1e6 else None
    if not isinstance(step, str):
        return None
    text = step.strip().lower()
    if text in SERIES_STEPS:
        return SERIES_STEPS[text], text in CALENDAR_STEPS
    m = STEP_TEXT.fullmatch(text)
    if m and 0 < float(m.group(1)) < 1e6:
        return float(m.group(1)), False
    return None


MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
          "december")
NAMED_MONTH = re.compile(r"\b(?:all of|for|in|during|through|over|the month of|until|to the end of)\s+(" + "|".join(MONTHS)
                         + r")(?:\s+(\d{4}))?\b", re.I)
NEXT_MONTH = re.compile(r"\bnext (?:calendar )?month\b", re.I)
WINDOW_SLACK_DAYS = 4      # a trading-day series has no weekend or holiday rows, so its last date can fall a few days short


def _month_end(year: int, month: int):
    import calendar
    import datetime as dt
    return dt.date(year, month, calendar.monthrange(year, month)[1])


def requested_window(goal: str, history: list[dict]) -> tuple[str, Any] | None:
    """The calendar month the person asked about, as (label, last date), read from the goal and from their answers to
    questions. "Next month" is the month after the data's last date; a named month is the next one that ends after it.
    None when no such month is named, or the data's last date is unknown.

    Live wti_budget (2026-11-23): asked for "next month", the person answered "All of December 2026"; the plan forecast 21
    trading days from 18 November, which end on 16 December, and the answer said the second half was not forecast."""
    import datetime as dt
    end = _data_facts(history).get("data_end")
    if not end:
        return None
    last = dt.date.fromisoformat(end)
    from .grounding import _as_text
    texts = [goal or ""] + [_as_text(h.get("answers")) for h in history
                            if h.get("kind") in ("ask_user", "evaluator_question") and h.get("answers")]
    found = None
    for text in texts:                                   # the person's answer, given later, wins over the goal's words
        for m in NAMED_MONTH.finditer(text):
            month = MONTHS.index(m.group(1).lower()) + 1
            year = int(m.group(2)) if m.group(2) else last.year
            if not m.group(2) and _month_end(year, month) <= last:
                year += 1
            if _month_end(year, month) > last:
                found = (f"{m.group(1).capitalize()} {year}", _month_end(year, month))
        if NEXT_MONTH.search(text) and found is None:
            y, mo = (last.year + 1, 1) if last.month == 12 else (last.year, last.month + 1)
            found = (f"next month ({MONTHS[mo - 1].capitalize()} {y})", _month_end(y, mo))
    return found


def _forecast_last_date(history: list[dict]):
    """The date of the last forecast step in the newest pack's table, or None when it has no dates."""
    import datetime as dt
    for h in reversed(history):
        res = contracts._ok_result(h, "forecast_report") if isinstance(h, dict) else None
        rows = ((res or {}).get("table") or {}).get("rows") if isinstance(res, dict) else None
        if rows:
            try:
                return dt.date.fromisoformat(str(rows[-1]["date"])[:10])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def window_checks(goal: str, history: list[dict]) -> list[dict]:
    """The pack reaches the end of the calendar month the person asked about. A block: the plan must forecast that far."""
    window = requested_window(goal, history)
    last = _forecast_last_date(history)
    if window is None or last is None:
        return []
    label, end = window
    import datetime as dt
    ok = last >= end - dt.timedelta(days=WINDOW_SLACK_DAYS)
    detail = f"the request covers {label}, which ends {end.isoformat()}; the forecast's last date is {last.isoformat()}"
    if not ok:
        detail += (". Forecast through the end of that month (count the steps from the data's last date), then report the "
                   "figures for that month's dates only")
    return [contracts._check("window_matches_request", ok, detail, "block")]


def _stored_health_verdict(history: list[dict]) -> str | None:
    """The health verdict on the pack the person receives, or None when no forecast stored one.

    A later timesfm_forecast is a diagnostic. It does not replace the newest forecast_report.
    """
    return contracts.delivered_verdict(history)


def level_demand_misapplied(reason: str, history: list[dict]) -> bool:
    """True when the evaluator demands the flat_expected level sentence for another verdict.

    SciRes-20261124-124543 stored health verdict check. The evaluator refused the finish three times
    because the 400-character rationale did not say the centre line is a level. That sentence is
    required only when the stored verdict is flat_expected.
    """
    text = reason or ""
    if _OTHER_GAP.search(text):
        return False
    if _LEVEL_SENTENCE.search(text) is None or _RATIONALE_JUDGED.search(text) is None:
        return False
    verdict = _stored_health_verdict(history)
    return verdict is not None and verdict != "flat_expected"


def _accept_despite_level_demand(history: list[dict]) -> dict:
    verdict = _stored_health_verdict(history) or "unknown"
    return {
        "verdict": "met",
        "reason": f"the stored health verdict is {verdict}, so the flat_expected level sentence is not required",
        "answer": (f"The forecast is in the evidence. The health verdict is {verdict}. "
                   "That verdict is not flat_expected, so the answer does not use the level sentence."),
    }


def horizon_checks(goal: str, history: list[dict]) -> list[dict]:
    """The check that the forecast that stands runs over the horizon the goal asks for. Empty when the goal states no horizon
    or no forecast has run: the check is left out, not reported as passing.

    Motivation vs Logic
    Motivation: before this, a plan of 14 steps for a request of 20 trading days could finish as "met", because no check put
    the plan's horizon beside the requested one. The mismatch has to be visible and has to be answered.
    Logic: a request in periods must equal the forecast's horizon. A calendar request is converted with the series step, and a
    daily series reads one period per trading day. A mismatch is a block: the finish is refused and the planner must forecast
    the requested span. A calendar request with no recorded step cannot be converted; that is a warning the answer carries,
    not a block, so every calendar forecast is not refused for a field the record does not yet hold.
    """
    window = window_checks(goal, history)
    request = requested_horizon(goal)
    plan = _plan_horizon(history)
    if window:
        return window          # a calendar month asked for by name is the span; "next month" counted in days is not
    if request is None or plan is None:
        return []
    count, unit = request
    horizon, rec = plan
    label = HORIZON_LABELS[unit] if count != 1 else HORIZON_LABELS[unit].rstrip("s")
    asked = f"the request asks for {count} {label}"
    if unit == "period":
        expected, slack = float(count), 0.5
    else:
        step = _series_step(rec)
        if step is None:
            return [contracts._check("horizon_matches_request", False,
                                     f"{asked}, but the series step is not recorded, so the request cannot be converted into "
                                     f"periods; the forecast runs {horizon} periods", "warn")]
        step_days, calendar = step
        if unit == "trading_day" and step_days == 1.0:
            expected = float(count)
        else:
            expected = count * UNIT_DAYS[unit] / step_days
        # A calendar step lines up with the calendar; a fixed-length step cannot, since month lengths vary by a day or two.
        slack = 0.5 if calendar or unit not in ("month", "quarter", "year") else 1.0
    ok = abs(expected - horizon) <= slack
    detail = f"{asked}, which is {expected:.1f} periods at the series step; the forecast runs {horizon} periods"
    if not ok:
        detail += ", so the plan must forecast the requested span"
    return [contracts._check("horizon_matches_request", ok, detail, "block")]


def horizon_note(goal: str, history: list[dict]) -> str | None:
    """The failing horizon checks as one sentence for the person, or None when every check holds or none applies."""
    failing = [c for c in horizon_checks(goal, history) if not c["ok"]]
    return ("Horizon check (computed): " + contracts.summarise(failing)) if failing else None


def _run(ctx: Ctx, brain: Any, goal: str, tools: dict[str, Tool], cfg: GoalConfig, context: str,
         history: list[dict]) -> dict:
    """The loop itself. Used for the goal and, recursively with depth 1, for a delegated sub-agent."""
    memo = Memo()
    if cfg.depth == 0:
        tools = {**tools, "notes": _notes_for(ctx, "main"), "read_observation": _observation_tool(history)}
        # Which instructions this mission ran under, recorded once so a resumed run reads it back unchanged.
        # The brief is stored with them: the goal, what counts as done, what must not change, and the budget. It is data a
        # resumed run and a reader can see, so a summary of the work can never drop it.
        def record(c: Ctx) -> dict:
            recorded = {n: {"version": prompts.current(n).version, "sha256": prompts.current(n).sha256}
                        for n in prompts.CURRENT}
            recorded["brief"] = {"goal": goal, "done_when": DONE_WHEN, "must_not_change": MUST_NOT_CHANGE,
                                 "budget": {"steps_per_part": cfg.max_iterations, "tool_calls": cfg.tool_budget}}
            return recorded
        ctx.step("contracts", record)
    tag = cfg.tag or f"d{cfg.depth}"
    seg_len = cfg.max_iterations
    started = time.time()
    rejected_finishes = 0     # consecutive finishes the evaluator turned down
    best_done, calls_at_best = -1, 0   # most checklist items ticked so far, and the call count when that was reached
    stuck = None                       # (next checklist item, calls since the last newly ticked item)
    rescue = False                     # set by the first stall: the next step hands the stuck item to a fresh analyst
    contract_blocks = 0       # times a deterministic check sent a "met" answer back for more work
    repair_granted = False    # the one repair allowance, granted when a blocked answer met a spent tool budget
    for i in range(cfg.end):
        if i and i % seg_len == 0:
            # A new part of the work starts here. The marker is an observation, so the planner and the replay see it.
            history.append({"kind": "continued", "segment": i // seg_len})
        out_of_calls = cfg.tool_budget is not None and _calls_used(history) >= cfg.tool_budget
        # Every step index is taken on every pass, including the forced finishes below. A finish that was decided outside
        # the ledger would shift the indices of later steps, and a resumed run would read the wrong stored outputs.
        _compact(ctx, brain, goal, history, memo, f"{tag}:{i}", view_chars=cfg.view_chars,
                 keep_recent=cfg.keep_recent, memory_chars=cfg.memory_chars,
                 boundary=i > 0 and i % seg_len == 0)
        note = _budget_note(i, cfg, history)
        if (read_note := _skills_read_note(ctx)):
            note += "\n\n" + read_note
        if (repeat_note := _repeat_error_note(history)):
            note += "\n\n" + repeat_note
        if (card := _state_card(history)):
            note += "\n\n" + card
        if cfg.depth == 0 and FORECAST_GOAL.search(goal):
            # The checklist is derived from the evidence on every step, so it cannot drift and a resumed run sees the same one.
            items = contracts.progress(history, run_root=SANDBOX_ROOT, wants_files=bool(FILES_GOAL.search(goal)),
                                       external=bool(EXTERNAL_GOAL.search(goal)))
            note += "\n\n" + contracts.render_progress(items)
            if (asked := _asked_block(goal, "planner")):
                note += "\n\n" + asked
            if (horizon_line := horizon_note(goal, history)):
                note += "\n\n" + horizon_line       # the planner sees a mismatch while it can still forecast the right span
            if (short := _short_calls_note(cfg, history, items)):
                note += "\n\n" + short
            done_now, calls_now = sum(it.get("status") == "done" for it in items), _calls_used(history)
            if done_now > best_done:
                best_done, calls_at_best = done_now, calls_now
            if (idle := _idle_note(calls_now - calls_at_best, items)):
                note += "\n\n" + idle
            nxt_item = next((it for it in items if it.get("next")), None)
            nxt = nxt_item["item"] if nxt_item else None
            # The item's detail rides along for the rescue brief only (live retail_biz rep6: the ledger item's analyst got
            # the item's name, not the cell that closes it, and printed the scan 14 times). The dedup key stays the item.
            stuck = ((nxt, STUCK_CALLS if rescue else calls_now - calls_at_best, nxt_item.get("detail") or "")
                     if nxt else None)
            rescue = False
            _status(ctx, i, history, items, started)
        if (stall_note := _stall_notice(history, stuck[0] if stuck else None)):
            note += "\n\n" + stall_note          # round 7: one repeat of the same result, so the planner is told it already exists
        stored = _stored_step_name(ctx)
        # A resumed run follows the decision the ledger holds at this position: the harness's choice depends on the call
        # allowance, which grows when a run continues, so recomputing it could take the other path (live retail: "replay
        # diverged at step 48: the ledger has plan:d0:21, the program asked for auto:d0:21", and the mission failed).
        auto = (_harness_plan(goal, history, tools, lambda t: ctx.store.tool_results(ctx.mission_id, t), stuck)
                if cfg.depth == 0 and not cfg.finish_now and not out_of_calls and cfg.auto_delegate
                and not (stored or "").startswith("plan:") else None)
        if auto is None and (stored or "").startswith("auto:") and cfg.depth == 0:
            auto = ctx.store.step_record(ctx.mission_id, ctx.step_idx + 1)["v"]      # the stored harness plan, replayed
        if auto is not None:
            plan = ctx.step(f"auto:{tag}:{i}", lambda c: auto)
        else:
            # The allowance decision is stored with the plan (A5-07). A resumed run after a top-up recomputes out_of_calls as
            # False, but the stored plan was made when the allowance was spent, so the gates below read the stored value.
            plan = ctx.step(f"plan:{tag}:{i}", lambda c: {**_decide_plan(brain, goal, tools, history, memo, context, note,
                                                                        finish_now=cfg.finish_now, out_of_calls=out_of_calls),
                                                          "out_of_calls": out_of_calls})
        spent = bool(plan.get("out_of_calls", out_of_calls))
        if plan["action"] == "finish":
            obs = {"kind": "finish", "rationale": plan["rationale"]}
        elif plan["action"] == "ask_user":
            open_qs, repeated = _already_asked(plan.get("questions", []) or [], history)
            payload = {"questions": open_qs}
            problems = (validate_questions(payload) if open_qs or not repeated
                        else ["every question was already asked and answered"])
            if problems and repeated:
                obs = {"kind": "ask_user", "error": "not asked again, the person already answered: " + "; ".join(repeated)
                       + ". The decision is in the decisions list; act on it."}
            elif problems:
                obs = {"kind": "ask_user", "error": "invalid questions: " + "; ".join(problems)}
            elif not HIGH_RISK.search(str(plan.get("rationale", ""))):
                obs = {"kind": "ask_user", "error": "not asked: a question goes to the person only for a high-risk decision, "
                       "one that changes the delivered numbers materially, that the data cannot settle and where a wrong guess "
                       "is costly. Decide this yourself with the option that has the better expected outcome, state the "
                       "assumption in the answer, and carry on. If it is high-risk, ask again with a rationale that starts "
                       "'high-risk:' and says how a wrong guess changes the numbers."}
            else:
                # Pauses the run here. The answer, once given, is returned as the step's value. Questions that repeat an
                # earlier one are left out; the earlier answer stands.
                answers = ctx.step(f"ask:{tag}:{i}", lambda c: c.ask(payload))
                obs = {"kind": "ask_user", "questions": open_qs, "answers": answers}
                if repeated:
                    obs["not_asked_again"] = repeated
        elif plan["action"] == "delegate":
            delegated = _delegate(ctx, brain, plan, tools, cfg, context, history)
            if plan.get("auto"):
                for d in delegated:
                    if d.get("kind") == "subagent":
                        d["auto"] = plan["auto"]      # started by the harness, so each trigger fires once
            history.extend(delegated[:-1])
            obs = delegated[-1]
        else:
            name = plan.get("tool", "")
            args = _repair_source(plan.get("args", {}), history)
            if name == "sandbox_python":
                args = _auto_carry(args, history)
            if name not in tools:
                obs = {"kind": "tool", "tool": name, "args": args, "error": f"unknown tool {name!r}"}
            else:
                if (pos := _repeat_of(name, args, history)) is not None:
                    obs = _repeat_obs(name, args, history, pos)     # not sent again: the earlier result stands
                else:
                    obs = ctx.step(f"act:{tag}:{i}", lambda c: _act(c, tools[name], name, args, tools))
        if obs.get("kind") == "tool":
            obs.setdefault("step", len(history) + 1)     # the number read_observation pages back by (1-based, as evidence shows)
            note = _coverage_note(obs, history)
            if note:
                obs["coverage_note"] = note
            if (verdicts := contracts.panel_verdicts(obs, history)) is not None:
                obs["verdicts"] = verdicts          # derived from stored results, so a replay derives the same
        history.append(obs)

        if obs.get("kind") == "tool" and _stalled(history, cfg.stall_window):
            if cfg.depth == 0 and cfg.auto_delegate and stuck and not any(h.get("kind") == "stall_rescue" for h in history):
                # The first stall is not the end: the next checklist item goes to a fresh analyst, and only a second stall
                # stops the run (live retail: four identical diagnostics cells ended the mission at 11 calls, no forecast).
                history.append({"kind": "stall_rescue", "note": "the same step repeated with the same result; the next "
                                "checklist item (" + stuck[0] + ") goes to a fresh analyst. Build on its report."})
                rescue = True
                continue
            # Repeating the same call will not help. Judge what was found so far: a best answer, labelled partial,
            # beats an empty "stalled".
            verdict = ctx.step(f"evalstall:{tag}:{i}", lambda c: brain.decide(_eval_messages(goal, history, memo, context),
                                                                          EVAL_SCHEMA, "eval"))
            draft = _gap_draft(verdict)
            written = _compose(ctx, brain, f"composestall:{tag}:{i}", goal, history, memo, context, draft)
            text = written if written is not None else verdict.get("answer", "")
            named = contracts.check_answer_files(text, history)
            if named is not None and not named["ok"] and verdict["verdict"] != "impossible":
                fixed = _compose(ctx, brain, f"filesstall:{tag}:{i}", goal, history, memo, context, text,
                                 fixes=contracts.summarise([named]))
                if fixed is not None and fixed.strip():
                    text = fixed
            # Round 7: the reason names the repeated step, and keeps the words the outcome mapping reads as reason "stalled".
            reason = ("The analysis kept repeating the same step (" + _stall_label(_run_of_same_result(history) or [obs])
                      + "), so this answer rests on what was found before that.")
            if text.strip() and verdict["verdict"] != "impossible":
                return {"status": "partial", "iterations": i + 1, "answer": text, "evidence": history, "reason": reason}
            return {"status": "stalled", "iterations": i + 1, "evidence": history, "reason": reason}

        last = i == cfg.end - 1
        # The gate must judge a step the same way in every part: `last` moves with cfg.end, so a finish that ended part 1 was
        # judged by the evaluator then but refused on resume (replay diverged). at_part_end is fixed by the step index.
        at_part_end = (i + 1) % seg_len == 0
        if plan["action"] == "finish" and cfg.depth == 0 and FORECAST_GOAL.search(goal) and not (at_part_end or spent or cfg.finish_now):
            # Round 3, A5-04: live, a forecast was finished with the checklist's data scan and forecast still undone, and the
            # evaluator wrote a verdict on it. A finish with items still to do is refused here, before the evaluator is asked,
            # and the refusal names the next item. Decided from the stored history, so a resumed run refuses the same finish.
            refused = sum(1 for h in history if h.get("kind") == "finish_refused")
            items = contracts.progress(history, run_root=SANDBOX_ROOT, wants_files=bool(FILES_GOAL.search(goal)),
                                       external=bool(EXTERNAL_GOAL.search(goal)))
            nxt = next((it["item"] for it in items if it.get("status") != "done"), None)
            if nxt and refused < MAX_PREMATURE_FINISHES:
                history.append({"kind": "finish_refused", "rationale": plan.get("rationale", ""), "next": nxt,
                                "note": f"the work is not finished: '{nxt}' is still to do. Do it before finishing."})
                continue
        # The evaluator is a judgement on a finished piece of work, so it runs when the planner says it is done, or on the
        # last step of this run to turn whatever was found into an answer or a checkpoint. Between those, it only costs time.
        # The evaluator also runs at the end of each part. The stored evaluation then replays at the same position when
        # the next part continues from there, so both runs ask the ledger for the same steps in the same order.
        if plan["action"] != "finish" and not last and not spent and not at_part_end:
            continue
        # A finish with a stored forecast is written before it is judged. SciRes-20261124-124543 refused the
        # 400-character rationale. The evaluator now sees the written answer. A resume replays the same step.
        judge_context = context
        prewritten = None
        if plan["action"] == "finish" and _stored_health_verdict(history):
            draft = _answer_channel(history).split(": ", 1)[-1].strip()
            prewritten = _compose(ctx, brain, f"precompose:{tag}:{i}", goal, history, memo, context, draft)
            if prewritten and prewritten.strip():
                judge_context = ((context + "\n\n") if context else "") + "Written answer to judge:\n" + prewritten.strip()
        eval_schema = SUB_EVAL_SCHEMA if cfg.depth else EVAL_SCHEMA
        verdict = ctx.step(f"eval:{tag}:{i}", lambda c: brain.decide(_eval_messages(goal, history, memo, judge_context),
                                                                 eval_schema, "eval"))
        # The evaluator may find that one decision by the person would change the answer. It is asked once per run,
        # then the evaluator judges again with the answer in the evidence.
        qs = _usable_questions(verdict) if verdict["verdict"] != "met" else None
        if qs:
            qs, _ = _already_asked(qs, history)
        if qs and not any(h.get("kind") == "evaluator_question" for h in history):
            answers = ctx.step(f"evalask:{tag}:{i}", lambda c: c.ask({"questions": qs}))
            history.append({"kind": "evaluator_question", "questions": qs, "answers": answers})
            verdict = ctx.step(f"eval2:{tag}:{i}", lambda c: brain.decide(_eval_messages(goal, history, memo, judge_context),
                                                                       eval_schema, "eval"))
        # SciRes-20261124-124543: the evaluator demanded the flat_expected level sentence for a check verdict,
        # and it judged the finish rationale. That refusal does not stand. The sentence stays required for flat_expected.
        if verdict.get("verdict") == "not_met" and level_demand_misapplied(str(verdict.get("reason") or ""), history):
            verdict = _accept_despite_level_demand(history)
        if verdict["verdict"] == "met":
            answer = verdict.get("answer", "")
            # A finish that already wrote the answer keeps that text. The evaluator judged it. A second write
            # would replace it with the evaluator's schema-bound draft.
            if prewritten and prewritten.strip():
                answer = prewritten.strip()
            else:
                # A verdict can say "met" and carry no answer (a reply cut short, or a model that only gave its reason).
                # The writer then works from the reason and the evidence; "met" with nothing to read is never returned.
                written = _compose(ctx, brain, f"compose:{tag}:{i}", goal, history, memo, judge_context,
                                   answer or verdict.get("reason", ""))
                if written is not None:
                    answer = written
            if not answer.strip():
                if not last and not spent:
                    history.append({"kind": "evaluation", "verdict": "not_met",
                                    "missing": "the goal looks met, but no written answer was given; state the answer"})
                    continue
                return {"status": "partial", "iterations": i + 1, "answer": "",
                        "reason": "The analysis finished but no written answer was produced.", "evidence": history}
            # A model saying it is finished is a claim; the checks run on the evidence. A blocking failure goes back to the
            # planner as work to do, at most MAX_CONTRACT_BLOCKS times, then the answer leaves labelled partial.
            # The horizon check is the goal's own (the request is in the goal text), so it runs for the goal and not for a
            # delegated sub-agent, whose brief may name a span of its own.
            horizon = horizon_checks(goal, history) if cfg.depth == 0 else []
            blocked = contracts.blocking(contracts.check_evidence(history, run_root=SANDBOX_ROOT) + horizon)
            if (blocked and out_of_calls and not repair_granted and cfg.depth == 0 and cfg.tool_budget is not None
                    and contract_blocks < MAX_CONTRACT_BLOCKS and not last and not cfg.finish_now):
                # The tool budget is spent, so the fix could not run and the answer would leave partial for want of one call.
                # Live (WTI rep2 and retail rep1, 2026-11-22) both blocked answers met a spent budget. One allowance, for this
                # fix only, goes to the config and to the runtime's own limit, so the planner's notes and the enforcement agree.
                repair_granted = True
                cfg.tool_budget += REPAIR_CALLS
                runtime_budget = getattr(getattr(ctx, "rt", None), "budget", None)
                if isinstance(runtime_budget, dict) and runtime_budget.get("max_tool_calls") is not None:
                    runtime_budget["max_tool_calls"] = int(runtime_budget["max_tool_calls"]) + REPAIR_CALLS
                contract_blocks += 1
                history.append({"kind": "evaluation", "verdict": "not_met", "contract": [c["name"] for c in blocked],
                                "missing": "A required check failed on the evidence: " + contracts.summarise(blocked)
                                + f". Fix the cause with a tool call before finishing. A repair allowance of {REPAIR_CALLS} "
                                "tool calls was granted for this fix only."})
                continue
            if blocked and contract_blocks < MAX_CONTRACT_BLOCKS and not last and not out_of_calls and not cfg.finish_now:
                contract_blocks += 1
                history.append({"kind": "evaluation", "verdict": "not_met", "contract": [c["name"] for c in blocked],
                                "missing": "A required check failed on the evidence: " + contracts.summarise(blocked)
                                + ". Fix the cause with a tool call before finishing."})
                continue
            if blocked:
                return {"status": "partial", "iterations": i + 1, "answer": answer, "evidence": history,
                        "reason": "This answer did not pass every required check: " + contracts.summarise(blocked),
                        "contract_failures": [c["name"] for c in blocked]}
            answer, fixed = snap(answer, history, goal)          # a copying slip in a large figure is corrected, not argued over
            # ...and a copying slip in a station table's day cell, set to the forecast's own value for that station and day.
            answer, table_fixed = contracts.snap_tables(answer, history)
            answer += snap_note([*fixed, *table_fixed])
            # The answer's plain claims (a cycle, the best rule, a total the pack withheld) are checked against the stored
            # output. The fix is in the words, so the composer rewrites once; a claim still wrong after that leaves partial.
            # Round 4: these checks were built (C3a) and tested but never called here, so none ran live.
            claims = contracts.blocking(contracts.check_answer(answer, history, run_root=SANDBOX_ROOT, goal=goal)) if cfg.depth == 0 else []
            if claims:
                rewritten = _compose(ctx, brain, f"claims:{tag}:{i}", goal, history, memo, context, answer,
                                     fixes=contracts.summarise(claims))
                if rewritten is not None and rewritten.strip():
                    answer = rewritten
                    claims = contracts.blocking(contracts.check_answer(answer, history, run_root=SANDBOX_ROOT, goal=goal))
            if claims:
                return {"status": "partial", "iterations": i + 1, "answer": answer, "evidence": history,
                        "reason": "This answer states something the analysis output does not support: "
                                  + contracts.summarise(claims),
                        "contract_failures": [c["name"] for c in claims]}
            bad = unverified(answer, history, goal)
            if bad and _corrections(history) < MAX_CORRECTIONS and not last and not out_of_calls and not cfg.finish_now:
                # Gather before repairing: the figure may simply be an output the analysis has not printed yet. The
                # planner is told which figures lack a source, and the answer is judged again after it works.
                history.append({"kind": "evaluation", "verdict": "not_met", "missing": _missing_figures_text(bad),
                                "unverified": bad})
                continue
            pre_repair = None
            if bad:
                # Repair pass: the figures must come from the evidence or be left out. Figures that still lack a source
                # after the gathering rounds are removed, and the answer says what is not available.
                pre_repair = answer
                repaired = _compose(ctx, brain, f"repair:{tag}:{i}", goal, history, memo, context, answer, bad)
                if repaired is None:
                    fixed = ctx.step(f"repair:{tag}:{i}", lambda c: brain.decide(
                        _repair_messages(goal, history, memo, answer, bad, context), EVAL_SCHEMA, "eval"))
                    repaired = fixed.get("answer", "") or ""
                answer = repaired
                if not answer.strip():
                    return {"status": "partial", "iterations": i + 1, "answer": "",
                            "reason": "The results did not support a written answer.", "evidence": history}
                bad = unverified(answer, history, goal)
            if pre_repair is not None and answer != pre_repair and cfg.depth == 0:
                # The claims above were checked on the text before the repair. The rewrite may have changed a claim the stored
                # output supports (a cycle, a benchmark), so the claims are checked again on the text that will be returned.
                stale = contracts.blocking(contracts.check_answer(answer, history, run_root=SANDBOX_ROOT, goal=goal))
                if stale:
                    return {"status": "partial", "iterations": i + 1, "answer": answer, "evidence": history,
                            "reason": "The repaired answer states something the analysis output does not support: "
                                      + contracts.summarise(stale),
                            "contract_failures": [c["name"] for c in stale]}
            if bad:
                return {"status": "partial", "iterations": i + 1, "answer": answer,
                        "reason": "Some figures in this answer could not be matched to the analysis output.",
                        "unverified": bad, "evidence": history}
            out = {"status": "met", "iterations": i + 1, "answer": answer, "evidence": history}
            if cfg.depth:
                out["findings"], out["figures"] = verdict.get("findings") or [], verdict.get("figures") or []
            return out
        if verdict["verdict"] == "impossible":
            return {"status": "impossible", "iterations": i + 1, "reason": verdict["reason"], "evidence": history}
        if cfg.finish_now:
            # The person asked for the answer now. A best answer from the evidence, with what is missing, beats no answer,
            # and it is labelled partial, because the person chose to stop before the goal was met.
            draft = _gap_draft(verdict)
            written = _compose(ctx, brain, f"compose:{tag}:{i}", goal, history, memo, context, draft)
            text = written if written is not None and written.strip() else verdict.get("answer", "")
            return {"status": "partial", "iterations": i + 1, "answer": text or "", "reason": verdict["reason"],
                    "evidence": history}
        if last or out_of_calls:
            # This part of the work is spent (its steps, or the tool calls allowed so far), and the goal is not met. That
            # is a checkpoint, not a failure: the person continues from here, and the stored steps replay before the loop
            # goes on. The draft is kept so "answer now" can still write what is established.
            return {"status": "checkpoint", "iterations": i + 1, "segment": cfg.segment, "reason": verdict["reason"],
                    "draft": verdict.get("answer", ""), "evidence": history, "memory": memo.text}
        if plan["action"] == "finish":
            rejected_finishes += 1
            if rejected_finishes >= MAX_REJECTED_FINISHES:
                # The planner keeps saying it is done and the evaluator keeps saying it is not: more passes will not settle
                # it. Write up what is established, say what is not, and label it partial (seen live: 90 rejected finishes).
                draft = _gap_draft(verdict)
                written = _compose(ctx, brain, f"composeloop:{tag}:{i}", goal, history, memo, context, draft)
                text = written if written is not None and written.strip() else verdict.get("answer", "")
                if text.strip():
                    return {"status": "partial", "iterations": i + 1, "answer": text, "evidence": history,
                            "reason": "The analysis could not close this gap: " + verdict["reason"]}
                return {"status": "stalled", "iterations": i + 1, "evidence": history}
            history.append({"kind": "evaluation", "verdict": "not_met", "missing": verdict["reason"]
                            + " Your last finish was refused. Call a tool that closes this gap; if no tool can, finish again and "
                              "say plainly what is established and what is not."})
        else:
            rejected_finishes = 0
            history.append({"kind": "evaluation", "verdict": "not_met", "missing": verdict["reason"]})
    # Unreachable: the last step of the part always returns above. Kept so a future edit cannot fall through silently.
    return {"status": "checkpoint", "iterations": cfg.end, "segment": cfg.segment,
            "reason": "this part of the work is spent", "evidence": history, "memory": memo.text}


DATE_TEXT = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?(?:\s+\d{4})?"
                       r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,?\s+\d{4})?", re.I)
# Any heading level, with or without a colon or bold (audit 2026-11-22, item 8): "### Outside context:", "**Outside context**".
OUTSIDE_HEAD = re.compile(r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?[ \t]*outside context[ \t]*(?:\*\*)?[ \t]*:?[ \t]*$", re.I | re.M)
SECTION_END = re.compile(r"^(?:#{1,6}[ \t]+\S|\*\*[^*\n]+\*\*[ \t]*:?[ \t]*$)", re.M)


def _web_evidence(ctx: Ctx) -> list[dict]:
    """The passages of every page the mission read, as one evidence item for grounding (the loop's history holds only the
    planner's clipped view, and compaction may have folded it away)."""
    texts: list[str] = []
    for tool in ("web_research", "web_fetch"):
        for res in ctx.store.tool_results(ctx.mission_id, tool):
            if not isinstance(res, dict):
                continue
            for src in res.get("sources") or []:
                texts.extend(p.get("text", "") for p in src.get("passages") or [] if isinstance(p, dict))
            if isinstance(res.get("text"), str):
                texts.append(res["text"])
    return [{"kind": "tool", "tool": "web_sources", "args": {}, "result": {"passages": texts}}] if texts else []


def _check_outside(ctx: Ctx, answer: str) -> tuple[str, list[str]]:
    """Removes lines of the answer's Outside context section whose figures appear in no fetched web passage, and says so.
    Live, an answer cited "IMF: EUR/JPY 162.34", a number from the agent's own analysis that no source contained, and
    attributed an inventory figure to a site whose passage credited someone else. Dates are not figures. Returns the
    answer and the figures removed."""
    m = OUTSIDE_HEAD.search(answer or "")
    if not m:
        return answer, []
    from prophet_agent.research.passages import verify_figures
    sources: list = []
    for tool in ("web_research", "web_fetch", "web_search"):
        for res in ctx.store.tool_results(ctx.mission_id, tool):
            if isinstance(res, dict):
                sources.extend(res.get("sources") or res.get("hits") or [])
                if isinstance(res.get("text"), str):
                    sources.append({"text": res["text"]})
    nxt = SECTION_END.search(answer[m.end():])
    end = m.end() + nxt.start() if nxt else len(answer)
    kept, removed, dropped = [], [], 0
    for line in answer[m.end():end].splitlines():
        bad = verify_figures(DATE_TEXT.sub(" ", line), sources) if line.strip() else []
        if bad:
            removed.extend(b for b in bad if b not in removed)
            dropped += 1
        else:
            kept.append(line)
    if not removed:
        return answer, []
    body = "\n".join(kept).rstrip()
    # The note gives the count, not the figures: listing them put unsourced numbers back into the answer, and the grounding
    # check then flagged its own note (live EUR/JPY round 6). The figures stay in the result's outside_removed.
    body += (f"\n\n_{dropped} outside claim(s) were left out because their figures were not found in the pages "
             "that were read._\n")
    return answer[:m.end()] + body + ("\n" + answer[end:] if end < len(answer) else ""), removed


def goal_program(goal: str, tools: dict[str, Tool], brain: Brain, cfg: GoalConfig | None = None,
                 context: str = "") -> Callable[[Ctx], dict]:
    cfg = cfg or GoalConfig()

    def program(ctx: Ctx) -> dict:
        history: list[dict] = []
        bind = getattr(brain, "bind", None)
        if bind is not None:
            # A provider that is busy is waited out inside a step. The checkpoint lets that wait feel a cancel or a spent
            # time allowance (and keeps the lease alive); the note tells the person plainly that the wait is happening.
            last_note = [0.0]

            def note(info: dict) -> None:
                if time.time() - last_note[0] >= 20.0:
                    last_note[0] = time.time()
                    ctx.store.append_event(ctx.mission_id, "provider.waiting", info)

            bind(ctx._guard_time, note, ctx.seconds_left)
        if hasattr(brain, "on_text"):
            def show(text: str) -> None:
                ctx.store.set_draft(ctx.mission_id, text)

            brain.on_text = show
            show("")                       # a draft left by a crashed run is not the answer of this run
        result = _run(ctx, brain, goal, tools, cfg, context, history)
        # Outside claims are checked against the pages first (lines whose figures no page holds are removed), then the whole
        # answer is grounded against the analysis evidence plus the text of the pages that were read.
        if isinstance(result.get("answer"), str):
            checked, removed = _check_outside(ctx, result["answer"])
            if removed:
                result = {**result, "answer": checked, "outside_removed": removed}
        result = _ground_result(result, history + _web_evidence(ctx), goal)
        # A horizon the plan did not use is a note the answer carries, whatever the status: a partial answer names it too.
        hz_note = horizon_note(goal, history)
        if hz_note and isinstance(result.get("answer"), str) and result["answer"].strip() and hz_note not in result["answer"]:
            result = {**result, "answer": result["answer"].rstrip() + "\n\n" + hz_note}
        trust = contracts.trust_statement(history)
        if trust and isinstance(result.get("answer"), str) and result["answer"].strip() and "Accuracy check (computed" not in result["answer"]:
            # Written by code from the stored backtest, after grounding: the figures in it are the ledger's own.
            result = {**result, "answer": result["answer"].rstrip() + "\n\n" + trust, "trust": contracts.trust_facts(history)}
        show = getattr(brain, "on_text", None)
        final = result.get("answer")
        if callable(show) and isinstance(final, str) and final:
            show(final)                    # the notes added after the model finished are in the text the person sees
        return {**result, "stop_reason": STOP_REASONS.get(result.get("status", ""), "unknown")}

    return program


def _ground_result(result: dict, history: list[dict], goal: str) -> dict:
    """Every exit path is checked, not only the "met" one: a stalled, budget-capped or impossible answer carries
    figures too. Figures that the evidence does not support are listed on the result. A "met" result that fails the
    check is downgraded to partial, so it is never presented as fully grounded. An image the answer names that no run made
    is a failed check on every exit path too."""
    out = result
    named = contracts.check_answer_files(result.get("answer"), history)
    if named is not None and not named["ok"]:
        failures = list(result.get("contract_failures") or [])
        out = {**result, "contract_failures": failures + [named["name"]] if named["name"] not in failures else failures}
        if out.get("status") == "met":
            out["status"] = "partial"
            out["reason"] = "This answer states something the analysis output does not support: " + contracts.summarise([named])
    bad: list[str] = []
    for text in (result.get("answer") or "", result.get("reason") or ""):
        for token in unverified(text, history, goal):
            if token not in bad:
                bad.append(token)
    if not bad:
        return out
    out = {**out, "unverified": bad}
    if out.get("status") == "met":
        out["status"] = "partial"
        out["reason"] = "Some figures in this answer could not be matched to the analysis output."
    return out


MIN_CALL_S = 45.0   # seconds of the segment's allowance below which no model call starts: it is a checkpoint instead
DRAFT_FLUSH_S = 0.08  # shortest gap between two updates of the text the person is watching
DRAFT_FLUSH_CHARS = 48


class NimBrain:
    """Adapter from the goal loop to the NIM transport. Output is schema-bound and parsed here."""

    def __init__(self, client: Any, models: list[str] | None = None, *, thinking: bool = False,
                 max_tokens: int = 4096, write_tokens: int = 3000):
        self.write_tokens = write_tokens
        self.client = client
        self.models = models
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.checkpoint: Callable[[], None] | None = None
        self.on_wait: Callable[[dict], None] | None = None
        self.remaining: Callable[[], float | None] | None = None
        # Set by the goal program. Called with the full visible answer so far, only while reveal() is active.
        self.on_text: Callable[[str], None] | None = None
        self._reveal = False

    @contextlib.contextmanager
    def reveal(self):
        """Let write() pass arriving text to on_text. Planner calls stay buffered: they use decide(), not write()."""
        self._reveal = True
        try:
            yield
        finally:
            self._reveal = False

    def bind(self, checkpoint: Callable[[], None] | None, on_wait: Callable[[dict], None] | None = None,
             remaining: Callable[[], float | None] | None = None) -> None:
        """Let a provider wait be stopped (cancel, spent time allowance) and reported. Called by the goal program. ``remaining``
        gives the segment's seconds left, so one model call cannot run past the allowance (round 4, A5-06)."""
        self.checkpoint, self.on_wait, self.remaining = checkpoint, on_wait, remaining

    def _extra(self) -> dict:
        extra: dict = {}
        if self.checkpoint is not None:
            extra["checkpoint"] = self.checkpoint
        if self.on_wait is not None:
            extra["on_wait"] = self.on_wait
        if self.remaining is not None and (left := self.remaining()) is not None:
            if left < MIN_CALL_S:
                # Live electricity rep4 (2026-11-23): a planner call started with 1 s of the 900 s allowance left, ran out at
                # once, and the provider's "budget spent" failed the whole mission after 49 delegated calls. Too little time
                # for a call is the allowance being spent: a checkpoint the person continues, with every step kept.
                raise BudgetExceeded("max_seconds")
            extra["budget_s"] = min(BUDGET_S, float(left))
        return extra

    def _last_error(self) -> str:
        import sys
        return str(sys.exc_info()[1] or "the model service gave no usable answer")

    def _models(self) -> list:
        return list(self.models or getattr(self.client, "chain", None) or DEFAULT_CHAIN)

    def _guarded(self, fn: Callable[[dict], Any]) -> Any:
        extra = self._extra()
        try:
            return fn(extra)
        except NotConfigured:
            raise                       # no key or provider: continuing cannot help, so the mission fails and says why
        except Exhausted:
            # A call cut short by the segment's allowance (not by the provider's own budget) ran out of the person's time,
            # not of models: that is a checkpoint too.
            if extra.get("budget_s", BUDGET_S) < BUDGET_S and (self.remaining() or 0.0) < MIN_CALL_S:
                raise BudgetExceeded("max_seconds") from None
            raise BudgetExceeded("model_service", redact(str(self._last_error()))) from None
        except NimError as exc:
            # Live electricity rep7 (2026-11-23): "nim stream cut 4 times" inside one analyst failed the whole mission after
            # 64 calls. A provider failure is not the analysis failing: it is a checkpoint. Nothing after the failed call was
            # stored, so a continued run replays the stored steps and retries that call; the service continues on its own
            # up to its auto_continue limit, then the person decides.
            raise BudgetExceeded("model_service", redact(str(exc))) from None

    def _complete(self, messages: list[dict], **kw: Any) -> Any:
        def run(extra: dict) -> Any:
            return self.client.complete(self._models(), messages, **kw, **extra)
        return self._guarded(run)

    def _stream_text(self, messages: list[dict], extra: dict) -> str:
        """Read a free-text stream and hand the visible text to on_text as it grows. Returns the finished text."""
        parts: list[str] = []
        shown, last = "", 0.0
        reply = None
        for kind, value in self.client.stream(self._models(), messages, thinking=False, max_tokens=self.write_tokens, **extra):
            if kind == "delta":
                parts.append(value)
                visible = preview_text("".join(parts))
                now = time.monotonic()
                if not visible or visible == shown:
                    continue
                if shown and now - last < DRAFT_FLUSH_S and len(visible) - len(shown) < DRAFT_FLUSH_CHARS:
                    continue
                self.on_text(visible)
                shown, last = visible, now
            elif kind == "done":
                reply = value
        if reply is None:
            raise NimError("stream ended without an answer")
        if reply.text != shown:
            self.on_text(reply.text)
        return reply.text

    def write(self, messages: list[dict]) -> str:
        """Free text, no schema, so Markdown keeps its line breaks (a schema-bound string comes back on one line).

        Inside reveal(), the text is passed to on_text as tokens arrive. Outside it, the call waits for the whole reply,
        which is what the memo compaction and every scripted brain do.
        """
        if self._reveal and self.on_text is not None and hasattr(self.client, "stream"):
            return self._guarded(lambda extra: self._stream_text(messages, extra))
        reply = self._complete(messages, thinking=False, max_tokens=self.write_tokens)
        return reply.text

    def decide(self, messages: list[dict], schema: dict, name: str) -> dict:
        reply = self._complete(messages, thinking=self.thinking, max_tokens=self.max_tokens, schema=schema, schema_name=name)
        return json.loads(reply.text)
