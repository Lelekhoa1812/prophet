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
from . import prediction_quality
from . import grounding
from .grounding import snap, snap_note, unverified
from . import agents
from . import lanes, names
from . import claims as claim_checks
from .questions import QUESTIONS_SCHEMA, expected_shape, repair_questions, validate_for_goal
from .runner import READ_ONLY, UNCHARGED, ArgumentRejected, BudgetExceeded, Cancelled, Ctx, Paused, Tool
from .store import LeaseLost, OutcomeUnknown, canonical

from prophet_agent.sandbox.runner import OUT_DIR, SANDBOX_ROOT
from prophet_agent.tools.sandbox_tool import stand_in_reason

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["action", "rationale"],
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["call_tool", "ask_user", "delegate", "lanes", "finish"]},
        "tool": {"type": "string", "minLength": 1},
        "args": {"type": "object"},
        "questions": QUESTIONS_SCHEMA["properties"]["questions"],
        "task": {"type": "string", "maxLength": 1200},
        "name": {"type": "string", "maxLength": 60},
        "tools": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        "agent_type": {"type": "string", "enum": sorted(agents.TYPES)},
        "deliverable": {"type": "string", "maxLength": 600},
        "constraints": {"type": "string", "maxLength": 600},
        "resume": {"type": "string", "maxLength": 12},
        "tasks": {"type": "array", "maxItems": agents.MAX_PARALLEL, "items": {
            "type": "object", "required": ["task"], "additionalProperties": False,
            "properties": {"task": {"type": "string", "maxLength": 1200},
                           "name": {"type": "string", "maxLength": 60},
                           "agent_type": {"type": "string", "enum": sorted(agents.TYPES)},
                           "deliverable": {"type": "string", "maxLength": 600},
                           "constraints": {"type": "string", "maxLength": 600},
                           "tools": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                           "resume": {"type": "string", "maxLength": 12}}}},
        "lanes": {"type": "array", "minItems": 1, "maxItems": lanes.MAX_LANES, "items": {
            "type": "object", "required": ["task"], "additionalProperties": False,
            "properties": {"name": {"type": "string", "maxLength": 60},
                           "task": {"type": "string", "maxLength": 1200},
                           "agent_type": {"type": "string", "enum": sorted(agents.TYPES)},
                           "deliverable": {"type": "string", "maxLength": 600},
                           "constraints": {"type": "string", "maxLength": 600},
                           "tools": {"type": "array", "items": {"type": "string"}, "maxItems": 12}}}},
        "rationale": {"type": "string", "maxLength": 400},
    },
    # A call_tool plan names the tool and its arguments. Without this a plan with no tool passed the check, ran nothing and
    # stalled the mission (seen live: Matilda sent three such plans in a row on the electricity scenario).
    "allOf": [{"if": {"properties": {"action": {"const": "call_tool"}}, "required": ["action"]},
               "then": {"required": ["tool", "args"]}},
              {"if": {"properties": {"action": {"const": "lanes"}}, "required": ["action"]},
               "then": {"required": ["lanes"]}}],
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
VIEW_CHARS = 32000        # uncompacted observations shown before the older ones fold into the memo
KEEP_RECENT = 6           # newest observations never compacted
MIN_FOLD = 3              # older observations folded per compaction, unless the view is far over its limit
HARD_VIEW_FACTOR = 2.5    # the view may grow to this many times VIEW_CHARS before a smaller fold is forced
MEMORY_CHARS = 20000       # hard cap on the memo; a memo over this is cut and says so
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
MAX_CONTRACT_BLOCKS = 5    # times a failed deterministic check may send a finished answer back before it leaves as partial
# Owner, 2026-11-27: complex missions ended partial while the work was still moving. A block repeats only while the failing
# checks change; the same failing set on consecutive blocks means the repairs are not landing (CRITIC's stop rule: unchanged
# across two corrections), and the answer leaves as partial then, or at MAX_CONTRACT_BLOCKS.
SAME_BLOCKS_TO_STOP = 2
MAX_PIVOTS = 2             # stalls after the rescue that turn into a change of strategy before the run is written up
REPAIR_CALLS = 8          # one extra allowance, granted once, when a blocked answer meets a spent tool budget (2026-11-22)
# Panel round 10 (EPL): the request said "Ask me which clubs to keep", and four of the planner's questions were refused for
# lacking a "high-risk:" rationale. A request that asks to be asked is the person's own decision that the question is worth it.
ASKED_TO_ASK = re.compile(r"\bask me\b|\bcheck with me\b|\bconfirm with me\b", re.I)
CLAIM_REWRITES = 2       # composer rewrites for an answer claim the stored output does not support
MAX_PREMATURE_FINISHES = 2  # round 3, A5-04: finishes refused for checklist items still to do, per run, before the evaluator judges
MAX_REJECTED_FINISHES = 5  # finishes the evaluator may reject in a row before the work is written up as it stands
PIVOT_AFTER_REJECTIONS = 3  # rejections in a row that first turn into a change of strategy
_LEVEL_SENTENCE = re.compile(r"centre line is a level|center line is a level", re.I)
_RATIONALE_JUDGED = re.compile(r"finish rationale|required language", re.I)
# A refusal that also names a missing result is not only the level sentence. Do not turn that into met.
_OTHER_GAP = re.compile(r"\b(total|workbook|backtest|chart|rows|table)\b", re.I)
# Decision R1 (the person, 2026-11-22): no fixed number of questions. A question reaches the person only when the planner
# declares it high-risk and says how a wrong guess changes the numbers; anything else the agent decides and states as an
# assumption. Replaces MAX_ASKS = 8, which was set after a mission asked 109 low-value questions (mission 0c7432c7).
HIGH_RISK = re.compile(r"\bhigh[- ]risk\b\s*[:,-]?\s*(.{20,})", re.I | re.S)


def as_question_action(plan: dict) -> dict:
    """A plan that calls a tool named ask_user with questions is the ask_user action it means; one named finish is finish.

    Motivation: panel round 4 (EPL, 2026-11-27), the planner sent action call_tool with tool ask_user three times; each got
    "unknown tool 'ask_user'" and the run stalled on the repeat. The questions and rationale are kept as written.
    """
    args = plan.get("args") if isinstance(plan.get("args"), dict) else {}
    # Round 7 (CO2): the planner also sent finish as a tool call; "unknown tool 'finish'" repeated until the run stalled.
    if plan.get("action") == "call_tool" and plan.get("tool") == "finish":
        return {**{k: v for k, v in plan.items() if k not in ("tool", "args")}, "action": "finish",
                "rationale": plan.get("rationale") or str(args.get("rationale") or "")}
    if plan.get("action") == "call_tool" and plan.get("tool") == "ask_user" and "questions" in args:
        return {**{k: v for k, v in plan.items() if k not in ("tool", "args")}, "action": "ask_user",
                "questions": args["questions"]}
    return plan


SUB_STEPS = 14           # default step budget of a delegated sub-agent
BRANCH_BLOCK = 100_000    # step numbers reserved per plan step for parallel sub-agents, and per agent within it
BRANCH_STRIDE = 20_000
# The horizon a goal asks for: "next 20 trading days", "60 days ahead", "horizon of 12 periods" (digits only). Units are
# read as written; converting them to the series' own periods needs the series step (see horizon_checks).
HORIZON_REQUEST = re.compile(
    r"\b(?:next|horizon of|ahead)\s+(\d{1,4})[\s-]*(trading\s+days?|business\s+days?|days?|weeks?|months?|quarters?|years?"
    r"|periods?|steps?)\b"
    r"|\b(\d{1,4})[\s-]+(trading\s+days?|business\s+days?|days?|weeks?|months?|quarters?|years?|periods?|steps?)\s+"
    r"(?:ahead|horizon)\b", re.I)
ONE_UNIT_REQUEST = re.compile(r"\b(?:over|for|during|within)\s+the\s+(?:next|coming)\s+(year|quarter|week)\b", re.I)
# How old the data may be before a forecast "from now" needs a newer source: a release is normally out within these many days.
CURRENT_DATA_DAYS = {"monthly": 150, "quarterly": 270, "yearly": 640, "short": 30}
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
    lane: str = ""                    # this loop's lane id when it runs as a research lane: it reads the board before each plan
    lane_workers: int = lanes.DEFAULT_WORKERS   # lanes that run at once in a fan-out

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
        view = {k: obs[k] for k in ("kind", "agent_id", "agent_type", "task", "tools", "status", "answer", "reason",
                                    "steps", "files", "unverified") if k in obs}
        if obs.get("lane") and len(str(view.get("answer", ""))) > LANE_VIEW_CHARS:
            # A fan-out of many lanes would fill the view with reports; the digest carries the findings, and
            # read_observation pages the rest.
            view["answer"] = str(view["answer"])[:LANE_VIEW_CHARS] + "...[read_observation pages the full report]"
        return view
    if "full" in obs:
        return {k: v for k, v in obs.items() if k != "full"}
    return obs


LANE_VIEW_CHARS = 1_500   # a lane report's answer as the lead's planner sees it


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


# Tools whose later successful call replaces an earlier one with the same key (the tool, the horizon, and for
# forecast_report its centre).
# Audit of round 20261128-173823: the finish evaluator judged AUUNEMP five times on a 24-step forecast_report the lead had
# already re-run at 25 steps (eval:d0:22-26), AURATE on a superseded scenario run, and SPX read one run's figures as
# another's. Each older result now says which step replaced it, and only its head is shown.
SUPERSEDED_TOOLS = ("forecast_report", "scenario_forecast", "simulate_season", "forecast", "season_backtest")
SUPERSEDED_CHARS = 400


def _supersede_key(obs: dict) -> tuple | None:
    if obs.get("kind") != "tool" or obs.get("tool") not in SUPERSEDED_TOOLS or "error" in obs or "result" not in obs:
        return None
    args = obs.get("args") if isinstance(obs.get("args"), dict) else {}
    # The horizon is part of the key: round 20261128-201606 (AUUNEMP) re-ran a 24-step combination after the 25-step one
    # the answer quoted, and hiding the 25-step run left its figures unverified. A re-run on the same horizon (a longer
    # series, new settings) replaces the earlier one.
    return (obs["tool"], str(args.get("centre") or "") if obs["tool"] == "forecast_report" else "", str(args.get("horizon")))


def superseded_by(history: list[dict], start: int = 1) -> dict[int, int]:
    """{position: position of the later call that replaced it}, positions numbered from ``start`` as in the evidence."""
    latest: dict[tuple, int] = {}
    out: dict[int, int] = {}
    for n in range(len(history) - 1, -1, -1):
        key = _supersede_key(history[n])
        if key is None:
            continue
        if key in latest:
            out[n + start] = latest[key]
        else:
            latest[key] = n + start
    return out


def _evidence_lines(history: list[dict], start: int = 1) -> str:
    # Each observation is clipped on its own, so the goal and the tool list are never cut off by a long result.
    if not history:
        return "(none yet)"
    replaced = superseded_by(history, start)
    lines = []
    for n, obs in enumerate(history, start=start):
        if n in replaced:
            head = canonical(_view(obs))[:SUPERSEDED_CHARS]
            lines.append(f"{n}. [SUPERSEDED by step {replaced[n]}: a later {obs.get('tool')} call replaced this result; "
                         f"judge and quote step {replaced[n]}, not this one] {head}...")
        else:
            lines.append(f"{n}. {_clip(_view(obs), obs.get('step'))}")
    return "\n".join(lines)


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
# VN Unemployment (2026-11-27): the same web_search ran twice in a row, twice, and the stall rule stopped the mission. The web
# does not change within one mission, so an identical web call is answered with the earlier result like any other call.
REPEAT_EXEMPT = frozenset(READ_ONLY)
# A call whose result depends only on its arguments, so an identical call after an unrelated cell is still the same result.
PURE_TOOLS = frozenset({"sandbox_python"})
WEB_TOOLS = frozenset({"web_research", "web_search", "web_fetch", "search_open_data"})


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
            if (not pure and name not in WEB_TOOLS
                    and any(x.get("kind") == "tool" and x.get("tool") == "sandbox_python" for x in history[pos:])):
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


OPTION_FIGURE = re.compile(r"[-+]?\d[\d,]*\.\d+|[-+]?\d[\d,]*\s*(?:%|percent|pts?\b|points?\b|wins?\b|losses\b|"
                           r"goals?\b|games?\b)", re.IGNORECASE)


RANK_CLAIM = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)\s+(?:place|after|in the (?:table|league|standings)|position)\b"
                        r"|\b(?:current |league )?leaders?\b|\bcurrent(?:ly)? (?:standing|position)s?\b|\bby current standing\b",
                        re.I)
STANDINGS_EVIDENCE = re.compile(r"\bstandings?\b|\bleague table\b|\bposition\b|\bpts\b", re.I)


def _option_figure_problems(questions: list, history: list[dict], goal: str) -> list[str]:
    """A figure in a question option must come from the evidence. Round 20 (EPL, 2026-11-27): the options named
    "Liverpool (28 points)" while the downloaded table said 21, the person chose on that figure, and the run ended partial."""
    problems = []
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            continue
        for j, o in enumerate(q.get("options") or []):
            if not isinstance(o, dict):
                continue
            text = f"{o.get('label') or ''} {o.get('description') or ''}"
            stated = [m.group(0) for m in OPTION_FIGURE.finditer(text)]
            if not stated:
                continue
            bad = unverified(" ".join(stated), history, goal)
            if bad:
                problems.append(f"questions[{i}].options[{j}] states {', '.join(bad[:3])}, which no tool printed: take "
                                "the figure from the fetched data or leave it out of the option")
    # Round 21 (EPL, 2026-11-27): the options said "Current leader", "2nd after 12 rounds", "4th after 12 rounds" before any
    # table was computed; the fetched results had Brighton 2nd and Chelsea 8th, and 110 matches were 11 rounds.
    ranked = [f"questions[{i}].options[{j}]" for i, q in enumerate(questions) if isinstance(q, dict)
              for j, o in enumerate(q.get("options") or []) if isinstance(o, dict)
              and RANK_CLAIM.search(f"{o.get('label') or ''} {o.get('description') or ''}")]
    if ranked and not any(STANDINGS_EVIDENCE.search(str(h.get("result"))) for h in history
                          if h.get("kind") == "tool" and "error" not in h):
        problems.append(f"{', '.join(ranked[:3])} name a league position, but no tool has computed the standings yet: compute "
                        "the table from the fetched results first (points, played, position), then build the options from it")
    return problems


def _last_failed_tool(history: list[dict]) -> str | None:
    """The tool of the newest failed call, or None."""
    for h in reversed(history):
        if h.get("kind") == "tool" and _error_signature(h):
            return h.get("tool")
    return None


def _stored_step_name(ctx: Ctx) -> str | None:
    """The name of the step the ledger already holds at the next position, or None on a fresh run."""
    store, mid = getattr(ctx, "store", None), getattr(ctx, "mission_id", None)
    if store is None or mid is None:
        return None
    rec = store.step_record(mid, ctx.step_idx + 1)
    return rec["name"] if rec is not None and rec.get("status") == "done" else None


def _ledger_skipped_act(ctx: Ctx, act_name: str) -> bool:
    """True when the ledger already holds a different step where this act step would go: the stored run did not call the
    tool at this point. Live GOLD a00e870d (2026-11-28): "replay diverged at step 18: the ledger has 'plan:d0:9', the program
    asked for 'act:d0:8'" after the tool set grew between parts."""
    stored = _stored_step_name(ctx)
    return stored is not None and stored != act_name


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
    if "table-claim" not in kinds and "extract_page_tables" in tools and contracts.html_download_urls(history):
        urls = contracts.extract_target_urls(history)
        claimed = any(h.get("kind") == "tool" and h.get("tool") == "state_table_claim" and not h.get("error") for h in history)
        if urls and not claimed:
            named = ", ".join(urls)
            return {"action": "delegate", "auto": "table-claim", "agent_type": "principal_analyst",
                    "rationale": "a download returned a page that web_fetch already stored; extract that page before a model step",
                    "task": "A file download returned an HTML page while working on this goal: " + goal
                            + "\nThe page is already stored. Call extract_page_tables once for each of these urls: " + named
                            + ". State the table claim first when it is not already in the history. "
                            "Do not store a different series under the requested name.",
                    "deliverable": "The claim and each candidate dataset id. Say when the series is still unbound.",
                    "constraints": "Do not call web_fetch. Do not call download_web_data on those urls. "
                                   "Do not bind the series. Do not forecast."}
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
        # VN Unemployment (2026-11-27): repeated HTTP 404s from downloads were handed to an analyst, who has no download
        # tools and could only retype numbers. A failing download is the planner's to reroute (the refusal names the next
        # call); only a failing analysis cell goes to a fresh analyst.
        if key not in kinds and "sandbox_python" in tools and _last_failed_tool(history) == "sandbox_python":
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
                            "(entity, series name, measure). Diagnose each failed download. When the signal is html_page, "
                            "call extract_page_tables on a url web_fetch already stored. When no page is stored, call "
                            "web_fetch with the failed download's https address and no queries argument, then call "
                            "extract_page_tables with that same url. Do not store a different series under the requested name. "
                            "Do not invent an address that no page listed.",
                    "deliverable": "The claim, each signal, and any candidate dataset id. Say when the series is still unbound.",
                    "constraints": "Do not forecast. Do not call bind_series."}
    # Official series and the league backtest need no file of the mission's own, so they run before any inspection: the
    # VNUNEMP missions never found the statistics office's quarterly rate on their own (Mission 10).
    if (step := _official_series_step(goal, history, tools)) is not None:
        return step
    if (step := _league_backtest_step(goal, history, tools)) is not None:
        return step
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
    if (step := _grain_step(goal, history, tools)) is not None:
        return step
    if (step := _market_driver_step(goal, history, tools)) is not None:
        return step
    if (step := _horizon_backtest_step(history, tools)) is not None:
        return step
    if (step := _combination_step(goal, history, tools)) is not None:
        return step
    if (step := _risk_regime_step(goal, history, tools)) is not None:
        return step
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


TARGET_TOOLS = ("backtest_series", "timesfm_forecast", "control_lab", "forecast_report")
SPX_GOAL = re.compile(r"s\s*&\s*p\s*500|\bspx\b|\bsp500\b|\^gspc\b", re.I)
GOLD_GOAL = re.compile(r"\bgold\b|\bxau\b|\bgc=f\b", re.I)
SCAN_HORIZON_CAP = 64


def _target_source(history: list[dict]) -> tuple[dict, int | None] | None:
    """The source {run_id, file[, name]} and horizon of the newest successful call that scored or forecast one saved series."""
    for h in reversed(contracts.with_delegated(history)):
        if not isinstance(h, dict) or h.get("kind") != "tool" or h.get("tool") not in TARGET_TOOLS or h.get("error"):
            continue
        a = h.get("args") if isinstance(h.get("args"), dict) else {}
        src = a.get("source")
        if isinstance(src, dict) and isinstance(src.get("run_id"), str):
            hz = a.get("horizon")
            return {k: src[k] for k in ("run_id", "file", "name") if k in src}, (hz if isinstance(hz, int) else None)
    return None


def _benchmark_kind(goal: str, domain: str) -> str | None:
    from prophet_agent.research import driver_families as dfam
    if domain == "policy_rate" and dfam.place_of(goal) == "Australia":
        return "au_cash_rate"
    if domain == "equity_index" and SPX_GOAL.search(goal):
        return "vix"
    if domain == "precious_metal" and GOLD_GOAL.search(goal):
        return "gvz"
    return None


def _market_driver_step(goal: str, history: list[dict], tools: dict[str, Tool]) -> dict | None:
    """The market benchmark and the driver scan, run by the harness once each for a market or macro target (owner decision
    2026-11-28, after round 20261128-110827: AUUNEMP refused 17 finishes and SPX 3 without scoring a driver or reading the
    market's own pricing, though the blocking checks named both every time). The benchmark runs as soon as it can (a cash
    rate path needs no level; a VIX or GVZ range needs the target's last value); the scan once a series has been backtested
    or forecast, on that series and its horizon. A call made earlier, by the model or a sub-agent, is not made again."""
    from prophet_agent.research import driver_families as dfam
    domain = dfam.detect(goal or "")
    if domain not in prediction_quality.DRIVEN_DOMAINS:
        return None
    made = {h.get("tool") for h in contracts.with_delegated(history) if isinstance(h, dict) and h.get("kind") == "tool"}
    target = _target_source(history)
    kind = _benchmark_kind(goal, domain)
    if kind and "market_benchmark" in tools and "market_benchmark" not in made and (kind == "au_cash_rate" or target):
        args: dict[str, Any] = {"kind": kind}
        if kind != "au_cash_rate" and target:
            from prophet_agent.tools.series_io import resolve_series
            try:
                named = resolve_series({"source": target[0]})
                if len(named) == 1 and len(named[0][1]) and float(named[0][1][-1]) > 0:
                    args["level"] = round(float(named[0][1][-1]), 4)
            except (ValueError, OSError):
                pass
        return {"action": "call_tool", "tool": "market_benchmark", "args": args, "auto": "market",
                "rationale": "the harness reads what the market prices for this target before the answer is written"}
    if target and "driver_scan" in tools and "driver_scan" not in made:
        horizon = min(target[1] or 12, SCAN_HORIZON_CAP)
        return {"action": "call_tool", "tool": "driver_scan", "auto": "drivers",
                "args": {"source": target[0], "goal": goal, "horizon": max(1, horizon)},
                "rationale": "the harness scores the domain's named outside drivers against the target series"}
    return None


LEAGUE_CODES = ((re.compile(r"premier league|\bepl\b", re.I), "E0"), (re.compile(r"\bchampionship\b", re.I), "E1"),
                (re.compile(r"la liga", re.I), "SP1"), (re.compile(r"bundesliga", re.I), "D1"),
                (re.compile(r"serie a\b", re.I), "I1"), (re.compile(r"ligue 1\b", re.I), "F1"))


def _league_backtest_step(goal: str, history: list[dict], tools: dict[str, Tool]) -> dict | None:
    """The season simulator scored on the league's past seasons, run once by the harness for a football league-table goal
    (Mission 10, owner review 2026-11-28: an EPL table was quoted with no measure of how far a final table is usually out).
    Runs once a mission, before the simulation, so the answer can state the error beside the table."""
    if "league_backtest" not in tools:
        return None
    made = {h.get("tool") for h in contracts.with_delegated(history) if isinstance(h, dict) and h.get("kind") == "tool"}
    if "league_backtest" in made:
        return None
    code = next((c for rx, c in LEAGUE_CODES if rx.search(goal or "")), None)
    if code is None:
        return None
    return {"action": "call_tool", "tool": "league_backtest", "auto": "league_backtest", "args": {"league": code},
            "rationale": "the harness scores the season model's final table on the league's past seasons"}


def _official_series_step(goal: str, history: list[dict], tools: dict[str, Tool]) -> dict | None:
    """An official series the harness assembles once, before any search, when the goal names one (Mission 10, M10-4):
    Vietnam's unemployment rate comes from the statistics office's quarterly releases, not a series that ends in 2024."""
    if "official_series" not in tools:
        return None
    from prophet_agent.research import driver_families as dfam
    if dfam.detect(goal or "") != "labour_market" or dfam.place_of(goal or "") != "Vietnam":
        return None
    made = {h.get("tool") for h in contracts.with_delegated(history) if isinstance(h, dict) and h.get("kind") == "tool"}
    if "official_series" in made:
        return None
    return {"action": "call_tool", "tool": "official_series", "auto": "official_series",
            "args": {"series": "vn_unemployment_quarterly"},
            "rationale": "the harness assembles the statistics office's quarterly rate, the latest published figures"}


GRAIN_MARK = "# harness: grain"
GRAIN_RULE = {"monthly": ("MS", 30.4), "quarterly": ("QS", 91.3), "weekly": ("W-FRI", 7.0), "yearly": ("YS", 365.25)}
LEVEL_DOMAINS = ("equity_index", "precious_metal", "commodity", "currency", "policy_rate")
GRAIN_CODE = '''{mark}
import json
import prophet_lib as pl

s = pl.load_series("in/carried/{fname}")
if isinstance(s, dict):
    s = s[{name!r}] if {name!r} in s else next(iter(s.values()))
s = s.dropna().sort_index()
g = s.resample("{rule}").last().dropna()          # the value at the end of each period: a close, or the rate in force
pl.save_series(g, "out/series_{grain}.json", name={label!r})
print(json.dumps({{"grain": "{grain}", "points": len(g), "from": str(g.index[0].date()), "to": str(g.index[-1].date()),
                  "last_value": float(g.iloc[-1]), "last_source_day": str(s.index[-1].date()),
                  "source_points": len(s)}}))
print("The last period holds the data up to %s only. Forecast out/series_{grain}.json from this run (source = this run id "
      "and that file) in {grain} periods; the finer series is not the grain the forecast is reported in." % s.index[-1].date())
'''


def _grain_step(goal: str, history: list[dict], tools: dict[str, Tool]) -> dict | None:
    """The target series resampled once to the grain the forecast is reported in, by the harness.

    Round 20261128-125459: SPX was asked for "month by month over the next 2 years" and built a 42-step and a 504-step daily
    pack before a monthly one, then spent its budget; the grain check refuses daily packs only at finish. AURATE was asked
    for five years of a daily cash rate, hit the 512-step limit and delivered two years. Logic: once a saved series has been
    backtested or forecast, when its median step is finer than the grain the goal names (monthly, quarterly, weekly, yearly),
    or the target is a policy rate kept daily, and the domain is a level (a price, an index, a rate), sandbox_python takes
    the value at the end of each period and saves out/series_<grain>.json. Runs once a mission; flows and other domains are
    left to the planner, because the aggregation (sum or mean) depends on what the series counts."""
    if "sandbox_python" not in tools:
        return None
    from prophet_agent.research import driver_families as dfam
    domain = dfam.detect(goal or "")
    if domain not in LEVEL_DOMAINS:
        return None
    flat = [h for h in contracts.with_delegated(history) if isinstance(h, dict) and h.get("kind") == "tool"]
    if any(h.get("tool") == "sandbox_python" and GRAIN_MARK in str((h.get("args") or {}).get("code") or "") for h in flat):
        return None
    asked = [name for name, pattern, _ in prediction_quality._GRAINS if pattern.search(goal or "")]
    grain = asked[0] if len(asked) == 1 else ("monthly" if domain == "policy_rate" and not asked else None)
    if grain is None or grain not in GRAIN_RULE:
        return None
    target = _target_source(history)
    if target is None or not isinstance(target[0].get("file"), str):
        return None
    from datetime import date as _date
    from prophet_agent.tools.series_io import resolve_series, run_dates
    try:
        named = resolve_series({"source": target[0]})
        dates = run_dates({"source": target[0]}, named[0][0]) if len(named) == 1 else None
    except (ValueError, OSError):
        return None
    if not dates or len(dates) < 3:
        return None
    days = sorted((_date.fromisoformat(str(b)[:10]) - _date.fromisoformat(str(a)[:10])).days for a, b in zip(dates, dates[1:]))
    rule, period = GRAIN_RULE[grain]
    if days[len(days) // 2] >= 0.8 * period:
        return None
    name = named[0][0]
    code = GRAIN_CODE.format(mark=GRAIN_MARK, fname=target[0]["file"].split("/")[-1], name=target[0].get("name") or name,
                             rule=rule, grain=grain, label=str(target[0].get("name") or name))
    return {"action": "call_tool", "tool": "sandbox_python", "auto": "grain",
            "args": {"code": code, "carry": [{"run_id": target[0]["run_id"], "file": target[0]["file"]}],
                     "name": f"Resample the target to {grain} periods"},
            "rationale": f"the forecast is reported in {grain} periods and the saved series steps every "
                         f"{days[len(days) // 2]} days; the harness takes the value at the end of each period"}


FORECAST_TOOLS = ("forecast_report", "timesfm_forecast")
CARRIED_CONTROL = ("past_covariates", "future_covariates", "context_points", "ignore_ranges", "season")


def _horizon_backtest_step(history: list[dict], tools: dict[str, Tool]) -> dict | None:
    """A backtest at the horizon of the newest forecast, run by the harness once when no backtest covers it.

    Round 20261128-125459: GOLD's data ended four months before the run, so its 12-month forecast ran 16 periods and the
    baseline check held every finish ("no backtest covers that horizon (backtests ran at [12])"); AUUNEMP the same at 25
    against 24. The repair is one deterministic call: backtest_series on the forecast's own source and horizon, with the
    drivers and context it used. A call already made at that horizon on that source (even a failed one) is not repeated."""
    if "backtest_series" not in tools:
        return None
    flat = [h for h in contracts.with_delegated(history) if isinstance(h, dict) and h.get("kind") == "tool"]
    def usable(h: dict, tools_: tuple) -> bool:
        return (h.get("tool") in tools_ and not h.get("error") and isinstance(h.get("args"), dict)
                and isinstance((h["args"].get("source") or None), dict)
                and isinstance(h["args"].get("horizon"), int) and not isinstance(h["args"].get("horizon"), bool))
    fc = next((h for h in reversed(flat) if usable(h, FORECAST_TOOLS)), None)
    # Round 20261128-201606 (GOLD): the lead backtested the saved series (backtest_series, horizon 12), its driver scan was
    # refused, and it stalled on sandbox errors before any forecast; the mission ended with no forecast at all. A finished
    # backtest of a saved series names the source and horizon as well as a forecast does.
    fc = fc or next((h for h in reversed(flat) if usable(h, ("backtest_series",))), None)
    if fc is None:
        return None
    a = fc["args"]
    horizon, src = a["horizon"], a["source"]
    for h in flat:
        if h.get("tool") != "backtest_series" or not isinstance(h.get("args"), dict):
            continue
        hz = h["args"].get("horizon")
        if not isinstance(hz, int):
            continue
        if (not h.get("error") and hz >= horizon) or (hz == horizon and h["args"].get("source") == src):
            return None
    args = {"source": src, "horizon": horizon, **{k: a[k] for k in CARRIED_CONTROL if k in a}}
    return {"action": "call_tool", "tool": "backtest_series", "auto": "horizon", "args": args,
            "rationale": f"the forecast runs {horizon} periods ahead and no backtest covers that horizon; the harness "
                         "backtests the same series at it"}


def _combination_step(goal: str, history: list[dict], tools: dict[str, Tool]) -> dict | None:
    """The backtest-weighted combination, run by the harness once per forecast source and horizon (Mission 10, owner decision
    2026-11-28, "earn it on backtest"): TimesFM, the simple rules and the domain's market and structural candidates are scored
    on the same origins, and the pack draws their combination. A ledger audit found every delivered centre came from the
    target's own history while the market path and the drivers never competed. Runs for the driven domains (rates, labour,
    prices, indices) once a forecast of a saved series exists; a call already made on that source and horizon is not repeated."""
    if "forecast_report" not in tools:
        return None
    from prophet_agent.research import driver_families as dfam
    if dfam.detect(goal or "") not in prediction_quality.DRIVEN_DOMAINS:
        return None
    flat = [h for h in contracts.with_delegated(history) if isinstance(h, dict) and h.get("kind") == "tool"]
    def usable(h: dict, tools_: tuple) -> bool:
        return (h.get("tool") in tools_ and not h.get("error") and isinstance(h.get("args"), dict)
                and isinstance((h["args"].get("source") or None), dict)
                and isinstance(h["args"].get("horizon"), int) and not isinstance(h["args"].get("horizon"), bool))
    fc = next((h for h in reversed(flat) if usable(h, FORECAST_TOOLS)), None)
    # Round 20261128-201606 (GOLD): the lead backtested the saved series (backtest_series, horizon 12), its driver scan was
    # refused, and it stalled on sandbox errors before any forecast; the mission ended with no forecast at all. A finished
    # backtest of a saved series names the source and horizon as well as a forecast does.
    fc = fc or next((h for h in reversed(flat) if usable(h, ("backtest_series",))), None)
    if fc is None:
        return None
    a = fc["args"]
    source = a["source"]
    # An adopted driver reaches the forecast (M10-5): the driver scan saved the target with its adopted drivers, and the
    # combination on that file adds TimesFM with them as a candidate.
    for h in reversed(flat):
        res = contracts._ok_result(h, "driver_scan")
        use = res.get("use") if isinstance(res, dict) else None
        if isinstance(use, dict) and isinstance(use.get("source"), dict) and use.get("points", 0) >= 60:
            source = {k: use["source"][k] for k in ("run_id", "file", "name") if k in use["source"]}
            break
    for h in flat:
        b = h.get("args") if isinstance(h.get("args"), dict) else {}
        if (h.get("tool") == "forecast_report" and b.get("centre") == "combination" and b.get("source") in (source, a["source"])
                and b.get("horizon") == a["horizon"]):
            return None
    args = {"source": source, "horizon": a["horizon"], "centre": "combination", "goal": goal,
            "title": str(a.get("title") or "Forecast")[:80]}
    return {"action": "call_tool", "tool": "forecast_report", "auto": "combination", "args": args,
            "rationale": "the harness scores every candidate (TimesFM, the rules, the market and structural models) on the "
                         "same origins and draws their backtest-weighted combination"}


def _risk_regime_step(goal: str, history: list[dict], tools: dict[str, Tool]) -> dict | None:
    """risk_regime once per mission, after the combination (Mission 10 batch 2): geopolitical and policy risk today and the
    target's measured moves after past spikes, so the answer states the regime and scenario shifts can cite a measurement.
    Runs for the driven domains on the combination's source and horizon; never repeated."""
    if "risk_regime" not in tools:
        return None
    from prophet_agent.research import driver_families as dfam
    if dfam.detect(goal or "") not in prediction_quality.DRIVEN_DOMAINS:
        return None
    flat = [h for h in contracts.with_delegated(history) if isinstance(h, dict) and h.get("kind") == "tool"]
    if any(h.get("tool") == "risk_regime" for h in flat):
        return None
    comb = next((h for h in reversed(flat) if h.get("tool") == "forecast_report" and not h.get("error")
                 and isinstance(h.get("args"), dict) and h["args"].get("centre") == "combination"
                 and isinstance(h["args"].get("source"), dict) and isinstance(h["args"].get("horizon"), int)), None)
    if comb is None:
        return None
    args = {"source": comb["args"]["source"], "horizon": comb["args"]["horizon"], "goal": goal}
    return {"action": "call_tool", "tool": "risk_regime", "auto": "risk", "args": args,
            "rationale": "the harness reads geopolitical and policy risk today and measures how the target moved after "
                         "past risk spikes"}


# The checks a refused finish names, and the tools that close each one (owner decision 2026-11-28, "Both": live AUUNEMP refused
# 17 finishes for these checks and spent its calls on 33 web_research and 20 web_fetch calls, never on the test they asked for).
FOCUS_CHECKS: dict[str, tuple[str, ...]] = {
    "signals_gathered": ("driver_scan", "control_lab", "download_open_data", "download_market_prices", "backtest_series"),
    "answer_signal_search": ("driver_scan", "control_lab", "download_open_data", "download_market_prices", "backtest_series"),
    "centre_not_flat": ("driver_scan", "control_lab", "backtest_series", "scenario_forecast"),
    "fundamentals_covered": ("driver_map", "scenario_forecast", "market_benchmark", "driver_scan", "web_research", "web_fetch"),
    "band_vs_market": ("market_benchmark", "calibrate_intervals"),
    "market_stated": ("market_benchmark",),
}
FOCUS_ALWAYS = ("forecast_report", "read_observation", "sandbox_python", "notes")
FOCUS_AFTER = 2


def _focus_tools(history: list[dict], tools: dict[str, Tool]) -> tuple[dict[str, Tool], str]:
    """After FOCUS_AFTER finishes refused for a driver, fundamentals or market check, and while the newest refusal still names
    one, the planner is offered only the tools that close the named checks (plus the pack, paging and analysis tools). A
    later evaluation that names none of them lifts the focus. Derived from the history alone, so a replay sees the same."""
    evals = [h for h in history if isinstance(h, dict) and h.get("kind") == "evaluation"]
    refused = [h for h in evals if set(h.get("contract") or ()) & FOCUS_CHECKS.keys()]
    if len(refused) < FOCUS_AFTER or evals[-1] is not refused[-1]:
        return tools, ""
    open_checks = [c for c in refused[-1]["contract"] if c in FOCUS_CHECKS]
    keep = set(FOCUS_ALWAYS) | {t for c in open_checks for t in FOCUS_CHECKS[c]}
    offered = {n: t for n, t in tools.items() if n in keep}
    if not offered:
        return tools, ""
    return offered, (f"The finish was refused {len(refused)} times for {', '.join(open_checks)}. Until those checks pass, only "
                     f"the tools that close them are offered: {', '.join(sorted(offered))}. Do that work (read each refusal "
                     "for what is missing), rebuild the pack if the forecast changed, then finish.")


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
    from prophet_agent.runtime import league_table
    if league_table.table_markdown(history):
        outside = ("A table of every team from the newest season simulation (points now, expected final points, 10-90 range, "
                   "expected position and probabilities) is appended after your answer by code. Do not write your own "
                   "table of expected points or probabilities; name the leading teams and their figures only as that "
                   "simulation gives them, and refer to the appended table for the rest."
                   + ("\n\n" + outside if outside else ""))
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


# A sub-agent (depth 1) cannot start lanes or delegate: those plans were refused every time and planned again. Live round
# 20261128-063203: 924 "lanes" decisions in VNUNEMP, 126 of 166 in AUUNEMP, 14 in a row in SPX and in EPL lane a5. The
# nested planner is not offered them.
PLAN_SCHEMA_NESTED: dict[str, Any] = {
    **PLAN_SCHEMA,
    "properties": {**PLAN_SCHEMA["properties"],
                   "action": {"type": "string", "enum": ["call_tool", "ask_user", "finish"]}},
    **({"allOf": [c for c in PLAN_SCHEMA["allOf"] if c.get("if", {}).get("properties", {}).get("action", {}).get("const")
                  not in ("lanes", "delegate")]} if "allOf" in PLAN_SCHEMA else {}),
}


def _decide_plan(brain: Any, goal: str, tools: dict[str, Tool], history: list[dict], memo: Memo, context: str,
                 note: str, *, finish_now: bool, out_of_calls: bool, nested: bool = False) -> dict:
    """The planner's decision for one step, made inside a stored step. Forced finishes never reach a tool: when the
    person asked for the answer now, or the tool allowance is spent, the decision is a finish."""
    if finish_now:
        return {"action": "finish", "rationale": "the person asked for the answer so far"}
    if out_of_calls:
        # No tool call is left, so planning another would only end the run with nothing to show.
        return {"action": "finish", "rationale": "tool budget used"}
    return brain.decide(_plan_messages(goal, tools, history, memo, context, note),
                        PLAN_SCHEMA_NESTED if nested else PLAN_SCHEMA, "plan")


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


ROLLUP_CHARS = 600
_STUB_LINE = re.compile(r"^(\d+)\. (?:sub-agent (a\d+)|([A-Za-z_]+)\()", re.M)


def _rollup(cut: str) -> str:
    """One line for the part of the memory cut to fit: which observations it held, by tool, so the planner still knows they
    exist and can page any back. Owner, 2026-11-27: the cut said only "[middle of the memory cut to fit]", and the steps in it
    were lost from the planner's sight."""
    found = _STUB_LINE.findall(cut)
    if not found:
        return "[part of the memory was cut to fit; the full record is kept, read_observation pages any step]"
    steps = [int(n) for n, _, _ in found]
    counts: dict[str, int] = {}
    for _, agent, tool in found:
        name = f"sub-agent {agent}" if agent else tool
        counts[name] = counts.get(name, 0) + 1
    tools = ", ".join(f"{k} x{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    return (f"[observations {min(steps)} to {max(steps)} were folded out of this memory to fit ({tools}); their results are "
            "kept, and read_observation with the step pages any of them]")[:ROLLUP_CHARS]


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
        tail = text[-(memory_chars - keep_head - ROLLUP_CHARS):]
        text = text[:keep_head] + "\n" + _rollup(text[keep_head:len(text) - len(tail)]) + "\n" + tail
    memo.text = text
    memo.covered = cut


def _usable_questions(verdict: dict, goal: str = "") -> list[dict] | None:
    """The evaluator's questions, if it asked any and they pass the same validator as the planner's (shape, and no option that
    offers invented data in place of the real series)."""
    qs = verdict.get("questions")
    if not qs or validate_for_goal({"questions": qs}, goal):
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
        args = {k: (_code_form(v) if k == "code" and isinstance(v, str) else v)
                for k, v in args.items() if k not in ("carry", "name")}      # a new name on the same code is a repeat
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
    seen: set[str] = set()
    for h in _with_subagent_steps(history):
        res = h.get("result") if h.get("kind") == "tool" else None
        if not isinstance(res, dict) or not isinstance(res.get("run_id"), str):
            continue
        paths: set[str] = set()
        for o in res.get("outputs") or res.get("files") or []:
            if isinstance(o, str):
                paths.add(o.split(" (")[0])
            elif isinstance(o, dict) and isinstance(o.get("path"), str):
                paths.add(o["path"])
        if res["run_id"] in seen:
            continue
        seen.add(res["run_id"])
        runs.append((res["run_id"], paths))
    return runs


def _with_subagent_steps(history: list[dict]) -> list[dict]:
    """The history with each sub-agent's own steps in place of its report. VN Unemployment (2026-11-27): a file an analyst
    wrote was not auto-mounted for the lead, which then opened in/carried/<that file> and failed."""
    out: list[dict] = []
    for h in history:
        if h.get("kind") == "subagent" and isinstance(h.get("evidence"), list):
            out.extend(e for e in h["evidence"] if isinstance(e, dict))
        out.append(h)
    return out


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
    # Panel round 5 (NGSTORAGE): three cells each carried a different earlier ledger, so every save held only some of the
    # stages and the checklist never saw four. A carried ledger is always the newest one of that name in the mission.
    given = [({**c, "run_id": newest[name][0], "file": newest[name][1]}
              if (name := str(c.get("file", "")).rsplit("/", 1)[-1]) in newest
              and contracts.LEDGER_FILE.fullmatch(f"out/{name}") else c) for c in given]
    have = {str(c.get("file", "")).rsplit("/", 1)[-1] for c in given}
    room = max(0, AUTO_CARRY_MAX - len(given))
    extra = [{"run_id": r, "file": p} for name, (r, p) in sorted(newest.items()) if name not in have][:room]
    if not extra:
        return args
    return {**args, "carry": given + extra}


TABLE_TEXT_TOOLS = ("validate_table", "bind_series")


def _typed_table_refusal(name: str, args: Any, history: list[dict], goal: str) -> str | None:
    """Why a table passed as CSV text is refused: its rows name members, or carry figures, that no fetched data holds.

    Motivation: panel round 10 (EPL). The planner typed a 22-row standings table into validate_table: clubs from memory
    (Leicester, Southampton, Wolves, West Ham) that no downloaded file names, with points no tool computed. A table check that
    accepts a typed table checks the model's memory, not the data. The grounding check caught the figures only at the end.
    Logic: the first cell of each data row must appear in the request or in what the tools returned (case and spacing
    ignored), and the figures in the text must pass the same grounding check as an answer. None when the call is not a
    table tool with csv text, or when the table traces to the evidence. Decided from stored history: a replay refuses alike.
    """
    if name not in TABLE_TEXT_TOOLS or not isinstance(args, dict) or not isinstance(args.get("csv"), str):
        return None
    lines = [ln for ln in args["csv"].splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    seen = grounding._norm_name(grounding.evidence_text(history, goal))
    names = [ln.split(",")[0].strip().strip('"') for ln in lines[1:]]
    unsourced = [n for n in names if n and not re.fullmatch(r"[\d.,%+-]+", n) and grounding._norm_name(n) not in seen]
    figures = unverified(args["csv"], history, goal)
    if not unsourced and not figures:
        return None
    what = (f"rows name {', '.join(unsourced[:5])}, which no fetched data names" if unsourced
            else f"figures {', '.join(figures[:5])} appear in no tool output")
    return (f"the csv table was not checked: {what}. A table comes from fetched data: build it in a sandbox run from the "
            "downloaded file and pass its dataset_id or the run's output, not text typed from memory")


def _drop_unsourced_entities(args: Any, history: list[dict], goal: str) -> tuple[Any, list[str]]:
    """A list of expected members (clubs, constituents) that the request and the fetched data do not name is left out.

    Motivation: panel round 9 (EPL). Before any download, the planner stated a table claim whose expected_entities were
    last season's clubs from memory (Burnley, Leicester City, Southampton, West Ham United). Two independent sources then
    named this season's promoted clubs; the claim, the planner and the evaluator all treated the files as wrong, and the
    mission ended with no forecast. The prompts already said members come from the data; a prompt rule was not enough.
    Logic: when any expected entity appears neither in the request nor in what the tools returned so far (case and spacing
    ignored), the whole list is dropped and expected_rows keeps its count. Decided from the stored history and the goal, so
    a replay builds the same arguments. Returns the arguments and the names that had no source.
    """
    if not isinstance(args, dict) or not isinstance(args.get("expected_entities"), list):
        return args, []
    seen = grounding._norm_name(grounding.evidence_text(history, goal))
    names = [n for n in args["expected_entities"] if isinstance(n, str) and n.strip()]
    unsourced = [n for n in names if grounding._norm_name(n) not in seen]
    if not unsourced:
        return args, []
    out = {k: v for k, v in args.items() if k != "expected_entities"}
    if names and not isinstance(out.get("expected_rows"), int):
        out["expected_rows"] = len(names)
    return out, unsourced


def _drop_blank_keys(args: Any) -> Any:
    """Arguments without blank names.

    Motivation: baseline panel 2026-11-26 (CPI, NEM, AUUNEMP) called list_open_data_sources with {"": {}}, and the tool
    refused it as an unknown argument. A key with no name carries nothing the tool could read.
    Logic: top-level keys that are empty or only whitespace are dropped; everything else is passed as sent. Deterministic,
    so a replay builds the same arguments.
    """
    if not isinstance(args, dict):
        return args
    return {k: v for k, v in args.items() if not (isinstance(k, str) and not k.strip())}


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
    _number_sources(ctx, result, full)
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


_SOURCE_ID = re.compile(r"^S\d+$")


def _number_sources(ctx: Ctx, *results: Any) -> None:
    """Gives every web source in a tool result its mission-wide number, in place: the same address is the same S-number for
    the lead and every lane, in this call and all later ones. A fetched page (web_fetch) gets one too, so it can be cited.

    Motivation: live AURATE (round 20261128-063203). Each web_research call numbered its sources from S1, so a lane's
    "S1" pointed at the Bank of England page while the quote came from the RBA's Statement on Monetary Policy; the claim
    check read the wrong page and the answer's [S1] meant something different in every call.
    Logic: runs inside the act step, so the numbers are stored with the observation and a replay reads the same ones. The
    registry lives in the store (one row per mission and address). Event rows that carry a source_id are renumbered alike.
    """
    store, mid = getattr(ctx, "store", None), getattr(ctx, "mission_id", None)
    if store is None or mid is None or not hasattr(store, "source_numbers"):
        return
    holders: list[tuple[dict, str]] = []
    for res in results:
        if not isinstance(res, dict):
            continue
        for src in res.get("sources") or []:
            if isinstance(src, dict) and isinstance(src.get("url"), str) and src["url"]:
                holders.append((src, src.get("final_url") or src["url"]))
        if "sources" not in res and isinstance(res.get("url"), str) and isinstance(res.get("text") or res.get("passages"),
                                                                                    (str, list)):
            holders.append((res, res.get("final_url") or res["url"]))
    if not holders:
        return
    try:
        numbers = store.source_numbers(mid, [u for _, u in holders])
    except Exception:
        return            # numbering is a convenience for citations; it must never fail the tool call
    remap: dict[str, str] = {}
    for holder, url in holders:
        new = f"S{numbers[url]}"
        old = holder.get("id")
        if isinstance(old, str) and _SOURCE_ID.match(old):
            remap[old] = new
        holder["id"] = new
    for res in results:
        if isinstance(res, dict):
            for ev in res.get("events") or []:
                if isinstance(ev, dict) and ev.get("source_id") in remap:
                    ev["source_id"] = remap[ev["source_id"]]


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


def _exact_run(history: list[dict]) -> list[dict]:
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


# Mission 9 (2026-11-26, live EPL 56465803). About 60 sandbox cells re-read the same carried files and printed the same standings
# table with small differences (a heading, a diagnostic line, a shorter print), so no outcome key matched and the stall rule
# waited until step 63 of 196. Near-duplicates are judged by what a cell reads and writes, the table shapes it prints, and its
# printed rows and words, not by its bytes.
NEAR_SIMILARITY = 0.85     # Jaccard of two cells' row tokens or word shingles, at or above which they are near-duplicates
NEAR_SHINGLE = 3           # words per shingle
_SHAPE = re.compile(r"\((\d+), (\d+)\)")                                        # a pandas-style shape, e.g. (11, 9)
_FILE_NAME = re.compile(r"[\w./-]+\.(?:csv|json|html?|txt|parquet|xlsx)\b")
_FILE_LINE = re.compile(r"^\s*(?:\S+\.(?:csv|json|html?|txt|parquet|xlsx)(?: \(\d+ bytes\))?|Files in\b.*|Available carried files\b.*)\s*$",
                        re.M)                                                    # a file-listing line, not a result
_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z][a-z.'-]*|[^\sa-z\d]")


def _resolved(obs: dict, history: list[dict]) -> dict:
    """The call a pointer (repeat_of) points to, or the call itself. The position is 1-based in the full, unfiltered history."""
    pos = obs.get("repeat_of")
    if isinstance(pos, int) and 1 <= pos <= len(history):
        return history[pos - 1]
    return obs


def _sandbox_files(obs: dict, history: list[dict]) -> frozenset[str]:
    """The files a sandbox cell names in its code, and the files it wrote. A cell that names none is named by nothing."""
    src = _resolved(obs, history)
    args = src.get("args")
    code = str(args.get("code") or "") if isinstance(args, dict) else ""
    res = src.get("full") if isinstance(src.get("full"), dict) else src.get("result")
    written: list[str] = []
    for o in (res.get("outputs") or []) if isinstance(res, dict) else []:
        if isinstance(o, dict) and isinstance(o.get("path"), str):
            written.append(o["path"])
        elif isinstance(o, str):
            written.append(o.split(" (")[0])
    return frozenset(_FILE_NAME.findall(code)) | frozenset(written)


def _sandbox_new_content(obs: dict, history: list[dict], earlier: list[dict], *, new_name_only: bool = False) -> bool:
    """Whether a cell wrote content no earlier cell wrote.

    Motivation: replay of EPL 56465803 (tests/prophet_agent/evals): the re-parse cells wrote the same table under new names
    (pl_standings_2026_27_full.csv, then _clean.csv, both sha256 a9f0ca43), and a new name counted as progress, so the
    loop ran 14 times. Logic: an output is new when its sha256 is new; an output with no sha256 is new when its name is.
    With ``new_name_only`` an output must also have a new name: a rewrite of a file seen before (a running ledger.json)
    is judged by what the cell printed instead.
    """
    def outputs(h: dict) -> list[tuple[str, str]]:
        src = _resolved(h, history)
        res = src.get("full") if isinstance(src.get("full"), dict) else src.get("result")
        out = []
        for o in (res.get("outputs") or []) if isinstance(res, dict) else []:
            if isinstance(o, dict) and isinstance(o.get("path"), str):
                out.append((o["path"], str(o.get("sha256") or "")))
            elif isinstance(o, str):
                out.append((o.split(" (")[0], ""))
        return out

    seen_names, seen_digests = set(), set()
    for h in earlier:
        for name, digest in outputs(h):
            seen_names.add(name)
            if digest:
                seen_digests.add(digest)
    if new_name_only:
        return any(name not in seen_names and (not digest or digest not in seen_digests) for name, digest in outputs(obs))
    return any((digest not in seen_digests) if digest else (name not in seen_names) for name, digest in outputs(obs))


def _no_new_information(profile: dict, earlier: list[dict]) -> bool:
    """A cell whose printed table rows (or, with no rows, its word shingles) were nearly all printed by earlier cells that
    showed the same table shapes. Motivation: the EPL loop printed the same 11-club table between cells that differed in
    other ways, so no two neighbours were near-duplicates. Logic: coverage of the newest cell's rows by the union of earlier
    cells with the same shapes, at least NEAR_SIMILARITY; a cell that prints no table shape is never counted."""
    if not profile["shapes"]:
        return False
    same = [p for p in earlier if p["shapes"] == profile["shapes"]]
    if not same:
        return False
    key = "lines" if profile["lines"] else "shingles"
    mine = profile[key]
    if not mine:
        return False
    union = frozenset().union(*(p[key] for p in same))
    return len(mine & union) / len(mine) >= NEAR_SIMILARITY


_ARTEFACT = re.compile(r"\.(?:png|svg|pdf|docx|xlsx|html?|jpe?g)$", re.I)


def _new_artefact(obs: dict, history: list[dict], earlier: list[dict]) -> bool:
    """A chart or document file no earlier cell wrote: work for the answer, whatever the cell printed."""
    seen = set()
    for h in earlier:
        seen |= _sandbox_files(h, history)
    return any(_ARTEFACT.search(name) and name not in seen for name in _sandbox_files(obs, history))


def _stale_run(history: list[dict]) -> list[dict]:
    """The newest sandbox cells, newest first, that printed no new information compared with every cell before them.

    Motivation: replay of EPL 56465803: six cells in a row printed the same 11-club table; each also wrote a newly named
    assessment JSON, and a rule that compared only neighbours let the loop run on. Logic: a cell that printed only table
    rows already printed under the same shapes is stale. A cell that wrote a new chart or document, wrote a file with a new
    name and new content, failed, or printed new rows ends the run. A cell that printed no table shape is skipped.
    """
    calls = [h for h in history if h.get("kind") == "tool" and not h.get("rejected")]
    sandbox = [h for h in calls if h.get("tool") == "sandbox_python"]
    profiles = [(h, _sandbox_profile(h, history)) for h in sandbox]
    run: list[dict] = []
    for k in range(len(profiles) - 1, 0, -1):
        h, p = profiles[k]
        if p is None or (p["ok"] and not p["shapes"] and not _sandbox_new_content(h, history, [x for x, _ in profiles[:k]])):
            continue
        before = [h2 for h2, _ in profiles[:k]]
        earlier = [p2 for _, p2 in profiles[:k] if p2 is not None]
        if (not p["ok"] or _new_artefact(h, history, before) or _sandbox_new_content(h, history, before, new_name_only=True)
                or not _no_new_information(p, earlier)):
            break
        run.append(h)
    return run


def _row_lines(text: str) -> set[str]:
    """The table rows a cell printed, each as one normalised line (tokens joined), so a changed number is a new row."""
    out: set[str] = set()
    for ln in text.splitlines():
        toks = _TOKEN.findall(ln.lower())
        if sum(t[0].isdigit() for t in toks) >= 3 and any(t[0].isalpha() for t in toks):
            out.add(" ".join(toks))
    return out


def _row_tokens(text: str) -> set[str]:
    """The tokens of the lines that carry numbers and a word (table rows): numbers kept, layout and spacing dropped."""
    out: set[str] = set()
    for ln in text.splitlines():
        toks = _TOKEN.findall(ln.lower())
        if sum(t[0].isdigit() for t in toks) >= 3 and any(t[0].isalpha() for t in toks):
            out.update(toks)
    return out


def _sandbox_profile(obs: dict, history: list[dict]) -> dict | None:
    """What a sandbox cell produced, for near-duplicate checks, or None when it produced nothing to compare.

    Motivation: a repeat that changes a few words of the same table is still a repeat, and a cell that writes a new file or
    prints a new table shape is work. Run ids and file listings say nothing about the result, so they are removed first.
    Logic: the profile holds ok, the files (_sandbox_files), the table shapes printed, the row tokens and the word shingles of the
    normalised stdout and stderr. A pointer is resolved to the cell it points to, so a repeat keeps its profile.
    """
    src = _resolved(obs, history)
    if src.get("tool") != "sandbox_python" or src.get("error") is not None:
        return None
    res = src.get("full") if isinstance(src.get("full"), dict) else src.get("result")
    if not isinstance(res, dict):
        return None
    stdout = str(res.get("stdout") or "")
    stderr = str(res.get("stderr") or res.get("stderr_tail") or "")
    if not (stdout.strip() or stderr.strip() or _sandbox_files(src, history)):
        return None
    text = _FILE_LINE.sub("", RUN_ID_ANY.sub("<run>", stdout + "\n" + stderr))
    tokens = _TOKEN.findall(text.lower())
    return {"ok": res.get("ok") is True, "files": _sandbox_files(src, history),
            "shapes": tuple(sorted(set(_SHAPE.findall(stdout)))), "rows": frozenset(_row_tokens(text)),
            "lines": frozenset(_row_lines(text)),
            "shingles": frozenset(tuple(tokens[i:i + NEAR_SHINGLE]) for i in range(max(0, len(tokens) - NEAR_SHINGLE + 1)))}


def _jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def _near_duplicate(a: dict, b: dict) -> bool:
    """Two sandbox profiles are near-duplicates when both ran the same way, print the same table shapes, and their rows or
    shingles overlap at least NEAR_SIMILARITY. The files are compared by the caller: a new file is new work."""
    if a["ok"] != b["ok"] or a["shapes"] != b["shapes"]:
        return False
    rows = _jaccard(a["rows"], b["rows"]) if a["rows"] and b["rows"] else 0.0
    return max(rows, _jaccard(a["shingles"], b["shingles"])) >= NEAR_SIMILARITY


def _near_run(history: list[dict]) -> list[dict]:
    """The newest sandbox cells that are near-duplicates of the newest one, newest first. Tool calls of other kinds and cells that
    produced nothing are skipped, not counted, so a re-parse with a fetch between its cells is still one loop; the first cell that
    is not near ends the run. A newest cell that names or writes a file no earlier cell named is progress, so it has no run."""
    calls = [h for h in history if h.get("kind") == "tool" and not h.get("rejected")]
    sandbox = [h for h in calls if h.get("tool") == "sandbox_python"]
    if not sandbox:
        return []
    newest = _sandbox_profile(sandbox[-1], history)
    if newest is None:
        return []
    earlier_names: set[str] = set()
    for h in sandbox[:-1]:
        earlier_names |= _sandbox_files(h, history)
    newest_obs = _resolved(sandbox[-1], history)
    code = str((newest_obs.get("args") or {}).get("code") or "") if isinstance(newest_obs.get("args"), dict) else ""
    # A file the code reads for the first time is new input; an output is new work unless it repeats content already written
    # under another name (EPL 56465803 replay: _full.csv and _clean.csv were the same table).
    if (frozenset(_FILE_NAME.findall(code)) - earlier_names
            or _sandbox_new_content(sandbox[-1], history, sandbox[:-1], new_name_only=True)):
        return []
    run = [sandbox[-1]]
    for h in reversed(sandbox[:-1]):
        p = _sandbox_profile(h, history)
        if p is None:
            continue
        if not _near_duplicate(newest, p):
            break
        run.append(h)
    return run


def _run_of_same_result(history: list[dict]) -> list[dict]:
    """The run the stall rule and the notice use: the longer of the exact run (same progress key) and the near-duplicate run."""
    exact = _exact_run(history)
    near = _near_run(history)
    stale = _stale_run(history)
    return max((near, stale, exact), key=len)


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
    same = "the same" if all(_progress_key(h) == _progress_key(run[-1]) for h in run) else "nearly the same"
    files = set.intersection(*(set(_sandbox_files(h, history)) for h in run)) if run[0].get("tool") == "sandbox_python" else set()
    named = (" Each call reads or writes " + ", ".join(sorted(files)[:6]) + " and prints the same table shape.") if files else ""
    return ("Stall notice: the last " + str(len(run)) + " calls gave " + same + " result as " + _stall_label(run) + "." + named +
            " That result already exists: do not run this step again, and do not re-parse it with another cell. " + todo +
            " If the same result comes back again, the run is stopped with what it has found.")


def _stalled(history: list[dict], window: int) -> bool:
    """Stalled when the newest ``window`` tool calls are the same call with the same outcome (run ids aside), or when the newest
    ERROR_STALL calls failed the same way even with different code: the planner is not converging. Two calls taking turns
    (A, B, A with a window of 3; A, B, A, B with 4) are a loop too when each returns the same outcome it returned before:
    a call that changed nothing it reads back is not progress. The outcome is compared by _progress_key, so the same output
    from a different sandbox cell is no progress either.
    A refused argument list ("rejected") is not a call: it is not charged, and it is not counted here either (GOAL.md
    defect 5). Before, the same invalid call sent again reached ERROR_STALL and stopped the mission as stalled."""
    full, history = history, [h for h in history if not h.get("rejected")]
    if _error_run(history) >= ERROR_STALL:
        return True
    # Mission 9: `window` sandbox cells that are near-duplicates of the newest one (same files, shapes and rows) are the same loop.
    # The pointers are resolved against the full history, so the positions they carry still point at the right calls.
    if len(_near_run(full)) >= window or len(_stale_run(full)) >= window:
        return True
    tool_obs = [h for h in history if h.get("kind") == "tool"]
    if len(tool_obs) < window:
        return False
    last = [_progress_key(h) for h in tool_obs[-window:]]
    if len(set(last)) == 1:
        return True
    return window >= 3 and len(set(last)) == 2 and all(last[k] == last[k - 2] for k in range(2, len(last)))


def _may_block_again(history: list[dict], blocked: list[dict]) -> bool:
    """Another repair round for a blocked answer while the failing checks keep changing, at most MAX_CONTRACT_BLOCKS."""
    earlier = [tuple(sorted(h["contract"])) for h in history if h.get("kind") == "evaluation" and h.get("contract")]
    if len(earlier) >= MAX_CONTRACT_BLOCKS:
        return False
    now = tuple(sorted(c["name"] for c in blocked))
    recent = earlier[-(SAME_BLOCKS_TO_STOP):]
    return not (len(recent) == SAME_BLOCKS_TO_STOP and all(r == now for r in recent))


def _pivot(history: list[dict], why: str, repeated: str = "") -> bool:
    """Turn a stop into a change of strategy while pivots remain. Returns True when a pivot was added."""
    if sum(1 for h in history if h.get("kind") == "pivot") >= MAX_PIVOTS:
        return False
    history.append({"kind": "pivot", "why": why, **({"repeated": repeated} if repeated else {}),
                    "note": "This path is exhausted; do not repeat it. Change strategy: another source or method for the "
                            "open item, a fan-out of research lanes on the open question (action lanes, each lane a different "
                            "angle), or a question to the person if a decision blocks you. Then finish when the checks pass."})
    return True


def _specs(plan: dict) -> list[dict]:
    """The delegations in a plan: ``tasks`` (parallel) or the single ``task`` with its optional fields."""
    keys = ("task", "agent_type", "deliverable", "constraints", "tools", "resume", "name")
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
    parts.append("Put each claim you rely on on its own line, in this form: CLAIM | value=<the value as the page prints it> | "
                 "unit=<unit> | as_of=<date the source gives> | entity=<what it is about> | source_id=<the S-id of the source> | "
                 "quote=\"<the words of the page around the value, copied exactly>\". Add locator=<a label next to the quote> only "
                 "when the value appears more than once on that page. A claim without a quote from a page a tool read is not checked.")
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


def _agent_started(ctx: Ctx, agent_id: str, agent_type: str, spec: dict, lane: bool) -> None:
    """One ``agent.started`` event per sub-agent, so the console can show a running lane's brief before its report exists.
    Written once: a resumed run that replays the agent finds the event and writes nothing. Never raises."""
    try:
        ctx.store.append_event_once(ctx.mission_id, "agent.started", {
            "agent_id": agent_id, "agent_type": agent_type, "task": str(spec.get("task") or "")[:2000],
            "deliverable": str(spec.get("deliverable") or "")[:600], "lane": lane, "resumed": bool(spec.get("resume")),
            "name": str(spec.get("name") or "")[:60]},
            key="agent_id")
    except Exception:
        pass


def _one_agent(ctx: Ctx, brain: Any, spec: dict, number: int, tools: dict[str, Tool], cfg: GoalConfig, context: str,
               history: list[dict], lane: dict | None = None) -> dict:
    """Runs one sub-agent with its own history, memo, notes and a tool list scoped by its type. ``lane`` (from
    runtime/lanes.py) makes it a research lane: it gets the board, shares identical web calls with the other lanes, reads its
    board messages before each step and cannot ask the person."""
    agent = agents.get(spec.get("agent_type")) or agents.TYPES["general"]
    agent_id = f"a{number}"
    sub_tools = agents.allowed(agent, tools)
    # The brief's tool list is the lead's decision and is read from the mission's tools, not only the type's: live AURATE
    # (round 20261128-063203) sent twenty analyst lanes to download and test drivers, and each reported "download_open_data
    # is not available" because the analyst type has no download tools.
    asked = [n for n in (spec.get("tools") or []) if isinstance(n, str) and n in tools and n not in ("notes", "board")]
    if asked:
        sub_tools = {n: tools[n] for n in asked}
    sub_tools.pop("notes", None)
    sub_tools["notes"] = _notes_for(ctx, agent_id)
    if lane:
        sub_tools = lanes.shared_tools(ctx, sub_tools)
        sub_tools["board"] = lanes.board_tool(ctx, agent_id, lane["roster"])
    prior = None
    if spec.get("resume"):
        prior = next((h for h in reversed(history) if h.get("kind") == "subagent" and h.get("agent_id") == spec["resume"]), None)
        if prior is None:
            return {"kind": "delegate", "error": f"no earlier sub-agent {spec['resume']!r} to resume; known: "
                    + (", ".join(sorted({h['agent_id'] for h in history if h.get('kind') == 'subagent' and h.get('agent_id')})) or "none")}
    # The step budget is the type's own (AgentType.steps). It was declared but never read (audit item 24), so every sub-agent
    # got cfg.sub_steps whatever its type said.
    sub_cfg = GoalConfig(max_iterations=agent.steps, stall_window=cfg.stall_window, depth=1, sub_steps=cfg.sub_steps,
                         tag=f"d1.{agent_id}", lane=agent_id if lane else "")
    # Live (Beijing 3c379255, 2026-11-23): four analysts started with an empty history and each ran the lead's inspect_data
    # again, then read the same pages, and the step stalled. They start with the lead's tool results so far, so an identical
    # call is answered with the earlier result, as it is for the lead.
    sub_history: list[dict] = list(prior["evidence"]) if prior else seed_history(history)
    sub_tools = _sub_observation_tools(sub_tools, sub_history)
    goal = _brief(spec, agent, agent_id, prior is not None)
    if lane:
        goal += "\n\n" + lane["brief"]
    label = names.for_task(spec.get("task"), spec.get("name"), fallback=f"{agent.name.capitalize()} {number}")
    _agent_started(ctx, agent_id, agent.name, {**spec, "name": label}, bool(lane))
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
    obs = {"kind": "subagent", "agent_id": agent_id, "name": label, "agent_type": agent.name, "task": task,
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
    # Mission 9 (task 1): each claim the report lists (value, source and quote) is checked against the page text the research cache
    # holds for its source. The check is a ledger step, so a resumed run reads the stored statuses and never reads the cache again.
    if claim_checks.parse_claims(answer):
        checked = ctx.step(f"claims:{agent_id}", lambda _c: claim_checks.check_report_claims(answer, sub_history))
        obs["claims"], obs["claim_summary"] = checked["claims"], checked["summary"]
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
    if not found:
        # Round 21 (GOLD): "over the next year" has no digit, so no horizon was read and a forecast that ended in 2018 passed.
        # One unit is read only after over, for, during or within, so "rates next year" (a calendar year) is not taken.
        for m in ONE_UNIT_REQUEST.finditer(goal or ""):
            found.add((1, _horizon_unit(m.group(1))))
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
        return _calendar_step(float(step)) if 0 < step < 1e6 else None
    if not isinstance(step, str):
        return None
    text = step.strip().lower()
    if text in SERIES_STEPS:
        return SERIES_STEPS[text], text in CALENDAR_STEPS
    m = STEP_TEXT.fullmatch(text)
    if m and 0 < float(m.group(1)) < 1e6:
        return _calendar_step(float(m.group(1)))
    return None


def _calendar_step(days: float) -> tuple[float, bool]:
    """A typical step in days, read as a calendar month, quarter or year when it is one. Round 21 (AURATE, 2026-11-27): month-end
    dates gave a typical step of 31 days, so five years read as 58.9 periods against the 60 the forecast ran, the finish was
    blocked again and again, and the run stalled."""
    for low, high, name in ((28, 31, "monthly"), (89, 92, "quarterly"), (365, 366, "yearly")):
        if low <= days <= high:
            return SERIES_STEPS[name], True
    return days, False


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


def _dated_span_check(asked: str, count: int, unit: str, rec: dict, history: list[dict]) -> dict | None:
    """The horizon check by calendar dates, for a request in days, weeks, months, quarters or years when the forecast's window
    names its first and last dates. None when the request is in periods or trading days, or the window has no dates.

    Motivation: panel round 4 (NVDA, 2026-11-27), "the next 2 months" on a trading-day series ran 40 steps from 27 November
    2026 to 26 January 2027, two calendar months, but the period count read 2 x 30.4 = 60.9 daily periods and blocked the
    finish; the evaluator repeated the wrong figure and the run stalled. Logic: the span runs from the data's last date (or
    the day before the first forecast date) to the last forecast date, and must match the requested days within 5 % or 4
    days, whichever is larger, which covers weekends and holidays at either end.
    """
    import datetime as dt
    # Round 6 (CO2): "the next 30 days" on a daily series kept on a business-day index ran 30 steps over 42 calendar days
    # and was blocked, and the loop wandered between 22 and 26 steps. A request in days counts daily values, as before.
    if unit in ("period", "trading_day", "day"):
        return None
    window = rec.get("window") if isinstance(rec.get("window"), dict) else None
    if window is None:
        return None
    try:
        first = dt.date.fromisoformat(str(window.get("first"))[:10])
        last = dt.date.fromisoformat(str(window.get("last"))[:10])
    except ValueError:
        return None
    # Only a series stepped in days or weeks. A monthly point is dated at its period's start, so six months from November
    # end on 1 April, 152 days on (panel round 5, CPI: the check asked for 7, then 8 months). The period count is exact there.
    step = _series_step(rec)
    steps = window.get("steps")
    gap = ((last - first).days / (steps - 1)) if isinstance(steps, int) and steps > 1 else None
    # Weekly too (round 6, NGSTORAGE: 8 weekly steps were read as 50 days against 56 and blocked). Only a series stepped in
    # single days, where trading days and calendar days part, is checked by its dates.
    if (step is not None and (step[1] or step[0] >= 2)) or (gap is not None and gap >= 2):
        return None
    end = _data_facts(history).get("data_end")
    try:
        start = dt.date.fromisoformat(str(end)[:10]) if end else first - dt.timedelta(days=1)
    except ValueError:
        start = first - dt.timedelta(days=1)
    if start >= first:
        start = first - dt.timedelta(days=1)
    wanted = count * UNIT_DAYS[unit]
    span = (last - start).days
    ok = abs(span - wanted) <= max(4.0, 0.05 * wanted)
    detail = (f"{asked}, about {wanted:.0f} calendar days; the forecast runs from {first.isoformat()} to {last.isoformat()}, "
              f"{span} days after the data's last date")
    if not ok:
        detail += ", so the plan must forecast the requested span"
    return contracts._check("horizon_matches_request", ok, detail, "block")


def horizon_checks(goal: str, history: list[dict], today: str | None = None) -> list[dict]:
    """The check that the forecast that stands runs over the horizon the goal asks for. Empty when the goal states no horizon
    or no forecast has run: the check is left out, not reported as passing.

    Motivation vs Logic
    Motivation: before this, a plan of 14 steps for a request of 20 trading days could finish as "met", because no check put
    the plan's horizon beside the requested one. The mismatch has to be visible and has to be answered.
    Logic: a request in periods must equal the forecast's horizon. A calendar request is converted with the series step, and a
    daily series reads one period per trading day. A mismatch is a block: the finish is refused and the planner must forecast
    the requested span. A calendar request with no recorded step cannot be converted; that is a warning the answer carries,
    not a block, so every calendar forecast is not refused for a field the record does not yet hold.
    "The next N units" runs from today (the run's stored date), not from the data's last date. Round 21 (VNUNEMP and AURATE,
    2026-11-27): series ending in 2024 were forecast for 2 and 5 years from their last point, so most of each forecast lay in
    the past and the check passed. When the data ends more than one step before today, the periods between them are added to
    the request.
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
    checks_before: list[dict] = []
    label = HORIZON_LABELS[unit] if count != 1 else HORIZON_LABELS[unit].rstrip("s")
    asked = f"the request asks for {count} {label}"
    if (dated := _dated_span_check(asked, count, unit, rec, history)) is not None:
        return [dated]
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
        if (stale := _stale_data_check(history, today, step_days, rec)) is not None:
            checks_before.append(stale)
        if (gap := _periods_before_today(history, today, step_days, rec)) is not None:
            expected += gap[0]
            asked += (f" from today ({today}); the data ends {gap[1]}, {gap[0]} periods earlier, so the forecast must also "
                      "cover those periods")
    ok = abs(expected - horizon) <= slack
    detail = f"{asked}, which is {expected:.1f} periods at the series step; the forecast runs {horizon} periods"
    if not ok:
        detail += ", so the plan must forecast the requested span"
    return checks_before + [contracts._check("horizon_matches_request", ok, detail, "block")]


def _series_end(history: list[dict], rec: dict | None, step_days: float) -> str | None:
    """The last observed date of the series the standing forecast was made from: the record's own last_observed date, else
    one step before its first forecast date, else the newest time column the files were inspected with."""
    import datetime as dt
    rec = rec if isinstance(rec, dict) else {}
    last = rec.get("last_observed")
    if isinstance(last, dict) and re.match(r"\d{4}-\d{2}-\d{2}", str(last.get("date") or "")):
        return str(last["date"])[:10]
    first = (rec.get("window") or {}).get("first") if isinstance(rec.get("window"), dict) else None
    first = first or (rec.get("future_dates") or [None])[0]
    if first and re.match(r"\d{4}-\d{2}-\d{2}", str(first)):
        try:
            return (dt.date.fromisoformat(str(first)[:10]) - dt.timedelta(days=round(step_days))).isoformat()
        except ValueError:
            pass
    return _data_facts(history).get("data_end")


def _stale_data_check(history: list[dict], today: str | None, step_days: float, rec: dict | None = None) -> dict | None:
    """A block when the series the forecast stands on ends long before today, None when it is current or a date is missing.

    Motivation: round 21 (2026-11-27), GOLD forecast October 2017 to September 2018 from a stopped series and was judged
    met; VNUNEMP and AURATE forecast from 2024 data while their own research had 2025 and 2026 figures. A forecast "from
    now" needs data that reaches close to now.
    Logic: the allowed age is how long a fresh release normally takes at the series step (150 days for monthly, 270 for
    quarterly, 640 for yearly, 30 for shorter steps). Older data blocks the finish with what to do instead."""
    import datetime as dt
    end = _series_end(history, rec, step_days)
    if not today or not end:
        return None
    try:
        days = (dt.date.fromisoformat(str(today)[:10]) - dt.date.fromisoformat(str(end)[:10])).days
    except ValueError:
        return None
    kind = "yearly" if step_days >= 360 else "quarterly" if step_days >= 85 else "monthly" if step_days >= 27 else "short"
    if days <= CURRENT_DATA_DAYS[kind]:
        return None
    return contracts._check(
        "data_is_current", False,
        f"the data ends {str(end)[:10]}, {days} days before today ({str(today)[:10]}); a {kind} series is normally published "
        f"within {CURRENT_DATA_DAYS[kind]} days. Fetch the series from a source that has the recent figures (the provider's "
        "own service, FRED, or the figures research found, added to the series with their sources), forecast from there, "
        "and if no newer figure exists, say in the answer how old the data is", "block")


def _periods_before_today(history: list[dict], today: str | None, step_days: float,
                          rec: dict | None = None) -> tuple[int, str] | None:
    """Whole periods between the data's last date and today, with that date, when the data ends more than one and a half
    steps before today (a monthly figure for last month is on time). None otherwise, or when a date is missing."""
    import datetime as dt
    end = _series_end(history, rec, step_days)
    if not today or not end or step_days < 1:
        return None
    try:
        last = dt.date.fromisoformat(str(end)[:10])
        now = dt.date.fromisoformat(str(today)[:10])
    except ValueError:
        return None
    days = (now - last).days
    if days <= 1.5 * step_days:
        return None
    return int(days // step_days), last.isoformat()


def horizon_note(goal: str, history: list[dict], today: str | None = None) -> str | None:
    """The failing horizon checks as one sentence for the person, or None when every check holds or none applies."""
    failing = [c for c in horizon_checks(goal, history, today) if not c["ok"]]
    return ("Horizon check (computed): " + contracts.summarise(failing)) if failing else None


def _dated(context: str, today: str | None) -> str:
    """The context with the run's date at its head, so the planner, the evaluator and the composer all read it.

    Motivation vs Logic
    Motivation: live Nvidia mission (2026-11-26), the planner used data_end 2026-07-24 because nothing told it the date. A data end
    that is guessed is a wrong forecast window.
    Logic: the date is the one stored in the contracts step (the ledger), so a resumed run reads the same line. A run without a
    stored date (a mission started before this change) keeps its context unchanged.
    """
    if not today:
        return context
    line = (f"The date today is {today}. Any data end, latest date or current figure is on or before this day; do not "
            "assume a date later than it.")
    return line + ("\n\n" + context if context else "")


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
            # Mission 9 (M9-7): the date is read once and stored here, so a resumed run sees the same date the first run did.
            recorded["today"] = time.strftime("%Y-%m-%d")
            return recorded
        stored = ctx.step("contracts", record)
        context = _dated(context, stored.get("today") if isinstance(stored, dict) else None)
        run_today = stored.get("today") if isinstance(stored, dict) else None
    else:
        run_today = None
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
                                       external=bool(EXTERNAL_GOAL.search(goal)), asks=bool(ASKED_TO_ASK.search(goal)))
            note += "\n\n" + contracts.render_progress(items)
            if (asked := _asked_block(goal, "planner")):
                note += "\n\n" + asked
            if (horizon_line := horizon_note(goal, history, run_today)):
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
            detail = nxt_item.get("detail") or "" if nxt_item else ""
            if rescue and (last := next((h for h in reversed(history) if h.get("kind") == "stall_rescue"), None)):
                # EPL 56465803 replay: the rescue analyst re-parsed the same 11-club table three more times. It is told
                # what the lead repeated, so it takes another route instead.
                detail = (detail + " " if detail else "") + (
                    f"The lead stalled repeating {last.get('repeated', 'one step')}. That result already exists: do not "
                    "re-read or re-parse it. Take a different route (another source, find_in_page on a page already "
                    "fetched, or a complete table), or report what is missing.")
            stuck = (nxt, STUCK_CALLS if rescue else calls_now - calls_at_best, detail) if nxt else None
            rescue = False
            _status(ctx, i, history, items, started)
        if (stall_note := _stall_notice(history, stuck[0] if stuck else None)):
            note += "\n\n" + stall_note          # round 7: one repeat of the same result, so the planner is told it already exists
        if cfg.lane:
            # A research lane reads the board before it plans. The read is a ledger step, so what the lane saw is stored and a
            # replay sees the same messages whatever the other lanes have posted since (runtime/lanes.py).
            seen = max((int(m.get("seq", 0)) for h in history if h.get("kind") == "board"
                        for m in h.get("messages") or []), default=0)
            got = ctx.step(f"inbox:{tag}:{i}", lambda c, _seen=seen: c.store.board_read(
                c.mission_id, cfg.lane, after=_seen, limit=lanes.INBOX_LIMIT))
            if got:
                history.append({"kind": "board", "messages": got,
                                "note": "new messages for you on the research board; answer a question sent to you"})
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
            plan_tools, focus = _focus_tools(history, tools) if cfg.depth == 0 else (tools, "")
            if focus:
                note += "\n\n" + focus
            plan = ctx.step(f"plan:{tag}:{i}", lambda c: {**_decide_plan(brain, goal, plan_tools, history, memo, context, note,
                                                                        finish_now=cfg.finish_now, out_of_calls=out_of_calls,
                                                                        nested=cfg.depth > 0),
                                                          "out_of_calls": out_of_calls})
        spent = bool(plan.get("out_of_calls", out_of_calls))
        plan = as_question_action(plan)
        if plan["action"] == "finish":
            obs = {"kind": "finish", "rationale": plan["rationale"]}
        elif plan["action"] == "ask_user" and cfg.lane:
            obs = {"kind": "ask_user", "error": "a research lane cannot ask the person. Post the question to the lead on the "
                   "board (board action=post to=lead kind=question) and carry on with what you can establish."}
        elif plan["action"] == "ask_user":
            # Mission 9 (M9-7): one question object, or more than four, is repaired in place; the notes ride on the observation.
            raw_qs, repair_notes = repair_questions(plan.get("questions", []))
            items = raw_qs if isinstance(raw_qs, list) else []
            open_dicts, repeated = _already_asked([q for q in items if isinstance(q, dict)], history)
            open_qs = open_dicts + [q for q in items if not isinstance(q, dict)]    # a non-object is left for the validator
            payload = {"questions": open_qs}
            # Mission 9 (task 2): an option that offers invented data in place of the real series is refused here, with the reason.
            problems = (validate_for_goal(payload, goal) + _option_figure_problems(open_qs, history, goal)
                        if open_qs or not repeated else ["every question was already asked and answered"])
            if problems and repeated:
                obs = {"kind": "ask_user", "error": "not asked again, the person already answered: " + "; ".join(repeated)
                       + ". The decision is in the decisions list; act on it."}
            elif problems:
                obs = {"kind": "ask_user", "error": "invalid questions: " + "; ".join(problems) + ". " + expected_shape()
                       + ("" if not repair_notes else " Repairs made: " + "; ".join(repair_notes) + ".")}
            elif not HIGH_RISK.search(str(plan.get("rationale", ""))) and not ASKED_TO_ASK.search(goal or ""):
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
                if repair_notes:
                    obs["repair"] = repair_notes
        elif plan["action"] == "lanes":
            ran = lanes.run_lanes(ctx, brain, plan, tools, cfg, context, history, _one_agent, BRANCH_BLOCK,
                                  workers=cfg.lane_workers)
            history.extend(ran[:-1])
            obs = ran[-1]
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
            args = _repair_source(_drop_blank_keys(plan.get("args", {})), history)
            if isinstance(args, dict) and "rationale" in args:
                # No tool takes a rationale; it is the plan's own field (panel round 4, NVDA: "unknown arguments ['rationale']"
                # five times on sandbox_python). Derived from the stored plan, so a replay drops it alike.
                args = {k: v for k, v in args.items() if k != "rationale"}
            if name == "sandbox_python":
                args = _auto_carry(args, history)
            args, unsourced = _drop_unsourced_entities(args, history, goal)
            stand_in = (stand_in_reason(str(args.get("code") or ""), goal)
                        if name == "sandbox_python" and isinstance(args, dict) else None) or \
                _typed_table_refusal(name, args, history, goal)
            if stand_in:
                obs = {"kind": "tool", "tool": name, "args": args, "error": stand_in}     # derived from args and goal: replays alike
            elif name not in tools:
                obs = {"kind": "tool", "tool": name, "args": args, "error": f"unknown tool {name!r}"}
            else:
                if (pos := _repeat_of(name, args, history)) is not None:
                    obs = _repeat_obs(name, args, history, pos)     # not sent again: the earlier result stands
                elif _ledger_skipped_act(ctx, f"act:{tag}:{i}"):
                    # Follows the ledger: a stored run that did not call the tool here (it was not offered then) is replayed
                    # the same way, so a tool set that grew between parts cannot shift the step indices.
                    obs = {"kind": "tool", "tool": name, "args": args, "error": f"unknown tool {name!r}"}
                else:
                    obs = ctx.step(f"act:{tag}:{i}", lambda c: _act(c, tools[name], name, args, tools))
        if obs.get("kind") == "tool" and obs.get("args") is args and unsourced:
            obs["entities_note"] = (f"expected_entities was left out: {', '.join(unsourced[:6])} appear in no fetched data and "
                                    "not in the request, so they came from memory, which is older than today. Read the "
                                    "members from the fetched table, then state them")
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
                                "checklist item (" + stuck[0] + ") goes to a fresh analyst. Build on its report.",
                                "repeated": _stall_label(_run_of_same_result(history) or [obs])})
                rescue = True
                continue
            if (cfg.depth == 0 and not cfg.finish_now and not out_of_calls and i < cfg.end - 1
                    and _pivot(history, "stall", _stall_label(_run_of_same_result(history) or [obs]))):
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
            failures: list[str] = []
            if text.strip() and verdict["verdict"] != "impossible" and cfg.depth == 0:
                # Audit M10 (2026-11-27): a stalled run returned its answer with no answer checks, and the VN notebook's third
                # turn sent a forecast of an interpolated series as "Vietnam's unemployment rate". The stall answer meets the
                # same claim checks and figure checks as a finished one, with one rewrite each.
                claims = contracts.blocking(contracts.check_answer(text, history, run_root=SANDBOX_ROOT, goal=goal))
                if claims:
                    fixed = _compose(ctx, brain, f"claimsstall:{tag}:{i}", goal, history, memo, context, text,
                                     fixes=contracts.summarise(claims))
                    if fixed is not None and fixed.strip():
                        text = contracts.finishing_sentences(fixed, history)
                        claims = contracts.blocking(contracts.check_answer(text, history, run_root=SANDBOX_ROOT, goal=goal))
                bad = unverified(text, history, goal)
                if bad:
                    repaired = _compose(ctx, brain, f"repairstall:{tag}:{i}", goal, history, memo, context, text, bad)
                    if repaired is not None and repaired.strip():
                        text = repaired
                        claims = contracts.blocking(contracts.check_answer(text, history, run_root=SANDBOX_ROOT, goal=goal))
                failures = [c["name"] for c in claims]
                if failures:
                    reason += " It still states something the analysis output does not support: " + contracts.summarise(claims)
            if text.strip() and verdict["verdict"] != "impossible":
                out = {"status": "partial", "iterations": i + 1, "answer": text, "evidence": history, "reason": reason}
                if failures:
                    out["contract_failures"] = failures
                return out
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
                                       external=bool(EXTERNAL_GOAL.search(goal)), asks=bool(ASKED_TO_ASK.search(goal)))
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
        qs = _usable_questions(verdict, goal) if verdict["verdict"] != "met" else None
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
            horizon = horizon_checks(goal, history, run_today) if cfg.depth == 0 else []
            if cfg.depth == 0:                       # the lead's prediction is drawn, and its centre line is not flat
                horizon = horizon + prediction_quality.prediction_checks(history, goal, answer)
            blocked = contracts.blocking(contracts.check_evidence(history, run_root=SANDBOX_ROOT) + horizon)
            if (blocked and out_of_calls and not repair_granted and cfg.depth == 0 and cfg.tool_budget is not None
                    and _may_block_again(history, blocked) and not last and not cfg.finish_now):
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
            if blocked and _may_block_again(history, blocked) and not last and not out_of_calls and not cfg.finish_now:
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
            answer = contracts.finishing_sentences(answer, history)    # follows from the stored flat_expected verdict
            # The answer's plain claims (a cycle, the best rule, a total the pack withheld) are checked against the stored
            # output. The fix is in the words, so the composer rewrites once; a claim still wrong after that leaves partial.
            # Round 4: these checks were built (C3a) and tested but never called here, so none ran live.
            claims = contracts.blocking(contracts.check_answer(answer, history, run_root=SANDBOX_ROOT, goal=goal)) if cfg.depth == 0 else []
            # Round 10 (AUUNEMP): the one rewrite moved "no level shifts" to another sentence and the run ended partial. A
            # second rewrite is told the claim it kept; the step names differ, so a replay reads each stored rewrite.
            fixed_before: list[str] = []
            for attempt in range(CLAIM_REWRITES if claims else 0):
                # Round 14 (GISTEMP): the first rewrite corrected the shift dates, a second claim then blocked it, and the
                # second rewrite brought the wrong dates back. Each rewrite is also told what earlier ones fixed.
                fixes = contracts.summarise(claims)
                if fixed_before:
                    fixes += " Keep these earlier corrections: " + " ".join(fixed_before)
                rewritten = _compose(ctx, brain, f"claims:{tag}:{i}" + (f":{attempt}" if attempt else ""), goal, history, memo,
                                     context, answer, fixes=fixes)
                fixed_before.append(contracts.summarise(claims))
                if rewritten is None or not rewritten.strip():
                    break
                # Round 12 (AUUNEMP): the rewrite dropped the level sentence added above, and the run ended partial on it.
                answer = contracts.finishing_sentences(rewritten, history)
                claims = contracts.blocking(contracts.check_answer(answer, history, run_root=SANDBOX_ROOT, goal=goal))
                if not claims:
                    break
            if claims and {c["name"] for c in claims} & MECHANICAL_FIXES:
                # Round 13 (AUUNEMP): both rewrites kept "kept as genuine regime changes". The fix the check asks for is to
                # take the cause out, so it is done here rather than ending the run on that phrase. Round 20261128-151151
                # (SPX, AUUNEMP): a moving line called flat survived both rewrites; its correct sentence is stored data.
                answer = _mechanical_fixes(answer, history, claims)
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
            if (rejected_finishes == PIVOT_AFTER_REJECTIONS and cfg.depth == 0 and not cfg.finish_now
                    and _pivot(history, "rejected finishes: " + str(verdict.get("reason") or "")[:200])):
                continue
            if rejected_finishes >= MAX_REJECTED_FINISHES:
                # The planner keeps saying it is done and the evaluator keeps saying it is not: more passes will not settle
                # it. Write up what is established, say what is not, and label it partial (seen live: 90 rejected finishes).
                draft = _gap_draft(verdict)
                written = _compose(ctx, brain, f"composeloop:{tag}:{i}", goal, history, memo, context, draft)
                text = written if written is not None and written.strip() else verdict.get("answer", "")
                if text.strip():
                    out = {"status": "partial", "iterations": i + 1, "answer": text, "evidence": history,
                           "reason": "The analysis could not close this gap: " + verdict["reason"]}
                    if cfg.depth == 0:
                        text, wrong = _checked_gap_answer(ctx, brain, tag, i, goal, history, memo, context, text)
                        out["answer"] = text
                        if wrong:
                            out["reason"] += " The answer still states something the analysis output does not support: " \
                                + contracts.summarise(wrong)
                            out["contract_failures"] = [c["name"] for c in wrong]
                    return out
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


ANSWER_SCOPED = frozenset({"answer_table_matches_data", "data_end_stated", "answer_band_matches_pack",
                           "league_claims_match_fit", "market_stated"})


def _gap_answer_claims(text: str, history: list[dict], goal: str) -> list[dict]:
    """The blocking checks on an answer's own words: the stated claims (latest value, data span, cycles...) and the answer
    checks among the prediction checks (table cells, data end, band, league claims, market)."""
    claims = contracts.blocking(contracts.check_answer(text, history, run_root=SANDBOX_ROOT, goal=goal))
    pq = [c for c in prediction_quality.prediction_checks(history, goal, text) if c.get("name") in ANSWER_SCOPED]
    return claims + contracts.blocking(pq)


def _checked_gap_answer(ctx: Ctx, brain: Any, tag: str, i: int, goal: str, history: list[dict], memo: Memo, context: str,
                        text: str) -> tuple[str, list[dict]]:
    """A partial answer written after too many refused finishes, checked like a finished one and rewritten once.

    Round 20261128-125459: three of six missions ended through this path, and none of the answer checks ran on it. EPL
    shipped a standings table with 101 cells that disagree with the printed table and four clubs not in the league, and
    AUUNEMP "4.1% in October 2026 (the latest official figure)" where the series it forecast ends at 4.865. The checks now
    run here; the composer rewrites once with the fixes; what is still wrong is returned for the reason."""
    wrong = _gap_answer_claims(text, history, goal)
    if not wrong:
        return text, []
    rewritten = _compose(ctx, brain, f"gapclaims:{tag}:{i}", goal, history, memo, context, text,
                         fixes=contracts.summarise(wrong))
    if rewritten is not None and rewritten.strip():
        text = rewritten
        wrong = _gap_answer_claims(text, history, goal)
    if {c["name"] for c in wrong} & MECHANICAL_FIXES:
        text = _mechanical_fixes(text, history, wrong)
        wrong = _gap_answer_claims(text, history, goal)
    return text, wrong


MECHANICAL_FIXES = frozenset({"answer_causes_cited", "answer_flat_claim_after_tilt"})


def _mechanical_fixes(text: str, history: list[dict], claims: list[dict]) -> str:
    """The answer with the fixes that follow from stored data alone applied: uncited causes dropped, and a moving centre
    line called flat described by its first and last values."""
    names = {c["name"] for c in claims}
    if "answer_causes_cited" in names:
        text = grounding.drop_uncited_causes(text)
    if "answer_flat_claim_after_tilt" in names:
        text = contracts.replace_flat_claims(text, history)
    return text


DATE_TEXT = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?(?:\s+\d{4})?"
                       r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,?\s+\d{4})?", re.I)
# Any heading level, with or without a colon or bold (audit 2026-11-22, item 8): "### Outside context:", "**Outside context**".
OUTSIDE_HEAD = re.compile(r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?[ \t]*outside context[ \t]*(?:\*\*)?[ \t]*:?[ \t]*$", re.I | re.M)
BULLET = re.compile(r"^[ \t]*[-*\u2022][ \t]+\S")
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
    fetched: list[dict] = []
    for tool in ("web_research", "web_fetch", "web_search", "download_web_data"):
        for res in ctx.store.tool_results(ctx.mission_id, tool):
            if isinstance(res, dict):
                fetched.append({"kind": "tool", "tool": tool, "result": res})
                if tool == "download_web_data":
                    continue
                sources.extend(res.get("sources") or res.get("hits") or [])
                if isinstance(res.get("text"), str):
                    sources.append({"text": res["text"]})
    nxt = SECTION_END.search(answer[m.end():])
    end = m.end() + nxt.start() if nxt else len(answer)
    kept, removed, dropped = [], [], 0
    texts = grounding._source_texts(fetched)
    for line in answer[m.end():end].splitlines():
        if BULLET.match(line) and not grounding.quote_found(line, texts):
            # Round 10 (GISTEMP): "ENSO-neutral conditions expected", where the cited CPC passage says a strong El Nino is
            # likely. A claim with no figure passed every check. Composer v7 asks for an exact quote of the passage on each
            # line; a line whose quote is missing or not in the cited site's text is left out.
            removed.append("(quote)")
            dropped += 1
            continue
        bad = verify_figures(DATE_TEXT.sub(" ", line), sources) if line.strip() else []
        # Round 10 (CPI): "(philadelphiafed.org, latest Q4 2026) ... 2.5-2.7%" passed because 2.5 and 2.7 were printed in
        # another source. A line that cites a domain is checked against what that domain returned.
        bad = bad + [b for b in (grounding.source_bound_mismatches(line, fetched) if line.strip() else []) if b not in bad]
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
    body += (f"\n\n_{dropped} outside claim(s) were left out because their figures or quotes were not found in the pages "
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
        stored = ctx.store.step_output(ctx.mission_id, 0)         # the contracts step ({"v": ...}): the run's stored date
        brief = stored.get("v") if isinstance(stored, dict) else None
        hz_note = horizon_note(goal, history, brief.get("today") if isinstance(brief, dict) else None)
        if hz_note and isinstance(result.get("answer"), str) and result["answer"].strip() and hz_note not in result["answer"]:
            result = {**result, "answer": result["answer"].rstrip() + "\n\n" + hz_note}
        from prophet_agent.runtime import league_table
        season = league_table.table_markdown(history)
        if season and isinstance(result.get("answer"), str) and result["answer"].strip() and league_table.HEADER not in result["answer"]:
            # Written by code from the stored simulation, after grounding (audit of round 20261128-173823, EPL).
            result = {**result, "answer": result["answer"].rstrip() + "\n\n" + season}
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
