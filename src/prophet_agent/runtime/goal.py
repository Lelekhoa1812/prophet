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

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from prophet_agent import prompts
from prophet_agent.nim.errors import redact
from prophet_agent.nim.transport import DEFAULT_CHAIN

from . import contracts
from .grounding import unverified
from . import agents
from .questions import QUESTIONS_SCHEMA, validate_questions
from .runner import BudgetExceeded, Cancelled, Ctx, Paused, Tool
from .store import OutcomeUnknown, canonical

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

OBS_CHARS = 4000          # one observation, as the planner sees it
VIEW_CHARS = 16000        # uncompacted observations shown before the older ones fold into the memo
KEEP_RECENT = 4           # newest observations never compacted
MIN_FOLD = 3              # older observations folded per compaction, unless the view is far over its limit
HARD_VIEW_FACTOR = 2.5    # the view may grow to this many times VIEW_CHARS before a smaller fold is forced
MEMORY_CHARS = 8000       # hard cap on the memo; a memo over this is cut and says so
ASKS_KEEP_CHARS = 2000    # head of the memo (the ASKS section) that a cut to fit never removes
EXTERNAL_GOAL = re.compile(r"\b(?:wti|brent|crude|oil|gas|gold|silver|copper|commodit\w*|fx|forex|exchange rate|usd|eur|gbp|jpy|aud|"
                           r"stock|equit\w*|share price|s&p|nasdaq|index|yield|bond|treasur\w*|bitcoin|btc|crypto\w*|spot price|"
                           r"interest rate|inflation|cpi|unemployment|gdp)\b", re.I)
FORECAST_GOAL = re.compile(r"forecast|predict|projection|project(?:ed)? (?:the|next)|next \d+|demand|outlook", re.I)
FILES_GOAL = re.compile(r"\b(?:chart|graph|plot|workbook|spreadsheet|xlsx|docx|pdf|png|svg|report|deliverable)s?\b", re.I)
MAX_CORRECTIONS = 2       # times an answer with unmatched figures goes back to gather before it is repaired
DONE_WHEN = ("the evaluator judges the goal met", "the deterministic checks pass on the evidence "
             "(data inspected, forecast compared with baselines, forecast health repaired, deliverables present)",
             "every figure in the answer traces to a tool result")
MUST_NOT_CHANGE = ("synthetic or expected numbers are never presented as observed", "the 15 percent improvement bar is not moved",
                   "TimesFM 3 outputs are research use only")
STOP_REASONS = {"met": "done_verified", "partial": "partial_unverified", "impossible": "impossible", "stalled": "stalled",
                "checkpoint": "budget_part_spent"}
MAX_CONTRACT_BLOCKS = 2    # times a failed deterministic check may send a finished answer back before it leaves as partial
MAX_REJECTED_FINISHES = 3  # finishes the evaluator may reject in a row before the work is written up as it stands
MAX_ASKS = 8              # questions a mission may put to the person; past this the work decides with what it has
SUB_STEPS = 14            # default step budget of a delegated sub-agent
BRANCH_BLOCK = 100_000    # step numbers reserved per plan step for parallel sub-agents, and per agent within it
BRANCH_STRIDE = 20_000


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
    """What the planner sees of an observation. A sub-agent shows its report, not its own working."""
    if obs.get("kind") == "subagent":
        return {k: obs[k] for k in ("kind", "agent_id", "agent_type", "task", "tools", "status", "answer", "reason",
                                    "steps", "files", "unverified") if k in obs}
    return obs


def _clip(obj: Any) -> str:
    s = canonical(obj)
    return s if len(s) <= OBS_CHARS else s[:OBS_CHARS] + "...[clipped]"


def _tool_lines(tools: dict[str, Tool]) -> str:
    if not tools:
        return "- (none)"
    return "\n".join(f"- {name}: {tools[name].doc}" if tools[name].doc else f"- {name}" for name in sorted(tools))


def _evidence_lines(history: list[dict], start: int = 1) -> str:
    # Each observation is clipped on its own, so the goal and the tool list are never cut off by a long result.
    if not history:
        return "(none yet)"
    return "\n".join(f"{n}. {_clip(_view(obs))}" for n, obs in enumerate(history, start=start))


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


def _asks_made(history: list[dict]) -> int:
    """Pauses for the person so far in this mission: every answered question set, planner or evaluator."""
    return sum(1 for h in history if h.get("kind") in ("ask_user", "evaluator_question") and h.get("answers"))


# Values a question is about: dates, and numbers of three or more digits. Two questions that name the same dates and
# figures are about the same decision even when the wording differs (a re-phrased promo-window question is one decision).
_ANCHOR = re.compile(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{3,}(?:[.,]\d+)?")
ANCHOR_SHARE = 0.5


def _same_anchors(a: str, b: str) -> bool:
    """True when two questions name at least two of the same values and most of the values either one names."""
    xa, xb = set(_ANCHOR.findall(a)), set(_ANCHOR.findall(b))
    shared = xa & xb
    return len(shared) >= 2 and len(shared) / len(xa | xb) >= ANCHOR_SHARE


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
        same = any((head and head == eh) or difflib.SequenceMatcher(None, text, et).ratio() >= SAME_QUESTION_RATIO
                   or _same_anchors(text, et)
                   for eh, et in earlier)
        (repeated if same else open_qs).append(q)
    return open_qs, [str(q.get("question", ""))[:160] for q in repeated]


READ_ONLY_TOOLS = ("analyst_skill", "notes")      # reading a playbook or a note does no work, so it is not charged to the budget


def _calls_used(history: list[dict]) -> int:
    """Tool calls made so far: unknown-tool slips never reached the runtime, and reads of playbooks and notes are not work."""
    return sum(1 for h in history if h.get("kind") == "tool" and h.get("tool") not in READ_ONLY_TOOLS
               and not str(h.get("error", "")).startswith("unknown tool"))


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


def _budget_note(i: int, cfg: "GoalConfig", history: list[dict]) -> str:
    left = cfg.end - i
    if cfg.tool_budget is not None:
        left = min(left, cfg.tool_budget - _calls_used(history))
    note = f"Budget: step {i + 1} of this part of the work; {max(left, 0)} steps left in it."
    streak, err, kind = _failure_streak(history)
    if streak >= 2 and kind == "ask_user":
        note += (f" The last {streak} questions were refused ({err}). Do not send that question again. Fix its shape, or "
                 "decide without asking and say what you assumed.")
    elif streak >= 3:
        note += (f" The last {streak} calls failed in a row ({err}). Stop writing the large step again. Run one small cell "
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


def _eval_messages(goal: str, history: list[dict], memo: Memo, context: str = "") -> list[dict]:
    system = prompts.current("evaluator").text
    user = f"Goal:\n{goal}\n\n{context + chr(10) + chr(10) if context else ''}{_evidence_block(history, memo)}"
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


def _compose_messages(goal: str, history: list[dict], memo: Memo, context: str, draft: str,
                      bad: list[str] | None = None) -> list[dict]:
    system = prompts.current("composer").text
    user = (f"Goal:\n{goal}\n\n{context + chr(10) + chr(10) if context else ''}{_evidence_block(history, memo)}"
            f"\n\nDraft (the content to keep; its formatting was lost):\n{draft}")
    if bad:
        user += (f"\n\nThese figures in the draft do not appear in the evidence: {', '.join(bad)}. Remove each of them, "
                 "row numbers included. Describe a row in words if the evidence gives no number for it.")
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


def _compact(ctx: Ctx, brain: Any, goal: str, history: list[dict], memo: Memo, tag: str, *,
             view_chars: int = VIEW_CHARS, keep_recent: int = KEEP_RECENT,
             memory_chars: int = MEMORY_CHARS) -> None:
    """Folds the observations older than the newest KEEP_RECENT into the memo once they outgrow the view. Durable: the
    memo text comes from a stored step, so a resumed run reads the same memo without asking the writer again."""
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
    write = getattr(brain, "write", None)
    if write is None:
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
             bad: list[str] | None = None) -> str | None:
    """The finished answer as plain text. A schema-bound reply cannot carry line breaks (the decoder drops them), so the
    verdict's answer is only a draft; the text the person reads is written in a free-text call. None: the brain cannot."""
    write = getattr(brain, "write", None)
    if write is None or not draft.strip():
        return None
    return ctx.step(step, lambda c: write(_compose_messages(goal, history, memo, context, draft, bad)))


def _signature(obs: dict) -> str:
    return canonical({"tool": obs.get("tool"), "args": obs.get("args"), "result": obs.get("result"),
                      "error": obs.get("error")})


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


def _act(ctx: Ctx, tool: Tool, name: str, args: dict) -> dict:
    try:
        result = ctx.call(tool, args)
    except (BudgetExceeded, Cancelled, OutcomeUnknown, Paused):
        raise                    # control flow, not evidence: the runtime must see these
    except Exception as exc:  # a tool failure is evidence for the planner; the ledger already recorded it
        return {"kind": "tool", "tool": name, "args": args, "error": redact(str(exc))}
    if tool.observe is not None:
        try:
            result = tool.observe(result)
        except Exception:  # a broken summariser must not hide the result from the planner
            pass
    return {"kind": "tool", "tool": name, "args": args, "result": result}


def _status(ctx: Ctx, i: int, history: list[dict], items: list[dict], started: float) -> None:
    """One small event per step: where the mission stands (step, tool calls, time, checklist). The console and a reader
    can follow a long run from these without reading the history. Never raises: a status line must not stop the work."""
    try:
        done = sum(1 for it in items if it["status"] == "done")
        nxt = next((it["item"] for it in items if it.get("next")), None)
        ctx.store.append_event(ctx.mission_id, "mission.progress", {
            "step": i + 1, "tool_calls": _calls_used(history), "elapsed_s": round(time.time() - started, 1),
            "checklist_done": done, "checklist_total": len(items), "next": nxt})
    except Exception:
        pass


def _stalled(history: list[dict], window: int) -> bool:
    tool_obs = [h for h in history if h.get("kind") == "tool"]
    if len(tool_obs) < window:
        return False
    last = [_signature(h) for h in tool_obs[-window:]]
    return len(set(last)) == 1


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
    sub_cfg = GoalConfig(max_iterations=cfg.sub_steps, stall_window=cfg.stall_window, depth=1, sub_steps=cfg.sub_steps,
                         tag=f"d1.{agent_id}")
    sub_history: list[dict] = list(prior["evidence"]) if prior else []
    goal = _brief(spec, agent, agent_id, prior is not None)
    play = agents.playbooks(agent)
    sub_context = (context + "\n\n" if context else "") + (play[:7000] if play else "")
    # Sub-agent steps are top-level steps of the same ledger, numbered in call order like any others, so a resumed run
    # replays them. They are not wrapped in an outer ctx.step, which would nest their numbers inside one another.
    sub = _run(ctx, brain, goal, sub_tools, sub_cfg, sub_context.strip(), sub_history)
    answer = sub.get("answer", "") or sub.get("draft", "")
    obs = {"kind": "subagent", "agent_id": agent_id, "agent_type": agent.name, "task": str(spec["task"]).strip(),
           "tools": sorted(sub_tools), "status": sub.get("status"), "answer": answer, "reason": sub.get("reason", ""),
           "steps": len(sub_history), "files": _files_of(sub_history), "evidence": sub_history}
    bad = unverified(answer, sub_history, str(spec["task"])) if answer else []
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


def _run(ctx: Ctx, brain: Any, goal: str, tools: dict[str, Tool], cfg: GoalConfig, context: str,
         history: list[dict]) -> dict:
    """The loop itself. Used for the goal and, recursively with depth 1, for a delegated sub-agent."""
    memo = Memo()
    if cfg.depth == 0:
        tools = {**tools, "notes": _notes_for(ctx, "main")}
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
    contract_blocks = 0       # times a deterministic check sent a "met" answer back for more work
    for i in range(cfg.end):
        if i and i % seg_len == 0:
            # A new part of the work starts here. The marker is an observation, so the planner and the replay see it.
            history.append({"kind": "continued", "segment": i // seg_len})
        out_of_calls = cfg.tool_budget is not None and _calls_used(history) >= cfg.tool_budget
        # Every step index is taken on every pass, including the forced finishes below. A finish that was decided outside
        # the ledger would shift the indices of later steps, and a resumed run would read the wrong stored outputs.
        _compact(ctx, brain, goal, history, memo, f"{tag}:{i}", view_chars=cfg.view_chars,
                 keep_recent=cfg.keep_recent, memory_chars=cfg.memory_chars)
        note = _budget_note(i, cfg, history)
        if (read_note := _skills_read_note(ctx)):
            note += "\n\n" + read_note
        if cfg.depth == 0 and FORECAST_GOAL.search(goal):
            # The checklist is derived from the evidence on every step, so it cannot drift and a resumed run sees the same one.
            items = contracts.progress(history, run_root=SANDBOX_ROOT, wants_files=bool(FILES_GOAL.search(goal)),
                                       external=bool(EXTERNAL_GOAL.search(goal)))
            note += "\n\n" + contracts.render_progress(items)
            _status(ctx, i, history, items, started)
        plan = ctx.step(f"plan:{tag}:{i}", lambda c: _decide_plan(brain, goal, tools, history, memo, context, note,
                                                                 finish_now=cfg.finish_now, out_of_calls=out_of_calls))
        if plan["action"] == "finish":
            obs = {"kind": "finish", "rationale": plan["rationale"]}
        elif plan["action"] == "ask_user":
            open_qs, repeated = _already_asked(plan.get("questions", []) or [], history)
            payload = {"questions": open_qs}
            problems = (validate_questions(payload) if open_qs or not repeated
                        else ["every question was already asked and answered"])
            asks = _asks_made(history)
            if asks >= MAX_ASKS:
                obs = {"kind": "ask_user", "error": f"no more questions: the person has already been asked {asks} times in this "
                       "mission. Decide with what is known, state each assumption in the answer, and finish."}
            elif problems and repeated:
                obs = {"kind": "ask_user", "error": "not asked again, the person already answered: " + "; ".join(repeated)
                       + ". The decision is in the decisions list; act on it."}
            elif problems:
                obs = {"kind": "ask_user", "error": "invalid questions: " + "; ".join(problems)}
            else:
                # Pauses the run here. The answer, once given, is returned as the step's value. Questions that repeat an
                # earlier one are left out; the earlier answer stands.
                answers = ctx.step(f"ask:{tag}:{i}", lambda c: c.ask(payload))
                obs = {"kind": "ask_user", "questions": open_qs, "answers": answers}
                if repeated:
                    obs["not_asked_again"] = repeated
        elif plan["action"] == "delegate":
            delegated = _delegate(ctx, brain, plan, tools, cfg, context, history)
            history.extend(delegated[:-1])
            obs = delegated[-1]
        else:
            name = plan.get("tool", "")
            args = _repair_source(plan.get("args", {}), history)
            if name not in tools:
                obs = {"kind": "tool", "tool": name, "args": args, "error": f"unknown tool {name!r}"}
            else:
                obs = ctx.step(f"act:{tag}:{i}", lambda c: _act(c, tools[name], name, args))
        history.append(obs)

        if obs.get("kind") == "tool" and _stalled(history, cfg.stall_window):
            # Repeating the same call will not help. Judge what was found so far: a best answer, labelled partial,
            # beats an empty "stalled".
            verdict = ctx.step(f"evalstall:{tag}:{i}", lambda c: brain.decide(_eval_messages(goal, history, memo, context),
                                                                          EVAL_SCHEMA, "eval"))
            draft = verdict.get("answer", "") or verdict.get("reason", "")
            written = _compose(ctx, brain, f"composestall:{tag}:{i}", goal, history, memo, context, draft)
            text = written if written is not None else verdict.get("answer", "")
            if text.strip() and verdict["verdict"] != "impossible":
                return {"status": "partial", "iterations": i + 1, "answer": text, "evidence": history,
                        "reason": "The analysis kept repeating the same step, so this answer rests on what was found before that."}
            return {"status": "stalled", "iterations": i + 1, "evidence": history}

        last = i == cfg.end - 1
        # The evaluator is a judgement on a finished piece of work, so it runs when the planner says it is done, or on the
        # last step of this run to turn whatever was found into an answer or a checkpoint. Between those, it only costs time.
        # The evaluator also runs at the end of each part. The stored evaluation then replays at the same position when
        # the next part continues from there, so both runs ask the ledger for the same steps in the same order.
        at_part_end = (i + 1) % seg_len == 0
        if plan["action"] != "finish" and not last and not out_of_calls and not at_part_end:
            continue
        verdict = ctx.step(f"eval:{tag}:{i}", lambda c: brain.decide(_eval_messages(goal, history, memo, context),
                                                                 EVAL_SCHEMA, "eval"))
        # The evaluator may find that one decision by the person would change the answer. It is asked once per run,
        # then the evaluator judges again with the answer in the evidence.
        qs = _usable_questions(verdict) if verdict["verdict"] != "met" else None
        if qs:
            qs, _ = _already_asked(qs, history)
        if qs and not any(h.get("kind") == "evaluator_question" for h in history):
            answers = ctx.step(f"evalask:{tag}:{i}", lambda c: c.ask({"questions": qs}))
            history.append({"kind": "evaluator_question", "questions": qs, "answers": answers})
            verdict = ctx.step(f"eval2:{tag}:{i}", lambda c: brain.decide(_eval_messages(goal, history, memo, context),
                                                                       EVAL_SCHEMA, "eval"))
        if verdict["verdict"] == "met":
            answer = verdict.get("answer", "")
            # A verdict can say "met" and carry no answer (a reply cut short, or a model that only gave its reason).
            # The writer then works from the reason and the evidence; "met" with nothing to read is never returned.
            written = _compose(ctx, brain, f"compose:{tag}:{i}", goal, history, memo, context, answer or verdict.get("reason", ""))
            if written is not None:
                answer = written
            if not answer.strip():
                if not last and not out_of_calls:
                    history.append({"kind": "evaluation", "verdict": "not_met",
                                    "missing": "the goal looks met, but no written answer was given; state the answer"})
                    continue
                return {"status": "partial", "iterations": i + 1, "answer": "",
                        "reason": "The analysis finished but no written answer was produced.", "evidence": history}
            # A model saying it is finished is a claim; the checks run on the evidence. A blocking failure goes back to the
            # planner as work to do, at most MAX_CONTRACT_BLOCKS times, then the answer leaves labelled partial.
            blocked = contracts.blocking(contracts.check_evidence(history, run_root=SANDBOX_ROOT))
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
            bad = unverified(answer, history, goal)
            if bad and _corrections(history) < MAX_CORRECTIONS and not last and not out_of_calls and not cfg.finish_now:
                # Gather before repairing: the figure may simply be an output the analysis has not printed yet. The
                # planner is told which figures lack a source, and the answer is judged again after it works.
                history.append({"kind": "evaluation", "verdict": "not_met", "missing": _missing_figures_text(bad),
                                "unverified": bad})
                continue
            if bad:
                # Repair pass: the figures must come from the evidence or be left out. Figures that still lack a source
                # after the gathering rounds are removed, and the answer says what is not available.
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
            if bad:
                return {"status": "partial", "iterations": i + 1, "answer": answer,
                        "reason": "Some figures in this answer could not be matched to the analysis output.",
                        "unverified": bad, "evidence": history}
            return {"status": "met", "iterations": i + 1, "answer": answer, "evidence": history}
        if verdict["verdict"] == "impossible":
            return {"status": "impossible", "iterations": i + 1, "reason": verdict["reason"], "evidence": history}
        if cfg.finish_now:
            # The person asked for the answer now. A best answer from the evidence, with what is missing, beats no answer,
            # and it is labelled partial, because the person chose to stop before the goal was met.
            draft = verdict.get("answer", "") or verdict.get("reason", "")
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
                draft = verdict.get("answer", "") or verdict.get("reason", "")
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

            bind(ctx._guard_time, note)
        result = _ground_result(_run(ctx, brain, goal, tools, cfg, context, history), history, goal)
        return {**result, "stop_reason": STOP_REASONS.get(result.get("status", ""), "unknown")}

    return program


def _ground_result(result: dict, history: list[dict], goal: str) -> dict:
    """Every exit path is checked, not only the "met" one: a stalled, budget-capped or impossible answer carries
    figures too. Figures that the evidence does not support are listed on the result. A "met" result that fails the
    check is downgraded to partial, so it is never presented as fully grounded."""
    bad: list[str] = []
    for text in (result.get("answer") or "", result.get("reason") or ""):
        for token in unverified(text, history, goal):
            if token not in bad:
                bad.append(token)
    if not bad:
        return result
    out = {**result, "unverified": bad}
    if out.get("status") == "met":
        out["status"] = "partial"
        out["reason"] = "Some figures in this answer could not be matched to the analysis output."
    return out


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

    def bind(self, checkpoint: Callable[[], None] | None, on_wait: Callable[[dict], None] | None = None) -> None:
        """Let a provider wait be stopped (cancel, spent time allowance) and reported. Called by the goal program."""
        self.checkpoint, self.on_wait = checkpoint, on_wait

    def _extra(self) -> dict:
        extra: dict = {}
        if self.checkpoint is not None:
            extra["checkpoint"] = self.checkpoint
        if self.on_wait is not None:
            extra["on_wait"] = self.on_wait
        return extra

    def write(self, messages: list[dict]) -> str:
        """Free text, no schema, so Markdown keeps its line breaks (a schema-bound string comes back on one line)."""
        reply = self.client.complete(list(self.models or getattr(self.client, "chain", None) or DEFAULT_CHAIN), messages, thinking=False,
                                     max_tokens=self.write_tokens, **self._extra())
        return reply.text

    def decide(self, messages: list[dict], schema: dict, name: str) -> dict:
        reply = self.client.complete(list(self.models or getattr(self.client, "chain", None) or DEFAULT_CHAIN), messages,
                                     thinking=self.thinking, max_tokens=self.max_tokens,
                                     schema=schema, schema_name=name, **self._extra())
        return json.loads(reply.text)
