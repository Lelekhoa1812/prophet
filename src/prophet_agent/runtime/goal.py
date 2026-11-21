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
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from prophet_agent import prompts
from prophet_agent.nim.errors import redact
from prophet_agent.nim.transport import DEFAULT_CHAIN

from .grounding import unverified
from .questions import QUESTIONS_SCHEMA, validate_questions
from .runner import BudgetExceeded, Cancelled, Ctx, Paused, Tool
from .store import OutcomeUnknown, canonical

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["action", "rationale"],
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["call_tool", "ask_user", "delegate", "finish"]},
        "tool": {"type": "string"},
        "args": {"type": "object"},
        "questions": QUESTIONS_SCHEMA["properties"]["questions"],
        "task": {"type": "string", "maxLength": 1200},
        "tools": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        "rationale": {"type": "string", "maxLength": 400},
    },
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
MEMORY_CHARS = 8000       # hard cap on the memo; a memo over this is cut and says so
MAX_CORRECTIONS = 2       # times an answer with unmatched figures goes back to gather before it is repaired
MAX_ASKS = 8              # questions a mission may put to the person; past this the work decides with what it has
SUB_STEPS = 12            # default step budget of a delegated sub-agent


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
        return {k: obs[k] for k in ("kind", "task", "tools", "status", "answer", "reason", "steps") if k in obs}
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


def _calls_used(history: list[dict]) -> int:
    """Tool calls made so far: unknown-tool slips never reached the runtime, so they are not counted."""
    return sum(1 for h in history if h.get("kind") == "tool" and not str(h.get("error", "")).startswith("unknown tool"))


def _corrections(history: list[dict]) -> int:
    return sum(1 for h in history if h.get("kind") == "evaluation" and h.get("unverified"))


def _failure_streak(history: list[dict]) -> tuple[int, str, str]:
    """How many of the latest tool calls or questions were refused in a row, the latest error, and the kind of step.

    A repeated failure must change the plan. A question the runtime refuses is a failure too: live run 06 sent the same
    refused question 80 times because only tool failures were counted.
    """
    streak, last, kind = 0, "", ""
    for h in reversed(history):
        if h.get("kind") not in ("tool", "ask_user"):
            continue
        if not h.get("error"):
            break
        streak += 1
        last = last or str(h["error"])[:200]
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
    if cut <= memo.covered or sum(len(_clip(_view(o))) for o in pending) <= view_chars:
        return
    chunk = history[memo.covered:cut]
    write = getattr(brain, "write", None)
    if write is None:
        text = (memo.text + f"\n[{len(chunk)} earlier observations were left out of this view; the full record is kept.]")
    else:
        text = ctx.step(f"compact:{tag}:{cut}", lambda c: write(_compact_messages(goal, memo.text, chunk)))
    text = str(text or "").strip()
    if len(text) > memory_chars:
        text = text[-memory_chars:]
        text = "[earlier memory cut to fit]\n" + text
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


def _stalled(history: list[dict], window: int) -> bool:
    tool_obs = [h for h in history if h.get("kind") == "tool"]
    if len(tool_obs) < window:
        return False
    last = [_signature(h) for h in tool_obs[-window:]]
    return len(set(last)) == 1


def _delegate(ctx: Ctx, brain: Any, plan: dict, tools: dict[str, Tool], cfg: GoalConfig, context: str) -> dict:
    """Runs a sub-agent on one task with its own history and memo. Only its report returns to the parent."""
    task = str(plan.get("task") or "").strip()
    if cfg.depth > 0:
        return {"kind": "delegate", "error": "a sub-agent cannot start another sub-agent; do the step yourself"}
    if not task:
        return {"kind": "delegate", "error": "a delegated task needs a task description"}
    names = [n for n in (plan.get("tools") or []) if n in tools]
    sub_tools = {n: tools[n] for n in names} if names else dict(tools)
    sub_cfg = GoalConfig(max_iterations=cfg.sub_steps, stall_window=cfg.stall_window, depth=1, sub_steps=cfg.sub_steps)
    sub_history: list[dict] = []
    # Sub-agent steps are top-level steps of the same ledger, numbered in call order like any others, so a resumed run
    # replays them. They are not wrapped in an outer ctx.step, which would nest their numbers inside one another.
    sub = _run(ctx, brain, task, sub_tools, sub_cfg, context, sub_history)
    return {"kind": "subagent", "task": task, "tools": sorted(sub_tools), "status": sub.get("status"),
            "answer": sub.get("answer", "") or sub.get("draft", ""), "reason": sub.get("reason", ""),
            "steps": len(sub_history), "evidence": sub_history}


def _run(ctx: Ctx, brain: Any, goal: str, tools: dict[str, Tool], cfg: GoalConfig, context: str,
         history: list[dict]) -> dict:
    """The loop itself. Used for the goal and, recursively with depth 1, for a delegated sub-agent."""
    memo = Memo()
    if cfg.depth == 0:
        # Which instructions this mission ran under, recorded once so a resumed run reads it back unchanged.
        ctx.step("contracts", lambda c: {n: {"version": prompts.current(n).version, "sha256": prompts.current(n).sha256}
                                         for n in prompts.CURRENT})
    tag = f"d{cfg.depth}"
    seg_len = cfg.max_iterations
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
            obs = _delegate(ctx, brain, plan, tools, cfg, context)
        else:
            name = plan.get("tool", "")
            args = plan.get("args", {})
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
        history.append({"kind": "evaluation", "verdict": "not_met", "missing": verdict["reason"]})
    # Unreachable: the last step of the part always returns above. Kept so a future edit cannot fall through silently.
    return {"status": "checkpoint", "iterations": cfg.end, "segment": cfg.segment,
            "reason": "this part of the work is spent", "evidence": history, "memory": memo.text}


def goal_program(goal: str, tools: dict[str, Tool], brain: Brain, cfg: GoalConfig | None = None,
                 context: str = "") -> Callable[[Ctx], dict]:
    cfg = cfg or GoalConfig()

    def program(ctx: Ctx) -> dict:
        history: list[dict] = []
        return _ground_result(_run(ctx, brain, goal, tools, cfg, context, history), history, goal)

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

    def write(self, messages: list[dict]) -> str:
        """Free text, no schema, so Markdown keeps its line breaks (a schema-bound string comes back on one line)."""
        reply = self.client.complete(list(self.models or getattr(self.client, "chain", None) or DEFAULT_CHAIN), messages, thinking=False,
                                     max_tokens=self.write_tokens)
        return reply.text

    def decide(self, messages: list[dict], schema: dict, name: str) -> dict:
        reply = self.client.complete(list(self.models or getattr(self.client, "chain", None) or DEFAULT_CHAIN), messages,
                                     thinking=self.thinking, max_tokens=self.max_tokens,
                                     schema=schema, schema_name=name)
        return json.loads(reply.text)
