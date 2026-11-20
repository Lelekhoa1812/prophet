"""Goal loop: plan, act with a tool or ask the user, then evaluate against the goal.

Motivation vs Logic
Motivation: a forecasting request is open-ended. The agent must decide the next action, use tools (including TimesFM),
know when the goal is met or cannot be met, and stop rather than spin when it is stuck or over budget.
Logic: each iteration has three durable steps: plan (model chooses an action), act (tool call or question), evaluate
(model returns met, not_met or impossible). Every model call lives inside ``ctx.step`` so a resumed run reads the
stored decision. Observations are rebuilt from those stored outputs, so the planner sees the same history after a
crash. Stall detection looks at the last ``stall_window`` actions: if they are identical, further iterations will not
change the outcome, so the loop stops. Budgets are the iteration cap here and ``max_tool_calls`` / ``max_seconds`` in
the runtime.

The loop's own ``status`` is returned inside the program result. The runtime status is ``completed`` whenever the
program finished, whatever the goal outcome was.
"""
from __future__ import annotations

import json
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
        "action": {"type": "string", "enum": ["call_tool", "ask_user", "finish"]},
        "tool": {"type": "string"},
        "args": {"type": "object"},
        "questions": QUESTIONS_SCHEMA["properties"]["questions"],
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

OBS_CHARS = 4000


class Brain(Protocol):
    def decide(self, messages: list[dict], schema: dict, name: str) -> dict: ...

    # Optional: ``write(messages) -> str`` returns free text. The final answer is written through it when present,
    # because a schema-bound reply loses its line breaks.


@dataclass
class GoalConfig:
    max_iterations: int = 14
    stall_window: int = 3
    tool_budget: int | None = None     # tool calls the runtime allows; the planner is told how many are left


@dataclass
class Trace:
    observations: list[dict] = field(default_factory=list)


def _clip(obj: Any) -> str:
    s = canonical(obj)
    return s if len(s) <= OBS_CHARS else s[:OBS_CHARS] + "...[clipped]"


def _tool_lines(tools: dict[str, Tool]) -> str:
    if not tools:
        return "- (none)"
    return "\n".join(f"- {name}: {tools[name].doc}" if tools[name].doc else f"- {name}" for name in sorted(tools))


def _evidence_lines(history: list[dict]) -> str:
    # Each observation is clipped on its own, so the goal and the tool list are never cut off by a long result.
    if not history:
        return "(none yet)"
    return "\n".join(f"{n}. {_clip(obs)}" for n, obs in enumerate(history[-12:], start=1))


def _calls_used(history: list[dict]) -> int:
    """Tool calls made so far: unknown-tool slips never reached the runtime, so they are not counted."""
    return sum(1 for h in history if h.get("kind") == "tool" and not str(h.get("error", "")).startswith("unknown tool"))


def _budget_note(i: int, cfg: "GoalConfig", history: list[dict]) -> str:
    left = cfg.max_iterations - i
    if cfg.tool_budget is not None:
        left = min(left, cfg.tool_budget - _calls_used(history))
    note = f"Budget: step {i + 1} of {cfg.max_iterations}; {max(left, 0)} steps left."
    if left <= 2:
        note += " Stop exploring. Finish with what is established and say plainly what is missing."
    return note


def _plan_messages(goal: str, tools: dict[str, Tool], history: list[dict], context: str = "",
                   budget: str = "") -> list[dict]:
    system = prompts.current("planner").text
    user = (f"Goal:\n{goal}\n\n{context + chr(10) + chr(10) if context else ''}Available tools:\n{_tool_lines(tools)}\n\n"
            f"Evidence so far (oldest first):\n{_evidence_lines(history)}" + (f"\n\n{budget}" if budget else ""))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _eval_messages(goal: str, history: list[dict], context: str = "") -> list[dict]:
    system = prompts.current("evaluator").text
    user = f"Goal:\n{goal}\n\n{context + chr(10) + chr(10) if context else ''}Evidence so far (oldest first):\n{_evidence_lines(history)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _repair_messages(goal: str, history: list[dict], answer: str, bad: list[str], context: str = "") -> list[dict]:
    system = prompts.current("evaluator").text
    user = (f"Goal:\n{goal}\n\n{context + chr(10) + chr(10) if context else ''}Evidence so far (oldest first):\n"
            f"{_evidence_lines(history)}\n\nYour draft answer:\n{answer}\n\n"
            f"These figures in the draft do not appear in the evidence: {', '.join(bad)}. Rewrite the answer and remove "
            "each of them, row numbers included. Describe a row in words if the evidence gives no number for it. Use only "
            "figures that appear in the evidence above, and say plainly when a figure is not available. Return the "
            "verdict met with the rewritten answer.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _compose_messages(goal: str, history: list[dict], context: str, draft: str, bad: list[str] | None = None) -> list[dict]:
    system = prompts.current("composer").text
    user = (f"Goal:\n{goal}\n\n{context + chr(10) + chr(10) if context else ''}Evidence (oldest first):\n"
            f"{_evidence_lines(history)}\n\nDraft (the content to keep; its formatting was lost):\n{draft}")
    if bad:
        user += (f"\n\nThese figures in the draft do not appear in the evidence: {', '.join(bad)}. Remove each of them, "
                 "row numbers included. Describe a row in words if the evidence gives no number for it.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user + "\n\nWrite the final answer in Markdown."}]


def _compose(ctx: Ctx, brain: Any, step: str, goal: str, history: list[dict], context: str, draft: str,
             bad: list[str] | None = None) -> str | None:
    """The finished answer as plain text. A schema-bound reply cannot carry line breaks (the decoder drops them), so the
    verdict's answer is only a draft; the text the person reads is written in a free-text call. None: the brain cannot."""
    write = getattr(brain, "write", None)
    if write is None or not draft.strip():
        return None
    return ctx.step(step, lambda c: write(_compose_messages(goal, history, context, draft, bad)))


def _usable_questions(verdict: dict) -> list[dict] | None:
    """The evaluator's questions, if it asked any and they pass the same validator as the planner's."""
    qs = verdict.get("questions")
    if not qs or validate_questions({"questions": qs}):
        return None
    return qs


def _signature(obs: dict) -> str:
    return canonical({"tool": obs.get("tool"), "args": obs.get("args"), "result": obs.get("result"),
                      "error": obs.get("error")})


def goal_program(goal: str, tools: dict[str, Tool], brain: Brain, cfg: GoalConfig | None = None,
                 context: str = "") -> Callable[[Ctx], dict]:
    cfg = cfg or GoalConfig()

    def program(ctx: Ctx) -> dict:
        # Which instructions this mission ran under, recorded once so a resumed run reads it back unchanged.
        ctx.step("contracts", lambda c: {n: {"version": prompts.current(n).version, "sha256": prompts.current(n).sha256}
                                         for n in prompts.CURRENT})
        history: list[dict] = []
        for i in range(cfg.max_iterations):
            out_of_calls = cfg.tool_budget is not None and _calls_used(history) >= cfg.tool_budget
            if out_of_calls:
                # No tool call is left, so planning another would only end the run with nothing to show. It is derived
                # from the history, so a resumed run reaches the same decision without a stored step.
                plan = {"action": "finish", "rationale": "tool budget used"}
            else:
                note = _budget_note(i, cfg, history)
                plan = ctx.step(f"plan:{i}", lambda c: brain.decide(_plan_messages(goal, tools, history, context, note),
                                                                   PLAN_SCHEMA, "plan"))
            if plan["action"] == "finish":
                obs = {"kind": "finish", "rationale": plan["rationale"]}
            elif plan["action"] == "ask_user":
                payload = {"questions": plan.get("questions", [])}
                problems = validate_questions(payload)
                if problems:
                    obs = {"kind": "ask_user", "error": "invalid questions: " + "; ".join(problems)}
                else:
                    # Pauses the run here. The answer, once given, is returned as the step's value.
                    answers = ctx.step(f"ask:{i}", lambda c: c.ask(payload))
                    obs = {"kind": "ask_user", "answers": answers}
            else:
                name = plan.get("tool", "")
                args = plan.get("args", {})
                if name not in tools:
                    obs = {"kind": "tool", "tool": name, "args": args, "error": f"unknown tool {name!r}"}
                else:
                    obs = ctx.step(f"act:{i}", lambda c: _act(c, tools[name], name, args))
            history.append(obs)

            if obs.get("kind") == "tool" and _stalled(history, cfg.stall_window):
                return {"status": "stalled", "iterations": i + 1, "evidence": history}

            # The evaluator is a judgement on a finished piece of work, so it runs when the planner says it is done, or
            # on the last iteration to turn whatever was found into an honest answer. Between those, it only costs time.
            if plan["action"] != "finish" and i < cfg.max_iterations - 1:
                continue
            verdict = ctx.step(f"eval:{i}", lambda c: brain.decide(_eval_messages(goal, history, context),
                                                                   EVAL_SCHEMA, "eval"))
            # The evaluator may find that one decision by the person would change the answer. It is asked once per
            # run, then the evaluator judges again with the answer in the evidence.
            qs = _usable_questions(verdict) if verdict["verdict"] != "met" else None
            if qs and not any(h.get("kind") == "evaluator_question" for h in history):
                answers = ctx.step(f"evalask:{i}", lambda c: c.ask({"questions": qs}))
                history.append({"kind": "evaluator_question", "questions": qs, "answers": answers})
                verdict = ctx.step(f"eval2:{i}", lambda c: brain.decide(_eval_messages(goal, history, context),
                                                                        EVAL_SCHEMA, "eval"))
            if verdict["verdict"] == "met":
                answer = verdict.get("answer", "")
                written = _compose(ctx, brain, f"compose:{i}", goal, history, context, answer)
                if written is not None:
                    answer = written
                bad = unverified(answer, history, goal)
                if bad:
                    # One repair pass: the figures must come from the evidence or be left out.
                    repaired = _compose(ctx, brain, f"repair:{i}", goal, history, context, answer, bad)
                    if repaired is None:
                        fixed = ctx.step(f"repair:{i}", lambda c: brain.decide(
                            _repair_messages(goal, history, answer, bad, context), EVAL_SCHEMA, "eval"))
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
                return {"status": "impossible", "iterations": i + 1, "reason": verdict["reason"],
                        "evidence": history}
            final = i == cfg.max_iterations - 1 or out_of_calls
            if final and verdict.get("answer", "").strip():
                # The budget is spent. A best answer from the evidence, with what is missing, beats no answer.
                written = _compose(ctx, brain, f"compose:{i}", goal, history, context, verdict["answer"])
                return {"status": "partial", "iterations": i + 1, "answer": written if written is not None else verdict["answer"],
                        "reason": verdict["reason"], "evidence": history}
            if final:
                return {"status": "iteration_budget", "iterations": i + 1, "evidence": history}
            history.append({"kind": "evaluation", "verdict": "not_met", "missing": verdict["reason"]})
        return {"status": "iteration_budget", "iterations": cfg.max_iterations, "evidence": history}

    return program


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
        reply = self.client.complete(list(self.models or DEFAULT_CHAIN), messages, thinking=False,
                                     max_tokens=self.write_tokens)
        return reply.text

    def decide(self, messages: list[dict], schema: dict, name: str) -> dict:
        reply = self.client.complete(list(self.models or DEFAULT_CHAIN), messages,
                                     thinking=self.thinking, max_tokens=self.max_tokens,
                                     schema=schema, schema_name=name)
        return json.loads(reply.text)
