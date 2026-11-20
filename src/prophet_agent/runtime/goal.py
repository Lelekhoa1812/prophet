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

from .questions import QUESTIONS_SCHEMA, validate_questions
from .runner import BudgetExceeded, Ctx, Paused, Tool
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
    },
}

OBS_CHARS = 4000


class Brain(Protocol):
    def decide(self, messages: list[dict], schema: dict, name: str) -> dict: ...


@dataclass
class GoalConfig:
    max_iterations: int = 14
    stall_window: int = 3


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


def _plan_messages(goal: str, tools: dict[str, Tool], history: list[dict], context: str = "") -> list[dict]:
    system = prompts.current("planner").text
    user = (f"Goal:\n{goal}\n\n{context + chr(10) + chr(10) if context else ''}Available tools:\n{_tool_lines(tools)}\n\n"
            f"Evidence so far (oldest first):\n{_evidence_lines(history)}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _eval_messages(goal: str, history: list[dict], context: str = "") -> list[dict]:
    system = prompts.current("evaluator").text
    user = f"Goal:\n{goal}\n\n{context + chr(10) + chr(10) if context else ''}Evidence so far (oldest first):\n{_evidence_lines(history)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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
            plan = ctx.step(f"plan:{i}", lambda c: brain.decide(_plan_messages(goal, tools, history, context),
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
            if verdict["verdict"] == "met":
                return {"status": "met", "iterations": i + 1, "answer": verdict.get("answer", ""),
                        "evidence": history}
            if verdict["verdict"] == "impossible":
                return {"status": "impossible", "iterations": i + 1, "reason": verdict["reason"],
                        "evidence": history}
            history.append({"kind": "evaluation", "verdict": "not_met", "missing": verdict["reason"]})
        return {"status": "iteration_budget", "iterations": cfg.max_iterations, "evidence": history}

    return program


def _act(ctx: Ctx, tool: Tool, name: str, args: dict) -> dict:
    try:
        result = ctx.call(tool, args)
    except (BudgetExceeded, OutcomeUnknown, Paused):
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
                 max_tokens: int = 4096):
        self.client = client
        self.models = models
        self.thinking = thinking
        self.max_tokens = max_tokens

    def decide(self, messages: list[dict], schema: dict, name: str) -> dict:
        reply = self.client.complete(list(self.models or DEFAULT_CHAIN), messages,
                                     thinking=self.thinking, max_tokens=self.max_tokens,
                                     schema=schema, schema_name=name)
        return json.loads(reply.text)
