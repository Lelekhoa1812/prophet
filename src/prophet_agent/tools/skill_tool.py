"""Analyst skills as a runtime tool: the planner lists the playbooks, then reads only the one it needs.

Motivation vs Logic
Motivation: the decision rules a principal analyst follows (profiling, gaps, outliers, censoring, leakage, backtests,
intervals) would bloat the planner prompt if they were always present. A playbook read on demand keeps the prompt short
and puts the detail exactly where a step needs it.
Logic: the tool is pure (same input, same output), so it is marked idempotent. With no name it returns every skill's
name and summary. With a name it returns that skill's summary and body. An unknown name is refused with the valid names,
so the planner corrects itself rather than guessing.
"""
from __future__ import annotations

from typing import Any

from prophet_agent.runtime.runner import Tool
from prophet_agent.skills import list_skills, read_skill

ALLOWED_ARGS = {"name"}


def _run(args: dict, _key: str) -> dict[str, Any]:
    extra = sorted(set(args) - ALLOWED_ARGS)
    if extra:
        raise ValueError(f"unknown arguments {extra}; allowed: name (optional)")
    name = args.get("name")
    skills = list_skills()
    if name is None:
        return {"skills": skills}
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"name must be one of: {', '.join(s['name'] for s in skills)}")
    key = name.strip()
    try:
        text = read_skill(key)
    except KeyError:
        raise ValueError(f"unknown skill {key!r}; valid names: {', '.join(s['name'] for s in skills)}") from None
    summary = next(s["summary"] for s in skills if s["name"] == key)
    return {"name": key, "summary": summary, "text": text}


def skill_tool() -> Tool:
    return Tool("analyst_skill", _run, idempotent=True)


ANALYST_SKILL_SPEC = {
    "name": "analyst_skill",
    "title": "Analyst skills",
    "activity": "Reading an analysis playbook",
    "description": "Read a short analysis playbook when a step needs a decision rule: profiling a new dataset, gaps, "
                   "outliers and level shifts, zeros and intermittent demand, seasonality, smoothing without leakage, "
                   "choosing a model by backtest, TimesFM control, drivers, intervals, and the handover. Call with no name "
                   "to list the playbooks and their summaries, then with a name to read one. Read one before deciding a "
                   "treatment you are not sure of. A playbook states rules and thresholds; the tools' own checks still "
                   "apply.",
    "args": {"name": "optional skill name from the list; leave it out to list all skills with their summaries"},
}
