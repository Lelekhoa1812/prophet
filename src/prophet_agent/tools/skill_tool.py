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
    asked = name.strip()
    key = resolve_name(asked, [s["name"] for s in skills])
    if key is None:
        raise ValueError(f"unknown skill {asked!r}; valid names: {', '.join(s['name'] for s in skills)}")
    text = read_skill(key)
    summary = next(s["summary"] for s in skills if s["name"] == key)
    out = {"name": key, "summary": summary, "text": text}
    if key != asked:
        out["resolved_from"] = asked
    return out


# Topic words from this tool's own description, mapped to the playbook that covers them. Live: two missions asked for
# "profiling_a_new_dataset" (the description's wording) and spent a call on the refusal.
TOPICS = {"profil": "first-contact-profile", "first-contact": "first-contact-profile", "gap": "missing-data-and-gaps",
          "missing": "missing-data-and-gaps", "outlier": "outliers-and-level-shifts", "level-shift": "outliers-and-level-shifts",
          "changepoint": "changepoints-and-regimes", "regime": "changepoints-and-regimes", "intermittent": "zeros-censoring-and-intermittent",
          "zero": "zeros-censoring-and-intermittent", "censor": "zeros-censoring-and-intermittent",
          "season": "seasonality-and-calendar", "calendar": "seasonality-and-calendar", "smooth": "smoothing-without-leakage",
          "leak": "smoothing-without-leakage", "backtest": "choose-model-by-backtest", "choos": "choose-model-by-backtest",
          "timesfm": "timesfm-control-and-flat-forecasts", "flat": "timesfm-control-and-flat-forecasts",
          "driver": "drivers-and-covariates", "covariate": "drivers-and-covariates", "interval": "intervals-and-trust-statement",
          "trust": "intervals-and-trust-statement", "handover": "deliverables-and-handover", "deliverable": "deliverables-and-handover",
          "conformal": "forecast-combination-and-conformal", "combin": "forecast-combination-and-conformal",
          "kalman": "kalman-and-gradient-features", "knn": "knn-and-analogue-imputation", "imput": "knn-and-analogue-imputation",
          "market": "market-and-macro-research", "macro": "market-and-macro-research", "web": "web-research-playbook",
          "research": "web-research-playbook", "bayes": "bayesian-shrinkage-hierarchies", "shrink": "bayesian-shrinkage-hierarchies",
          "hierarch": "bayesian-shrinkage-hierarchies"}


def resolve_name(asked: str, names: list[str]) -> str | None:
    """The playbook a name refers to: exact, then with spaces and underscores as hyphens, then the closest name, then the
    first topic word it contains. None when nothing matches, so the caller is shown the list."""
    import difflib
    import re
    if asked in names:
        return asked
    norm = re.sub(r"[^a-z0-9]+", "-", asked.lower()).strip("-")
    if norm in names:
        return norm
    close = difflib.get_close_matches(norm, names, n=1, cutoff=0.75)
    if close:
        return close[0]
    for word, target in TOPICS.items():
        if word in norm and target in names:
            return target
    return None


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
