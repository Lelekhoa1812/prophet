"""Agent types: what a delegated sub-agent may use, know and be asked to return.

Motivation vs Logic
Motivation: one prompt cannot make a good researcher, a careful analyst and a sceptical verifier at the same time. Claude
Code, Codex and OpenCode all define agent types as small files (a description, a tool allowlist, a prompt) so the parent can
pick a role and the role limits what the agent can touch. Prophet does the same, in code, because its tools are code.
Logic: an ``AgentType`` names the tools it may call (always intersected with the tools the mission has), the playbooks it is
shown first, a short role prompt, and a step budget. The parent chooses a type and writes a brief; the brief is what the
sub-agent works from, because it does not see the parent's history. A type never widens the parent's tools. ``notes`` is
every agent's own working memory and is added to each.
"""
from __future__ import annotations

from dataclasses import dataclass

from prophet_agent.skills import read_skill

MAX_PARALLEL = 4          # sub-agents that may run at once from one plan step
MAX_AGENTS = 24           # sub-agents a mission may start in total (cost control: each is a full loop of model calls)


@dataclass(frozen=True)
class AgentType:
    name: str
    description: str
    tools: tuple[str, ...]
    skills: tuple[str, ...] = ()
    steps: int = 12
    role: str = ""


_COMMON = ("notes", "analyst_skill")

TYPES: dict[str, AgentType] = {t.name: t for t in (
    AgentType(
        "general", "A flexible helper for a self-contained step that fits no other type. Gets every tool the mission has.",
        tools=("*",), steps=12,
        role="You are a general sub-agent. Do the brief, nothing more, and report what you found with the evidence."),
    AgentType(
        "researcher",
        "Finds and reads outside sources on the web (news, policy, central banks, market reports) and returns dated, "
        "cited findings. Use before forecasting anything driven by events: FX, commodities, equities, demand tied to "
        "policy, weather or news.",
        tools=("web_research", "web_search", "web_fetch", "download_web_data", "download_market_prices", *_COMMON),
        skills=("web-research-playbook",), steps=14,
        role="You are a research sub-agent. Plan several different angles first, search them together, read the best "
             "passages, follow up on what is missing, and stop when new searches add nothing. Every claim you report has "
             "a source id, a date and a quote that appears in the fetched text. Say what you could not find. Never put a "
             "number in your report that a tool did not print."),
    AgentType(
        "analyst",
        "Takes one slice of the data (one column, entity, period or defect) and analyses it down to the atomic level with "
        "Python: profile, decompose, test, compare, and report what it means for the forecast.",
        tools=("sandbox_python", "inspect_data", "detect_anomalies", "build_drivers", "save_table", *_COMMON),
        skills=("first-contact-profile",), steps=14,
        role="You are an analysis sub-agent. Work in small cells that print what they find. Read the data before you "
             "decide. Report findings with the numbers the cells printed, the decision you recommend and why, and the "
             "files you wrote (run id and path)."),
    AgentType(
        "modeler",
        "Runs forecasts and backtests and measures what helps TimesFM: baselines, control options, combinations.",
        tools=("sandbox_python", "timesfm_forecast", "backtest_series", "panel_backtest", "control_lab", "forecast_report",
               *_COMMON),
        skills=("choose-model-by-backtest", "timesfm-control-and-flat-forecasts"), steps=14,
        role="You are a modelling sub-agent. Compare against baselines on the same origins, adopt a treatment only when "
             "the lab or backtest says it helped, and report scores with the baseline each was measured against."),
    AgentType(
        "verifier",
        "A sceptical second reader. Re-computes the figures and checks the claims of a draft answer or another agent's "
        "report against the data and tool outputs, and returns what holds and what does not.",
        tools=("sandbox_python", "inspect_data", "backtest_series", "validate_table", "bind_series", *_COMMON), steps=10,
        role="You are a verification sub-agent. You do not build anything new. For each claim you were given, re-compute "
             "it from the data or from a stored output (carry the earlier run's file), and report holds, fails or cannot "
             "check, with the number you got. A claim you cannot reproduce is reported as such, never as holding. "
             "bind_series only when validate_table has accepted that same file."),
    AgentType(
        "principal_analyst",
        "States what table the question needs, collects pages, and reads why a download failed. It does not bind the "
        "series name and it does not forecast.",
        tools=("web_search", "web_fetch", "web_research", "state_table_claim", "diagnose_source", "extract_page_tables",
               "download_web_data", "download_market_prices", *_COMMON),
        steps=12,
        role="You state the table claim, collect pages, and read each diagnosis. A file name is not the series. "
             "You do not bind a series and you do not forecast. Report the claim, the signals, and the candidate ids."),
    AgentType(
        "data_scientist",
        "Maps columns, reshapes a candidate in the sandbox, and checks it. A cell that runs is not an accepted table.",
        tools=("sandbox_python", "inspect_data", "validate_table", "fuse_sources", "calibrate_overlap", *_COMMON),
        steps=12,
        role="You map columns to the claim, reshape bytes in the sandbox, and call validate_table. You do not download "
             "and you do not bind. Report the verdict and the files the cells wrote."),
)}


def get(name: str | None) -> AgentType | None:
    return TYPES.get(str(name or "general").strip().lower())


def describe() -> str:
    """The agent types as planner-facing lines."""
    return "\n".join(f"- {t.name}: {t.description} Tools: {'all' if t.tools == ('*',) else ', '.join(t.tools)}."
                     for t in TYPES.values())


def allowed(agent: AgentType, available: dict) -> dict:
    """The mission's tools this type may use."""
    if agent.tools == ("*",):
        return dict(available)
    return {n: available[n] for n in agent.tools if n in available}


def playbooks(agent: AgentType) -> str:
    """The playbooks an agent type is shown first, as text. A missing playbook is skipped, not an error: a type can name
    one before it is written."""
    parts = []
    for name in agent.skills:
        try:
            parts.append(f"Playbook {name}:\n{read_skill(name)}")
        except KeyError:
            continue
    return "\n\n".join(parts)
