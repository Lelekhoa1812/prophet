"""Per-notebook data-sharing controls (A7-12).

A notebook may switch web research off and may limit which configured model providers receive its data. The defaults keep
everything on. The rules here are pure: the service stores the setting and applies them when a turn is created, a mission
starts, continues or is answered now, and before an automatic part runs.
"""
from __future__ import annotations

from typing import Any

from prophet_agent.nim.providers import PROVIDERS
from prophet_agent.tools.research_tools import tools as research_tool_pairs

# Every tool that sends a query or a page request to the web. Taken from the research module, so a new research tool is
# covered without a list to keep in step.
RESEARCH_TOOL_NAMES: frozenset[str] = frozenset(tool.name for tool, _ in research_tool_pairs())

DEFAULT_SHARING: dict[str, Any] = {"web_research": True, "providers": None}   # None: every configured provider


class PolicyRefused(ValueError):
    """A turn, start or continue that the notebook's sharing settings do not allow. A ValueError, so the API maps it to 409."""


def normalise_patch(body: dict) -> dict:
    """Check a PUT body. Only the keys present change. Raises ValueError with a message a person can act on."""
    out: dict[str, Any] = {}
    if "web_research" in body:
        if not isinstance(body["web_research"], bool):
            raise ValueError("web_research must be true or false")
        out["web_research"] = body["web_research"]
    if "providers" in body:
        value = body["providers"]
        if value is None:
            out["providers"] = None
        elif isinstance(value, list) and all(isinstance(p, str) for p in value):
            unknown = sorted(set(value) - set(PROVIDERS))
            if unknown:
                raise ValueError(f"unknown provider: {', '.join(unknown)}")
            out["providers"] = sorted(set(value))
        else:
            raise ValueError("providers must be a list of provider names, or null for all")
    unknown_keys = sorted(set(body) - {"web_research", "providers"})
    if unknown_keys:
        raise ValueError(f"unknown field: {', '.join(unknown_keys)}")
    return out


def tools_allowed(tools: dict, sharing: dict) -> dict:
    """The mission's tool set under the notebook's rules. Research tools are left out when web research is off."""
    if sharing["web_research"]:
        return tools
    return {name: tool for name, tool in tools.items() if name not in RESEARCH_TOOL_NAMES}
