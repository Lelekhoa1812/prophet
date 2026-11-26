"""Map header cells onto the claim. A missing measure stays unmapped.

Motivation vs Logic
Motivation: several numeric columns are not a reason to pick the last one.
Logic: a header matches a role only when its normalised text is one of the aliases the claim lists.
"""
from __future__ import annotations

import re

_PUNCT = re.compile(r"[^a-z0-9]+")


def normalise(value: str) -> str:
    return _PUNCT.sub("", str(value).casefold())


def choose_columns(claim: dict, headers: list[str]) -> dict:
    """Role -> header index. A role with no alias in the header is absent."""
    roles = {
        "time": claim.get("time_aliases") or [],
        "entity": claim.get("entity_aliases") or [],
        "measure": claim.get("measure_aliases") or [],
    }
    chosen = {}
    used = set()
    for role, aliases in roles.items():
        wanted = {normalise(item) for item in aliases if normalise(item)}
        for index, header in enumerate(headers):
            if index in used:
                continue
            if normalise(header) in wanted:
                chosen[role] = index
                used.add(index)
                break
    return {"mapped": chosen, "unmapped": [role for role in ("time", "measure") if role not in chosen and _required(claim, role)]}


def _required(claim: dict, role: str) -> bool:
    """A league table has one row per club and no date column, so only its measure is required."""
    if role == "time":
        return claim.get("table_kind") != "league"
    if role == "measure":
        return bool(claim.get("measure"))
    return False
