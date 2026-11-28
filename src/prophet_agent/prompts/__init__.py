"""Versioned prompt contracts: the planner and evaluator instructions live in files, not in code.

Motivation vs Logic
Motivation: a prompt change is a behaviour change. It must be reviewable as a diff, testable, and traceable to the
missions that ran under it.
Logic: each contract is ``<name>.v<N>.md`` with a first line ``version: <name>.v<N>``. ``load`` returns the text (without
the version line), the version id and a short sha256, and refuses a file whose first line disagrees with its name. The
goal loop records the ids in the ledger at the start of every mission.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
CURRENT = {"planner": "planner.v42", "evaluator": "evaluator.v13", "composer": "composer.v9", "compactor": "compactor.v2"}


@dataclass(frozen=True)
class Contract:
    version: str
    text: str
    sha256: str


def load(version: str) -> Contract:
    raw = (HERE / f"{version}.md").read_text(encoding="utf-8")
    first, _, body = raw.partition("\n")
    if first.strip() != f"version: {version}":
        raise ValueError(f"{version}.md must start with 'version: {version}'")
    return Contract(version, body.strip(), hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12])


def current(name: str) -> Contract:
    return load(CURRENT[name])
