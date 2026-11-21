"""Analyst skills: short playbooks the planner reads on demand, not in the planner prompt.

Motivation vs Logic
Motivation: the planner prompt must stay short, but a principal analyst's how-to knowledge (decision rules, thresholds,
anti-patterns) is long. Each skill is one markdown file with a two-line front matter. The planner sees the names and
one-line summaries cheaply, and reads a single body only when it needs it (progressive disclosure).
Logic: files are named by slug and parsed the same way every time. ``read_skill`` returns the body with the front matter
removed. An unknown name raises KeyError listing the valid names, so a typo is corrected, not guessed.
"""
from __future__ import annotations

from pathlib import Path

SKILLS_DIR = Path(__file__).parent


def _parse(path: Path) -> tuple[dict[str, str], str]:
    """Front matter (``key: value`` lines up to the first blank line) and the body after it."""
    lines = path.read_text(encoding="utf-8").split("\n")
    meta: dict[str, str] = {}
    i = 0
    while i < len(lines) and lines[i].strip():
        key, _, value = lines[i].partition(":")
        meta[key.strip()] = value.strip()
        i += 1
    return meta, "\n".join(lines[i:]).strip()


def _index() -> dict[str, tuple[str, Path]]:
    out: dict[str, tuple[str, Path]] = {}
    for path in sorted(SKILLS_DIR.glob("*.md")):
        meta, _ = _parse(path)
        out[meta.get("name", path.stem)] = (meta.get("summary", ""), path)
    return out


def list_skills() -> list[dict]:
    """Every skill as {name, summary}, sorted by name."""
    return [{"name": name, "summary": summary} for name, (summary, _) in _index().items()]


def read_skill(name: str) -> str:
    """The body of one skill (front matter removed). KeyError lists the valid names for an unknown one."""
    index = _index()
    if name not in index:
        raise KeyError(f"unknown skill {name!r}; valid names: {', '.join(sorted(index))}")
    return _parse(index[name][1])[1]


__all__ = ["SKILLS_DIR", "list_skills", "read_skill"]
