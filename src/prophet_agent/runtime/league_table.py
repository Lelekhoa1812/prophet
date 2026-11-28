"""The simulated final table of every team, written by code from the stored simulate_season result.

Motivation vs Logic
Motivation (audit of round 20261128-173823, EPL): the planner's view of simulate_season holds six team rows (it must fit
one observation), and the goal asked for the final table of twenty. The answer named Arsenal favourites on 86.2 points;
the simulation had Brighton at 44.9 per cent for the title and Arsenal on 78.9 points, and 19 of 20 rows differed from it by
3 to 26 points. The figure check flagged them, but the composer could not repair a table it never saw.
Logic: from the newest successful simulate_season whose full result lists the teams, a Markdown table of every team in
order of expected final points: points now, expected final points, the 10-90 range of the final total, expected position,
and each probability the rows carry as a percentage. It is appended to the answer after grounding, like the accuracy
paragraph, so its figures are the ledger's own; the composer is told it will be there.
"""
from __future__ import annotations

from typing import Any

MIN_TEAMS = 10          # a table is appended for a league season, not for a two-team question
HEADER = "Simulated final table (computed from the season simulation, every team)"
_LABELS = {"title": "Title", "top_places": "Top places", "relegation": "Relegation", "playoff": "Play-offs",
           "playoffs": "Play-offs"}


def newest_season(history: list[dict]) -> dict | None:
    """The full result of the newest successful simulate_season call that lists its teams."""
    for h in reversed(history):
        if not isinstance(h, dict) or h.get("kind") != "tool" or h.get("tool") != "simulate_season" or "error" in h:
            continue
        full = h.get("full") if isinstance(h.get("full"), dict) else h.get("result")
        if isinstance(full, dict) and isinstance(full.get("teams"), list):
            return full
    return None


def _pct(v: Any) -> str:
    return f"{100 * float(v):.1f}%" if isinstance(v, (int, float)) else ""


def _num(v: Any, nd: int = 1) -> str:
    return f"{float(v):.{nd}f}" if isinstance(v, (int, float)) else ""


def table_markdown(history: list[dict]) -> str:
    """The appended table, or "" when there is no season simulation of at least MIN_TEAMS teams."""
    full = newest_season(history)
    if full is None:
        return ""
    rows = [r for r in full["teams"] if isinstance(r, dict) and r.get("team")]
    if len(rows) < MIN_TEAMS:
        return ""
    probs = [k for k in _LABELS if any(isinstance(r.get(k), (int, float)) for r in rows)]
    probs = list(dict.fromkeys(_LABELS[k] for k in probs))
    keys = {lab: [k for k, v in _LABELS.items() if v == lab] for lab in probs}
    rows.sort(key=lambda r: -float(r.get("expected_final") or 0))
    head = ["Pos", "Team", "Points now", "Expected final", "10-90 range", "Expected position", *probs]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, r in enumerate(rows, start=1):
        fin = r.get("final") if isinstance(r.get("final"), dict) else {}
        rng = f"{_num(fin.get('p10'), 0)} to {_num(fin.get('p90'), 0)}" if fin else ""
        cells = [str(i), str(r["team"]), _num(r.get("current"), 0), _num(r.get("expected_final")), rng,
                 _num(r.get("expected_position"))]
        for lab in probs:
            cells.append(next((_pct(r[k]) for k in keys[lab] if isinstance(r.get(k), (int, float))), ""))
        lines.append("| " + " | ".join(cells) + " |")
    s = full.get("settings") if isinstance(full.get("settings"), dict) else {}
    sig = s.get("signals") if isinstance(s.get("signals"), dict) else {}
    basis = ", ".join(f"{k} {v}" for k, v in sig.items() if isinstance(v, (int, float)))
    tail = (f"{s.get('n_sims')} simulated seasons from {s.get('n_played')} played and {s.get('n_remaining')} remaining "
            f"matches" if s.get("n_sims") else "")
    note = "; ".join(x for x in (tail, f"signals counted: {basis}" if basis else "") if x)
    return f"{HEADER}:\n\n" + "\n".join(lines) + (f"\n\n{note}." if note else "")
