"""Dollars-protected what-if: what the alerts shown in a replay could be worth, under stated assumptions.

Motivation vs Logic
Motivation: executives ask 'what is it worth'. The replay knows which problems an alert matched (flagged), not what a
person then did. Reporting flagged dollars as saved would overclaim.
Logic: expected net = acceptance x effectiveness x dollars of recorded problems the alerts matched
                      - review cost for every alert shown
                      - acceptance x false-alarm cost for every shown alert that matched nothing.
Only agents with a recorded dollar truth are valued; the others are listed as unmeasured, with their review cost shown but
not netted. The alert's own claimed dollars are shown beside the matched truth, because summing claimed dollars overstates
(one stockout can be alerted on several days).
Acceptance (a person acts on a shown alert) and effectiveness (share of the problem's dollars recovered when they do)
are assumptions the reader sets; nothing here is observed. It also ignores that an alert can arrive after the loss
(median alert delay is days for shortfall), so it leans optimistic. A live shadow period replaces the assumptions.
"""
from __future__ import annotations

GRID_ACCEPTANCE = (0.25, 0.5, 0.75, 1.0)
GRID_EFFECTIVENESS = (0.25, 0.5, 0.75)

CAVEAT = ("Expected value under stated assumptions on a synthetic replay. Not observed savings. Alerts that arrive after the "
          "loss are still counted, so it leans optimistic.")


def measured(s: dict) -> bool:
    """Whether the replay has a recorded dollar truth for this agent.

    Motivation vs Logic: customer and margin dollars are estimates from a pre-event baseline (sim/event_dollars.py). Putting
    them in the headline net would mix estimates with recorded losses, so they are reported separately and left unmeasured here.
    """
    if s.get("dollar_basis") == "estimate":
        return False
    return float(s.get("dollars_truth_total", 0.0) or 0.0) > 0.0


def _net(agents: dict, econ, a: float, e: float) -> tuple[list[dict], dict]:
    rows = []
    for name, s in agents.items():
        if not measured(s):
            rows.append({"agent": name, "shown": int(s.get("proposals", 0)), "useful": int(s.get("useful", 0)), "value_measured": False,
                         "review_cost": round(int(s.get("proposals", 0)) * econ.review_cost, 2)})
            continue
        shown, useful = int(s.get("proposals", 0)), int(s.get("useful", 0))
        gross = float(s.get("dollars_truth_caught", 0.0)) * a * e
        review = shown * econ.review_cost
        false = max(shown - useful, 0) * a * econ.false_alarm_cost
        rows.append({"agent": name, "shown": shown, "useful": useful, "value_measured": True,
                     "claimed_dollars": round(float(s.get("dollars_at_stake_total", 0.0)), 2),
                     "truth_dollars_total": round(float(s.get("dollars_truth_total", 0.0)), 2), "truth_dollars_matched": round(float(s.get("dollars_truth_caught", 0.0)), 2),
                     "gross": round(gross, 2), "review_cost": round(review, 2), "false_alarm_cost": round(false, 2), "net": round(gross - review - false, 2)})
    m = [r for r in rows if r["value_measured"]]
    tot = {k: round(sum(r[k] for r in m), 2) for k in ("gross", "review_cost", "false_alarm_cost", "net")}
    return rows, tot


def whatif(agents: dict, econ, acceptance: float = 0.5, effectiveness: float | None = None) -> dict:
    e = econ.effectiveness if effectiveness is None else effectiveness
    rows, tot = _net(agents, econ, acceptance, e)
    # net(a) = a * (e*T - F) - R is linear in acceptance, so break-even is closed form
    ms = [s for s in agents.values() if measured(s)]
    T = sum(float(s.get("dollars_truth_caught", 0.0)) for s in ms)
    F = sum(max(int(s.get("proposals", 0)) - int(s.get("useful", 0)), 0) for s in ms) * econ.false_alarm_cost
    R = sum(int(s.get("proposals", 0)) for s in ms) * econ.review_cost
    slope = e * T - F
    be = R / slope if slope > 0 else None
    grid = [{"acceptance": a, "effectiveness": ee, "net": _net(agents, econ, a, ee)[1]["net"]} for ee in GRID_EFFECTIVENESS for a in GRID_ACCEPTANCE]
    return {"acceptance": acceptance, "effectiveness": e, "agents": rows, "total": tot,
            "break_even_acceptance": round(be, 4) if be is not None else None, "grid": grid,
            "unmeasured_agents": [r["agent"] for r in rows if not r["value_measured"]],
            "evidence": "Synthetic replay", "caveat": CAVEAT}


def problem_level(rows: list, gap_days: int = 10) -> dict:
    """Collapse alerts on the same thing into problems: alerts for one entity less than `gap_days` apart are one problem,
    claimed at the largest single alert. rows: (agent, entity_id, as_of 'YYYY-MM-DD', dollars_at_stake).

    Motivation: one stockout is alerted on several days, so summing alert dollars counts it several times (alert-level sums
    were 2.3-2.5x the real problem dollars on the twin). The gap matches the scorer's matching window.
    """
    from datetime import date
    by: dict[tuple, list] = {}
    for agent, ent, d, dollars in rows:
        by.setdefault((agent, ent), []).append((date.fromisoformat(str(d)[:10]), float(dollars)))
    out: dict[str, dict] = {}
    for (agent, _), items in by.items():
        items.sort()
        a = out.setdefault(agent, {"alerts": 0, "problems": 0, "claimed_dollars": 0.0})
        last, best = None, 0.0
        for d, dollars in items:
            a["alerts"] += 1
            if last is None or (d - last).days > gap_days:
                a["claimed_dollars"] += best
                a["problems"] += 1
                best = dollars
            else:
                best = max(best, dollars)
            last = d
        a["claimed_dollars"] += best
    for a in out.values():
        a["claimed_dollars"] = round(a["claimed_dollars"], 2)
    return out
