"""Score a run from its tool results. The score does not read the question text.

Motivation vs Logic
Motivation: a future point is not knowable yet. The run can still be scored on whether it bound a series it had accepted.
Logic: each check is true or false from the tool records. A bind with a name and a status other than accepted fails.
"""
from __future__ import annotations


def process_score(events: list[dict]) -> dict:
    """``events`` are dicts with ``tool`` and the tool result fields. The question text is not an input."""
    claims = [event for event in events if event.get("tool") == "state_table_claim" and not event.get("error")]
    verdicts = [event for event in events if event.get("tool") == "validate_table"]
    binds = [event for event in events if event.get("tool") == "bind_series"]
    diagnoses = [event for event in events if event.get("tool") == "diagnose_source"]
    false_bind = any(event.get("bound_name") and event.get("status") != "accepted" for event in verdicts + binds)
    accepted = any(event.get("status") == "accepted" and event.get("bound_name") for event in binds)
    known = {
        "auth_denied", "not_found", "html_page", "unsupported_type", "not_a_table", "wrong_entity",
        "schema_gap", "entity_unstated", "accepted",
    }
    checks = {
        "claim_stored": bool(claims),
        "no_false_bind": not false_bind,
        "signals_closed": all(event.get("signal") in known for event in diagnoses),
        "bind_follows_accept": _bind_follows(verdicts, binds),
    }
    return {"passed": all(checks.values()), "checks": checks, "accepted_bind": accepted}


def _bind_follows(verdicts: list[dict], binds: list[dict]) -> bool:
    """A named bind exists only when some verdict for that series was accepted."""
    for event in binds:
        if not event.get("bound_name"):
            continue
        if event.get("status") != "accepted":
            return False
        if not any(item.get("status") == "accepted" and item.get("series_name") == event.get("series_name") for item in verdicts):
            return False
    return True
