"""Structured questions in the AskUserQuestion shape, validated before the run pauses.

Motivation vs Logic
Motivation: when the agent needs a decision, the question must be answerable in a click. A malformed question
(too many options, an overlong header, no question mark) wastes the user's time and makes the UI guess.
Logic: one schema for the model to fill, and one validator that the runtime applies before pausing. The validator is
the authority; the schema only helps the model produce valid output on the first try.
"""
from __future__ import annotations

import json
import re
from typing import Any

MAX_QUESTIONS = 4
MIN_OPTIONS = 2
MAX_OPTIONS = 4
MAX_HEADER = 12

QUESTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["questions"],
    "additionalProperties": False,
    "properties": {
        "questions": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_QUESTIONS,
            "items": {
                "type": "object",
                "required": ["question", "header", "options", "multiSelect"],
                "additionalProperties": False,
                "properties": {
                    "question": {"type": "string", "minLength": 1},
                    "header": {"type": "string", "minLength": 1, "maxLength": MAX_HEADER},
                    "multiSelect": {"type": "boolean"},
                    "options": {
                        "type": "array",
                        "minItems": MIN_OPTIONS,
                        "maxItems": MAX_OPTIONS,
                        "items": {
                            "type": "object",
                            "required": ["label", "description"],
                            "additionalProperties": False,
                            "properties": {
                                "label": {"type": "string", "minLength": 1, "maxLength": 60},
                                "description": {"type": "string", "maxLength": 240},
                            },
                        },
                    },
                },
            },
        }
    },
}


EXAMPLE_QUESTIONS: list[dict[str, Any]] = [{
    "question": "Which horizon should the forecast cover?", "header": "Horizon", "multiSelect": False,
    "options": [{"label": "4 weeks", "description": "Short range"}, {"label": "12 weeks", "description": "One quarter"}]}]


def expected_shape() -> str:
    """The shape an ask_user call must take, with a minimal example, for an error the planner can act on in one step."""
    return ("Send the questions as a list of 1 to 4 objects, each with question (ending in '?'), header (12 characters or "
            "fewer), multiSelect (true or false) and options (2 to 4 objects with label and description). For example: "
            + json.dumps({"questions": EXAMPLE_QUESTIONS}, separators=(",", ":")))


def repair_questions(raw: Any) -> tuple[Any, list[str]]:
    """The questions of an ask_user call with the easy shape errors fixed, and a note for each fix.

    Motivation vs Logic
    Motivation: live Nvidia mission (2026-11-26), an ask_user call failed with "questions must be a list of 1 to 4 items" because
    the planner sent one question object, not a list. The person was not asked and a planner step was spent. A single object, a
    JSON string of either, or more than four questions are shape errors the runtime can repair without another model call.
    Logic: a JSON string is decoded; a single object (or one holding a "questions" value) becomes a list of one; a list longer
    than MAX_QUESTIONS keeps its first four and notes the rest. Anything else is returned unchanged for validate_questions to
    explain, with expected_shape() in the error.
    """
    notes: list[str] = []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return raw, notes
        notes.append("the questions were sent as text and decoded")
    if isinstance(raw, dict) and isinstance(raw.get("questions"), (list, dict)):
        raw = raw["questions"]                  # the payload itself was sent instead of its questions
    if isinstance(raw, dict):
        raw = [raw]
        notes.append("a single question object was wrapped in a list")
    if isinstance(raw, list) and len(raw) > MAX_QUESTIONS:
        notes.append(f"{len(raw)} questions were sent; the first {MAX_QUESTIONS} are asked, and the others can be asked "
                     "after these answers")
        raw = raw[:MAX_QUESTIONS]
    return raw, notes


def validate_questions(payload: Any) -> list[str]:
    """Return a list of problems. An empty list means the payload is a valid question set."""
    problems: list[str] = []
    if not isinstance(payload, dict) or set(payload) - {"questions"} or "questions" not in payload:
        return ["payload must be an object with only a 'questions' key"]
    qs = payload["questions"]
    if not isinstance(qs, list) or not 1 <= len(qs) <= MAX_QUESTIONS:
        return [f"questions must be a list of 1 to {MAX_QUESTIONS} items"]
    for i, q in enumerate(qs):
        where = f"questions[{i}]"
        if not isinstance(q, dict):
            problems.append(f"{where} must be an object")
            continue
        text = q.get("question")
        if not isinstance(text, str) or not text.strip():
            problems.append(f"{where}.question must be a non-empty string")
        elif not text.rstrip().endswith("?"):
            problems.append(f"{where}.question should end with a question mark")
        header = q.get("header")
        if not isinstance(header, str) or not header.strip():
            problems.append(f"{where}.header must be a non-empty string")
        elif len(header) > MAX_HEADER:
            problems.append(f"{where}.header is {len(header)} characters; the limit is {MAX_HEADER}")
        if not isinstance(q.get("multiSelect"), bool):
            problems.append(f"{where}.multiSelect must be a boolean")
        opts = q.get("options")
        if not isinstance(opts, list) or not MIN_OPTIONS <= len(opts) <= MAX_OPTIONS:
            problems.append(f"{where}.options must be a list of {MIN_OPTIONS} to {MAX_OPTIONS} items")
            continue
        labels = []
        for j, o in enumerate(opts):
            if not isinstance(o, dict) or not isinstance(o.get("label"), str) or not o["label"].strip():
                problems.append(f"{where}.options[{j}].label must be a non-empty string")
                continue
            labels.append(o["label"])
            if len(o["label"]) > 60:
                problems.append(f"{where}.options[{j}].label is longer than 60 characters")
            if not isinstance(o.get("description", ""), str):
                problems.append(f"{where}.options[{j}].description must be a string")
        if len(set(labels)) != len(labels):
            problems.append(f"{where}.options labels must be distinct")
        for a in range(len(labels)):
            for b in range(a + 1, len(labels)):
                if _near_duplicates(labels[a], labels[b]):
                    problems.append(f"{where}.options {labels[a]!r} and {labels[b]!r} are near-duplicates; make each "
                                    "option a clearly different choice")
    return problems


def _terms(label: str) -> frozenset[str]:
    """Words of a label, lower-cased, with a plural 's' dropped, so 'Monthly totals' and 'Monthly total' match."""
    words = re.findall(r"[a-z0-9]+", label.lower())
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words)


def _split(label: str) -> tuple[str, str]:
    """A label's main words and its parenthetical qualifier, e.g. 'Day-first (DD/MM/YYYY)'."""
    main = re.sub(r"\([^)]*\)", " ", label)
    qualifier = " ".join(re.findall(r"\(([^)]*)\)", label))
    return main, qualifier


def _similar(x: str, y: str) -> bool:
    tx, ty = _terms(x), _terms(y)
    if not tx or not ty:
        return False
    return tx == ty or len(tx & ty) / len(tx | ty) >= 0.6


def _near_duplicates(a: str, b: str) -> bool:
    # The main words decide whether two options are the same choice. A parenthetical is an example or a qualifier:
    # 'Day-first (DD/MM/YYYY)' and 'Month-first (MM/DD/YYYY)' share the format text but are different choices, so the
    # qualifiers only count when both labels have one that is similar.
    (main_a, qual_a), (main_b, qual_b) = _split(a), _split(b)
    if not _terms(main_a) or not _terms(main_b):
        return _similar(a, b)
    if not _similar(main_a, main_b):
        return False
    return not (qual_a and qual_b) or _similar(qual_a, qual_b)


# Mission 9 (task 2): a question may not offer invented data in place of the real series the person asked for.
# Live (Nvidia, baseline run 2026-11-26): the planner asked "Use synthetic data based on known NVDA price ranges" as an option.
INVENTED_DATA = re.compile(
    r"\b(?:synthetic|simulated|simulation|made[- ]up|invented|fabricated|fake|placeholder|illustrative)\s+"
    r"(?:[\w/-]+\s+){0,3}?(?:data|prices?|series|closes?|values?|history|figures|numbers|returns|observations)\b"
    r"|\b(?:approximat\w*|rough|reconstructed)\s+(?:[\w/-]+\s+)?(?:prices?|closes?|series|history|data|values?|figures)\b",
    re.I)
NEGATED = re.compile(r"\b(?:not|no|never|without|none|neither|nor|cannot|don't|do not|did not|was not|were not)\b", re.I)
# A goal that itself asks for a synthetic or simulated series (a twin, a test, a mock) may be offered one.
SYNTHETIC_GOAL = re.compile(r"synthetic|simulat|\btwins?\b|made[- ]up|\bfake\b|\bdummy\b|\bmock\b|sample data|test data", re.I)
SYNTHETIC_OPTION_PROBLEM = ("{where} offers {phrase!r} in place of the real series. Remove that option, and keep researching "
                            "public sources for the real series (price endpoints, filings, official tables). A synthetic "
                            "series may be offered only for a test the person asked for.")


def invented_data_phrase(text: str) -> str | None:
    """The first phrase in ``text`` that offers invented, simulated or approximate data, or None.

    Motivation vs Logic
    Motivation: an option such as "Use synthetic data based on known price ranges" asks the person to accept a made-up series
    in place of the real one. The person has not asked for that, and the answer would then be a forecast of invented numbers.
    Logic: a phrase of an invented-data word (synthetic, simulated, made-up, approximate, rough) next to a data word (data, prices,
    series, closes, values). A sentence that negates the phrase ("do not use synthetic data") is not an offer and is skipped.
    """
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", text or ""):
        if NEGATED.search(sentence):
            continue
        m = INVENTED_DATA.search(sentence)
        if m:
            return " ".join(m.group(0).split())
    return None


def synthetic_substitute_problems(payload: Any, goal: str = "") -> list[str]:
    """Problems for each question option that offers invented data in place of the real series. Empty when the goal itself asks
    for a synthetic or simulated series, because then the option is the test the person asked for."""
    if not isinstance(payload, dict) or not isinstance(payload.get("questions"), list) or SYNTHETIC_GOAL.search(goal or ""):
        return []
    problems: list[str] = []
    for i, q in enumerate(payload["questions"]):
        if not isinstance(q, dict):
            continue
        for j, o in enumerate(q.get("options") or []):
            if not isinstance(o, dict):
                continue
            text = f"{o.get('label') or ''}. {o.get('description') or ''}"
            phrase = invented_data_phrase(text)
            if phrase:
                problems.append(SYNTHETIC_OPTION_PROBLEM.format(where=f"questions[{i}].options[{j}]", phrase=phrase))
    return problems


def validate_for_goal(payload: Any, goal: str = "") -> list[str]:
    """The shape problems of validate_questions plus the invented-data refusals. The one check an ask_user call must pass."""
    return validate_questions(payload) + synthetic_substitute_problems(payload, goal)
