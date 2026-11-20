"""Structured questions in the AskUserQuestion shape, validated before the run pauses.

Motivation vs Logic
Motivation: when the agent needs a decision, the question must be answerable in a click. A malformed question
(too many options, an overlong header, no question mark) wastes the user's time and makes the UI guess.
Logic: one schema for the model to fill, and one validator that the runtime applies before pausing. The validator is
the authority; the schema only helps the model produce valid output on the first try.
"""
from __future__ import annotations

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
    return problems
