"""JSON-schema helpers for NIM's strict grammar.

Motivation vs Logic
Motivation: NIM's strict `json_schema` mode rejects keywords it cannot express (`uniqueItems`, `$schema`, `title`, and
whatever the 400 body names). Stripping `title` everywhere once deleted a *property called title* from a real schema,
so every answer was rejected. Keywords and property names live in different places and must be treated differently.
Logic: `strip_schema` removes keywords only; the values of `properties`, `patternProperties`, `$defs`, `definitions`
and `dependentSchemas` are name->schema maps whose keys are data and are never touched. `validate` always checks the
ORIGINAL schema locally, because server-side enforcement is best effort.
"""
from __future__ import annotations

import json
import re
from typing import Any

import jsonschema

STRIP_ALWAYS = frozenset({"uniqueItems", "$schema", "$id", "title"})
NAME_MAPS = frozenset({"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"})
_UNIMPL = re.compile(r"Unimplemented keys?:\s*(\[[^\]]*\])", re.I)


def strip_schema(node: Any, drop: frozenset[str] = STRIP_ALWAYS) -> Any:
    """Drop grammar keywords, never property names."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k in drop:
                continue
            if k in NAME_MAPS and isinstance(v, dict):
                out[k] = {name: strip_schema(sub, drop) for name, sub in v.items()}
            else:
                out[k] = strip_schema(v, drop)
        return out
    if isinstance(node, list):
        return [strip_schema(i, drop) for i in node]
    return node


def unimplemented_keys(body_text: str) -> set[str]:
    """Parse `Unimplemented keys: ["minItems"]` out of a 400 body. Empty set when the text does not say."""
    # The 400 body is JSON, so the list arrives with escaped quotes (\"minItems\"); unescape before matching
    m = _UNIMPL.search((body_text or "").replace('\\"', '"'))
    if not m:
        return set()
    try:
        return {str(k) for k in json.loads(m.group(1).replace("'", '"'))}
    except ValueError:
        return set()


def is_format_error(body_text: str) -> bool:
    t = (body_text or "").lower()
    return any(s in t for s in ("grammar", "response_format", "json_schema", "guided", "bad request"))


def validate(instance: Any, schema: dict) -> str | None:
    """None when valid, otherwise a one-line reason (first error, path included)."""
    try:
        jsonschema.validate(instance, schema)
        return None
    except jsonschema.ValidationError as e:
        path = "/".join(str(p) for p in e.absolute_path) or "(root)"
        return f"{path}: {e.message}"[:300]
    except jsonschema.SchemaError as e:
        return f"invalid schema: {e.message}"[:300]
