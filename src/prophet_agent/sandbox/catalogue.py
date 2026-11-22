"""A signature catalogue of the helpers the planner calls, generated from the code so it cannot drift.

Motivation vs Logic
Motivation: the planner guessed helper names and keyword arguments in about half of its live cells (save_series(name=),
hampel(n_sigma=), scan(time=)). The fix is to show the exact signatures in the spec the planner reads, and the spec must
never disagree with the code.
Logic: ``build_catalogue`` reads each listed helper with inspect and writes one line per helper: the qualified name, the
parameters with their defaults, and the return type taken from the return annotation. The list holds the helpers used
most in the ledger (tool_calls for sandbox_python), plus the panel helpers. The file is not copied into the sandbox
(runner copies only prophet_*.py), so it runs in the main process only.
"""
from __future__ import annotations

import inspect
import re
from typing import Any

from prophet_agent.sandbox import prophet_lib, prophet_scan, prophet_stats

# (alias the planner uses, module, function name). Order is the order in the spec.
CATALOGUE: tuple[tuple[str, Any, str], ...] = (
    ("pl", prophet_lib, "load"),
    ("pl", prophet_lib, "save_series"),
    ("pl", prophet_lib, "prepare_series"),
    ("pl", prophet_lib, "panel"),
    ("pl", prophet_lib, "save_panel"),
    ("pl", prophet_lib, "load_series"),
    ("sc", prophet_scan, "scan"),
    ("sc", prophet_scan, "write_scan"),
    ("ps", prophet_stats, "changepoints"),
    ("ps", prophet_stats, "hampel"),
    ("ps", prophet_stats, "stl_outliers"),
    ("pl", prophet_lib, "detect_anomalies"),
    ("pl", prophet_lib, "dominant_period"),
    ("pl", prophet_lib, "find_season"),
    ("pl", prophet_lib, "read_table"),
    ("pl", prophet_lib, "build_series"),
    ("pl", prophet_lib, "chart"),
    ("pl", prophet_lib, "list_holidays"),
)
# The class the planner uses most (ps.Ledger, 192 cells in the ledger) gets its own line.
LEDGER = ("ps", prophet_stats, "Ledger")
_LEDGER_METHODS = ("add", "save", "to_dict")
_LEDGER_PROPERTIES = ("entries",)

_RETURN_TEXT = re.compile(r"\b(?:pd|np|prophet_\w+)\.")


def _signature_text(fn: Any) -> str:
    """Parameter list without annotations, as the planner types it: ``x, window=7, causal=True``."""
    parts: list[str] = []
    keyword_only_marked = False
    params = [p for p in inspect.signature(fn).parameters.values() if p.name != "self"]
    for p in params:
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            parts.append("*" + p.name)
            keyword_only_marked = True
            continue
        if p.kind is inspect.Parameter.KEYWORD_ONLY and not keyword_only_marked:
            parts.append("*")
            keyword_only_marked = True
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            parts.append("**" + p.name)
            continue
        parts.append(p.name if p.default is inspect.Parameter.empty else f"{p.name}={p.default!r}")
    return ", ".join(parts)


def _return_text(fn: Any) -> str:
    """The return annotation in short form: ``tuple[pd.Series, dict]`` becomes ``(Series, dict)``."""
    ann = inspect.signature(fn).return_annotation
    if ann is inspect.Signature.empty:
        return "None"
    text = ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann))
    text = _RETURN_TEXT.sub("", text).replace(" ", "")
    if text.startswith("tuple[") and text.endswith("]"):
        text = "(" + text[len("tuple["):-1] + ")"
    return text


def signature_line(alias: str, fn: Any, name: str) -> str:
    """One catalogue line for a function: ``alias.name(params) -> returns``."""
    return f"{alias}.{name}({_signature_text(fn)}) -> {_return_text(fn)}"


def ledger_line() -> str:
    """One catalogue line for ps.Ledger: its constructor and its methods and property."""
    alias, module, cls_name = LEDGER
    cls = getattr(module, cls_name)
    methods = ", ".join(f"{m}({_signature_text(getattr(cls, m))})" for m in _LEDGER_METHODS)
    props = ", ".join(_LEDGER_PROPERTIES)
    return f"{alias}.{cls_name}() -> methods {methods}; read-only property {props} (list of dicts)"


def build_catalogue() -> str:
    """The whole catalogue, one helper per line, in the order of CATALOGUE, then the Ledger line."""
    lines = [signature_line(alias, getattr(module, name), name) for alias, module, name in CATALOGUE]
    lines.append(ledger_line())
    return "\n".join(lines)


def catalogue_entries() -> list[tuple[str, Any, str]]:
    """The (alias, module, name) triples the catalogue lists, for tests and the smoke run."""
    return list(CATALOGUE)
