"""Where a tool's series comes from: numbers in the call, or a file an earlier analysis step wrote.

Motivation vs Logic
Motivation: a real series can have thousands of points. The planner should not have to copy them into a tool call, and a
copy can silently drop or alter values. Analysis code in the sandbox cleans and shapes the data, then hands the result
over by reference.
Logic: ``values`` is a list of numbers. ``source`` names a sandbox run and a file in its ``out/`` folder (written with
``prophet_lib.save_series``), and optionally one series inside it. The run id must be a hex id and the file must stay
inside ``out/``. Anything that is not a finite number is refused with a message that says what to fix.
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from prophet_agent.sandbox.runner import OUT_DIR, SANDBOX_ROOT

MAX_SERIES = 100_000
MAX_NAMED = 100
RUN_ID = re.compile(r"^[0-9a-f]{32}$")

VALUES_HELP = "list of numbers, oldest first (give this or source, not both)"
SOURCE_HELP = ("object {run_id: id of the analysis run that wrote the file, file: 'out/series.json', name: optional "
               "series name} to read the series from that run instead of values")


def _clean(name: str, raw: Any) -> np.ndarray:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"series {name!r} must be a non-empty list of numbers")
    if len(raw) > MAX_SERIES:
        raise ValueError(f"series {name!r} has {len(raw)} points; the limit is {MAX_SERIES}")
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"series {name!r} contains values that are not numbers")
    bad = int((~np.isfinite(arr)).sum())
    if bad:
        raise ValueError(f"series {name!r} has {bad} missing or non-finite values; fill or drop them in the analysis "
                         "step and save again, so the choice is explicit")
    return arr


def resolve_series(args: dict) -> list[tuple[str, np.ndarray]]:
    values, source = args.get("values"), args.get("source")
    if (values is None) == (source is None):
        raise ValueError("give exactly one of values or source")
    if values is not None:
        return [("series", _clean("series", values))]
    if not isinstance(source, dict) or not RUN_ID.match(str(source.get("run_id", ""))):
        raise ValueError("source needs run_id (the 32-character id of the analysis run that wrote the file)")
    rel = str(source.get("file", f"{OUT_DIR}/series.json"))
    base = (SANDBOX_ROOT / source["run_id"]).resolve()
    target = (base / rel).resolve()
    if not str(target).startswith(str((base / OUT_DIR).resolve()) + "/") or not target.is_file():
        raise ValueError(f"no file {rel!r} in the outputs of run {source['run_id']}")
    try:
        doc = json.loads(target.read_text(encoding="utf-8"))
    except ValueError:
        raise ValueError(f"{rel} is not valid JSON; write it with prophet_lib.save_series")
    body = doc.get("series") if isinstance(doc, dict) else doc
    if isinstance(body, list):
        named = {"series": body}
    elif isinstance(body, dict):
        named = body
    else:
        raise ValueError(f"{rel} must hold a list of numbers or a dict of named lists")
    want = source.get("name")
    if want is not None:
        if want not in named:
            raise ValueError(f"series {want!r} not in the file; available: {sorted(named)[:20]}")
        named = {want: named[want]}
    if len(named) > MAX_NAMED:
        raise ValueError(f"{len(named)} series in the file; the limit per call is {MAX_NAMED}. Pick some with source.name")
    return [(str(k), _clean(str(k), v)) for k, v in named.items()]


def source_covariates(args: dict) -> dict[str, Any]:
    """Named covariates stored with a series file (``covariates`` key written by ``prophet_lib.save_series``). Empty when
    there is no source or the file holds none; lengths are checked later against the horizon."""
    source = args.get("source")
    if not isinstance(source, dict) or not RUN_ID.match(str(source.get("run_id", ""))):
        return {}
    try:
        target = (SANDBOX_ROOT / source["run_id"] / str(source.get("file", f"{OUT_DIR}/series.json"))).resolve()
        doc = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    cov = doc.get("covariates") if isinstance(doc, dict) else None
    return cov if isinstance(cov, dict) else {}


def run_dates(args: dict, name: str) -> list[str] | None:
    """Dates saved next to a series, if any, for chart axes."""
    source = args.get("source")
    if not isinstance(source, dict) or not RUN_ID.match(str(source.get("run_id", ""))):
        return None
    try:
        target = (SANDBOX_ROOT / source["run_id"] / str(source.get("file", f"{OUT_DIR}/series.json"))).resolve()
        doc = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    d = doc.get("dates") if isinstance(doc, dict) else None
    if isinstance(d, dict):
        d = d.get(name)
    return d if isinstance(d, list) else None
