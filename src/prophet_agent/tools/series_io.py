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
from pathlib import Path
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
    # Round 6m: numpy turned a nested list into a two-dimensional array and text such as "1" into a number without a word.
    # Both are refused here, with the form the values must take.
    if any(isinstance(v, (list, tuple, dict)) for v in raw):
        raise ValueError(f"series {name!r} must be one flat list of numbers, not a list of lists or objects")
    if any(isinstance(v, (str, bool)) for v in raw):
        raise ValueError(f"series {name!r} contains values that are not numbers (text or true/false); give numbers only")
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"series {name!r} contains values that are not numbers")
    bad = int((~np.isfinite(arr)).sum())
    if bad:
        raise ValueError(f"series {name!r} has {bad} missing or non-finite values; fill or drop them in the analysis "
                         "step and save again, so the choice is explicit")
    return arr


def missing_output(run_id: str, rel: str) -> str:
    """Why a named output is not there, with what would fix it: the run does not exist (an id that no tool returned), or
    it exists and these are the files it did write.

    Live electricity rep4 (2026-11-23): three calls named files their runs never wrote, one with a run id no tool had
    returned. "no file" alone left the planner guessing which run held the file.
    """
    base = SANDBOX_ROOT / run_id
    if not base.is_dir():
        return (f"no run {run_id} exists. A run id is the run_id a sandbox_python result returned; copy it from that "
                "result, or re-run the step that writes the file")
    out = base / OUT_DIR
    wrote = sorted(str(f.relative_to(base)) for f in out.rglob("*") if f.is_file())[:12] if out.is_dir() else []
    return (f"no file {rel!r} in the outputs of run {run_id}; that run wrote "
            + (", ".join(wrote) if wrote else "no files") + ". Name one of those, or the run that saved this file")


def _out_file(run_id: str, rel: str) -> Path:
    """The resolved path of ``rel`` inside the run's ``out/`` folder. Raises ValueError when it lies elsewhere. Every reader of
    a source file goes through here: an absolute or ``../`` name used to read any file on disk (round 6l, audit item 6)."""
    base = (SANDBOX_ROOT / run_id).resolve()
    target = (base / rel).resolve()
    if not str(target).startswith(str((base / OUT_DIR).resolve()) + "/"):
        raise ValueError(missing_output(run_id, rel))
    return target


def source_path(args: dict) -> Path:
    """The file a ``source`` argument names, after the same checks as resolve_series: a hex run id and a file inside the
    run's ``out/`` folder. Raises ValueError with the fix when either check fails."""
    source = args.get("source")
    if not isinstance(source, dict) or not RUN_ID.match(str(source.get("run_id", ""))):
        raise ValueError("source needs run_id (the 32-character id of the analysis run that wrote the file)")
    rel = str(source.get("file", f"{OUT_DIR}/series.json"))
    target = _out_file(source["run_id"], rel)
    if not target.is_file():
        raise ValueError(missing_output(source["run_id"], rel))
    return target


def resolve_series(args: dict) -> list[tuple[str, np.ndarray]]:
    values, source = args.get("values"), args.get("source")
    if (values is None) == (source is None):
        raise ValueError("give exactly one of values or source")
    if values is not None:
        return [("series", _clean("series", values))]
    if not isinstance(source, dict) or not RUN_ID.match(str(source.get("run_id", ""))):
        raise ValueError("source needs run_id (the 32-character id of the analysis run that wrote the file)")
    rel = str(source.get("file", f"{OUT_DIR}/series.json"))
    target = _out_file(source["run_id"], rel)
    if not target.is_file():
        raise ValueError(missing_output(source["run_id"], rel))
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
    if want is not None and not isinstance(want, (str, list)):
        # A dict is unhashable and raised a bare TypeError in the lookup below; a number is not a series name either (round 6m).
        raise ValueError("source.name is one series name (text), or a list of series names")
    if isinstance(want, list):
        # A subset of a panel (live electricity rep5: a list of ten names failed with "unhashable type: 'list'", three times).
        if not want or not all(isinstance(n, str) for n in want):
            raise ValueError("source.name is one series name, or a list of series names")
        absent = [n for n in want if n not in named]
        if absent:
            raise ValueError(f"series {absent[:10]} not in the file; available: {sorted(named)[:20]}")
        named, want = {n: named[n] for n in dict.fromkeys(want)}, None
    if want is not None:
        if want not in named and len(named) == 1:
            want = next(iter(named))          # one series in the file: no ambiguity, so a wrong name is not an error
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
        target = _out_file(source["run_id"], str(source.get("file", f"{OUT_DIR}/series.json")))
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
        target = _out_file(source["run_id"], str(source.get("file", f"{OUT_DIR}/series.json")))
        doc = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    d = doc.get("dates") if isinstance(doc, dict) else None
    if isinstance(d, dict):
        d = d.get(name)
    return d if isinstance(d, list) else None
