"""A mission's reproducibility bundle: a read-only zip a person can keep, check or rerun (A7-11).

Motivation vs Logic
Motivation: the forecasts are deterministic on the same input and code, but nothing let a reviewer see the code that ran,
the input it ran on, or the versions it ran with. Without that the claim cannot be checked from outside.
Logic: the bundle is built from what is already stored. The ledger (every tool call with its arguments and result), the
events and the final answer come from the store; each run folder's program, lib, inputs, JSON and CSV outputs, and manifest
come from the sandbox. Every file has a sha256 in hashes.json. Only named files are read, never the data folder or .env.
Text is scrubbed for key-shaped strings, and the build refuses to finish if one survives.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from importlib import metadata
from pathlib import Path
from typing import Any

from prophet_agent.nim.errors import scrub_keys
from prophet_agent.prompts import CURRENT, load

RUN_ID = re.compile(r"^[0-9a-f]{32}$")
KEY_SHAPED = re.compile(r"nvapi-[A-Za-z0-9_\-]{4,}")
PACKAGES = ("timesfm", "numpy", "pandas", "scipy", "lightgbm", "statsmodels")
MAX_INPUT_BYTES = 200 * 1024 * 1024        # inputs beyond this are listed by hash and not copied
RERUN_NOTE = ("Rerunning the same code on the same input should give the same figures. Each file's sha256 is in hashes.json; "
              "check the input and program hashes before comparing figures. Package versions are in configuration.json.")
BUDGET_SECRET_WORDS = ("key", "token", "secret", "password")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _scrub(data: bytes) -> bytes:
    """Replace key-shaped text in a text file. Binary files are returned as they are; they are checked below."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    return scrub_keys(KEY_SHAPED.sub("[key removed]", text), mark="[key removed]").encode("utf-8")


def _key_shaped(text: str) -> bool:
    """True while any key-shaped text is left: NVIDIA keys, other providers' key formats, or a Bearer token."""
    return bool(KEY_SHAPED.search(text)) or scrub_keys(text) != text


def _packages() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _run_ids(tool_rows: list[dict]) -> list[str]:
    seen: list[str] = []
    for row in tool_rows:
        res = row.get("result")
        rid = res.get("run_id") if isinstance(res, dict) else None
        if isinstance(rid, str) and RUN_ID.match(rid) and rid not in seen:
            seen.append(rid)
    return seen


def _contained_file(path: Path, root: Path) -> bool:
    """True for a regular file whose real location is inside the run folder. Analysis code writes into the folder, so an
    entry may be a symlink; one that leads outside the folder is not a file of the run and is never read."""
    try:
        return path.is_file() and path.resolve(strict=True).is_relative_to(root)
    except OSError:
        return False


def _run_files(run_dir: Path) -> list[tuple[str, Path]]:
    """The files of one run folder that a rerun needs: program, lib, inputs, the manifest and the numeric outputs."""
    root = run_dir.resolve()
    picked: list[tuple[str, Path]] = []
    for rel in ("program.py", "manifest.json"):
        if _contained_file(run_dir / rel, root):
            picked.append((rel, run_dir / rel))
    for sub, exts in (("lib", (".py",)), ("in", None), ("out", (".json", ".csv"))):
        base = run_dir / sub
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if (exts is None or p.suffix.lower() in exts) and _contained_file(p, root):
                picked.append((p.relative_to(run_dir).as_posix(), p))
    return picked


def build_bundle(store, mission_id: str, sandbox_root: Path, prompts: dict | None = None) -> bytes:
    """The zip for one mission. Raises KeyError for an unknown mission. Read-only."""
    mission = store.mission(mission_id)                       # KeyError for an unknown mission
    with store.lock:
        rows = store.conn.execute(
            "SELECT step_idx, tool, args_json, status, result_json, error FROM tool_calls WHERE mission_id=? ORDER BY step_idx",
            (mission_id,)).fetchall()
    tool_rows = [{"step": int(r["step_idx"]), "tool": r["tool"], "status": r["status"],
                  "args": json.loads(r["args_json"] or "null"), "result": json.loads(r["result_json"] or "null"),
                  "error": r["error"]} for r in rows]
    events = [{"seq": e.get("seq"), "kind": e["kind"], "payload": e["payload"]} for e in store.events(mission_id)]
    budget = {k: v for k, v in (mission.get("budget") or {}).items() if not any(w in k.lower() for w in BUDGET_SECRET_WORDS)}
    answer = mission.get("answer") if isinstance(mission.get("answer"), dict) else {}
    contracts = prompts if prompts is not None else {
        name: {"version": load(version).version, "sha256": load(version).sha256} for name, version in CURRENT.items()}

    entries: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    missing: list[str] = []
    input_bytes = 0
    skipped_inputs: list[str] = []

    def add(name: str, data: bytes) -> None:
        data = _scrub(data)
        entries[name] = data
        hashes[name] = _sha(data)

    for run_id in _run_ids(tool_rows):
        run_dir = sandbox_root / run_id
        if not run_dir.is_dir():
            missing.append(run_id)
            continue
        for rel, path in _run_files(run_dir):
            data = path.read_bytes()
            if rel.startswith("in/"):
                if input_bytes + len(data) > MAX_INPUT_BYTES:
                    skipped_inputs.append(rel)
                    hashes[f"runs/{run_id}/{rel}"] = _sha(data)
                    continue
                input_bytes += len(data)
            add(f"runs/{run_id}/{rel}", data)

    config = {"prompts": contracts, "budget": budget, "packages": _packages(), "runs": _run_ids(tool_rows),
              "missing_runs": missing, "inputs_not_copied": skipped_inputs,
              "tool_arguments": [{"step": r["step"], "tool": r["tool"], "args": r["args"]} for r in tool_rows]}
    add("configuration.json", json.dumps(config, indent=2, sort_keys=True, default=str).encode("utf-8"))
    add("ledger.json", json.dumps(tool_rows, indent=2, sort_keys=True, default=str).encode("utf-8"))
    add("events.json", json.dumps(events, indent=2, sort_keys=True, default=str).encode("utf-8"))
    add("mission.json", json.dumps({"id": mission_id, "goal": mission["goal"], "status": mission["status"],
                                    "budget": budget, "answer_status": answer.get("status"),
                                    "stop_reason": answer.get("stop_reason")}, indent=2, sort_keys=True, default=str).encode("utf-8"))
    add("answer.md", str(answer.get("answer") or "").encode("utf-8"))
    readme = (f"Prophet reproducibility bundle for mission {mission_id}.\n\n{RERUN_NOTE}\n\n"
              "runs/<run id>/ holds each analysis folder: program.py (the code cells), lib/ (the helper modules), in/ (the\n"
              "inputs as they were read), out/ (numeric outputs), manifest.json (what the sandbox recorded).\n"
              "ledger.json holds every tool call with its arguments and result. events.json holds the status history.\n"
              "configuration.json holds prompt versions and hashes, package versions, the budget and the tool arguments.\n"
              "answer.md is the final answer as written. Key-shaped text is removed; no .env file is included.\n")
    add("README.txt", readme.encode("utf-8"))
    # hashes.json lists every file above except itself.
    hashes_doc = json.dumps(dict(sorted(hashes.items())), indent=2).encode("utf-8")
    entries["hashes.json"] = hashes_doc

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(entries):
            zf.writestr(name, entries[name])
    data = buf.getvalue()
    _assert_clean(data)
    return data


def _assert_clean(data: bytes) -> None:
    """Refuse to return a bundle that still holds a key-shaped string or an .env entry."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            if name == ".env" or name.endswith("/.env"):
                raise RuntimeError("bundle would include an .env file")
            if _key_shaped(zf.read(name).decode("utf-8", errors="ignore")):
                raise RuntimeError(f"bundle would include key-shaped text in {name}")


__all__ = ["build_bundle"]
