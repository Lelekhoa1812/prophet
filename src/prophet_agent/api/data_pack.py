"""A zip of the files a session or a notebook holds, for the person to download.

Motivation vs Logic
Motivation: the answer names files such as in/NVDA_daily.csv, and the charts a run made, but the screen did not hand
those bytes back. A person should download every file the session or the notebook holds, from the page they are on.
Logic: the zip is built from stored paths only. Uploaded files come from the dataset store. Kept tables come from the
notebook folder. Each analysis run contributes its in/ and out/ trees, which is where images and tables land. The
sandbox lib and program are not data, so they stay out. A path that does not sit inside its own root is skipped.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

from prophet_agent.cleanup import mission_run_ids
from prophet_agent.datasets import ID as DATASET_ID

RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


def _inside(path: Path, root: Path) -> bool:
    """True when path is a real file and its resolved location stays inside root."""
    try:
        return path.is_file() and path.resolve(strict=True).is_relative_to(root.resolve())
    except OSError:
        return False


def _inside_dir(path: Path, root: Path) -> bool:
    try:
        return path.is_dir() and path.resolve(strict=True).is_relative_to(root.resolve())
    except OSError:
        return False


def _slug(text: str) -> str:
    slug = _SLUG.sub("-", text).strip("-")[:40]
    return slug or "files"


def output_files(sandbox_root: Path, run_id: str) -> list[dict]:
    """Every file under a run's out/ folder. Empty when the run id or the folder is not one this root owns."""
    if not isinstance(run_id, str) or not RUN_ID.match(run_id):
        return []
    run_dir = sandbox_root / run_id
    if not _inside_dir(run_dir, sandbox_root):
        return []
    base = run_dir / "out"
    if not base.is_dir():
        return []
    found = []
    for path in sorted(base.rglob("*")):
        if not _inside(path, run_dir):
            continue
        rel = path.relative_to(run_dir).as_posix()
        found.append({"path": rel, "bytes": path.stat().st_size, "sha256": ""})
    return found


def merge_output_files(recorded: list, sandbox_root: Path, run_id: str) -> list[dict]:
    """The files the tool result listed, plus any out/ file that exists on disk and was left off that list."""
    by_path: dict[str, dict] = {}
    for item in recorded or []:
        if isinstance(item, dict) and str(item.get("path", "")).startswith("out/"):
            by_path[item["path"]] = {"path": item["path"], "sha256": item.get("sha256") or "", "bytes": item.get("bytes")}
    for extra in output_files(sandbox_root, run_id):
        by_path.setdefault(extra["path"], extra)
    return [by_path[key] for key in sorted(by_path)]


def _run_ids(store, mission_id: str) -> list[str]:
    with store.lock:
        found = mission_run_ids(store.conn, [mission_id])
    return sorted(found)


def _dataset_entries(datasets, dataset_ids: list[str], prefix: str) -> tuple[list[tuple[str, Path]], list[str]]:
    entries: list[tuple[str, Path]] = []
    missing: list[str] = []
    kept: list[str] = []
    for did in dataset_ids:
        if not isinstance(did, str) or not DATASET_ID.match(did):
            continue
        try:
            datasets.get(did)
        except KeyError:
            missing.append(did)
            continue
        kept.append(did)
    if not kept:
        return entries, missing
    for name, path in datasets.mount(kept).items():
        if _inside(path, datasets.root):
            entries.append((prefix + name, path))
        else:
            missing.append(name)
    return entries, missing


def _run_entries(sandbox_root: Path, run_id: str, prefix: str) -> list[tuple[str, Path]]:
    if not RUN_ID.match(run_id):
        return []
    run_dir = sandbox_root / run_id
    if not _inside_dir(run_dir, sandbox_root):
        return []
    entries = []
    for sub in ("in", "out"):
        base = run_dir / sub
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if _inside(path, run_dir):
                entries.append((prefix + path.relative_to(run_dir).as_posix(), path))
    return entries


def _table_entries(notebooks, notebook_id: str) -> list[tuple[str, Path]]:
    root = (notebooks.root / notebook_id).resolve()
    with notebooks.store.lock:
        rows = notebooks.store.conn.execute(
            "SELECT t.name, v.version, v.file_path FROM derived_versions v "
            "JOIN derived_tables t ON t.id=v.table_id WHERE t.notebook_id=? AND v.dropped_by IS NULL ORDER BY t.name, v.version",
            (notebook_id,)).fetchall()
    entries = []
    for row in rows:
        path = Path(row["file_path"])
        if _inside(path, root):
            entries.append((f"tables/{row['name']}_v{row['version']}.csv", path))
    return entries


def _zip(entries: list[tuple[str, Path]], missing: list[str]) -> bytes:
    used: set[str] = set()
    buf = io.BytesIO()
    listed = []
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arc, path in entries:
            name = arc
            stem, n = name, 2
            while name in used:
                dot = stem.rfind(".")
                base, ext = (stem[:dot], stem[dot:]) if dot > 0 else (stem, "")
                name = f"{base}_{n}{ext}"
                n += 1
            used.add(name)
            zf.write(path, arcname=name)
            listed.append({"path": name, "bytes": path.stat().st_size})
        zf.writestr("manifest.json", json.dumps({"files": listed, "missing": missing}, indent=2))
    return buf.getvalue()


def build_mission_pack(service, mission_id: str, sandbox_root: Path) -> tuple[bytes, str]:
    """Zip for one session: attached files, and each run's inputs and outputs. Raises KeyError when the session is unknown."""
    mission = service.store.mission(mission_id)
    entries: list[tuple[str, Path]] = []
    missing: list[str] = []
    ids = [d for d in ((mission.get("budget") or {}).get("dataset_ids") or []) if isinstance(d, str)]
    notebook_id = mission.get("notebook_id")
    if notebook_id:
        ids = [s["dataset_id"] for s in service.notebooks.sources(notebook_id) if not s["removed"]] + ids
        entries.extend(_table_entries(service.notebooks, notebook_id))
    file_entries, gone = _dataset_entries(service.datasets, list(dict.fromkeys(ids)), "files/")
    entries.extend(file_entries)
    missing.extend(gone)
    for run_id in _run_ids(service.store, mission_id):
        entries.extend(_run_entries(sandbox_root, run_id, f"runs/{run_id}/"))
    title = str(mission.get("title") or mission_id[:12])
    return _zip(entries, missing), f"prophet-session-{_slug(title)}.zip"


def build_notebook_pack(service, notebook_id: str, sandbox_root: Path) -> tuple[bytes, str]:
    """Zip for one notebook: its files, kept tables, and every question's inputs and outputs. Raises KeyError when unknown."""
    notebook = service.notebooks.get(notebook_id)
    sources = [s["dataset_id"] for s in service.notebooks.sources(notebook_id) if not s["removed"]]
    entries, missing = _dataset_entries(service.datasets, sources, "files/")
    entries.extend(_table_entries(service.notebooks, notebook_id))
    for turn in service.notebooks.turns(notebook_id):
        for run_id in _run_ids(service.store, turn["id"]):
            entries.extend(_run_entries(sandbox_root, run_id, f"questions/{turn['seq']}/runs/{run_id}/"))
    return _zip(entries, missing), f"prophet-notebook-{_slug(str(notebook.get('title') or notebook_id[:12]))}.zip"
