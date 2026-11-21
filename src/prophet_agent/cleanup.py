"""Files that a deleted notebook or quick forecast leaves behind.

Motivation vs Logic
Motivation: deleting something in the app should delete what it owns. Sandbox run folders keep copies of the inputs and
outputs of every run, and dataset folders keep the uploaded file. Left behind, they are still downloadable and take up
disk, and they outlive the thing the person deleted.
Logic: callers delete the database rows inside their own transaction, while they still know which run ids and dataset
ids belonged to the item. They then call the helpers here after the commit, so a failed delete never removes files.
A dataset is removed only when no remaining source or mission still names it. Run ids are read from the stored ledger
JSON, so the lookup needs no new bookkeeping for runs created by mission tools.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Iterable

from prophet_agent.datasets import ID as DATASET_ID
from prophet_agent.sandbox.runner import SANDBOX_ROOT

RUN_ID = re.compile(r"^[0-9a-f]{32}$")
# canonical() writes JSON without spaces, json.dumps with them; accept both.
_RUN_REF = re.compile(r'"run_id"\s*:\s*"([0-9a-f]{32})"')

# (table, column) pairs of the ledger that can name a run folder, all keyed by mission_id.
_MISSION_LEDGER = (("tool_calls", "result_json"), ("steps", "output_json"), ("events", "payload_json"))


def run_ids_in(*texts: object) -> set[str]:
    found: set[str] = set()
    for text in texts:
        if text:
            found.update(_RUN_REF.findall(str(text)))
    return found


def mission_run_ids(c, mission_ids: Iterable[str]) -> set[str]:
    """Run ids named anywhere in the ledger of these missions. Call before their rows are deleted."""
    found: set[str] = set()
    for mid in mission_ids:
        for table, column in _MISSION_LEDGER:
            found |= run_ids_in(*(r[0] for r in c.execute(f"SELECT {column} FROM {table} WHERE mission_id=?", (mid,))))
    return found


def unreferenced_datasets(c, dataset_ids: Iterable[str]) -> list[str]:
    """Dataset ids that no remaining source row or mission budget names. Call after the owner's rows are deleted."""
    orphans = []
    for did in dict.fromkeys(dataset_ids):
        if not DATASET_ID.match(did):
            continue
        in_source = c.execute("SELECT 1 FROM sources WHERE dataset_id=? LIMIT 1", (did,)).fetchone()
        in_mission = c.execute("SELECT 1 FROM missions WHERE budget_json LIKE ? LIMIT 1", (f"%{did}%",)).fetchone()
        if in_source is None and in_mission is None:
            orphans.append(did)
    return orphans


def remove_runs(run_ids: Iterable[str]) -> None:
    for rid in run_ids:
        if RUN_ID.match(rid):
            shutil.rmtree(SANDBOX_ROOT / rid, ignore_errors=True)


def remove_datasets(root: Path, dataset_ids: Iterable[str]) -> None:
    for did in dataset_ids:
        if DATASET_ID.match(did):
            shutil.rmtree(Path(root) / did, ignore_errors=True)
