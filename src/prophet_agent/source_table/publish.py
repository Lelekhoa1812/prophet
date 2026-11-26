"""Bind a series name only after validate_table accepts the bytes.

Motivation vs Logic
Motivation: the person downloads the name they asked for. That name has to mean the entity they asked for.
Logic: the verdict is computed again here. A rejection writes nothing under the requested name. The same claim and
the same hash return the dataset already stored.
"""
from __future__ import annotations

import hashlib

from prophet_agent.datasets import DatasetStore
from prophet_agent.source_table.validate import validate_table


def bind_series(store: DatasetStore, claim: dict, csv_bytes: bytes, *, final_url: str = "",
                retrieved_at: str = "", page_text: str | None = None, reference_csvs: list[bytes] | None = None) -> dict:
    """Store ``csv_bytes`` under the claim's series name when the verdict is accepted. Otherwise bind nothing."""
    verdict = validate_table(claim, csv_bytes, final_url=final_url, page_text=page_text, reference_csvs=reference_csvs)
    digest = hashlib.sha256(csv_bytes).hexdigest() if csv_bytes else ""
    if verdict["status"] != "accepted":
        return {**verdict, "dataset_id": None, "sha256": digest}
    existing = _find(store, claim.get("claim_id") or "", digest)
    if existing is not None:
        return {**verdict, "dataset_id": existing["id"], "sha256": digest, "name": existing["name"]}
    provenance = {
        "source": "validated_table",
        "series_name": claim["series_name"],
        "claim_id": claim.get("claim_id") or "",
        "sha256": digest,
        "validation": "accepted",
        "entity": claim.get("entity") or "",
        "observed_entity": verdict.get("observed_entity") or "",
        "url": final_url,
        "retrieved_at": retrieved_at,
        "columns": [{"name": name, "origin": "observed"} for name in verdict.get("columns") or []],
    }
    meta = store.add(f"{claim['series_name']}.csv", csv_bytes, provenance=provenance)
    return {**verdict, "dataset_id": meta["id"], "sha256": digest, "name": meta["name"]}


def _find(store: DatasetStore, claim_id: str, digest: str) -> dict | None:
    if not claim_id or not digest or not store.root.is_dir():
        return None
    for folder in store.root.iterdir():
        try:
            meta = store.get(folder.name)
        except KeyError:
            continue
        provenance = meta.get("provenance") or {}
        if provenance.get("source") == "validated_table" and provenance.get("claim_id") == claim_id and provenance.get("sha256") == digest:
            return meta
    return None
