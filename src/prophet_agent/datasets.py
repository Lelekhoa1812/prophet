"""Uploaded files: stored once, hashed, handed to a mission read-only.

Motivation vs Logic
Motivation: forecasting starts from the person's own file. It must reach the analysis code unchanged, be traceable by
hash, and never be executed or served back from a path the client chose.
Logic: each upload gets a random id and its own folder under the datasets root, holds one file under a cleaned name,
and has a small ``meta.json`` (name, size, sha256, time). Names are reduced to a safe basename. Size is capped. The
store never opens or parses the file: reading is the sandbox's job (see sandbox/prophet_lib.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path

MAX_BYTES = 50 * 1024 * 1024
ID = re.compile(r"^[0-9a-f]{32}$")


def clean_name(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(str(name or "data")).name).strip(" .") or "data"
    return base[:120]


class DatasetStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def add(self, name: str, data: bytes) -> dict:
        if not data:
            raise ValueError("the file is empty")
        if len(data) > MAX_BYTES:
            raise ValueError(f"the file is {len(data) // (1024 * 1024)} MB; the limit is {MAX_BYTES // (1024 * 1024)} MB")
        did, fname = uuid.uuid4().hex, clean_name(name)
        folder = self.root / did
        folder.mkdir(parents=True)
        (folder / fname).write_bytes(data)
        meta = {"id": did, "name": fname, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "created": time.time()}
        # Written under a temp name and renamed, so meta.json is either absent or complete.
        tmp = folder / "meta.json.tmp"
        tmp.write_text(json.dumps(meta), encoding="utf-8")
        os.replace(tmp, folder / "meta.json")
        return meta

    def get(self, dataset_id: str) -> dict:
        if not ID.match(dataset_id):
            raise KeyError(dataset_id)
        try:
            return json.loads((self.root / dataset_id / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):            # a missing or half-written metadata file is an unknown dataset
            raise KeyError(dataset_id)

    def path(self, dataset_id: str) -> Path:
        return self.root / dataset_id / self.get(dataset_id)["name"]

    def list(self, limit: int = 50) -> list[dict]:
        if not self.root.is_dir():
            return []
        metas = []
        for d in self.root.iterdir():
            try:
                metas.append(self.get(d.name))
            except KeyError:
                continue
        return sorted(metas, key=lambda m: m["created"], reverse=True)[:limit]

    def mount(self, dataset_ids: list[str]) -> dict[str, Path]:
        """File name -> path for a mission. A repeated name gets a numeric suffix so no file hides another."""
        out: dict[str, Path] = {}
        for did in dataset_ids:
            meta = self.get(did)
            name, stem, n = meta["name"], Path(meta["name"]).stem, 1
            while name in out:
                n += 1
                name = f"{stem}_{n}{Path(meta['name']).suffix}"
            out[name] = self.path(did)
        return out
