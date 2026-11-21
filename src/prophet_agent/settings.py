"""Settings for Prophet: integrations, agent limits, context management and data utilities.

Motivation vs Logic
Motivation: a person running Prophet locally needs one place to see what is connected (NIM keys, TimesFM 3, the sandbox),
to bound how long the agent works before it pauses, and to export or clear what it has stored. Hard-coded limits were the
cause of "reached its limit" stops, so the limits live here and can be changed without a code edit.
Logic: non-secret preferences are bounded integers in ``data/settings.json``, written atomically. NIM keys are written to
the git-ignored ``.env`` (where ``load_keys`` reads them) with owner-only permissions. They are never returned: the view
shows a masked form (``nvapi-...abcd``) and the probe shows status codes only. Clearing data keeps settings and keys.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from prophet_agent.config import DATA, DOTENV, TIMESFM_CHECKPOINT
from prophet_agent.nim.transport import DEFAULT_CHAIN
from prophet_agent.runtime.goal import GoalConfig
from prophet_agent.sandbox.runner import memory_limit_enforced, run_python

DEFAULTS: dict[str, dict[str, int]] = {
    "agent": {"part_steps": 200, "tool_calls_per_part": 20, "part_seconds": 900, "sub_steps": 12},
    "context": {"view_chars": 16000, "keep_recent": 4, "memory_chars": 8000},
}
# (low, high) for every numeric setting. Anything outside is refused, never clamped silently.
BOUNDS: dict[tuple[str, str], tuple[int, int]] = {
    ("agent", "part_steps"): (5, 1000),
    ("agent", "tool_calls_per_part"): (1, 500),
    ("agent", "part_seconds"): (60, 14400),
    ("agent", "sub_steps"): (1, 60),
    ("context", "view_chars"): (2000, 200000),
    ("context", "keep_recent"): (1, 50),
    ("context", "memory_chars"): (1000, 100000),
}
MAX_KEYS = 16
KEY_PATTERN = re.compile(r"^nvapi-[A-Za-z0-9_\-]{8,}$")
ENV_KEY_LINE = re.compile(r"^\s*(export\s+)?NVIDIA_NIM_API_KEY(S|_[1-9])\s*=")
CLEAR_PHRASE = "delete all data"
LICENCE = ("TimesFM 3 weights are non-commercial: research use only. Do not use outputs for commercial decisions, client "
           "deliverables or paid services.")


def mask(key: str) -> str:
    """A key as a person may see it: the prefix and the last four characters. Never the whole value."""
    return f"{key[:6]}...{key[-4:]}" if len(key) > 12 else "nvapi-..."


class Settings:
    def __init__(self, data_root: Path | None = None, dotenv: Path | None = None):
        # data_root is the folder clear_data may empty (settings.json is kept). It defaults to the real data/ folder.
        self.data_root = data_root or DATA
        self.path = self.data_root / "settings.json"
        self.dotenv = dotenv or DOTENV
        self._lock = threading.Lock()

    # -- non-secret preferences --------------------------------------------------------------------------------

    def values(self) -> dict[str, dict[str, int]]:
        merged = {section: dict(vals) for section, vals in DEFAULTS.items()}
        try:
            stored = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return merged
        for (section, name), (low, high) in BOUNDS.items():
            block = stored.get(section) if isinstance(stored, dict) else None
            value = block.get(name) if isinstance(block, dict) else None
            if isinstance(value, int) and not isinstance(value, bool) and low <= value <= high:
                merged[section][name] = value
        return merged

    def update(self, patch: dict[str, Any]) -> dict[str, dict[str, int]]:
        """Validate and persist a partial update. Raises ValueError naming the first problem."""
        if not isinstance(patch, dict):
            raise ValueError("settings must be an object")
        current = self.values()
        for section, body in patch.items():
            if section == "nim":
                continue                             # keys are handled by set_keys
            if section not in DEFAULTS or not isinstance(body, dict):
                raise ValueError(f"unknown settings section {section!r}")
            for name, value in body.items():
                if (section, name) not in BOUNDS:
                    raise ValueError(f"unknown setting {section}.{name}")
                low, high = BOUNDS[(section, name)]
                if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                    raise ValueError(f"{section}.{name} must be a whole number from {low} to {high}")
                current[section][name] = value
        with self._lock:
            self._write_json(self.path, current)
        return current

    def goal_config(self, base: GoalConfig) -> GoalConfig:
        """The goal loop's limits as the person set them. Read at every start, so a change applies to the next part."""
        v = self.values()
        return replace(base, max_iterations=v["agent"]["part_steps"], sub_steps=v["agent"]["sub_steps"],
                       view_chars=v["context"]["view_chars"], keep_recent=v["context"]["keep_recent"],
                       memory_chars=v["context"]["memory_chars"])

    def default_budget(self) -> dict[str, int]:
        v = self.values()["agent"]
        return {"max_tool_calls": v["tool_calls_per_part"], "max_seconds": v["part_seconds"]}

    # -- NIM keys ----------------------------------------------------------------------------------------------

    def key_slots(self, environ: dict[str, str] | None = None) -> list[str]:
        from prophet_agent.nim.transport import load_keys
        return load_keys(environ)

    def masked_keys(self) -> list[dict[str, Any]]:
        return [{"slot": i, "masked": mask(k)} for i, k in enumerate(self.key_slots(), 1)]

    def set_keys(self, keys: list[str]) -> list[str]:
        """Replace the NIM key list. An entry that is the masked form of an existing key keeps that key, so a form that
        echoes masked values back cannot erase a key. Writes .env (owner-only) and the process environment."""
        if not isinstance(keys, list) or len(keys) > MAX_KEYS:
            raise ValueError(f"keys must be a list of at most {MAX_KEYS} entries")
        existing = self.key_slots()
        resolved: list[str] = []
        for item in keys:
            if not isinstance(item, str):
                raise ValueError("each key must be a string")
            value = item.strip()
            if "..." in value:
                match = [k for k in existing if mask(k) == value]
                if not match:
                    raise ValueError("a masked key does not match any stored key; paste the full key")
                value = match[0]
            if not KEY_PATTERN.match(value):
                raise ValueError("a NIM key starts with nvapi- and contains only letters, digits, '-' and '_'")
            if value not in resolved:
                resolved.append(value)
        self._write_dotenv(resolved)
        os.environ["NVIDIA_NIM_API_KEYS"] = ",".join(resolved)
        for i in range(1, 10):                       # blank the indexed form so an old value cannot come back
            os.environ[f"NVIDIA_NIM_API_KEY_{i}"] = ""
        from prophet_agent.nim.transport import reset_shared_client
        reset_shared_client()
        return resolved

    def _write_dotenv(self, keys: list[str]) -> None:
        try:
            lines = self.dotenv.read_text().splitlines()
        except OSError:
            lines = []
        kept = [line for line in lines if not ENV_KEY_LINE.match(line)]
        if keys:
            kept.append(f"NVIDIA_NIM_API_KEYS={','.join(keys)}")
        body = "\n".join(kept).rstrip() + "\n"
        self._write_text(self.dotenv, body, mode=0o600)

    # -- data utilities ----------------------------------------------------------------------------------------

    def summary(self, store, datasets) -> dict[str, int]:
        with store.lock:
            missions = store.conn.execute("SELECT COUNT(*) AS n FROM missions").fetchone()["n"]
            notebooks = store.conn.execute("SELECT COUNT(*) AS n FROM notebooks").fetchone()["n"]
        size = sum(p.stat().st_size for p in self.data_root.rglob("*") if p.is_file()) if self.data_root.exists() else 0
        return {"missions": int(missions), "notebooks": int(notebooks), "datasets": len(datasets.list(limit=10_000)),
                "bytes": int(size)}

    def export(self, store, datasets) -> dict[str, Any]:
        """Everything the ledger holds, as plain JSON. Keys and .env are not part of it."""
        with store.lock:
            tables = {name: [dict(r) for r in store.conn.execute(f"SELECT * FROM {name}").fetchall()]
                      for name in ("missions", "steps", "events", "questions", "notebooks", "sources")}
        return {"exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "settings": self.values(),
                "datasets": datasets.list(limit=10_000), **tables}

    def clear(self, store, active_statuses: tuple[str, ...], confirm: str) -> dict[str, int]:
        """Delete missions, notebooks, uploads and sandbox output. Settings and keys stay. Refused while work is running."""
        if confirm != CLEAR_PHRASE:
            raise ValueError(f"type '{CLEAR_PHRASE}' to confirm")
        with store.lock:
            running = store.conn.execute(
                f"SELECT COUNT(*) AS n FROM missions WHERE status IN ({','.join('?' * len(active_statuses))})",
                active_statuses).fetchone()["n"]
        if running:
            raise RuntimeError("a mission is still running; stop it before clearing data")
        removed = store.wipe_ledger()
        for child in self.data_root.iterdir() if self.data_root.exists() else []:
            if child.name == "settings.json" or child.suffix in (".sqlite", ".sqlite-wal", ".sqlite-shm") \
                    or child.name.endswith((".sqlite-wal", ".sqlite-shm")):
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        return removed

    # -- helpers -----------------------------------------------------------------------------------------------

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        Settings._write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")

    @staticmethod
    def _write_text(path: Path, text: str, mode: int = 0o644) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(text)
            os.chmod(tmp, mode)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def _sandbox_limits() -> dict[str, Any]:
    """The sandbox's real defaults, read from its signature so the page cannot drift from the code that enforces them."""
    params = inspect.signature(run_python).parameters
    return {
        "timeout_s": params["timeout_s"].default,
        "mem_mb": params["mem_mb"].default,
        "file_mb": params["file_mb"].default,
        "memory_limit_enforced": memory_limit_enforced(),
        "isolation": "Child process with a scrubbed environment, a write guard and resource limits. A guardrail, not a container.",
    }


def view(settings: Settings, store, datasets, pool_status: dict | None) -> dict[str, Any]:
    """What the Settings page shows. No key material: masked keys and counts only."""
    keys = settings.masked_keys()
    return {
        "nim": {"configured": bool(keys), "keys": keys, "models": list(DEFAULT_CHAIN), "pool": pool_status},
        "agent": settings.values()["agent"],
        "context": settings.values()["context"],
        "defaults": {section: dict(vals) for section, vals in DEFAULTS.items()},
        "bounds": {section: {name: [low, high] for (s, name), (low, high) in BOUNDS.items() if s == section}
                   for section in DEFAULTS},
        "timesfm": {"checkpoint": TIMESFM_CHECKPOINT, "engine_installed": importlib.util.find_spec("timesfm") is not None,
                    "licence": LICENCE},
        "sandbox": _sandbox_limits(),
        "data": settings.summary(store, datasets),
        "about": {"name": "Prophet", "licence": LICENCE, "clear_phrase": CLEAR_PHRASE},
    }
