"""Model verification: one tiny real completion per catalog model, stored in data/model_checks.json for the app to read.

Motivation vs Logic
Motivation: the catalog lists model ids that a person may not have access to yet. Settings should say which ones answered,
and with what HTTP status, without anyone pasting keys into a chat.
Logic: ``verify_models`` makes exactly one call per model it is asked to check, on one usable key, through
``NimClient.check_model`` (no failover, no retries). A model is "verified" on HTTP 200 with a completion, "failed" on any
other outcome (with the status and a redacted, short body), and "untested" when no call was made: no key, or the provider
was not in this run. Results are written once per run. Nothing is hard-coded: re-running the function re-checks.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import DATA, DOTENV, load_dotenv_values
from .providers import CATALOG, PROVIDERS

MODEL_CHECK_FILE = "model_checks.json"
CHECK_PROMPT = [{"role": "user", "content": "Reply with the single word: ok"}]
CHECK_MAX_TOKENS = 64          # the smallest budget that still lets a reasoning model finish a word


def _now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_model_checks(path: Path | None = None) -> dict[str, dict]:
    """The last stored check per model id: {"status", "http_status", "error", "checked_at", "provider"}. Empty when the
    file is missing or unreadable, so the app shows every model as untested rather than failing."""
    import json
    try:
        data = json.loads((path or DATA / MODEL_CHECK_FILE).read_text())
    except (OSError, ValueError):
        return {}
    models = data.get("models") if isinstance(data, dict) else None
    return models if isinstance(models, dict) else {}


def model_check(model_id: str, checks: Mapping[str, dict]) -> dict:
    """The status the app shows for one model: its stored check, or untested."""
    row = checks.get(model_id) or {}
    return {"status": row.get("status", "untested"), "checked_at": row.get("checked_at")}


def verify_models(*, providers: Iterable[str] | None = None, data_root: Path | None = None,
                  dotenv: Path | None = None, environ: Mapping[str, str] | None = None,
                  path: Path | None = None, transport: Any = None) -> dict:
    """Check the catalog and write data/model_checks.json. Returns the same document.

    ``providers``: the providers to call now (default: every provider that has a usable key). A provider with a key that
    is not listed is written as untested, so a partial run never claims a result it did not get.
    ``environ``: the variables to read keys from (default: the process environment over ``dotenv``, default the repo .env).
    """
    from ..settings import Settings
    from ..usage import ledger_for
    from .transport import NimClient

    root = data_root or DATA
    env = environ if environ is not None else {**load_dotenv_values(dotenv or DOTENV), **os.environ}
    settings = Settings(data_root=root, dotenv=dotenv or DOTENV)
    requested = set(providers) if providers is not None else set(PROVIDERS)
    checked_at = _now()
    results: dict[str, dict] = {}
    for name, spec in PROVIDERS.items():
        models = [m for m in CATALOG if m.provider == name]
        keys = settings.usable_keys(name, environ=env)
        if not keys:
            for info in models:
                results[info.id] = {"provider": name, "status": "untested", "http_status": None,
                                    "error": "no key configured", "checked_at": checked_at}
            continue
        if name not in requested:
            for info in models:
                results[info.id] = {"provider": name, "status": "untested", "http_status": None,
                                    "error": "not called in this run", "checked_at": checked_at}
            continue
        # One call per model on the first usable key; no failover and no retries (NimClient.check_model).
        hook = (lambda reply, p=name: ledger_for(root).record(p, reply.model, reply.usage))
        with NimClient(keys, provider=spec, on_reply=hook, transport=transport) as client:
            for info in models:
                out = client.check_model(info.id, CHECK_PROMPT, max_tokens=CHECK_MAX_TOKENS)
                error = out["error"]
                for key in keys:                       # a vendor's error body must never store a key value
                    error = error.replace(key, "[key]") if error else error
                results[info.id] = {"provider": name, "status": out["status"], "http_status": out["http_status"],
                                    "error": error, "checked_at": checked_at}
    doc = {"checked_at": checked_at, "models": results}
    Settings._write_json(path or root / MODEL_CHECK_FILE, doc)
    return doc
