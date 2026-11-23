"""Settings for Prophet: integrations, agent limits, context management, model providers and data utilities.

Motivation vs Logic
Motivation: a person running Prophet locally needs one place to see what is connected (NIM keys, Claude, ChatGPT, Gemini,
Grok, Matilda, TimesFM 3, the sandbox), to switch providers and models on or off, to set a token ceiling per model, and to
bound how long the agent works before it pauses. Hard-coded limits were the cause of "reached its limit" stops, so the
limits live here and can be changed without a code edit.
Logic: non-secret preferences are bounded integers and the model settings in ``data/settings.json``, written atomically.
Provider keys are written to the git-ignored ``.env`` (where ``load_keys`` reads them) with owner-only permissions. They
are never returned: the view shows a masked form (``nvapi-...abcd``) and the last key check (valid, invalid or untested
with its HTTP status). A key whose last check was invalid is left out of ``usable_keys``, so the planner never uses it.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from prophet_agent.config import DATA, DOTENV, TIMESFM_CHECKPOINT
from prophet_agent.nim.errors import NotConfigured
from prophet_agent.nim.model_checks import MODEL_CHECK_FILE, model_check, read_model_checks
from prophet_agent.nim.providers import CATALOG, DEFAULT_PROVIDER, MODELS, PROVIDERS, Provider, get_provider
from prophet_agent.nim.transport import DEFAULT_CHAIN, reset_shared_client
from prophet_agent.runtime.goal import GoalConfig
from prophet_agent.sandbox.runner import memory_limit_enforced, run_python
from prophet_agent.usage import WINDOWS

# The Settings page shows the size of the data folder. Walking tens of thousands of files takes about a second, so the
# answer is kept for SIZE_TTL_S seconds; clearing data drops it at once.
SIZE_TTL_S = 60.0
_SIZE_CACHE: dict[str, tuple[float, int]] = {}
_SIZE_LOCK = threading.Lock()


def _data_bytes(root: Path) -> int:
    key = str(root)
    now = time.monotonic()
    with _SIZE_LOCK:
        hit = _SIZE_CACHE.get(key)
        if hit is not None and now - hit[0] < SIZE_TTL_S:
            return hit[1]
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0
    with _SIZE_LOCK:
        _SIZE_CACHE[key] = (now, size)
    return size


def _forget_data_bytes(root: Path) -> None:
    with _SIZE_LOCK:
        _SIZE_CACHE.pop(str(root), None)


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
KEY_PATTERN = re.compile(r"^nvapi-[A-Za-z0-9_\-]{8,}$")                # NVIDIA NIM keys
GENERIC_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_\-.]{8,}$")             # other providers' key formats vary
LLM_SECTION = "llm"
LLM_FIELDS = ("provider", "providers", "models", "budgets")           # provider: legacy single-provider form
KEY_CHECKS = "key_checks"                                              # top level of settings.json, per provider
KEY_SECTIONS = ("nim", "providers")                                     # handled by set_keys, not by update()
CLEAR_PHRASE = "delete all data"
# data/ files that "clear data" keeps: settings, usage (not mission data) and the model verification results.
KEPT_ON_CLEAR = ("settings.json", MODEL_CHECK_FILE)


def mask(key: str) -> str:
    """A key as a person may see it: up to six leading and four trailing characters, and never more than a quarter of
    the key together. A full NIM key (about 64 characters) shows the familiar "nvapi-...abcd" form; a short key shows less."""
    edge = len(key) // 8
    if edge == 0:
        return "..."
    return f"{key[:min(6, edge)]}...{key[-min(4, edge):]}"


def key_status(http_status: int | None) -> str:
    """Map a key test's HTTP status: 200 is valid, 401 and 403 invalid, anything else (429, 5xx, timeouts) untested."""
    if http_status == 200:
        return "valid"
    if http_status in (401, 403):
        return "invalid"
    return "untested"


def _now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _valid_budget(value: Any) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)


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
        """Validate and persist a partial update. Raises ValueError naming the first problem; nothing is written then."""
        if not isinstance(patch, dict):
            raise ValueError("settings must be an object")
        current = self.values()
        llm_patch: dict[str, Any] | None = None
        for section, body in patch.items():
            if section in KEY_SECTIONS:
                continue                             # keys are handled by set_keys and save_key_checks
            if section == LLM_SECTION:
                if not isinstance(body, dict) or set(body) - set(LLM_FIELDS):
                    raise ValueError("llm accepts only providers, models, budgets and provider")
                llm_patch = body
                continue
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
            stored = self._raw()                     # keep sections this method does not own, such as key_checks
            if llm_patch is not None:
                stored[LLM_SECTION] = self._llm_after(stored, llm_patch)   # validates before anything is written
            stored.update(current)
            self._write_json(self.path, stored)
        if llm_patch is not None:
            reset_shared_client()
        return current

    def _raw(self) -> dict[str, Any]:
        try:
            stored = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return stored if isinstance(stored, dict) else {}

    def goal_config(self, base: GoalConfig) -> GoalConfig:
        """The goal loop's limits as the person set them. Read at every start, so a change applies to the next part."""
        v = self.values()
        return replace(base, max_iterations=v["agent"]["part_steps"], sub_steps=v["agent"]["sub_steps"],
                       view_chars=v["context"]["view_chars"], keep_recent=v["context"]["keep_recent"],
                       memory_chars=v["context"]["memory_chars"])

    def default_budget(self) -> dict[str, int]:
        v = self.values()["agent"]
        return {"max_tool_calls": v["tool_calls_per_part"], "max_seconds": v["part_seconds"]}

    # -- providers and models -----------------------------------------------------------------------------------

    def _llm_stored(self, stored: dict[str, Any]) -> dict[str, dict]:
        """The llm section as stored, normalised. A legacy {"provider": X} means X is on and every other provider is off."""
        raw = stored.get(LLM_SECTION) if isinstance(stored.get(LLM_SECTION), dict) else {}
        providers: dict[str, dict] = {}
        for name, entry in (raw.get("providers") or {}).items():
            if name in PROVIDERS and isinstance(entry, dict) and isinstance(entry.get("enabled"), bool):
                providers[name] = {"enabled": entry["enabled"]}
        if not providers and raw.get("provider") in PROVIDERS:
            providers = {name: {"enabled": name == raw["provider"]} for name in PROVIDERS}
        models = {m: {"enabled": e["enabled"]} for m, e in (raw.get("models") or {}).items()
                  if m in MODELS and isinstance(e, dict) and isinstance(e.get("enabled"), bool)}
        budgets: dict[str, dict] = {}
        for m, b in (raw.get("budgets") or {}).items():
            if m in MODELS and isinstance(b, dict):
                budgets[m] = {w: b[w] for w in WINDOWS if w in b and _valid_budget(b[w])}
        return {"providers": providers, "models": models, "budgets": budgets}

    def _llm_after(self, stored: dict[str, Any], patch: dict[str, Any]) -> dict[str, dict]:
        llm = self._llm_stored(stored)
        if "provider" in patch:
            name = patch["provider"]
            if name not in PROVIDERS:
                raise ValueError(f"unknown provider {name!r}; choose one of {', '.join(PROVIDERS)}")
            llm["providers"] = {p: {"enabled": p == name} for p in PROVIDERS}      # the legacy meaning: only this one
        if "providers" in patch:
            body = patch["providers"]
            if not isinstance(body, dict):
                raise ValueError("llm.providers maps a provider name to {enabled: true or false}")
            for name, entry in body.items():
                if name not in PROVIDERS:
                    raise ValueError(f"unknown provider {name!r}; choose one of {', '.join(PROVIDERS)}")
                if not isinstance(entry, dict) or set(entry) != {"enabled"} or not isinstance(entry["enabled"], bool):
                    raise ValueError(f"llm.providers.{name} accepts only enabled: true or false")
                llm["providers"][name] = {"enabled": entry["enabled"]}
        if "models" in patch:
            body = patch["models"]
            if not isinstance(body, dict):
                raise ValueError("llm.models maps a model id to {enabled: true or false}")
            for model, entry in body.items():
                if model not in MODELS:
                    raise ValueError(f"unknown model {model!r}")
                if not isinstance(entry, dict) or set(entry) != {"enabled"} or not isinstance(entry["enabled"], bool):
                    raise ValueError(f"llm.models.{model} accepts only enabled: true or false")
                llm["models"][model] = {"enabled": entry["enabled"]}
        if "budgets" in patch:
            body = patch["budgets"]
            if not isinstance(body, dict):
                raise ValueError("llm.budgets maps a model id to {1d, 7d, 30d}")
            for model, entry in body.items():
                if model not in MODELS:
                    raise ValueError(f"unknown model {model!r}")
                if not isinstance(entry, dict) or set(entry) - set(WINDOWS):
                    raise ValueError(f"budgets for {model} accept only 1d, 7d and 30d")
                for window, value in entry.items():
                    if not _valid_budget(value):
                        raise ValueError(f"budget {model} {window} must be null or a whole number of tokens, 0 or more")
                llm["budgets"][model] = {**llm["budgets"].get(model, {}), **entry}
        return llm

    def budgets(self) -> dict[str, dict[str, int | None]]:
        """Every catalog model with its token ceilings. None means no limit."""
        stored = self._llm_stored(self._raw())["budgets"]
        return {m.id: {w: stored.get(m.id, {}).get(w) for w in WINDOWS} for m in CATALOG}

    def provider_enabled(self, name: str, environ: Mapping[str, str] | None = None) -> bool:
        """A provider the person switched on or off. Without a choice it is on only when it has a usable key."""
        explicit = self._llm_stored(self._raw())["providers"].get(name)
        if explicit is not None:
            return explicit["enabled"]
        return bool(self.usable_keys(name, environ))

    def model_enabled(self, model_id: str) -> bool:
        entry = self._llm_stored(self._raw())["models"].get(model_id)
        return True if entry is None else entry["enabled"]

    def provider_ready(self, name: str, environ: Mapping[str, str] | None = None) -> bool:
        """Ready = at least one usable (not invalid) key, and an endpoint. Enabled is checked separately."""
        spec = PROVIDERS[name]
        return bool(self.usable_keys(name, environ)) and bool(spec.resolve_base_url(environ)) and bool(spec.models())

    def _not_ready_reason(self, name: str, environ: Mapping[str, str] | None) -> str:
        spec = PROVIDERS[name]
        if not self.usable_keys(name, environ):
            if self.key_slots(environ, name):
                return f"{spec.label} keys were all marked invalid; test them again or add a new key"
            return f"{spec.label} keys are not configured; the planner cannot run"
        if not spec.resolve_base_url(environ):
            return f"{spec.label} has no endpoint yet; set {spec.base_url_var} in .env"
        return f"{spec.label} has no models"

    def first_enabled(self, environ: Mapping[str, str] | None = None) -> str | None:
        """The first enabled provider in catalog order. This is what ``llm.provider`` reports."""
        return next((n for n in PROVIDERS if self.provider_enabled(n, environ)), None)

    def check_model_choice(self, model_id: str, environ: Mapping[str, str] | None = None) -> None:
        """Raise ValueError for an unknown model, or one whose model or provider is switched off."""
        if model_id not in MODELS:
            raise ValueError(f"unknown model {model_id!r}")
        info = MODELS[model_id]
        if not self.provider_enabled(info.provider, environ):
            raise ValueError(f"{PROVIDERS[info.provider].label} is turned off in Settings")
        if not self.model_enabled(model_id):
            raise ValueError(f"{info.label} is turned off in Settings")

    def planner_choice(self, model: str | None = None, environ: Mapping[str, str] | None = None) -> tuple[str, tuple[str, ...]]:
        """The provider and the model chain a mission uses. With a model: that model alone. Without: the enabled, ready
        provider that comes first in catalog order, with its enabled models in catalog order (the first is tried first;
        the rest are failover within that provider). Raises ValueError for an unknown or switched-off model and
        NotConfigured when nothing can plan yet."""
        if model is not None:
            self.check_model_choice(model, environ)
            name = MODELS[model].provider
            if not self.provider_ready(name, environ):
                raise NotConfigured(self._not_ready_reason(name, environ))
            # The chosen model is tried first. When it is rate-limited or overloaded the call cascades to the provider's
            # other enabled models; it never leaves the provider the person chose.
            others = tuple(m.id for m in CATALOG if m.provider == name and m.id != model and self.model_enabled(m.id))
            return name, (model, *others)
        enabled = [n for n in PROVIDERS if self.provider_enabled(n, environ)]
        if not enabled:
            raise NotConfigured("no model provider is turned on; turn one on in Settings and add its key")
        for name in enabled:
            if not self.provider_ready(name, environ):
                continue
            chain = tuple(m.id for m in CATALOG if m.provider == name and self.model_enabled(m.id))
            if chain:
                return name, chain
        first = enabled[0]
        if not self.provider_ready(first, environ):
            raise NotConfigured(self._not_ready_reason(first, environ))
        raise NotConfigured(f"{PROVIDERS[first].label} has no models turned on")

    def active_provider(self, environ: Mapping[str, str] | None = None) -> str:
        """The provider the planner uses by default; the first enabled one, or NIM when nothing is enabled."""
        try:
            return self.planner_choice(environ=environ)[0]
        except NotConfigured:
            return self.first_enabled(environ) or DEFAULT_PROVIDER

    def planner_problem(self, environ: Mapping[str, str] | None = None, model: str | None = None) -> str | None:
        """Why the planner cannot run, in words a person can act on. None when it can."""
        try:
            self.planner_choice(model=model, environ=environ)
        except (ValueError, NotConfigured) as exc:
            return str(exc)
        return None

    # -- NIM and provider keys ---------------------------------------------------------------------------------

    def key_slots(self, environ: dict[str, str] | None = None, provider: str = DEFAULT_PROVIDER) -> list[str]:
        from prophet_agent.nim.transport import load_keys
        return load_keys(environ, provider)

    def _checks_by_slot(self, provider: str) -> dict[int, dict]:
        rows = (self._raw().get(KEY_CHECKS) or {}).get(provider) or []
        return {row["slot"]: row for row in rows if isinstance(row, dict) and isinstance(row.get("slot"), int)}

    def usable_keys(self, provider: str = DEFAULT_PROVIDER, environ: Mapping[str, str] | None = None) -> list[str]:
        """The provider's keys minus any whose last check was invalid (401 or 403). This is what the planner uses."""
        keys = self.key_slots(None if environ is None else dict(environ), provider)
        checks = self._checks_by_slot(provider)
        return [k for slot, k in enumerate(keys, 1) if checks.get(slot, {}).get("status") != "invalid"]

    def masked_keys(self, provider: str = DEFAULT_PROVIDER) -> list[dict[str, Any]]:
        checks = self._checks_by_slot(provider)
        out = []
        for slot, key in enumerate(self.key_slots(None, provider), 1):
            c = checks.get(slot, {})
            out.append({"slot": slot, "masked": mask(key), "status": c.get("status", "untested"),
                        "http_status": c.get("http_status"), "checked_at": c.get("checked_at")})
        return out

    def save_key_checks(self, provider: str, rows: list[dict]) -> list[dict]:
        """Persist one key test per slot. ``rows`` are probe_provider rows, whose ``status`` is the HTTP code (None on a
        transport error). Returns the rows with status valid, invalid or untested, the HTTP code, and the masked key."""
        get_provider(provider)
        checked_at = _now()
        keys = self.key_slots(None, provider)
        stored_rows, out = [], []
        for row in rows:
            http = row.get("status") if isinstance(row.get("status"), int) else None
            slot = row["slot"]
            status = key_status(http)
            stored_rows.append({"slot": slot, "status": status, "http_status": http, "checked_at": checked_at})
            out.append({**row, "status": status, "http_status": http, "checked_at": checked_at, "ok": status == "valid",
                        "masked": mask(keys[slot - 1]) if slot <= len(keys) else None})
        with self._lock:
            stored = self._raw()
            checks = stored.get(KEY_CHECKS) if isinstance(stored.get(KEY_CHECKS), dict) else {}
            checks[provider] = stored_rows
            stored[KEY_CHECKS] = checks
            self._write_json(self.path, stored)
        reset_shared_client(provider)
        return out

    def _drop_key_checks(self, provider: str) -> None:
        with self._lock:
            stored = self._raw()
            checks = stored.get(KEY_CHECKS) if isinstance(stored.get(KEY_CHECKS), dict) else {}
            if checks.pop(provider, None) is not None:
                stored[KEY_CHECKS] = checks
                self._write_json(self.path, stored)

    def set_keys(self, keys: list[str], provider: str = DEFAULT_PROVIDER) -> list[str]:
        """Replace one provider's key list. An entry that is the masked form of an existing key keeps that key, so a form
        that echoes masked values back cannot erase a key. Writes .env (owner-only) and the process environment. The
        provider's key checks are cleared, because the slots now mean different keys."""
        spec = get_provider(provider)
        if not isinstance(keys, list) or len(keys) > MAX_KEYS:
            raise ValueError(f"keys must be a list of at most {MAX_KEYS} entries")
        existing = self.key_slots(None, provider)
        pattern = KEY_PATTERN if provider == DEFAULT_PROVIDER else GENERIC_KEY_PATTERN
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
            if not pattern.match(value):
                if provider == DEFAULT_PROVIDER:
                    raise ValueError("a NIM key starts with nvapi- and contains only letters, digits, '-' and '_'")
                raise ValueError(f"a {spec.label} key contains only letters, digits, '-', '_' and '.' (8 or more)")
            if value not in resolved:
                resolved.append(value)
        if not spec.multi_key and len(resolved) > 1:
            raise ValueError(f"{spec.label} takes one key")
        self._write_dotenv(spec, resolved)
        os.environ[spec.key_list_var] = ",".join(resolved)
        for alias in spec.key_aliases:
            os.environ[alias] = ""
        if spec.key_prefix:
            for i in range(1, 10):                   # blank the indexed form so an old value cannot come back
                os.environ[f"{spec.key_prefix}_{i}"] = ""
                for alias in spec.key_aliases:
                    os.environ[f"{alias}_{i}"] = ""
        self._drop_key_checks(provider)
        reset_shared_client(provider)
        return resolved

    def _write_dotenv(self, spec: Provider, keys: list[str]) -> None:
        names = [re.escape(spec.key_list_var), *(re.escape(a) for a in spec.key_aliases)]
        if spec.key_prefix:
            names.append(re.escape(spec.key_prefix) + r"_[1-9]")
        for alias in spec.key_aliases:
            names.append(re.escape(alias) + r"_[1-9]")
        env_line = re.compile(r"^\s*(export\s+)?(" + "|".join(names) + r")\s*=")
        try:
            lines = self.dotenv.read_text().splitlines()
        except OSError:
            lines = []
        kept = [line for line in lines if not env_line.match(line)]
        if keys:
            kept.append(f"{spec.key_list_var}={','.join(keys)}")
        body = "\n".join(kept).rstrip() + "\n"
        self._write_text(self.dotenv, body, mode=0o600)

    # -- data utilities ----------------------------------------------------------------------------------------

    def summary(self, store, datasets) -> dict[str, int]:
        with store.lock:
            missions = store.conn.execute("SELECT COUNT(*) AS n FROM missions").fetchone()["n"]
            notebooks = store.conn.execute("SELECT COUNT(*) AS n FROM notebooks").fetchone()["n"]
        return {"missions": int(missions), "notebooks": int(notebooks), "datasets": len(datasets.list(limit=10_000)),
                "bytes": int(_data_bytes(self.data_root))}

    def export(self, store, datasets) -> dict[str, Any]:
        """Everything the ledger holds, as plain JSON. Keys and .env are not part of it."""
        with store.lock:
            tables = {name: [dict(r) for r in store.conn.execute(f"SELECT * FROM {name}").fetchall()]
                      for name in ("missions", "steps", "events", "questions", "notebooks", "sources")}
        return {"exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "settings": self.values(),
                "datasets": datasets.list(limit=10_000), **tables}

    def clear(self, store, active_statuses: tuple[str, ...], confirm: str) -> dict[str, int]:
        """Delete missions, notebooks, uploads and sandbox output. Settings, keys, usage and model checks stay. Refused while
        work is running."""
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
            if child.name in KEPT_ON_CLEAR or child.suffix in (".sqlite", ".sqlite-wal", ".sqlite-shm") \
                    or child.name.endswith((".sqlite-wal", ".sqlite-shm")):
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        _forget_data_bytes(self.data_root)
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
    """What the Settings page shows. No key material: masked keys, key checks and model checks only."""
    keys = settings.masked_keys()
    active = settings.active_provider()
    checks = read_model_checks(settings.data_root / MODEL_CHECK_FILE)
    providers = []
    for name, spec in PROVIDERS.items():
        pkeys = settings.masked_keys(name)
        models = []
        for info in (m for m in CATALOG if m.provider == name):
            status = model_check(info.id, checks)
            models.append({"id": info.id, "label": info.label, "enabled": settings.model_enabled(info.id),
                           "status": status["status"], "checked_at": status["checked_at"]})
        endpoint = spec.resolve_base_url()
        providers.append({
            "name": name, "label": spec.label, "enabled": settings.provider_enabled(name),
            "ready": settings.provider_ready(name), "endpoint": endpoint, "multi_key": spec.multi_key,
            "key_hint": spec.key_hint, "keys": pkeys, "models": models,
            # Kept from the earlier shape so existing readers keep working.
            "configured": bool(pkeys), "endpoint_set": bool(endpoint), "active": name == active,
        })
    first = settings.first_enabled()
    return {
        "nim": {"configured": bool(keys), "keys": keys, "models": list(DEFAULT_CHAIN), "pool": pool_status},
        "llm": {"provider": first, "providers": providers, "budgets": settings.budgets(), "pool": pool_status},
        "agent": settings.values()["agent"],
        "context": settings.values()["context"],
        "defaults": {section: dict(vals) for section, vals in DEFAULTS.items()},
        "bounds": {section: {name: [low, high] for (s, name), (low, high) in BOUNDS.items() if s == section}
                   for section in DEFAULTS},
        "timesfm": {"checkpoint": TIMESFM_CHECKPOINT, "engine_installed": importlib.util.find_spec("timesfm") is not None},
        "sandbox": _sandbox_limits(),
        "data": settings.summary(store, datasets),
        "about": {"name": "Prophet", "clear_phrase": CLEAR_PHRASE},
    }
