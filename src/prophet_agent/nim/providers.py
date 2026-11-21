"""Model providers Prophet can call: where each one lives, which env vars hold its keys, and which models form its chain.

Motivation vs Logic
Motivation: NVIDIA NIM stays the default, but a person may want Claude, ChatGPT, Gemini, Grok or Matilda for planning.
Each one needs its own keys, endpoint and model list, and Settings must show them one at a time, not as one long list.
Logic: one frozen ``Provider`` per vendor. Every entry except Claude speaks the OpenAI chat-completions wire format, so the
single transport in ``transport.py`` serves all of them; the per-provider fields only change the handful of request
details that differ (token parameter, temperature, schema handling, NIM-only extras). Keys are read from environment
variables and the git-ignored ``.env``; the active provider is stored in ``data/settings.json``. Values are never logged.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..config import DATA, load_dotenv_values

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_DEFAULT_CHAIN: tuple[str, ...] = (
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "poolside/laguna-xs-2.1",
    "z-ai/glm-5.3-flash",
)


@dataclass(frozen=True)
class Provider:
    name: str                              # stable id used in settings and the API: nim, claude, openai, gemini, grok, matilda
    label: str                             # what a person reads in Settings
    key_list_var: str                      # env var holding one key or a comma list
    key_prefix: str | None                 # env prefix for indexed keys: <prefix>_1 .. <prefix>_9 (multi-key providers only)
    base_url: str | None                   # fixed endpoint; None means it must come from base_url_var
    base_url_var: str | None = None        # env var that supplies or overrides the endpoint
    default_models: tuple[str, ...] = ()   # chain tried in order; empty means the provider is not usable until configured
    probe_model: str | None = None         # cheapest model for a key test; defaults to the last model in the chain
    nim_extras: bool = False               # NIM-only request fields (chat_template_kwargs, GLM reasoning_effort)
    token_param: str = "max_tokens"        # OpenAI's newer models need max_completion_tokens
    send_temperature: bool = True          # reasoning-model endpoints reject a custom temperature
    schema_mode: str = "json_schema"       # "prompt": no response_format; the schema goes into the prompt and is checked locally

    @property
    def multi_key(self) -> bool:
        return self.key_prefix is not None

    @property
    def key_hint(self) -> str:
        if self.multi_key:
            return f"{self.key_list_var} and {self.key_prefix}_1..9"
        return self.key_list_var

    @property
    def models_var(self) -> str:
        return f"PROPHET_{self.name.upper()}_MODELS"

    def resolve_base_url(self, environ: Mapping[str, str] | None = None) -> str | None:
        env = _environ(environ)
        if self.base_url_var and env.get(self.base_url_var, "").strip():
            return env[self.base_url_var].strip().rstrip("/")
        return self.base_url

    def models(self, environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
        """The chain for this provider: PROPHET_<NAME>_MODELS (comma list) if set, else the built-in default."""
        raw = _environ(environ).get(self.models_var, "")
        names = [m.strip() for m in raw.split(",") if m.strip()]
        return tuple(dict.fromkeys(names)) or self.default_models

    def probe_target(self, environ: Mapping[str, str] | None = None) -> str:
        if self.probe_model:
            return self.probe_model
        chain = self.models(environ)
        return chain[-1] if chain else ""

    def usable(self, environ: Mapping[str, str] | None = None) -> bool:
        """Endpoint and model chain are known. Keys are checked separately."""
        return bool(self.resolve_base_url(environ)) and bool(self.models(environ))


PROVIDERS: dict[str, Provider] = {
    "nim": Provider(
        name="nim", label="NVIDIA NIM", key_list_var="NVIDIA_NIM_API_KEYS", key_prefix="NVIDIA_NIM_API_KEY",
        base_url=NIM_BASE_URL, default_models=NIM_DEFAULT_CHAIN, probe_model="poolside/laguna-xs-2.1",
        nim_extras=True,
    ),
    "claude": Provider(
        name="claude", label="Claude", key_list_var="ANTHROPIC_API_KEY", key_prefix=None,
        base_url="https://api.anthropic.com/v1",           # Anthropic's OpenAI-compatible endpoint
        default_models=("claude-sonnet-5-5", "claude-haiku-5-5"), schema_mode="prompt",
    ),
    "openai": Provider(
        name="openai", label="ChatGPT", key_list_var="OPENAI_API_KEY", key_prefix=None,
        base_url="https://api.openai.com/v1", default_models=("gpt-5", "gpt-5-mini"),
        token_param="max_completion_tokens", send_temperature=False,
    ),
    "gemini": Provider(
        name="gemini", label="Gemini", key_list_var="GOOGLE_API_KEY", key_prefix="GOOGLE_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_models=("gemini-2.5-pro", "gemini-2.5-flash"),
    ),
    "grok": Provider(
        name="grok", label="Grok", key_list_var="XAI_API_KEY", key_prefix=None,
        base_url="https://api.x.ai/v1", default_models=("grok-4", "grok-3-mini"),
    ),
    "matilda": Provider(
        # Motivation: Maincode's Matilda is OpenAI-compatible at this endpoint with a Bearer "mc_live_" key and one model id.
        # Logic: MATILDA_BASE_URL overrides the verified default. Matilda rejects sampling controls (temperature, top_p,
        # stop) and reasoning_effort, so no temperature is sent. Live-verified 2026-11-21: stream_options is accepted.
        # Sources: FreeRouter docs/API_CONTRACTS.md and src/lib/providers/openai.ts (buildMatildaBody).
        name="matilda", label="Matilda", key_list_var="MATILDA_API_KEY", key_prefix="MATILDA_API_KEY",
        base_url="https://matilda.maincode.com/api/v1", base_url_var="MATILDA_BASE_URL", default_models=("matilda",),
        send_temperature=False,
    ),
}
DEFAULT_PROVIDER = "nim"


def get_provider(name: str) -> Provider:
    try:
        return PROVIDERS[name]
    except KeyError:
        raise ValueError(f"unknown provider {name!r}; choose one of {', '.join(PROVIDERS)}") from None


def _environ(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    if environ is None:
        return {**load_dotenv_values(), **os.environ}
    return environ


def load_provider_keys(provider: Provider, environ: Mapping[str, str] | None = None) -> list[str]:
    """The provider's keys: the comma/space list first, then indexed names for multi-key providers. De-duplicated, order
    kept. The process environment wins over the git-ignored .env file. Values are returned, never printed."""
    import re
    env = _environ(environ)
    raw = re.split(r"[,\s]+", env.get(provider.key_list_var, ""))
    if provider.key_prefix:
        raw += [env.get(f"{provider.key_prefix}_{i}", "") for i in range(1, 10)]
    return list(dict.fromkeys(k.strip() for k in raw if k and k.strip()))


def active_provider_name(data_root: Path | None = None) -> str:
    """The provider the planner uses, from data/settings.json ("llm": {"provider": ...}). Defaults to NIM."""
    path = (data_root or DATA) / "settings.json"
    try:
        stored = json.loads(path.read_text())
    except (OSError, ValueError):
        return DEFAULT_PROVIDER
    llm = stored.get("llm") if isinstance(stored, dict) else None
    name = llm.get("provider") if isinstance(llm, dict) else None
    return name if name in PROVIDERS else DEFAULT_PROVIDER


def planner_problem(data_root: Path | None = None, environ: Mapping[str, str] | None = None) -> str | None:
    """Why the active provider cannot plan yet, in words a person can act on. None when it is ready."""
    spec = PROVIDERS[active_provider_name(data_root)]
    if not load_provider_keys(spec, environ):
        return f"{spec.label} keys are not configured; the planner cannot run"
    if not spec.usable(environ):
        return f"{spec.label} has no endpoint or model list yet; set {spec.base_url_var or spec.models_var} in .env"
    return None
