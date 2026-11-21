"""Model providers Prophet can call: where each one lives, which env vars hold its keys, and which models it offers.

Motivation vs Logic
Motivation: NVIDIA NIM stays the default, but a person may want Claude, ChatGPT, Gemini, Grok or Matilda for planning.
Each one needs its own keys, endpoint and model list, and Settings must show them one at a time, not as one long list.
Logic: one frozen ``Provider`` per vendor and one ``ModelInfo`` per model in ``CATALOG``. The catalog is the only list of
models Prophet offers: Settings, the planner and the usage page all read it. Every provider except Claude speaks the
OpenAI chat-completions wire format, so the single transport in ``transport.py`` serves all of them. Keys are read from
environment variables and the git-ignored ``.env``. Which providers and models are switched on, key checks and budgets
live in ``data/settings.json`` (see ``settings.py``). Values are never logged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..config import load_dotenv_values

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_DEFAULT_CHAIN: tuple[str, ...] = (
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "poolside/laguna-xs-2.1",
    "z-ai/glm-5.3-flash",
)


@dataclass(frozen=True)
class ModelInfo:
    id: str                                # the id sent to the vendor
    provider: str                          # a key of PROVIDERS
    label: str                             # what a person reads


# The model catalog, in planner order: providers in PROVIDERS order, models in the order listed here.
CATALOG: tuple[ModelInfo, ...] = (
    ModelInfo("nvidia/nemotron-3-ultra-550b-a55b", "nim", "Nemotron 3 Ultra 550B"),
    ModelInfo("nvidia/nemotron-3.5-super-vl-120b-a12b", "nim", "Nemotron 3.5 Super VL 120B"),
    ModelInfo("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "nim", "Nemotron 3 Nano Omni 30B (reasoning)"),
    ModelInfo("poolside/laguna-xs-2.1", "nim", "Poolside Laguna XS 2.1"),
    ModelInfo("z-ai/glm-5.3-flash", "nim", "GLM 5.3 Flash"),
    ModelInfo("claude-haiku-5-5", "claude", "Claude Haiku 5.5"),
    ModelInfo("claude-sonnet-5-5", "claude", "Claude Sonnet 5.5"),
    ModelInfo("claude-opus-5-5", "claude", "Claude Opus 5.5"),
    ModelInfo("claude-fable-5-2", "claude", "Claude Fable 5.2"),
    ModelInfo("gpt-6-luna", "openai", "GPT-6 Luna"),
    ModelInfo("gpt-6.1-terra", "openai", "GPT-6.1 Terra"),
    ModelInfo("gpt-6.2-sol", "openai", "GPT-6.2 Sol"),
    ModelInfo("gpt-6.2-astra", "openai", "GPT-6.2 Astra"),
    ModelInfo("gemini-3.5-flash", "gemini", "Gemini 3.5 Flash"),
    ModelInfo("gemini-3.6-flash", "gemini", "Gemini 3.6 Flash"),
    ModelInfo("gemini-3.7-flash", "gemini", "Gemini 3.7 Flash"),
    ModelInfo("gemini-3.8-flash", "gemini", "Gemini 3.8 Flash"),
    ModelInfo("gemini-3.1-flash-lite", "gemini", "Gemini 3.1 Flash-Lite"),
    ModelInfo("gemini-3.5-flash-lite", "gemini", "Gemini 3.5 Flash-Lite"),
    ModelInfo("grok-4.7", "grok", "Grok 4.7"),
    ModelInfo("grok-4.6", "grok", "Grok 4.6"),
    ModelInfo("grok-4.3", "grok", "Grok 4.3"),
    ModelInfo("matilda", "matilda", "Matilda"),
)
MODELS: dict[str, ModelInfo] = {m.id: m for m in CATALOG}


def catalog_ids(provider: str) -> tuple[str, ...]:
    return tuple(m.id for m in CATALOG if m.provider == provider)


@dataclass(frozen=True)
class Provider:
    name: str                              # stable id used in settings and the API: nim, claude, openai, gemini, grok, matilda
    label: str                             # what a person reads in Settings
    key_list_var: str                      # env var holding one key or a comma list
    key_prefix: str | None                 # env prefix for indexed keys: <prefix>_1 .. <prefix>_9 (multi-key providers only)
    base_url: str | None                   # fixed endpoint; None means it must come from base_url_var
    base_url_var: str | None = None        # env var that supplies or overrides the endpoint
    default_models: tuple[str, ...] = ()   # the catalog's model ids for this provider, in order
    probe_model: str | None = None         # cheapest model for a key test; defaults to the last model in the list
    nim_extras: bool = False               # NIM-only request fields (chat_template_kwargs, GLM reasoning_effort)
    token_param: str = "max_tokens"        # OpenAI's newer models need max_completion_tokens
    send_temperature: bool = True          # reasoning-model endpoints reject a custom temperature
    schema_mode: str = "json_schema"       # "prompt": no response_format; the schema goes into the prompt and is checked locally
    key_aliases: tuple[str, ...] = ()      # other names read the same way (list and indexed); the primary name is written

    @property
    def multi_key(self) -> bool:
        return self.key_prefix is not None

    @property
    def key_hint(self) -> str:
        if self.multi_key:
            return f"{self.key_list_var} and {self.key_prefix}_1..9"
        return self.key_list_var

    def resolve_base_url(self, environ: Mapping[str, str] | None = None) -> str | None:
        env = _environ(environ)
        if self.base_url_var and env.get(self.base_url_var, "").strip():
            return env[self.base_url_var].strip().rstrip("/")
        return self.base_url

    def models(self, environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
        """The catalog's model ids for this provider. The catalog is the only source; there is no per-provider override."""
        return self.default_models

    def probe_target(self, environ: Mapping[str, str] | None = None) -> str:
        if self.probe_model:
            return self.probe_model
        chain = self.models(environ)
        return chain[-1] if chain else ""

    def usable(self, environ: Mapping[str, str] | None = None) -> bool:
        """Endpoint and model list are known. Keys are checked separately."""
        return bool(self.resolve_base_url(environ)) and bool(self.models(environ))


PROVIDERS: dict[str, Provider] = {
    "nim": Provider(
        name="nim", label="NVIDIA NIM", key_list_var="NVIDIA_NIM_API_KEYS", key_prefix="NVIDIA_NIM_API_KEY",
        base_url=NIM_BASE_URL, default_models=NIM_DEFAULT_CHAIN, probe_model="poolside/laguna-xs-2.1",
        nim_extras=True,
    ),
    # Anthropic's OpenAI-compatible endpoint (verified 2026-11-21, platform.claude.com/docs/en/api/openai-sdk). Sonnet 5.5
    # and later reject any temperature below 1 with a 400, so temperature is not sent. Schema is checked locally (prompt
    # mode): the compatibility layer accepts json_schema strict, but its docs say strict tool schemas are ignored.
    # The probe is Haiku, the cheapest Claude in the catalog.
    "claude": Provider(
        name="claude", label="Claude", key_list_var="ANTHROPIC_API_KEY", key_prefix=None,
        base_url="https://api.anthropic.com/v1", default_models=catalog_ids("claude"), probe_model="claude-haiku-5-5",
        send_temperature=False, schema_mode="prompt",
    ),
    # OpenAI (verified 2026-11-21, developers.openai.com model pages). GPT-6 models are reasoning models: chat completions
    # takes max_completion_tokens and the pages say to omit temperature. The probe is GPT-6 Luna, the smallest in the catalog.
    "openai": Provider(
        name="openai", label="ChatGPT", key_list_var="OPENAI_API_KEY", key_prefix=None,
        base_url="https://api.openai.com/v1", default_models=catalog_ids("openai"), probe_model="gpt-6-luna",
        token_param="max_completion_tokens", send_temperature=False,
    ),
    # Google AI Studio. Keys are GOOGLE_API_KEY (list) or GOOGLE_API_KEY_1..9; GEMINI_API_KEY_n is also read, as free-router uses it.
    # Quotas apply per Google project, so keys from different projects add capacity. The probe is the 3.1 Flash-Lite model.
    "gemini": Provider(
        name="gemini", label="Gemini", key_list_var="GOOGLE_API_KEY", key_prefix="GOOGLE_API_KEY",
        key_aliases=("GEMINI_API_KEY",),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai", default_models=catalog_ids("gemini"),
        probe_model="gemini-3.5-flash-lite",     # verified 200 in the one-call check; 3.1-flash-lite gave 403
    ),
    # xAI (verified 2026-11-21, docs.x.ai): Bearer auth at https://api.x.ai/v1; chat completions documents tools and
    # response_format json_schema. Streaming on the raw endpoint is not stated in the docs; confirm live.
    "grok": Provider(
        name="grok", label="Grok", key_list_var="XAI_API_KEY", key_prefix=None,
        base_url="https://api.x.ai/v1", default_models=catalog_ids("grok"), probe_model="grok-4.3",
    ),
    "matilda": Provider(
        # Motivation: Maincode's Matilda is OpenAI-compatible at this endpoint with a Bearer "mc_live_" key and one model id.
        # Logic: MATILDA_BASE_URL overrides the verified default. Matilda rejects sampling controls (temperature, top_p,
        # stop) and reasoning_effort, so no temperature is sent. Live-verified 2026-11-21: stream_options is accepted.
        # Sources: FreeRouter docs/API_CONTRACTS.md and src/lib/providers/openai.ts (buildMatildaBody).
        name="matilda", label="Matilda", key_list_var="MATILDA_API_KEY", key_prefix="MATILDA_API_KEY",
        base_url="https://matilda.maincode.com/api/v1", base_url_var="MATILDA_BASE_URL", default_models=catalog_ids("matilda"),
        send_temperature=False,
    ),
}
DEFAULT_PROVIDER = "nim"


def get_provider(name: str) -> Provider:
    try:
        return PROVIDERS[name]
    except KeyError:
        raise ValueError(f"unknown provider {name!r}; choose one of {', '.join(PROVIDERS)}") from None


def model_info(model_id: str) -> ModelInfo:
    try:
        return MODELS[model_id]
    except KeyError:
        raise ValueError(f"unknown model {model_id!r}") from None


def _environ(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    if environ is None:
        return {**load_dotenv_values(), **os.environ}
    return environ


def load_provider_keys(provider: Provider, environ: Mapping[str, str] | None = None) -> list[str]:
    """The provider's keys: the comma/space list first, then indexed names for multi-key providers. De-duplicated, order
    kept. The process environment wins over the git-ignored .env file. Values are returned, never printed.
    This is the lower-level loader; the usable (non-invalid) keys come from ``Settings.usable_keys``."""
    import re
    env = _environ(environ)
    raw: list[str] = []
    for name in (provider.key_list_var, *provider.key_aliases):
        raw += re.split(r"[,\s]+", env.get(name, ""))
    for prefix in (provider.key_prefix, *provider.key_aliases):
        if prefix and provider.multi_key:
            raw += [env.get(f"{prefix}_{i}", "") for i in range(1, 10)]
    return list(dict.fromkeys(k.strip() for k in raw if k and k.strip()))


def active_provider_name(data_root: Path | None = None, environ: Mapping[str, str] | None = None) -> str:
    """The provider the planner uses by default: the first enabled, ready provider in catalog order (see Settings)."""
    from ..settings import Settings
    return Settings(data_root=data_root).active_provider(environ)


def planner_problem(data_root: Path | None = None, environ: Mapping[str, str] | None = None) -> str | None:
    """Why the planner cannot run with the current Settings, in words a person can act on. None when it can."""
    from ..settings import Settings
    return Settings(data_root=data_root).planner_problem(environ=environ)
