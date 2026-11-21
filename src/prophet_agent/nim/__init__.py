"""Model access for Prophet: NIM (default) plus Claude, ChatGPT, Gemini, Grok, Matilda and FreeRouter. Public surface: NimClient,
Reply, load_keys, the provider registry, DEFAULT_CHAIN and the error types."""
from .errors import Exhausted, NimError, NotConfigured, SchemaViolation, redact
from .providers import PROVIDERS, Provider
from .transport import DEFAULT_CHAIN, PROBE_MODEL, NimClient, Reply, clean_text, load_keys, probe_provider

__all__ = [
    "DEFAULT_CHAIN", "PROBE_MODEL", "PROVIDERS", "Provider", "probe_provider", "Exhausted", "NimClient", "NimError", "NotConfigured", "Reply",
    "SchemaViolation", "clean_text", "load_keys", "redact",
]
