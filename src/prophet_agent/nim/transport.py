"""NVIDIA NIM chat transport: key pool x model fallback chain, client-side pacing, failure policy, schema fallbacks.

Contract: NIM.md §3 (wire body), §4 (routing and the failure matrix §4.2), §5 (streaming), §8 (probes), §10 (defects
F1-F6, all fixed here). Verified live 2026-11-20 by the reference client in NIM.md §9.

Motivation vs Logic
Motivation: the agent runtime calls NIM many times per mission. A single busy key must not stall the call, a dead
model must not burn the whole budget, a key must never reach a log or an exception, and a bad schema must not
silently become a bad answer.
Logic: one shared client per process. A logical call tries models in order (demoted models last, never dropped);
each model may use up to 3 keys and a pass at most 7 dispatches. A 429 or 503 is waited out, not counted as final. Every HTTP request gets a timeout equal to the
time left on the wall-clock budget (F1). Each status code has one consequence: cool (key, model) for 429/408/425/402/
5xx; retire the key for 401/403 (re-admitted after 10 min, so a rotated key recovers, F4); demote the model for
404/410/503; a 413 (request too large) moves to the next model without demotion; reshape the schema on a 400/422 that names keywords (same key, same model); otherwise next model.
Requests always stream upstream and are assembled here (NIM.md §5 rule 4). Text is released to the caller only when
no schema or tool output needs buffering, and failover happens only before the first released token.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import httpx

from .errors import Exhausted, NimError, NotConfigured, SchemaViolation, redact
from .providers import NIM_BASE_URL, NIM_DEFAULT_CHAIN, PROVIDERS, Provider, active_provider_name, load_provider_keys
from .schema import STRIP_ALWAYS, is_format_error, strip_schema, unimplemented_keys, validate

log = logging.getLogger("prophet.nim")

BASE_URL = NIM_BASE_URL
# NIM.md §2 ranking. Thinking off is the default because it is the speed lever (2-6 s instead of 25-60 s).
DEFAULT_CHAIN: tuple[str, ...] = NIM_DEFAULT_CHAIN
PROBE_MODEL = "poolside/laguna-xs-2.1"       # NIM.md §8: fastest, cheapest key probe
_NIM = PROVIDERS["nim"]

MAX_DISPATCH = 7                  # per logical call (NIM.md §4.2)
MAX_KEYS_PER_MODEL = 3
MAX_RESHAPES = 2                  # schema reshapes per model before demotion
PER_MINUTE = 30                   # client-side pacing per key; free tier is about 40 rpm (NIM.md §4.3)
DEMOTE_S = 120.0                  # model demotion, process-wide (§4.4)
RETIRE_S = 600.0                  # 401/403 retirement, re-admitted afterwards (F4)
COOL_DEFAULT_S = 10.0             # other 5xx and unknown statuses
COOL_KEY_S = 30.0                 # 408/425, transport errors, unreadable JSON
COOL_402_S = 120.0
COOL_429_S = 30.0                 # used when Retry-After is absent or unreadable
RETRY_AFTER_CAP_S = 120.0
MAX_TOKENS_DEFAULT = 16384
GLM_MIN_TOKENS = 1024             # GLM spends small budgets on reasoning (NIM.md §3)
STREAM_IDLE_S = 120.0              # longest silence tolerated inside one streamed response, then fail over
BUDGET_S = 420.0                  # wall-clock budget for one logical call (long schema-bound answers need it)
ATTEMPT_BUDGET_S = 240.0          # one attempt's slice of that budget: a slow model fails over with time left for the next
MAX_CONTINUATIONS = 3             # a free-text answer cut mid-stream is continued this many times, each on a new budget
CONTINUE_PROMPT = ("Your reply was cut off. Continue exactly from where it stopped. Do not repeat anything already "
                   "written and do not add an introduction.")
RATE_STATUSES = frozenset({429, 503})   # rate limit or overload: waited out until the provider is depleted, never final
RATE_WAIT_MAX_S = 60.0            # longest single sleep while every usable key and model is rate-limited; then a new pass
RATE_WAIT_LIMIT_S = 1800.0        # one logical call waits at most this long on rate limits before it reports the refusal
SILENCE_DEMOTE_S = 600.0          # a model that sent nothing within the provider's silence limit goes last this long
MAX_REPAIRS = 2                   # an answer that fails the local schema is re-asked on the same model this many times
REPAIR_QUOTE_CHARS = 4000         # how much of the refused reply the repair turn shows the model
REPAIR_PROMPT = ("Your reply was not usable: {reason}. Reply again with only the JSON object the schema asks for. "
                 "No prose, no code fence.")
OVERLOAD_BASE_S = 30.0            # first 503 on a model: it goes last this long; doubles per repeat inside the window
OVERLOAD_MAX_S = 300.0
OVERLOAD_WINDOW_S = 600.0         # repeats further apart than this start again at the base
RESET_CAP_S = 86400.0             # a limit that names its own reset may cool a key until then, never longer
BILLING_COOL_S = 3600.0           # a spend cap or billing refusal is not a rate limit: do not retry for a long while
DEPLETED_WAIT_S = 3600.0          # when the soonest key is further away than this, the provider is spent: report it
MAX_TRUNCATION_RETRIES = 2        # a schema-bound reply cut by max_tokens is asked again with a larger allowance this often
TRUNCATION_TOKEN_CAP = 32768
SLOW_CALL_S = 15.0                # log calls at least this slow, or any >= 400
# Error codes a gateway puts in an in-stream error frame, mapped to the HTTP status they stand for (FreeRouter's
# envelope: free-router src/lib/http.ts). An unknown code is treated as a 502: the key cools briefly and the call moves on.
_IN_STREAM_STATUS: dict[str, int] = {
    "invalid_api_key": 401, "insufficient_scope": 403, "model_forbidden": 403, "model_not_found": 404,
    "rate_limit_exceeded": 429, "quota_exhausted": 429, "upstream_unavailable": 503, "upstream_invalid_output": 502,
    "upstream_request_rejected": 400, "unsupported_capability": 400, "unsupported_parameter": 400, "invalid_request": 400,
}

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"\s*```(?:json)?\s*(.*?)\s*```\s*", re.DOTALL)
_KEY_SPLIT = re.compile(r"[,\s]+")
_WIRE_NAME = re.compile(r"[^A-Za-z0-9_-]")


# -- value objects -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Reply:
    text: str                          # cleaned: <think> removed, one json fence unwrapped
    tool_calls: tuple[dict, ...]       # {"id", "name", "arguments": dict}; unparseable calls are dropped
    model: str
    finish_reason: str | None
    usage: dict                        # prompt/completion/reasoning/cached token counts when reported
    key_slot: int                      # 1-based position in the pool; never the key itself
    elapsed_s: float


@dataclass
class _Opts:
    thinking: bool | None
    max_tokens: int
    schema: dict | None
    schema_name: str
    tools: list[dict] | None


@dataclass(frozen=True)
class _Shape:
    mode: str                          # "strict" (json_schema) or "json_object" (fallback)
    drop: frozenset = frozenset()      # extra keywords the server named as unimplemented

    def with_drop(self, names: set[str]) -> "_Shape":
        return _Shape(self.mode, self.drop | frozenset(names))


@dataclass
class _Outcome:
    kind: str                          # ok | key | retire | reshape | model | budget | cut
    reason: str = ""
    reply: Reply | None = None
    demote: bool = False
    invalid: bool = False
    shape: _Shape | None = None
    partial: str = ""                  # "cut" only: the raw text that was already released before the stream broke
    status: int | None = None          # the HTTP status of the attempt, when one was received
    detail: str = ""                   # redacted, shortened response body for a failed attempt (model checks show it)
    hung: bool = False                 # the provider sent nothing within its silence limit (not a refusal)
    text: str = ""                     # "model" with invalid=True: the reply that failed the schema, for the repair turn


# -- pure helpers --------------------------------------------------------------------------------------------------

def load_keys(environ: Mapping[str, str] | None = None, provider: str = "nim") -> list[str]:
    """NVIDIA NIM keys by default: NVIDIA_NIM_API_KEYS (comma or space list), then NVIDIA_NIM_API_KEY_1..9.
    Other providers read their own variables (see ``providers.py``). De-duplicated, order kept.

    The process environment wins over the git-ignored .env file. Values are returned, never printed.
    """
    return load_provider_keys(PROVIDERS[provider] if isinstance(provider, str) else provider, environ)


def clean_text(raw: str) -> str:
    """NIM.md §3.2: drop <think> blocks (or everything up to a lone closing tag), unwrap one json fence."""
    text = _THINK.sub("", raw or "")
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.strip()
    m = _FENCE.fullmatch(text)
    return m.group(1).strip() if m else text


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type", "text") == "text")
    return str(content)


def _usage(u: Mapping[str, Any] | None) -> dict:
    if not u:
        return {}
    pd = u.get("prompt_tokens_details") or {}
    cd = u.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "cached_tokens": pd.get("cached_tokens"),
        "reasoning_tokens": cd.get("reasoning_tokens"),
    }


def _wire_name(name: str) -> str:
    return _WIRE_NAME.sub("_", name)[:64] or "output"


def _with_schema_hint(messages: Sequence[Mapping[str, Any]], schema: dict) -> list[dict]:
    """json_object mode cannot enforce the schema, so the schema goes into the prompt (NIM.md §3.1)."""
    hint = ("Reply with one JSON value that validates against this JSON Schema. No prose, no code fence.\n"
            + json.dumps(schema, ensure_ascii=False))
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system":
            m["content"] = f"{m.get('content', '')}\n\n{hint}"
            return out
    return [{"role": "system", "content": hint}, *out]


def build_body(model: str, messages: Sequence[Mapping[str, Any]], opts: _Opts, shape: _Shape,
               provider: Provider = _NIM) -> dict:
    """The wire body for one attempt (NIM.md §3). Always streams upstream. Provider fields only change what differs."""
    max_tokens = opts.max_tokens or MAX_TOKENS_DEFAULT
    is_glm = provider.nim_extras and "glm" in model
    if is_glm:
        max_tokens = max(max_tokens, GLM_MIN_TOKENS)
    body: dict[str, Any] = {
        "model": model,
        "messages": [dict(m) for m in messages],
        provider.token_param: max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if provider.send_temperature:
        body["temperature"] = 0.2
    if provider.nim_extras and opts.thinking is False:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if is_glm:
        body["reasoning_effort"] = "low"
    if opts.tools:
        body["tools"] = [
            {"type": "function", "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object"}),
            }} for t in opts.tools
        ]
        body["tool_choice"] = "auto"            # response_format is omitted when tools are present (NIM.md §3)
    elif opts.schema is not None and provider.schema_mode != "prompt":
        if shape.mode == "json_object":
            body["response_format"] = {"type": "json_object"}
        else:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": _wire_name(opts.schema_name),
                "schema": strip_schema(opts.schema, STRIP_ALWAYS | shape.drop),
                "strict": True,
            }}
    return body


def _tool_calls(acc: Mapping[int, dict]) -> tuple[dict, ...]:
    out = []
    for idx in sorted(acc):
        item = acc[idx]
        name = item.get("name") or ""
        try:
            args = json.loads(item.get("args") or "{}")
        except ValueError:
            continue                              # an unparseable call is dropped, not guessed (NIM.md §3.2)
        if name and isinstance(args, dict):
            out.append({"id": item.get("id") or f"call_{idx}", "name": name, "arguments": args})
    return tuple(out)


def _accumulate(acc: dict[int, dict], deltas: Sequence[Mapping[str, Any]] | None) -> None:
    for tc in deltas or []:
        idx = int(tc.get("index") or 0)
        item = acc.setdefault(idx, {"id": None, "name": "", "args": ""})
        if tc.get("id"):
            item["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name") and not item["name"]:
            item["name"] = fn["name"]
        args = fn.get("arguments")
        if isinstance(args, dict):
            args = json.dumps(args)
        item["args"] += args or ""


def _retry_after(value: str | None, default: float) -> float:
    try:
        seconds = float(value) if value is not None else default
    except ValueError:
        seconds = default
    return max(0.0, min(RETRY_AFTER_CAP_S, seconds))


_DAILY = re.compile(r"per[ _-]?day|daily|free-models-per-day|requests?_per_day", re.I)
_BILLING = re.compile(r"enforced_spend_limit|insufficient_quota|billing|spend[ _]limit", re.I)
# Google answers a bad key with HTTP 400 ("Please pass a valid API key", reason API_KEY_INVALID), not 401. That is the key's
# fault, not the model's or the request's, so the key is retired and the next one serves.
_BAD_KEY = re.compile(r"valid api key|api[_ ]key[_ ]?(not valid|invalid)|API_KEY_INVALID", re.I)
_RETRY_DELAY = re.compile(r'"retryDelay"\s*:\s*"([0-9.]+)s"')


def _seconds_to_utc_midnight(now_epoch: float) -> float:
    return 86400.0 - (now_epoch % 86400.0)


def _header_reset_seconds(headers: Mapping[str, str] | None, now_epoch: float) -> float | None:
    """Seconds until the window the response says it is spent in reopens. Only read when the response says nothing
    remains. Reset may be seconds from now, epoch seconds or epoch milliseconds, so the size of the number decides."""
    if not headers:
        return None
    low = {k.lower(): v for k, v in headers.items()}
    remaining = low.get("x-ratelimit-remaining")
    reset = low.get("x-ratelimit-reset")
    try:
        if reset is None or (remaining is not None and float(remaining) > 0):
            return None
        value = float(reset)
    except ValueError:
        return None
    if value > 1e11:
        value /= 1000.0               # epoch milliseconds
    if value > 1e8:
        value -= now_epoch            # epoch seconds
    return value if value > 0 else None


def rate_cooldown(retry_after: str | None, raw: str, headers: Mapping[str, str] | None, now_epoch: float) -> float:
    """How long a key rests after a 429, read from the strongest signal the provider gave. Order: a billing or spend-cap
    refusal (long, it is not a rate limit); a daily quota (until the day resets); a reset header that says nothing remains;
    Google's retryDelay; Retry-After; the default. Capped at a day."""
    text = raw or ""
    if _BILLING.search(text):
        return BILLING_COOL_S
    if _DAILY.search(text):
        return min(RESET_CAP_S, _seconds_to_utc_midnight(now_epoch))
    reset = _header_reset_seconds(headers, now_epoch)
    if reset is not None:
        return min(RESET_CAP_S, reset)
    m = _RETRY_DELAY.search(text)
    if m:
        return min(RETRY_AFTER_CAP_S * 2.5, float(m.group(1)) + 1.0)
    return _retry_after(retry_after, COOL_429_S)


def repair_hint(reason: str) -> str:
    """What to tell the model beyond the validator's line, by what went wrong. The line itself already names the path."""
    r = reason.lower()
    if "not valid json" in r:
        return " Start with { and end with }; do not write a sentence before or after the JSON, and do not explain."
    if "required" in r:
        return " Include every field the schema marks required."
    if "additional properties" in r or "unexpected" in r:
        return " Remove every field the schema does not list."
    if "is not of type" in r or "is not one of" in r:
        return " Use the exact type, or one of the exact values, the schema allows."
    return ""


def _diagnose(reason: str) -> str:
    """A short category for a refused reply, for the log: what kind of failure it was."""
    r = reason.lower()
    if "not valid json" in r:
        return "not_json"
    if "required" in r:
        return "missing_field"
    if "additional properties" in r:
        return "extra_field"
    if "is not of type" in r or "is not one of" in r:
        return "wrong_type"
    return "schema_other"


def _check_schema(text: str, schema: dict) -> str | None:
    try:
        value = json.loads(text)
    except ValueError:
        return "output is not valid JSON"
    return validate(value, schema)


def _json_slice(text: str) -> str:
    """A reply that is a JSON value followed or preceded by prose (a lead-in, a closing remark) is cut down to the value.
    Text that is already JSON, or holds no JSON value, comes back unchanged. Structure decides, not wording."""
    try:
        json.loads(text)
        return text
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for start, ch in enumerate(text):
        if ch in "{[":
            try:
                _, end = decoder.raw_decode(text, start)
            except ValueError:
                continue
            return text[start:end]
    return text


def _timeout(remaining: float, silence: float | None = None) -> httpx.Timeout:
    # `read` is the longest silence between two bytes, not the whole response: a long answer keeps streaming up to the
    # wall-clock budget, while a stalled one fails over after STREAM_IDLE_S instead of holding the call until the end.
    return httpx.Timeout(connect=min(10.0, remaining), read=min(silence or STREAM_IDLE_S, remaining), write=min(30.0, remaining),
                         pool=min(30.0, remaining))


class _AllBusy(Exhausted):
    """Every usable key is cooling or paced for this model, past this attempt's slice of the budget. A rate limit, not a failure:
    the call waits for the soonest key instead of reporting it (seen live: three missions failed with this message while
    five were running against four working keys)."""


class _BudgetSpent(Exception):
    """The wall-clock budget ran out inside a response. Internal: never escapes the transport."""


# -- client --------------------------------------------------------------------------------------------------------

class NimClient:
    """Thread-safe NIM chat client. Create one per process and reuse it (keep-alive connections).

    Cooldowns, pacing windows and in-flight counts live on the client, so every mission must share it (see
    ``shared_client``). A client per mission would give each one its own rate window and overrun the keys."""

    def __init__(self, keys: Sequence[str], *, provider: Provider = _NIM, per_minute: int | None = None,
                 transport: httpx.BaseTransport | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 rng: random.Random | None = None,
                 on_reply: Callable[[Reply], None] | None = None,
                 allow: Callable[[str], str | None] | None = None):
        self.provider = provider
        self.on_reply = on_reply               # called once per successful model call (the usage ledger hook); never raises
        # Budget gate: returns a reason when a model may not be called now (its token budget is used up), else None.
        self.allow = allow
        self.keys: list[str] = list(dict.fromkeys(k for k in keys if k))
        if not self.keys:
            raise NotConfigured(f"no {provider.label} key configured (looked for {provider.key_hint} in the "
                                "environment and .env)")
        base_url = provider.resolve_base_url()
        if not base_url:
            raise NotConfigured(f"{provider.label} has no endpoint yet; set {provider.base_url_var} in .env")
        self.chain: tuple[str, ...] = provider.models()
        # Pacing per key: the caller's value, else the provider's documented limit, else the NIM default.
        self.per_minute = per_minute or provider.rpm or PER_MINUTE
        self.clock = clock
        self.sleep = sleep
        self.rng = rng or random.Random()
        self.lock = threading.Lock()
        # Pacing window per key; one shared window when the vendor's limit belongs to the account (see Provider.shared_quota).
        shared_window: deque[float] = deque()
        self.sent: dict[str, deque[float]] = {k: (shared_window if provider.shared_quota else deque()) for k in self.keys}
        self.cool: dict[tuple[str, str], float] = {}
        self.bad: dict[str, float] = {}           # key -> re-admitted at
        self.demoted: dict[str, float] = {}       # model -> usable again at
        self.overload: dict[str, tuple[int, float]] = {}   # model -> (503s in a row, when the last one came)
        self.inflight: dict[str, int] = {k: 0 for k in self.keys}   # requests running now, per key
        self.http = httpx.Client(
            base_url=base_url, transport=transport,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=40),
            timeout=httpx.Timeout(10.0, read=120.0, write=30.0, pool=30.0),
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> "NimClient":
        return cls(load_keys(), **kwargs)

    @classmethod
    def for_provider(cls, name: str, **kwargs: Any) -> "NimClient":
        provider = PROVIDERS[name]
        return cls(load_keys(None, name), provider=provider, **kwargs)

    def __repr__(self) -> str:                    # never show key material
        return f"NimClient(provider={self.provider.name}, keys={len(self.keys)})"

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> "NimClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- public API --------------------------------------------------------------------------------------------

    def stream(self, models: Sequence[str], messages: Sequence[Mapping[str, Any]], *,
               thinking: bool | None = False, max_tokens: int = MAX_TOKENS_DEFAULT,
               schema: dict | None = None, schema_name: str = "output",
               tools: Sequence[Mapping[str, Any]] | None = None,
               budget_s: float = BUDGET_S, checkpoint: Callable[[], None] | None = None,
               on_wait: Callable[[dict], None] | None = None) -> Iterator[tuple[str, Any]]:
        """Yield ("delta", text) while output is released, then one ("done", Reply).

        With a schema or tools, nothing is released early: the answer is buffered, validated and then returned in
        "done", so a rejected answer never reaches the caller and failover stays possible.

        A rate limit (429) or an overload (503) is not a failure of the request. When a pass over the chain met only those,
        the call waits for the soonest key or model to be ready and goes round again, until the provider is depleted (every
        key rejected) or RATE_WAIT_LIMIT_S has passed. A model that refuses the request for good is dropped for this call.

        ``checkpoint`` is called at least once a second while the call waits and before each attempt; it may raise to stop the
        call (a cancelled mission, a spent time allowance), and the exception passes through. With a checkpoint the
        RATE_WAIT_LIMIT_S ceiling does not apply, because the caller can stop the wait. ``on_wait`` receives a plain-language
        dict ({"provider", "seconds", "waited", "reason"}) each time the call starts a wait, so a person sees "busy, retrying".
        Raises Exhausted (or SchemaViolation if the last failure was a local schema check) once the chain is spent.
        """
        chain = self._order(models)
        if not chain:
            raise NimError("no models given")
        if self.allow is not None:
            # Models over their token budget are skipped before any call is made; if none is left, the call stops here.
            verdicts = {m: self.allow(m) for m in chain}
            reasons = [why for why in verdicts.values() if why]
            if reasons:
                chain = [m for m in chain if verdicts[m] is None]
                if not chain:
                    raise Exhausted("token budget used up: " + "; ".join(reasons))
        deadline = self.clock() + budget_s
        opts = _Opts(thinking, max_tokens, schema, schema_name, [dict(t) for t in tools] if tools else None)
        last, last_invalid = "no attempt", False
        base = list(messages)
        carried, cuts = "", 0             # text already released from streams that were cut, and how often that happened
        waited = 0.0                      # seconds spent waiting out rate limits and overloads, across passes

        while True:
            dispatched, waitable, dead = 0, False, set()
            for model in chain:
                shape = _Shape("strict")
                reshapes, tried, key, reuse, repairs, truncations = 0, 0, None, False, 0, 0
                repair: list[dict] = []   # the model's last reply and the reason it was refused, when repairing
                while tried < MAX_KEYS_PER_MODEL and dispatched < MAX_DISPATCH:
                    if self.clock() >= deadline:
                        raise Exhausted(f"nim budget spent before a usable answer; last: {last}")
                    if checkpoint is not None:
                        checkpoint()
                    try:
                        key = self._acquire(model, deadline, only=key if reuse else None)
                    except _AllBusy as busy:
                        last, waitable = str(busy), True      # try the next model, then wait for the soonest key (see _rate_wait)
                        break
                    reuse = False
                    tried += 1
                    dispatched += 1
                    prompt_only = self.provider.schema_mode == "prompt" or shape.mode == "json_object"
                    sent = [*messages, *repair]
                    msgs = (_with_schema_hint(sent, schema)
                            if schema is not None and not opts.tools and prompt_only
                            else list(sent))
                    attempt_started = self.clock()
                    # One attempt gets a slice of the logical budget. Without the slice, a slow first model spends the whole
                    # budget, every later model starts with no time, and the call fails with nothing tried after it.
                    attempt_deadline = min(deadline, self.clock() + ATTEMPT_BUDGET_S)
                    try:
                        out = yield from self._attempt(key, model, msgs, opts, shape, attempt_deadline)
                    finally:
                        self._release(key)
                    log.info("nim attempt: model=%s outcome=%s elapsed=%.1fs reason=%s", model, out.kind,
                             self.clock() - attempt_started, out.reason)
                    if out.kind == "ok":
                        self.overload.pop(model, None)
                        reply = replace(out.reply, text=clean_text(carried) + out.reply.text) if carried else out.reply
                        self._record(reply)
                        yield "done", reply
                        return
                    last, last_invalid = out.reason, out.invalid
                    if out.status in RATE_STATUSES:
                        waitable = True
                    if out.hung:
                        waited += self.clock() - attempt_started    # time lost to a silent model counts toward the ceiling
                    if out.kind == "cut":
                        # The stream broke after text had already been released. Continue from that text on a fresh budget
                        # rather than failing: the model sees its own partial reply and writes only the rest. Bounded.
                        cuts += 1
                        if cuts > MAX_CONTINUATIONS:
                            raise NimError(f"nim stream cut {cuts} times; the continuation limit is reached")
                        carried += out.partial
                        messages = [*base, {"role": "assistant", "content": carried},
                                    {"role": "user", "content": CONTINUE_PROMPT}]
                        deadline = self.clock() + budget_s
                        dispatched -= 1               # a continuation is not a new attempt against the dispatch limit
                        tried -= 1
                        continue
                    if out.kind == "budget":
                        break
                    if out.kind == "reshape":
                        reshapes += 1
                        if reshapes > MAX_RESHAPES:
                            self._demote(model)
                            dead.add(model)
                            break
                        shape = out.shape or shape
                        tried -= 1                    # a reshape re-queues the same key and does not count as a new key
                        reuse = True
                        continue
                    if out.kind == "truncated" and truncations < MAX_TRUNCATION_RETRIES and opts.max_tokens < TRUNCATION_TOKEN_CAP:
                        # Cut by the allowance, not wrong: asking the model to shorten itself would only lose content. Ask
                        # again with more room on the same key.
                        truncations += 1
                        opts = replace(opts, max_tokens=min(TRUNCATION_TOKEN_CAP, (opts.max_tokens or MAX_TOKENS_DEFAULT) * 2))
                        tried -= 1
                        reuse = True
                        continue
                    if out.invalid and repairs < MAX_REPAIRS:
                        # The model answered, but not in the schema. Show it that reply and the reason, on the same key.
                        repairs += 1
                        if repairs == 2 and shape.mode == "strict" and schema is not None and not opts.tools:
                            # Strict decoding has now produced two unusable replies. Ask in plain JSON mode with the schema in the
                            # prompt and the local check, which some models follow where their constrained decoding fails.
                            shape = _Shape("json_object", shape.drop)
                        repair = [{"role": "assistant", "content": out.text[:REPAIR_QUOTE_CHARS]},
                                  {"role": "user", "content": REPAIR_PROMPT.format(reason=out.reason) + repair_hint(out.reason)}]
                        tried -= 1
                        reuse = True
                        continue
                    if out.kind == "model":
                        if out.demote:
                            self._demote(model)
                        if out.status not in RATE_STATUSES:
                            dead.add(model)
                        break
                    # "key" or "retire": try the next key for this model
            # The pass is over. Models that refused for good are dropped; rate limits and overloads wait, then go round again.
            chain = [m for m in chain if m not in dead]
            ceiling = RATE_WAIT_LIMIT_S if checkpoint is None else float("inf")
            if not waitable or not chain or waited >= ceiling:
                break
            wait = self._rate_wait(chain)
            if wait is None:
                break                                  # every key is rejected: the provider is depleted
            wait = min(wait, ceiling - waited)
            log.warning("nim: rate limited or overloaded on every usable model; waiting %.0fs then trying again (%s)",
                        wait, last)
            if on_wait is not None:
                try:
                    on_wait({"provider": self.provider.label, "seconds": round(wait), "waited": round(waited),
                             "reason": "busy"})
                except Exception:                      # a progress note must never stop the call
                    log.debug("on_wait failed", exc_info=True)
            waited += wait
            self._wait(wait, checkpoint)
            deadline = max(deadline, self.clock() + budget_s)   # waiting for a rate limit does not spend the work budget

        if checkpoint is None and waited >= RATE_WAIT_LIMIT_S:
            message = f"nim-rate-limited: still refused after {waited:.0f}s of waiting; last: {last}"
        else:
            message = f"nim-exhausted: {last}"
        if last_invalid:
            raise SchemaViolation(message)
        raise Exhausted(message)

    def _wait(self, seconds: float, checkpoint: Callable[[], None] | None) -> None:
        """Sleep in slices of at most a second, calling the checkpoint between them so a cancel or a spent allowance is felt."""
        left = seconds
        while left > 0:
            if checkpoint is not None:
                checkpoint()
            step = min(1.0, left)
            self.sleep(step)
            left -= step

    def _rate_wait(self, chain: Sequence[str]) -> float | None:
        """Seconds until the soonest (key, model) pair in the chain can be used again, capped at RATE_WAIT_MAX_S. None when
        every key is rejected (401/403) and none is re-admitted, i.e. the provider is depleted."""
        now = self.clock()
        usable = [k for k in self.keys if self.bad.get(k, 0.0) <= now]
        if not usable:
            return None
        waits = []
        for model in chain:
            key_wait = min(max(0.0, self._ready_at(k, model, now) - now) for k in usable)
            waits.append(max(key_wait, max(0.0, self.demoted.get(model, 0.0) - now)))
        soonest = min(waits)
        if soonest > DEPLETED_WAIT_S:
            return None                  # every key is at a daily or billing limit that resets hours from now: report it
        return min(max(1.0, soonest), RATE_WAIT_MAX_S)

    def complete(self, models: Sequence[str], messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> Reply:
        done: Reply | None = None
        for kind, value in self.stream(models, messages, **kwargs):
            if kind == "done":
                done = value
        if done is None:
            raise NimError("stream ended without an answer")
        return done

    def check_model(self, model: str, messages: Sequence[Mapping[str, Any]], *, max_tokens: int = 64,
                    budget_s: float = 60.0) -> dict:
        """One tiny completion for one model on one key: no failover, no retries, no demotion. Returns
        {"model", "status": "verified" | "failed", "http_status", "error"}. ``error`` is redacted and short; it never
        holds the key. A 200 with no text is "failed", because the check asks for a completion."""
        deadline = self.clock() + budget_s
        try:
            key = self._acquire(model, deadline)
        except NimError as exc:
            return {"model": model, "status": "failed", "http_status": None, "error": str(exc)}
        opts = _Opts(thinking=False, max_tokens=max_tokens, schema=None, schema_name="check", tools=None)
        try:
            gen = self._attempt(key, model, [dict(m) for m in messages], opts, _Shape("strict"), deadline)
            out: _Outcome | None = None
            while out is None:
                try:
                    next(gen)
                except StopIteration as stop:
                    out = stop.value
        finally:
            self._release(key)
        if out is not None and out.kind == "ok" and out.reply is not None:
            self._record(out.reply)
            return {"model": model, "status": "verified", "http_status": out.status or 200, "error": None}
        error = out.detail or out.reason if out is not None else "no answer"
        return {"model": model, "status": "failed", "http_status": out.status if out is not None else None,
                "error": redact(error, limit=200)}

    def _record(self, reply: Reply) -> None:
        """Hand a successful call to the usage hook. A broken hook must never fail the call that produced the answer."""
        if self.on_reply is None:
            return
        try:
            self.on_reply(reply)
        except Exception:
            log.warning("usage hook failed; the call itself succeeded", exc_info=True)

    def status(self) -> dict:
        """What the pool is doing right now, per key slot. Never includes key material: slots are 1-based positions."""
        now = self.clock()
        with self.lock:
            keys = []
            for slot, k in enumerate(self.keys, 1):
                self._purge(k, now)
                cooling = [{"model": m, "for_s": round(until - now, 1)}
                           for (kk, m), until in sorted(self.cool.items()) if kk == k and until > now]
                retired_for = max(0.0, self.bad.get(k, 0.0) - now)
                keys.append({"slot": slot, "in_flight": self.inflight.get(k, 0),
                             "sent_last_minute": len(self.sent[k]), "per_minute": self.per_minute,
                             "retired": retired_for > 0, "retired_for_s": round(retired_for, 1), "cooling": cooling})
            demoted = [{"model": m, "for_s": round(until - now, 1)}
                       for m, until in sorted(self.demoted.items()) if until > now]
        return {"keys": keys, "demoted_models": demoted, "in_flight": sum(k["in_flight"] for k in keys)}

    def probe(self, slot: int, model: str | None = None) -> dict:
        """One raw 1-token chat call on a single key. Returns the status only (NIM.md §8). Never retries."""
        key = self.keys[slot - 1]
        model = model or self.provider.probe_target() or PROBE_MODEL
        started = self.clock()
        try:
            with self.http.stream("POST", "/chat/completions",
                                  headers={"authorization": f"Bearer {key}", "accept": "text/event-stream",
                                          **dict(self.provider.extra_headers)},
                                  json={"model": model, "messages": [{"role": "user", "content": "ping"}],
                                        self.provider.token_param: 4 if self.provider.nim_extras else 64,
                                        "stream": True},
                                  timeout=_timeout(30.0)) as r:
                status = r.status_code
        except httpx.TransportError as exc:
            status = None
            err = type(exc).__name__
        else:
            err = None
        return {"slot": slot, "model": model, "status": status, "error": err,
                "elapsed_s": round(self.clock() - started, 2)}

    # -- internals ---------------------------------------------------------------------------------------------

    def _order(self, models: Sequence[str]) -> list[str]:
        now = self.clock()
        uniq = list(dict.fromkeys(m.strip() for m in models if m and m.strip()))
        return sorted(uniq, key=lambda m: self.demoted.get(m, 0.0) > now)   # demoted last, never dropped

    def _slot(self, key: str) -> int:
        return self.keys.index(key) + 1

    def _purge(self, key: str, now: float) -> None:
        w = self.sent[key]
        while w and now - w[0] >= 60.0:
            w.popleft()

    def _ready_at(self, key: str, model: str, now: float) -> float:
        w = self.sent[key]
        window = now if len(w) < self.per_minute else w[0] + 60.0
        return max(self.cool.get((key, model), 0.0), window)

    def _acquire(self, model: str, deadline: float, only: str | None = None) -> str:
        """Pick a key that has window capacity and no cooldown for this model. Sleeps only if every candidate is busy."""
        while True:
            with self.lock:
                now = self.clock()
                pool = [only] if only is not None else [k for k in self.keys if self.bad.get(k, 0.0) <= now]
                if not pool:
                    raise Exhausted("no usable key: every key was rejected (401/403) and none is re-admitted yet")
                self.rng.shuffle(pool)
                for k in pool:
                    self._purge(k, now)
                ready = [k for k in pool if self._ready_at(k, model, now) <= now]
                if ready:
                    # Least loaded first: fewest requests running now, then fewest sent in the last minute. The shuffle
                    # above breaks ties at random, so equally idle keys share the work.
                    ready.sort(key=lambda k: (self.inflight.get(k, 0), len(self.sent[k])))
                    chosen = ready[0]
                    self.sent[chosen].append(now)
                    self.inflight[chosen] = self.inflight.get(chosen, 0) + 1
                    return chosen
                soonest = min(pool, key=lambda k: self._ready_at(k, model, now))
                at = self._ready_at(soonest, model, now)
            if at > deadline:
                raise _AllBusy("all keys are cooling or paced past the budget")
            self.sleep(max(0.01, at - now))

    def _release(self, key: str) -> None:
        with self.lock:
            self.inflight[key] = max(0, self.inflight.get(key, 0) - 1)

    def _cool(self, key: str, model: str, seconds: float) -> None:
        with self.lock:
            self.cool[(key, model)] = self.clock() + seconds

    def _retire(self, key: str) -> None:
        with self.lock:
            self.bad[key] = self.clock() + RETIRE_S

    def _demote(self, model: str, seconds: float = DEMOTE_S) -> None:
        with self.lock:
            self.demoted[model] = self.clock() + seconds

    def _overload_duration(self, model: str) -> float:
        """A model that keeps answering 503 rests longer each time (30 s, 60 s, ... 300 s), and starts again from 30 s once it
        has been quiet for a while or has answered once."""
        with self.lock:
            now = self.clock()
            count, last = self.overload.get(model, (0, float("-inf")))
            count = 1 if now - last > OVERLOAD_WINDOW_S else count + 1
            self.overload[model] = (count, now)
            return min(OVERLOAD_MAX_S, OVERLOAD_BASE_S * 2 ** (count - 1))

    def _attempt(self, key: str, model: str, messages: list[dict], opts: _Opts, shape: _Shape,
                 deadline: float) -> Iterator[tuple[str, Any]]:
        """One HTTP attempt. Yields visible deltas; returns an _Outcome (the generator's return value)."""
        slot = self._slot(key)
        started = self.clock()
        emitted = False
        parts: list[str] = []
        live = opts.schema is None and not opts.tools        # release text as it arrives only when nothing is buffered
        body = build_body(model, messages, opts, shape, self.provider)
        try:
            with self.http.stream("POST", "/chat/completions",
                                  headers={"authorization": f"Bearer {key}", "accept": "text/event-stream",
                                          **dict(self.provider.extra_headers)},
                                  json=body,
                                  timeout=_timeout(max(0.1, deadline - self.clock()), self.provider.silence_s)) as r:
                status = r.status_code
                if status >= 400:
                    raw = r.read().decode("utf-8", "replace")
                    out = self._classify_http(key, model, slot, status, r.headers.get("retry-after"), raw,
                                              opts, shape, started, headers=r.headers)
                    out.status, out.detail = status, redact(raw, limit=200)
                    return out
                acc: dict[int, dict] = {}
                finish: str | None = None
                usage: dict = {}
                for line in r.iter_lines():
                    if self.clock() > deadline:
                        raise _BudgetSpent()
                    if not line.startswith("data:"):
                        continue                        # comments and keep-alives
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    ev = json.loads(data)
                    if isinstance(ev, dict) and ev.get("error") is not None:
                        # A gateway can fail after HTTP 200 (FreeRouter sends an error frame and no [DONE]). It is
                        # classified as the HTTP status its code stands for, so a refusal is never read as an empty reply.
                        return self._stream_error(key, model, slot, ev["error"], emitted, parts, opts, shape, started)
                    usage = ev.get("usage") or usage
                    for ch in ev.get("choices") or []:
                        finish = ch.get("finish_reason") or finish
                        delta = ch.get("delta") or {}
                        piece = _content_text(delta.get("content"))
                        if piece:
                            parts.append(piece)
                            if live:
                                emitted = True
                                yield "delta", piece
                        _accumulate(acc, delta.get("tool_calls"))
        except _BudgetSpent:
            if emitted:
                # Text has been released, so this attempt cannot fail over; the caller continues it on a new budget.
                return _Outcome("cut", f"{model}: budget spent while streaming", partial="".join(parts))
            return _Outcome("budget", f"{model}: budget spent while reading")
        except httpx.ReadTimeout as exc:
            if emitted:
                return _Outcome("cut", f"{model}: stream interrupted ({type(exc).__name__})", partial="".join(parts))
            if self.provider.silence_s:
                # Silence is a slow or hung model, not a bad key: rotating keys on the same model would repeat the wait.
                self._demote(model, SILENCE_DEMOTE_S)
                return _Outcome("model", f"{model}: no reply within {self.provider.silence_s:.0f}s", status=503, hung=True)
            self._cool(key, model, COOL_KEY_S)
            return _Outcome("key", f"{model}: {type(exc).__name__}")
        except (httpx.TransportError, ValueError) as exc:  # ValueError covers unreadable JSON in a stream
            if emitted:
                return _Outcome("cut", f"{model}: stream interrupted ({type(exc).__name__})",
                                partial="".join(parts))
            self._cool(key, model, COOL_KEY_S)
            return _Outcome("key", f"{model}: {type(exc).__name__}")

        elapsed = self.clock() - started
        if elapsed >= SLOW_CALL_S:
            log.info("nim call: %s slot=%d thinking=%s HTTP %s %.1fs", model, slot,
                     "on" if opts.thinking else "off", status, elapsed)
        text = clean_text("".join(parts))
        calls = _tool_calls(acc)
        if finish == "error" and not text and not calls:
            # Some gateways report a limit or an upstream fault inside a 200 stream as finish_reason "error", with no
            # text. That is a refusal to classify, not an empty answer that should demote the model.
            return self._stream_error(key, model, slot, {"code": "upstream_unavailable",
                                                         "message": "the stream ended with finish_reason error"},
                                      emitted, parts, opts, shape, started)
        if not text and not calls:
            if emitted:
                raise NimError("nim stream ended with no answer after output began; no failover")
            return _Outcome("model", f"{model}: empty reply (finish_reason={finish})", demote=finish != "length",
                            status=status)
        if opts.schema is not None and not opts.tools:
            text = _json_slice(text)
            err = _check_schema(text, opts.schema)
            if err:
                # What the model sent is what a repair needs to see; the log keeps a short redacted head and tail.
                truncated = finish == "length"
                log.warning("nim: %s reply refused locally (%s); category=%s finish=%s len=%d max_tokens=%s shape=%s "
                            "head=%r tail=%r", model, err, "truncated" if truncated else _diagnose(err), finish, len(text),
                            opts.max_tokens, shape.mode, redact(text[:160]), redact(text[-120:]))
                return _Outcome("truncated" if truncated else "model", f"{model}: local schema check failed: {err}",
                                invalid=True, status=status, text=text)
        reply = Reply(text=text, tool_calls=calls, model=model, finish_reason=finish,
                      usage=_usage(usage), key_slot=slot, elapsed_s=round(elapsed, 2))
        return _Outcome("ok", reply=reply, status=status)

    def _stream_error(self, key: str, model: str, slot: int, error: Any, emitted: bool, parts: list[str],
                      opts: _Opts, shape: _Shape, started: float) -> _Outcome:
        """An error frame inside a 200 stream. After text was released the attempt cannot fail over, so it is cut and
        continued; before that it is classified like the HTTP status its code stands for."""
        code = error.get("code") if isinstance(error, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        if emitted:
            return _Outcome("cut", f"{model}: stream error ({code or 'unknown'})", partial="".join(parts))
        status = _IN_STREAM_STATUS.get(code, 502)
        raw = json.dumps({"error": {"code": code, "message": str(message or "")[:200]}})
        return self._classify_http(key, model, slot, status, None, raw, opts, shape, started)

    def _classify_http(self, key: str, model: str, slot: int, status: int, retry_after: str | None, raw: str,
                       opts: _Opts, shape: _Shape, started: float, headers: Mapping[str, str] | None = None) -> _Outcome:
        """NIM.md §4.2, one row per branch. Reason strings carry the model and status, never the key."""
        note = f"{model}: HTTP {status}"
        log.warning("nim: %s slot=%d HTTP %s after %.1fs", model, slot, status, self.clock() - started)
        if status in (401, 403) or (status == 400 and _BAD_KEY.search(raw or "")):   # NIM answers 403, Google 400, to a bad key
            self._retire(key)
            return _Outcome("retire", note)
        if status == 503:
            self._demote(model, self._overload_duration(model))      # repeated overloads rest the model longer each time
            return _Outcome("model", note)
        if status in (404, 410):
            return _Outcome("model", note, demote=True)
        if status == 413:                        # this request is too large for this model's limit, not the model's health
            return _Outcome("model", note)
        if status in (400, 422):
            if opts.schema is not None and not opts.tools:
                names = unimplemented_keys(raw)
                if names - shape.drop:
                    return _Outcome("reshape", note, shape=shape.with_drop(names))
                if shape.mode != "json_object" and is_format_error(raw):
                    return _Outcome("reshape", note, shape=_Shape("json_object", shape.drop))
            return _Outcome("model", note)       # the request may be at fault; do not demote the model globally
        if status == 429:
            wait = rate_cooldown(retry_after, raw, headers, time.time())
        elif status in (408, 425):
            wait = COOL_KEY_S
        elif status == 402:
            wait = COOL_402_S
        else:
            wait = COOL_DEFAULT_S
        self._cool(key, model, wait)
        if status == 429 and self.provider.shared_quota:      # an account-wide limit binds every key, so rest them all
            for other in self.keys:
                self._cool(other, model, wait)
        return _Outcome("key", note)


# -- the process-wide client ----------------------------------------------------------------------------------------

_SHARED: dict[str, NimClient] = {}
_SHARED_LOCK = threading.Lock()


def _usage_hook(provider: str) -> Callable[[Reply], None]:
    """The callback every shared client gets: one usage row per successful model call, in data/usage.sqlite."""
    def record(reply: Reply) -> None:
        from ..usage import ledger_for
        from ..config import DATA
        ledger_for(DATA).record(provider, reply.model, reply.usage)
    return record


def _budget_gate() -> Callable[[str], str | None]:
    """The gate every shared client gets: a model whose token budget is used up in any window (1d, 7d, 30d) is not
    called. Budgets are read on each call, so a change in Settings applies at once. Read, write and cache tokens count together."""
    def gate(model: str) -> str | None:
        from ..settings import Settings
        from ..usage import WINDOWS, ledger_for
        from ..config import DATA
        budget = Settings().budgets().get(model) or {}
        if not any(budget.get(w) for w in WINDOWS):
            return None
        ledger = ledger_for(DATA)
        now = ledger.clock()
        over = [f"{w} {ledger.used(model, w, now):,}/{budget[w]:,} tokens" for w in WINDOWS
                if budget.get(w) and ledger.used(model, w, now) >= budget[w]]
        return f"{model} over budget ({'; '.join(over)})" if over else None
    return gate


def shared_client(provider: str | None = None) -> NimClient:
    """The one client of this process for one provider (default: the planner's active provider), created on first use.
    Every mission and sub-agent calls through it, so the rate windows, cooldowns and in-flight counts cover all of them
    together. Only keys that are not marked invalid are used (Settings.usable_keys). Raises NotConfigured without keys
    or endpoint."""
    from ..settings import Settings
    name = provider or active_provider_name()
    with _SHARED_LOCK:
        client = _SHARED.get(name)
        if client is None:
            client = NimClient(Settings().usable_keys(name), provider=PROVIDERS[name], on_reply=_usage_hook(name),
                               allow=_budget_gate())
            _SHARED[name] = client
        return client


def reset_shared_client(provider: str | None = None) -> None:
    """Drop the shared client of one provider, or of all, so the next call re-reads its keys (after a change in Settings).
    Calls already in flight keep their reference and finish normally."""
    with _SHARED_LOCK:
        if provider is None:
            _SHARED.clear()
        else:
            _SHARED.pop(provider, None)


def probe_provider(name: str) -> list[dict]:
    """One 1-token call per key of one provider, independent of the active one. Status and timing only, never text."""
    with NimClient.for_provider(name) as client:
        return [client.probe(slot) for slot in range(1, len(client.keys) + 1)]
