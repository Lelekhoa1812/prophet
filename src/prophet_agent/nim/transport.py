"""NVIDIA NIM chat transport: key pool x model fallback chain, client-side pacing, failure policy, schema fallbacks.

Contract: NIM.md §3 (wire body), §4 (routing and the failure matrix §4.2), §5 (streaming), §8 (probes), §10 (defects
F1-F6, all fixed here). Verified live 2026-11-20 by the reference client in NIM.md §9.

Motivation vs Logic
Motivation: the agent runtime calls NIM many times per mission. A single busy key must not stall the call, a dead
model must not burn the whole budget, a key must never reach a log or an exception, and a bad schema must not
silently become a bad answer.
Logic: one shared client per process. A logical call tries models in order (demoted models last, never dropped);
each model may use up to 3 keys and the call at most 7 dispatches. Every HTTP request gets a timeout equal to the
time left on the wall-clock budget (F1). Each status code has one consequence: cool (key, model) for 429/408/425/402/
5xx; retire the key for 401/403 (re-admitted after 10 min, so a rotated key recovers, F4); demote the model for
404/410/413/503; reshape the schema on a 400/422 that names keywords (same key, same model); otherwise next model.
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

from ..config import load_dotenv_values
from .errors import Exhausted, NimError, NotConfigured, SchemaViolation
from .schema import STRIP_ALWAYS, is_format_error, strip_schema, unimplemented_keys, validate

log = logging.getLogger("prophet.nim")

BASE_URL = "https://integrate.api.nvidia.com/v1"
# NIM.md §2 ranking. Thinking off is the default because it is the speed lever (2-6 s instead of 25-60 s).
DEFAULT_CHAIN: tuple[str, ...] = (
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "poolside/laguna-xs-2.1",
    "z-ai/glm-5.3-flash",
)
PROBE_MODEL = "poolside/laguna-xs-2.1"       # NIM.md §8: fastest, cheapest key probe

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
MAX_CONTINUATIONS = 3             # a free-text answer cut mid-stream is continued this many times, each on a new budget
CONTINUE_PROMPT = ("Your reply was cut off. Continue exactly from where it stopped. Do not repeat anything already "
                   "written and do not add an introduction.")
SLOW_CALL_S = 15.0                # log calls at least this slow, or any >= 400

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


# -- pure helpers --------------------------------------------------------------------------------------------------

def load_keys(environ: Mapping[str, str] | None = None) -> list[str]:
    """NVIDIA_NIM_API_KEYS (comma or space list), then NVIDIA_NIM_API_KEY_1..9; de-duplicated, order kept.

    The process environment wins over the git-ignored .env file. Values are returned, never printed.
    """
    if environ is None:
        environ = {**load_dotenv_values(), **os.environ}
    raw = _KEY_SPLIT.split(environ.get("NVIDIA_NIM_API_KEYS", ""))
    raw += [environ.get(f"NVIDIA_NIM_API_KEY_{i}", "") for i in range(1, 10)]
    return list(dict.fromkeys(k.strip() for k in raw if k and k.strip()))


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


def build_body(model: str, messages: Sequence[Mapping[str, Any]], opts: _Opts, shape: _Shape) -> dict:
    """The wire body for one attempt (NIM.md §3). Always streams upstream."""
    max_tokens = opts.max_tokens or MAX_TOKENS_DEFAULT
    is_glm = "glm" in model
    if is_glm:
        max_tokens = max(max_tokens, GLM_MIN_TOKENS)
    body: dict[str, Any] = {
        "model": model,
        "messages": [dict(m) for m in messages],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if opts.thinking is False:
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
    elif opts.schema is not None:
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


def _check_schema(text: str, schema: dict) -> str | None:
    try:
        value = json.loads(text)
    except ValueError:
        return "output is not valid JSON"
    return validate(value, schema)


def _timeout(remaining: float) -> httpx.Timeout:
    # `read` is the longest silence between two bytes, not the whole response: a long answer keeps streaming up to the
    # wall-clock budget, while a stalled one fails over after STREAM_IDLE_S instead of holding the call until the end.
    return httpx.Timeout(connect=min(10.0, remaining), read=min(STREAM_IDLE_S, remaining), write=min(30.0, remaining),
                         pool=min(30.0, remaining))


class _BudgetSpent(Exception):
    """The wall-clock budget ran out inside a response. Internal: never escapes the transport."""


# -- client --------------------------------------------------------------------------------------------------------

class NimClient:
    """Thread-safe NIM chat client. Create one per process and reuse it (keep-alive connections).

    Cooldowns, pacing windows and in-flight counts live on the client, so every mission must share it (see
    ``shared_client``). A client per mission would give each one its own rate window and overrun the keys."""

    def __init__(self, keys: Sequence[str], *, per_minute: int = PER_MINUTE,
                 transport: httpx.BaseTransport | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 rng: random.Random | None = None):
        self.keys: list[str] = list(dict.fromkeys(k for k in keys if k))
        if not self.keys:
            raise NotConfigured("no NVIDIA_NIM_API_KEY_n configured (looked for NVIDIA_NIM_API_KEYS and "
                                "NVIDIA_NIM_API_KEY_1..9 in the environment and .env)")
        self.per_minute = per_minute
        self.clock = clock
        self.sleep = sleep
        self.rng = rng or random.Random()
        self.lock = threading.Lock()
        self.sent: dict[str, deque[float]] = {k: deque() for k in self.keys}
        self.cool: dict[tuple[str, str], float] = {}
        self.bad: dict[str, float] = {}           # key -> re-admitted at
        self.demoted: dict[str, float] = {}       # model -> usable again at
        self.inflight: dict[str, int] = {k: 0 for k in self.keys}   # requests running now, per key
        self.http = httpx.Client(
            base_url=BASE_URL, transport=transport,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=40),
            timeout=httpx.Timeout(10.0, read=120.0, write=30.0, pool=30.0),
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> "NimClient":
        return cls(load_keys(), **kwargs)

    def __repr__(self) -> str:                    # never show key material
        return f"NimClient(keys={len(self.keys)})"

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
               budget_s: float = BUDGET_S) -> Iterator[tuple[str, Any]]:
        """Yield ("delta", text) while output is released, then one ("done", Reply).

        With a schema or tools, nothing is released early: the answer is buffered, validated and then returned in
        "done", so a rejected answer never reaches the caller and failover stays possible.
        Raises Exhausted (or SchemaViolation if the last failure was a local schema check) once the chain is spent.
        """
        chain = self._order(models)
        if not chain:
            raise NimError("no models given")
        deadline = self.clock() + budget_s
        opts = _Opts(thinking, max_tokens, schema, schema_name, [dict(t) for t in tools] if tools else None)
        dispatched, last, last_invalid = 0, "no attempt", False
        base = list(messages)
        carried, cuts = "", 0             # text already released from streams that were cut, and how often that happened

        for model in chain:
            shape = _Shape("strict")
            reshapes, tried, key, reuse = 0, 0, None, False
            while tried < MAX_KEYS_PER_MODEL and dispatched < MAX_DISPATCH:
                if self.clock() >= deadline:
                    raise Exhausted(f"nim budget spent before a usable answer; last: {last}")
                key = self._acquire(model, deadline, only=key if reuse else None)
                reuse = False
                tried += 1
                dispatched += 1
                msgs = (_with_schema_hint(messages, schema)
                        if schema is not None and not opts.tools and shape.mode == "json_object"
                        else list(messages))
                attempt_started = self.clock()
                try:
                    out = yield from self._attempt(key, model, msgs, opts, shape, deadline)
                finally:
                    self._release(key)
                log.info("nim attempt: model=%s outcome=%s elapsed=%.1fs reason=%s", model, out.kind,
                         self.clock() - attempt_started, out.reason)
                if out.kind == "ok":
                    reply = replace(out.reply, text=clean_text(carried) + out.reply.text) if carried else out.reply
                    yield "done", reply
                    return
                last, last_invalid = out.reason, out.invalid
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
                        break
                    shape = out.shape or shape
                    tried -= 1                    # a reshape re-queues the same key and does not count as a new key
                    reuse = True
                    continue
                if out.kind == "model":
                    if out.demote:
                        self._demote(model)
                    break
                # "key" or "retire": try the next key for this model
            # next model

        message = f"nim-exhausted: {last}"
        if last_invalid:
            raise SchemaViolation(message)
        raise Exhausted(message)

    def complete(self, models: Sequence[str], messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> Reply:
        done: Reply | None = None
        for kind, value in self.stream(models, messages, **kwargs):
            if kind == "done":
                done = value
        if done is None:
            raise NimError("stream ended without an answer")
        return done

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

    def probe(self, slot: int, model: str = PROBE_MODEL) -> dict:
        """One raw 1-token chat call on a single key. Returns the status only (NIM.md §8). Never retries."""
        key = self.keys[slot - 1]
        started = self.clock()
        try:
            with self.http.stream("POST", "/chat/completions",
                                  headers={"authorization": f"Bearer {key}", "accept": "text/event-stream"},
                                  json={"model": model, "messages": [{"role": "user", "content": "ping"}],
                                        "max_tokens": 4, "stream": True},
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
                raise Exhausted("all keys are cooling or paced past the budget")
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

    def _demote(self, model: str) -> None:
        with self.lock:
            self.demoted[model] = self.clock() + DEMOTE_S

    def _attempt(self, key: str, model: str, messages: list[dict], opts: _Opts, shape: _Shape,
                 deadline: float) -> Iterator[tuple[str, Any]]:
        """One HTTP attempt. Yields visible deltas; returns an _Outcome (the generator's return value)."""
        slot = self._slot(key)
        started = self.clock()
        emitted = False
        parts: list[str] = []
        live = opts.schema is None and not opts.tools        # release text as it arrives only when nothing is buffered
        body = build_body(model, messages, opts, shape)
        try:
            with self.http.stream("POST", "/chat/completions",
                                  headers={"authorization": f"Bearer {key}", "accept": "text/event-stream"},
                                  json=body, timeout=_timeout(max(0.1, deadline - self.clock()))) as r:
                status = r.status_code
                if status >= 400:
                    raw = r.read().decode("utf-8", "replace")
                    return self._classify_http(key, model, slot, status, r.headers.get("retry-after"), raw,
                                               opts, shape, started)
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
        if not text and not calls:
            if emitted:
                raise NimError("nim stream ended with no answer after output began; no failover")
            return _Outcome("model", f"{model}: empty reply (finish_reason={finish})", demote=finish != "length")
        if opts.schema is not None and not opts.tools:
            err = _check_schema(text, opts.schema)
            if err:
                return _Outcome("model", f"{model}: local schema check failed: {err}", invalid=True)
        reply = Reply(text=text, tool_calls=calls, model=model, finish_reason=finish,
                      usage=_usage(usage), key_slot=slot, elapsed_s=round(elapsed, 2))
        return _Outcome("ok", reply=reply)

    def _classify_http(self, key: str, model: str, slot: int, status: int, retry_after: str | None, raw: str,
                       opts: _Opts, shape: _Shape, started: float) -> _Outcome:
        """NIM.md §4.2, one row per branch. Reason strings carry the model and status, never the key."""
        note = f"{model}: HTTP {status}"
        log.warning("nim: %s slot=%d HTTP %s after %.1fs", model, slot, status, self.clock() - started)
        if status in (401, 403):                 # NIM answers 403, not 401, to a bad key on chat
            self._retire(key)
            return _Outcome("retire", note)
        if status in (404, 410, 413, 503):
            return _Outcome("model", note, demote=True)
        if status in (400, 422):
            if opts.schema is not None and not opts.tools:
                names = unimplemented_keys(raw)
                if names - shape.drop:
                    return _Outcome("reshape", note, shape=shape.with_drop(names))
                if shape.mode != "json_object" and is_format_error(raw):
                    return _Outcome("reshape", note, shape=_Shape("json_object", shape.drop))
            return _Outcome("model", note)       # the request may be at fault; do not demote the model globally
        if status == 429:
            wait = _retry_after(retry_after, COOL_429_S)
        elif status in (408, 425):
            wait = COOL_KEY_S
        elif status == 402:
            wait = COOL_402_S
        else:
            wait = COOL_DEFAULT_S
        self._cool(key, model, wait)
        return _Outcome("key", note)


# -- the process-wide client ----------------------------------------------------------------------------------------

_SHARED: NimClient | None = None
_SHARED_LOCK = threading.Lock()


def shared_client() -> NimClient:
    """The one NIM client of this process, created on first use. Every mission and sub-agent calls through it, so the
    rate windows, cooldowns and in-flight counts cover all of them together. Raises NotConfigured without keys."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = NimClient.from_env()
        return _SHARED


def reset_shared_client() -> None:
    """Drop the shared client so the next call re-reads the keys (after a key change in Settings). Calls already in
    flight keep their reference and finish normally."""
    global _SHARED
    with _SHARED_LOCK:
        _SHARED = None
