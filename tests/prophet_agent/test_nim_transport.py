"""NIM transport tests: one per row of NIM.md §4.2, plus pacing, budget, schema, streaming and redaction.

No network. A scripted httpx.MockTransport answers each (key slot, model) pair, and a fake clock replaces time so
cooldowns, demotions and pacing windows are exact. Run with: PYTHONPATH=src .venv/bin/python -m pytest tests/prophet_agent -q
"""
from __future__ import annotations

import json
import random

import httpx
import pytest

from prophet_agent.nim import Exhausted, NimClient, NimError, NotConfigured, SchemaViolation, load_keys
from prophet_agent.nim.transport import ATTEMPT_BUDGET_S, BUDGET_S
from prophet_agent.nim.transport import COOL_DEFAULT_S, DEMOTE_S, MAX_CONTINUATIONS, RETIRE_S, clean_text

KEYS = ("nvapi-AAAA1111", "nvapi-BBBB2222", "nvapi-CCCC3333")
M1 = "nvidia/nemotron-3-ultra-550b-a55b"
M2 = "poolside/laguna-xs-2.1"
M3 = "z-ai/glm-5.3-flash"
MSGS = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Say hi."}]


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class NoShuffle(random.Random):
    """Keeps key order 1, 2, 3 so tests can state exactly which slot is used."""

    def shuffle(self, x) -> None:  # noqa: D401
        return None


def sse(texts: list[str] | None = None, *, finish: str = "stop", usage: dict | None = None,
        raw_chunks: list[dict] | None = None) -> httpx.Response:
    chunks: list[dict] = list(raw_chunks or [])
    for t in texts or []:
        chunks.append({"choices": [{"delta": {"content": t}, "finish_reason": None}]})
    chunks.append({"choices": [{"delta": {}, "finish_reason": finish}],
                   "usage": usage or {"prompt_tokens": 10, "completion_tokens": 3,
                                      "prompt_tokens_details": {"cached_tokens": 4}}})
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def status(code: int, body: str = "", headers: dict | None = None):
    return lambda request, payload: httpx.Response(code, text=body, headers=headers or {})


def answer(*texts: str, finish: str = "stop"):
    return lambda request, payload: sse(list(texts), finish=finish)


def raise_(exc: Exception):
    def _raise(request, payload):
        raise exc
    return _raise


class Script:
    """Routes by (key slot, model); the most specific pattern wins; unmatched calls get a plain answer."""

    def __init__(self) -> None:
        self.routes: dict[tuple[int | None, str | None], object] = {}
        self.calls: list[dict] = []

    def on(self, responder, *, slot: int | None = None, model: str | None = None) -> "Script":
        self.routes[(slot, model)] = responder
        return self

    def __call__(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        key = request.headers.get("authorization", "").removeprefix("Bearer ")
        slot = KEYS.index(key) + 1 if key in KEYS else 0
        self.calls.append({"slot": slot, "model": payload["model"], "body": payload, "request": request})
        for pattern in ((slot, payload["model"]), (None, payload["model"]), (slot, None), (None, None)):
            if pattern in self.routes:
                return self.routes[pattern](request, payload)
        return sse(["ok"])

    def sequence(self) -> list[tuple[int, str]]:
        return [(c["slot"], c["model"]) for c in self.calls]


def make(script: Script, clock: FakeClock | None = None, keys=KEYS, **kwargs) -> tuple[NimClient, FakeClock]:
    clock = clock or FakeClock()
    client = NimClient(list(keys), transport=httpx.MockTransport(script), clock=clock, sleep=clock.sleep,
                       rng=NoShuffle(), **kwargs)
    return client, clock


# -- happy path and cleaning ---------------------------------------------------------------------------------------

def test_happy_path_uses_first_model_and_first_key():
    script = Script()
    client, _ = make(script)
    reply = client.complete([M1, M2], MSGS)
    assert reply.text == "ok"
    assert reply.model == M1
    assert reply.key_slot == 1
    assert script.sequence() == [(1, M1)]
    assert reply.usage["cached_tokens"] == 4


def test_think_block_and_json_fence_are_removed_from_streamed_text():
    script = Script().on(lambda r, p: sse(["<think>plan it", " out</think>", "```json\n", '{"a": 1}', "\n```"]))
    client, _ = make(script)
    reply = client.complete([M1], MSGS)
    assert reply.text == '{"a": 1}'


def test_clean_text_keeps_text_after_a_lone_closing_think_tag():
    assert clean_text("reasoning here</think>\nFinal answer") == "Final answer"


def test_thinking_off_is_sent_by_default_and_glm_gets_low_effort_and_minimum_tokens():
    script = Script()
    client, _ = make(script)
    client.complete([M1], MSGS)
    client.complete([M3], MSGS, max_tokens=256)
    assert script.calls[0]["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    glm = script.calls[1]["body"]
    assert glm["reasoning_effort"] == "low"
    assert glm["max_tokens"] >= 1024


def test_thinking_none_sends_no_template_switch():
    script = Script()
    client, _ = make(script)
    client.complete([M1], MSGS, thinking=None)
    assert "chat_template_kwargs" not in script.calls[0]["body"]


# -- NIM.md §4.2 failure matrix ------------------------------------------------------------------------------------

@pytest.mark.parametrize(("code", "expected_cool"), [(402, 120.0), (408, 30.0), (425, 30.0), (500, COOL_DEFAULT_S),
                                                     (502, COOL_DEFAULT_S), (418, COOL_DEFAULT_S)])
def test_key_errors_cool_the_key_for_that_model_only(code, expected_cool):
    script = Script().on(status(code), slot=1, model=M1)
    client, clock = make(script)
    reply = client.complete([M1], MSGS)
    assert reply.key_slot == 2
    assert client.cool[(KEYS[0], M1)] - 1000.0 == pytest.approx(expected_cool)
    assert (KEYS[0], M2) not in client.cool        # cooldown is per (key, model)


def test_429_honours_retry_after_and_caps_it():
    script = Script().on(status(429, headers={"retry-after": "7"}), slot=1, model=M1)
    client, clock = make(script)
    assert client.complete([M1], MSGS).key_slot != 1
    assert client.cool[(KEYS[0], M1)] == pytest.approx(1007.0)

    # key 1 is still cooling for M1, so the next M1 call avoids it; key 1 is free for M2 at once
    assert client.complete([M1], MSGS).key_slot != 1
    assert client.complete([M2], MSGS).model == M2
    assert (KEYS[0], M2) not in client.cool


def test_429_retry_after_is_capped_at_two_minutes():
    script = Script().on(status(429, headers={"retry-after": "9999"}), slot=1, model=M1)
    client, clock = make(script)
    client.complete([M1], MSGS)
    assert client.cool[(KEYS[0], M1)] - 1000.0 == pytest.approx(120.0)


def test_429_without_retry_after_cools_for_thirty_seconds():
    script = Script().on(status(429), slot=1, model=M1)
    client, clock = make(script)
    client.complete([M1], MSGS)
    assert client.cool[(KEYS[0], M1)] - 1000.0 == pytest.approx(30.0)


@pytest.mark.parametrize("code", [401, 403])
def test_401_and_403_retire_the_key_until_readmitted(code):
    script = Script().on(status(code), slot=1)
    client, clock = make(script)
    assert client.complete([M1], MSGS).key_slot != 1
    assert client.complete([M1], MSGS).key_slot != 1       # key 1 is retired, not tried again
    assert [s for s, _ in script.sequence()].count(1) == 1

    clock.t += RETIRE_S + 1                                # re-admitted after the retirement window
    before = len(script.calls)
    for _ in range(3):
        client.complete([M1], MSGS)
    assert any(c["slot"] == 1 for c in script.calls[before:])   # key 1 tried again (and fails again)


def test_transport_error_cools_key_thirty_seconds_then_next_key():
    script = Script().on(raise_(httpx.ConnectError("refused")), slot=1, model=M1)
    client, clock = make(script)
    assert client.complete([M1], MSGS).key_slot == 2
    assert client.cool[(KEYS[0], M1)] - 1000.0 == pytest.approx(30.0)


def test_unreadable_stream_json_cools_key_then_next_key():
    garbage = httpx.Response(200, text="data: {not json\n\n", headers={"content-type": "text/event-stream"})
    script = Script().on(lambda r, p: garbage, slot=1, model=M1)
    client, clock = make(script)
    assert client.complete([M1], MSGS).key_slot == 2
    assert client.cool[(KEYS[0], M1)] - 1000.0 == pytest.approx(30.0)


def test_request_too_large_moves_on_without_demoting_the_model():
    script = Script().on(status(413), model=M1)
    client, clock = make(script)
    assert client.complete([M1, M2], MSGS).model == M2
    assert M1 not in client.demoted, "a 413 is about this request, so the model stays first in the chain"


@pytest.mark.parametrize("code", [404, 410, 503])
def test_model_errors_demote_the_model_and_move_on(code):
    script = Script().on(status(code), model=M1)
    client, clock = make(script)
    reply = client.complete([M1, M2], MSGS)
    assert reply.model == M2
    assert [m for _, m in script.sequence()] == [M1, M2]  # one attempt per model: a model error is not a key error
    assert client.demoted[M1] - 1000.0 == pytest.approx(DEMOTE_S)


def test_demoted_model_moves_last_but_is_never_dropped():
    script = Script().on(status(503), model=M1)
    client, clock = make(script)
    client.complete([M1, M2], MSGS)                        # M1 demoted after its 503
    script.calls.clear()
    assert client.complete([M1, M2], MSGS).model == M2     # M2 is tried first now
    assert [m for _, m in script.sequence()] == [M2]
    script.routes.pop((None, M1))                          # M1 recovers
    script.calls.clear()
    assert client.complete([M1], MSGS).model == M1         # demoted, but still usable as the only option


def test_other_400_moves_to_next_model_without_demoting_it():
    script = Script().on(status(400, "messages not accepted"), model=M1)
    client, clock = make(script)
    assert client.complete([M1, M2], MSGS).model == M2
    assert M1 not in client.demoted


def test_empty_reply_with_length_finish_moves_on_without_demoting():
    script = Script().on(lambda r, p: sse([], finish="length"), model=M1)
    client, clock = make(script)
    assert client.complete([M1, M2], MSGS).model == M2
    assert M1 not in client.demoted


def test_empty_reply_with_stop_finish_demotes_the_model():
    script = Script().on(lambda r, p: sse([], finish="stop"), model=M1)
    client, clock = make(script)
    client.complete([M1, M2], MSGS)
    assert M1 in client.demoted


# -- schema handling -----------------------------------------------------------------------------------------------

MIN_ITEMS_SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "minItems": 1,
                                                              "items": {"type": "string"}}},
                    "required": ["items"]}


def test_unimplemented_keys_reshape_same_key_and_model():
    def reject_min_items(request, payload):
        schema = payload.get("response_format", {}).get("json_schema", {}).get("schema", {})
        if "minItems" in json.dumps(schema):
            return httpx.Response(400, text='{"detail": "Unimplemented keys: [\\"minItems\\"]"}')
        return sse(['{"items": ["a"]}'])

    script = Script().on(reject_min_items)
    client, _ = make(script)
    reply = client.complete([M1], MSGS, schema=MIN_ITEMS_SCHEMA)
    assert json.loads(reply.text) == {"items": ["a"]}
    assert script.sequence() == [(1, M1), (1, M1)]         # same key, same model
    assert "minItems" not in json.dumps(script.calls[1]["body"]["response_format"])
    assert "minItems" in json.dumps(MIN_ITEMS_SCHEMA)      # the caller's schema is not mutated


def test_format_error_falls_back_to_json_object_with_schema_in_prompt():
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}

    def grammar_rejects(request, payload):
        if payload.get("response_format", {}).get("type") == "json_schema":
            return httpx.Response(400, text="Grammar error: response_format not supported")
        return sse(['{"ok": true}'])

    script = Script().on(grammar_rejects)
    client, _ = make(script)
    reply = client.complete([M1], MSGS, schema=schema)
    assert reply.text == '{"ok": true}'
    second = script.calls[1]["body"]
    assert second["response_format"] == {"type": "json_object"}
    system = next(m for m in second["messages"] if m["role"] == "system")
    assert "JSON Schema" in system["content"]
    assert script.sequence() == [(1, M1), (1, M1)]


def test_local_schema_violation_moves_on_and_finally_raises_schema_violation():
    script = Script().on(answer('{"ok": "yes"}'))
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    client, _ = make(script)
    with pytest.raises(SchemaViolation):
        client.complete([M1, M2], MSGS, schema=schema)
    assert [m for _, m in script.sequence()] == [M1, M2]


def test_local_schema_check_uses_the_original_schema_not_the_stripped_one():
    # minItems is stripped from the wire schema, but the local check still enforces it
    script = Script().on(answer('{"items": []}'))
    client, _ = make(script)
    with pytest.raises(SchemaViolation):
        client.complete([M1], MSGS, schema=MIN_ITEMS_SCHEMA)


def test_a_property_named_title_survives_the_wire_schema():
    schema = {"title": "Doc", "type": "object", "properties": {"title": {"type": "string"}},
              "required": ["title"]}
    script = Script().on(answer('{"title": "x"}'))
    client, _ = make(script)
    assert json.loads(client.complete([M1], MSGS, schema=schema).text) == {"title": "x"}
    wire = script.calls[0]["body"]["response_format"]["json_schema"]["schema"]
    assert "title" in wire["properties"]                  # property name kept
    assert "title" not in wire                            # top-level keyword stripped


def test_tool_call_arguments_are_assembled_from_deltas_and_bad_ones_dropped():
    chunks = [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1",
                                                "function": {"name": "ask", "arguments": '{"q'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '":"Hi"}'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "c2",
                                                "function": {"name": "broken", "arguments": "not json"}}]}}]},
    ]
    script = Script().on(lambda r, p: sse(raw_chunks=chunks, finish="tool_calls"))
    client, _ = make(script)
    tools = [{"name": "ask", "description": "Ask the user", "parameters": {"type": "object"}}]
    reply = client.complete([M1], MSGS, tools=tools)
    assert reply.tool_calls == ({"id": "c1", "name": "ask", "arguments": {"q": "Hi"}},)
    body = script.calls[0]["body"]
    assert body["tool_choice"] == "auto" and "response_format" not in body


# -- streaming -----------------------------------------------------------------------------------------------------

def test_stream_releases_deltas_then_done():
    script = Script().on(lambda r, p: sse(["Hel", "lo"]))
    client, _ = make(script)
    events = list(client.stream([M1], MSGS))
    assert [e for e in events if e[0] == "delta"] == [("delta", "Hel"), ("delta", "lo")]
    assert events[-1][0] == "done" and events[-1][1].text == "Hello"


def _cut_after(text: str):
    """A stream that releases one delta and then breaks, as a dropped connection does."""
    yield f'data: {{"choices": [{{"delta": {{"content": "{text}"}}, "finish_reason": null}}]}}\n\n'.encode()
    raise httpx.ReadError("connection reset")


def test_a_cut_stream_is_continued_from_the_released_text_not_failed_over():
    seen: list[dict] = []

    def cut_then_finish(request, payload):
        seen.append(payload)
        if len(seen) == 1:
            return httpx.Response(200, content=_cut_after("Hel"), headers={"content-type": "text/event-stream"})
        return sse(["lo"])

    script = Script().on(cut_then_finish, model=M1)
    client, _ = make(script)
    done = list(client.stream([M1, M2], MSGS))[-1][1]
    assert done.text == "Hello"                           # released "Hel" + continuation "lo"
    assert done.model == M1
    assert [m for _, m in script.sequence()] == [M1, M1]  # continued on the same model; M2 never spliced on
    assert seen[1]["messages"][-2] == {"role": "assistant", "content": "Hel"}   # the model sees its own partial reply


def test_a_stream_cut_again_and_again_stops_at_the_continuation_limit_without_failover():
    script = Script().on(lambda r, p: httpx.Response(200, content=_cut_after("Hel"),
                                                     headers={"content-type": "text/event-stream"}), model=M1)
    client, _ = make(script)
    with pytest.raises(NimError, match="continuation limit"):
        list(client.stream([M1, M2], MSGS))
    assert [m for _, m in script.sequence()] == [M1] * (MAX_CONTINUATIONS + 1)   # M2 never spliced on


def test_failover_before_first_token_is_allowed_when_reading_fails_early():
    def breaks_immediately():
        raise httpx.ReadError("reset")
        yield b""  # pragma: no cover

    script = Script().on(lambda r, p: httpx.Response(200, content=breaks_immediately(),
                                                    headers={"content-type": "text/event-stream"}), model=M1)
    client, _ = make(script)
    assert client.complete([M1, M2], MSGS).model == M2


# -- keys, pacing, budget ------------------------------------------------------------------------------------------

def test_exhaustion_is_redacted_and_never_names_a_key():
    leaked = "nvapi-LEAKED987654321"
    script = Script().on(status(500, f"upstream echoed {leaked}"))
    client, _ = make(script)
    with pytest.raises(Exhausted) as info:
        client.complete([M1], MSGS)
    text = str(info.value)
    assert leaked not in text and KEYS[0] not in text
    assert script.sequence() == [(1, M1), (2, M1), (3, M1)]   # at most three keys per model


def test_dispatch_cap_is_seven_per_call():
    script = Script()
    for m in (M1, M2, M3):
        script.on(status(500), model=m)
    client, _ = make(script)
    with pytest.raises(Exhausted):
        client.complete([M1, M2, M3], MSGS)
    assert len(script.calls) == 7


def test_budget_bounds_each_request_and_stops_the_call():
    clock = FakeClock()

    def slow_failure(request, payload):
        clock.t += 400.0                                  # the upstream took longer than the whole budget
        return httpx.Response(500, text="slow")

    script = Script().on(slow_failure)
    client, _ = make(script, clock=clock)
    with pytest.raises(Exhausted):
        client.complete([M1, M2], MSGS, budget_s=300.0)
    assert len(script.calls) == 1
    read_timeout = script.calls[0]["request"].extensions["timeout"]["read"]
    assert read_timeout <= 300.0


def test_pacing_waits_for_a_window_slot_instead_of_overdriving_a_key():
    script = Script()
    client, clock = make(script, per_minute=2)
    for _ in range(6):                                    # 3 keys x 2 per minute fills every window
        client.complete([M1], MSGS)
    assert clock.t == 1000.0
    client.complete([M1], MSGS)                           # seventh call must wait for the oldest slot to age out
    assert clock.t == pytest.approx(1060.0)


def test_repeated_unimplemented_report_gives_up_on_the_model_without_demoting_it():
    body = '{"detail": "Unimplemented keys: [\\"minItems\\"]"}'
    script = Script().on(status(400, body), model=M1)
    client, _ = make(script)
    with pytest.raises(Exhausted):
        client.complete([M1], MSGS, schema=MIN_ITEMS_SCHEMA)
    # the same keyword again adds nothing new, so the model is abandoned; a 400 alone does not demote it
    assert script.sequence() == [(1, M1), (1, M1)]
    assert M1 not in client.demoted


def test_third_new_unimplemented_keyword_demotes_the_model_after_two_reshapes():
    reports = iter(["minItems", "maxItems", "pattern"])

    def name_another_keyword(request, payload):
        return httpx.Response(400, text='{"detail": "Unimplemented keys: [\\"%s\\"]"}' % next(reports))

    script = Script().on(name_another_keyword, model=M1).on(answer('{"items": ["a"]}'), model=M2)
    client, _ = make(script)
    reply = client.complete([M1, M2], MSGS, schema=MIN_ITEMS_SCHEMA)
    assert reply.model == M2
    assert [m for _, m in script.sequence()] == [M1, M1, M1, M2]        # two reshapes, then demotion
    assert len({s for s, m in script.sequence() if m == M1}) == 1        # reshapes re-use the same key
    assert M1 in client.demoted


def test_load_keys_orders_dedupes_and_keeps_only_nim_names():
    keys = load_keys({"NVIDIA_NIM_API_KEYS": "nvapi-X, nvapi-Y", "NVIDIA_NIM_API_KEY_1": "nvapi-Y",
                      "NVIDIA_NIM_API_KEY_3": "nvapi-Z", "UNRELATED": "nvapi-NOPE"})
    assert keys == ["nvapi-X", "nvapi-Y", "nvapi-Z"]


def test_no_keys_is_a_configuration_error_without_key_material():
    with pytest.raises(NotConfigured) as info:
        NimClient([])
    assert "NVIDIA_NIM_API_KEY_1" in str(info.value)


def test_repr_never_shows_key_material():
    client, _ = make(Script())
    assert "nvapi-" not in repr(client)
    client.close()


def test_every_request_streams_with_an_idle_timeout_inside_the_budget():
    from prophet_agent.nim.transport import BUDGET_S, STREAM_IDLE_S
    script = Script().on(answer('{"a": 1}'))
    client, _ = make(script)
    client.complete([M1], MSGS)
    client.complete([M1], MSGS, schema={"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]})
    client.probe(1)
    assert len(script.calls) == 3
    for c in script.calls:
        assert c["body"]["stream"] is True, "no request may wait for a whole response in one piece"
    for c in script.calls[:2]:
        assert c["request"].extensions["timeout"]["read"] <= STREAM_IDLE_S
    assert BUDGET_S >= 300.0, "long schema-bound answers need room beyond five minutes of silence-free streaming"


def test_probe_reports_status_only_and_never_retries():
    script = Script().on(status(403), slot=1)
    client, _ = make(script)
    result = client.probe(1)
    assert result["status"] == 403
    assert set(result) == {"slot", "model", "status", "error", "elapsed_s"}
    assert len(script.calls) == 1
    client.close()


def test_nim_error_messages_are_redacted_on_construction():
    err = NimError("auth failed for nvapi-SECRETVALUE123")
    assert "SECRETVALUE" not in str(err) and "[key]" in str(err)


def test_slow_first_model_fails_over_while_time_is_left():
    # Live run (missions 5a091d2d, b2108a71): the first model in the chain ran past the whole logical budget, so every
    # later model started with no time and the mission failed with "budget spent" after 426 s. One attempt now has a
    # slice of the budget, so the next model in the chain still gets its turn.
    clock = FakeClock()

    def slow_first(request, payload):
        clock.t += ATTEMPT_BUDGET_S + 30       # the stream runs past its slice before its first line is read
        return sse(["late"])

    script = Script().on(slow_first, model=M1)
    client, _ = make(script, clock=clock)
    reply = client.complete([M1, M2], MSGS)
    assert reply.model == M2
    assert reply.text == "ok"


def test_a_single_slow_model_still_ends_as_exhausted_not_hangs():
    clock = FakeClock()

    def slow(request, payload):
        clock.t += BUDGET_S + 30
        return sse(["late"])

    client, _ = make(Script().on(slow), clock=clock)
    with pytest.raises(Exhausted):
        client.complete([M1], MSGS)
