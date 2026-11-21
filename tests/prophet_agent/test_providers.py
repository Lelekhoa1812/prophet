"""Provider registry: key variables, endpoints, request shapes per provider, settings persistence and planner gating.

Offline only. No test calls a real provider; the HTTP layer is an httpx MockTransport.
"""
from __future__ import annotations

import json
import random

import httpx
import pytest

from prophet_agent.nim import NimClient
from prophet_agent.nim.providers import PROVIDERS, active_provider_name, load_provider_keys, planner_problem
from prophet_agent.nim.transport import _Opts, _Shape, build_body
from prophet_agent.settings import Settings, mask


def _sse(text: str) -> bytes:
    chunks = [{"choices": [{"delta": {"content": text}, "finish_reason": None}]},
              {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}]
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


def _clock():
    now = [0.0]
    return (lambda: now[0]), (lambda s: now.__setitem__(0, now[0] + s))


def test_key_variables_follow_the_brief():
    env = {"GOOGLE_API_KEY": "AIza-one,AIza-two", "GOOGLE_API_KEY_3": "AIza-three", "GOOGLE_API_KEY_1": "AIza-two",
           "ANTHROPIC_API_KEY": "sk-ant-1234567890", "ANTHROPIC_API_KEY_2": "sk-ant-NOT-READ",
           "MATILDA_API_KEY_1": "mat-key-111111", "OPENAI_API_KEY": "sk-openai-0000000"}
    assert load_provider_keys(PROVIDERS["gemini"], env) == ["AIza-one", "AIza-two", "AIza-three"]
    assert load_provider_keys(PROVIDERS["gemini"], {"GEMINI_API_KEY_2": "AIza-alias"}) == ["AIza-alias"], \
        "GEMINI_API_KEY_n is read too, the name the working .env and free-router use"
    assert load_provider_keys(PROVIDERS["claude"], env) == ["sk-ant-1234567890"], "Claude is a single key"
    assert load_provider_keys(PROVIDERS["matilda"], env) == ["mat-key-111111"]
    assert load_provider_keys(PROVIDERS["openai"], env) == ["sk-openai-0000000"]
    assert load_provider_keys(PROVIDERS["grok"], env) == []


def test_gemini_uses_the_openai_compatible_endpoint_and_the_verified_chain():
    gemini = PROVIDERS["gemini"]
    assert gemini.resolve_base_url({}) == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert gemini.models({}) == ("gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
                                 "gemini-3.1-flash-lite", "gemini-3.5-flash-lite")
    assert not any(m.startswith("gemini-2.5") for m in gemini.models({})), "2.5 models were retired on 3 Nov 2026"
    assert gemini.probe_target({}) == "gemini-3.5-flash-lite", "a model that verified, so a 403 means the key"


def test_gemini_request_shape_uses_max_tokens_and_a_strict_json_schema():
    schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    opts = _Opts(thinking=False, max_tokens=500, schema=schema, schema_name="plan", tools=None)
    body = build_body("gemini-3.6-flash", [{"role": "user", "content": "hi"}], opts, _Shape("strict"),
                      PROVIDERS["gemini"])
    assert body["model"] == "gemini-3.6-flash" and body["stream"] is True
    assert body["max_tokens"] == 500 and "max_completion_tokens" not in body
    assert body["temperature"] == 0.2
    assert "chat_template_kwargs" not in body and "reasoning_effort" not in body, "NIM-only fields stay out"
    assert body["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "plan", "schema": schema, "strict": True}}


def test_gemini_client_sends_one_bearer_key_and_fails_over_past_a_rejected_key():
    seen: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization")
        seen.append((str(request.url), auth))
        if auth == "Bearer AIza-rejected-0000":
            return httpx.Response(403, json={"error": {"status": "PERMISSION_DENIED"}})
        return httpx.Response(200, content=_sse("pong"), headers={"content-type": "text/event-stream"})

    clock, sleep = _clock()
    client = NimClient(["AIza-rejected-0000", "AIza-good-1111"], provider=PROVIDERS["gemini"],
                       transport=httpx.MockTransport(handler), clock=clock, sleep=sleep)
    reply = client.complete(["gemini-3.6-flash"], [{"role": "user", "content": "hi"}])
    assert reply.text == "pong" and reply.key_slot in (1, 2)
    assert {url for url, _ in seen} == {"https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"}
    assert all(auth and auth.startswith("Bearer ") for _, auth in seen), "Gemini's OpenAI layer takes a Bearer key"
    assert ("Bearer AIza-good-1111") in {auth for _, auth in seen}, "the good key answered after the rejection"
    assert "AIza-" not in repr(client), "the client never shows key material"


def test_gemini_and_matilda_are_multi_key_and_the_others_are_not():
    assert PROVIDERS["gemini"].multi_key and PROVIDERS["matilda"].multi_key and PROVIDERS["nim"].multi_key
    for name in ("claude", "openai", "grok"):
        assert not PROVIDERS[name].multi_key


def test_request_shape_differs_only_where_the_vendor_differs():
    opts = _Opts(thinking=False, max_tokens=500, schema={"type": "object"}, schema_name="plan", tools=None)
    openai = build_body("gpt-5", [{"role": "user", "content": "hi"}], opts, _Shape("strict"), PROVIDERS["openai"])
    assert openai["max_completion_tokens"] == 500 and "max_tokens" not in openai
    assert "temperature" not in openai and "chat_template_kwargs" not in openai
    assert openai["response_format"]["type"] == "json_schema"

    claude = build_body("claude-sonnet-5-5", [{"role": "user", "content": "hi"}], opts, _Shape("strict"),
                        PROVIDERS["claude"])
    assert "response_format" not in claude, "Claude gets the schema in the prompt, checked locally"

    nim = build_body("nvidia/nemotron-3-ultra-550b-a55b", [{"role": "user", "content": "hi"}], opts,
                     _Shape("strict"), PROVIDERS["nim"])
    assert nim["chat_template_kwargs"] == {"enable_thinking": False}
    assert nim["max_tokens"] == 500 and nim["temperature"] == 0.2


def test_claude_and_openai_send_no_temperature_and_grok_keeps_the_default_shape():
    # Claude Sonnet 5.5 and later return 400 for any temperature below 1; GPT-6 pages say to omit it.
    opts = _Opts(thinking=False, max_tokens=300, schema=None, schema_name="x", tools=None)
    claude = build_body("claude-sonnet-5-5", [{"role": "user", "content": "hi"}], opts, _Shape("strict"),
                        PROVIDERS["claude"])
    assert "temperature" not in claude and claude["max_tokens"] == 300
    openai = build_body("gpt-6.2-astra", [{"role": "user", "content": "hi"}], opts, _Shape("strict"),
                        PROVIDERS["openai"])
    assert "temperature" not in openai and openai["max_completion_tokens"] == 300
    grok = build_body("grok-4.7", [{"role": "user", "content": "hi"}], opts, _Shape("strict"), PROVIDERS["grok"])
    assert grok["temperature"] == 0.2 and grok["stream"] is True


def test_verified_chains_match_the_current_model_ids():
    assert PROVIDERS["claude"].default_models == ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-2")
    assert PROVIDERS["openai"].default_models == ("gpt-6-luna", "gpt-6.1-terra", "gpt-6.2-sol", "gpt-6.2-astra")
    assert PROVIDERS["grok"].default_models == ("grok-4.7", "grok-4.6", "grok-4.3")
    assert PROVIDERS["claude"].probe_target({}) == "claude-haiku-5-5", "the probe stays on the cheapest Claude"
    assert PROVIDERS["openai"].probe_target({}) == "gpt-6-luna"
    assert PROVIDERS["grok"].base_url == "https://api.x.ai/v1"
    assert PROVIDERS["claude"].base_url == "https://api.anthropic.com/v1"


def test_grok_endpoint_and_bearer_header(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, content=_sse("ok"), headers={"content-type": "text/event-stream"})

    clock, sleep = _clock()
    client = NimClient(["xai-000000000"], provider=PROVIDERS["grok"], transport=httpx.MockTransport(handler),
                       clock=clock, sleep=sleep)
    assert client.complete(["grok-4.7"], [{"role": "user", "content": "hi"}]).text == "ok"
    assert seen["url"] == "https://api.x.ai/v1/chat/completions"
    assert seen["auth"] == "Bearer xai-000000000"


def test_client_sends_bearer_key_to_the_vendor_endpoint_and_returns_the_answer():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, content=_sse("hello"), headers={"content-type": "text/event-stream"})

    clock, sleep = _clock()
    client = NimClient(["sk-openai-0000000"], provider=PROVIDERS["openai"], transport=httpx.MockTransport(handler),
                       clock=clock, sleep=sleep)
    reply = client.complete(["gpt-5"], [{"role": "user", "content": "hi"}])
    assert reply.text == "hello" and reply.model == "gpt-5"
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-openai-0000000"
    assert "sk-openai" not in repr(client), "the client never shows key material"


def test_matilda_defaults_to_the_verified_endpoint_and_model(monkeypatch):
    # Source: FreeRouter docs/API_CONTRACTS.md; live-verified 2026-11-21 (model id "matilda", OpenAI-compatible).
    monkeypatch.delenv("MATILDA_BASE_URL", raising=False)
    assert PROVIDERS["matilda"].resolve_base_url() == "https://matilda.maincode.com/api/v1"
    assert PROVIDERS["matilda"].models({}) == ("matilda",)
    assert PROVIDERS["matilda"].probe_target({}) == "matilda"
    assert PROVIDERS["matilda"].usable({})


def test_matilda_endpoint_can_be_overridden(monkeypatch):
    monkeypatch.setenv("MATILDA_BASE_URL", "https://matilda.example.test/api/v1/")
    assert PROVIDERS["matilda"].resolve_base_url() == "https://matilda.example.test/api/v1"


def test_matilda_request_shape_and_auth_follow_the_openai_wire_format(monkeypatch):
    monkeypatch.delenv("MATILDA_BASE_URL", raising=False)
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=_sse('{"status": "ok"}'), headers={"content-type": "text/event-stream"})

    schema = {"type": "object", "properties": {"status": {"type": "string", "enum": ["ok"]}},
              "required": ["status"], "additionalProperties": False}
    clock, sleep = _clock()
    client = NimClient(["mat-key-111111"], provider=PROVIDERS["matilda"], transport=httpx.MockTransport(handler),
                       clock=clock, sleep=sleep)
    reply = client.complete(["matilda"], [{"role": "user", "content": "hi"}], thinking=False, max_tokens=50,
                            schema=schema, schema_name="status_reply")
    assert reply.model == "matilda" and reply.text == '{"status": "ok"}'
    assert seen["url"] == "https://matilda.maincode.com/api/v1/chat/completions"
    assert seen["auth"] == "Bearer mat-key-111111"
    body = seen["body"]
    assert body["model"] == "matilda" and body["stream"] is True and body["max_tokens"] == 50
    assert body["stream_options"] == {"include_usage": True}
    for banned in ("temperature", "top_p", "stop", "reasoning_effort", "chat_template_kwargs", "max_completion_tokens"):
        assert banned not in body, f"Matilda rejects or ignores {banned}"
    assert body["response_format"]["type"] == "json_schema" and body["response_format"]["json_schema"]["strict"] is True
    assert "mat-key-111111" not in repr(client), "the client never shows key material"


def test_matilda_rotates_to_the_next_key_after_a_rate_limit(monkeypatch):
    monkeypatch.delenv("MATILDA_BASE_URL", raising=False)
    used: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.headers["authorization"].removeprefix("Bearer ")
        used.append(key)
        if key == "mat-key-111111":
            return httpx.Response(429, headers={"retry-after": "30"}, content=b'{"error": "rate limited"}')
        return httpx.Response(200, content=_sse("READY"), headers={"content-type": "text/event-stream"})

    class _InOrder(random.Random):                  # keep key order so the first attempt is always key 1
        def shuffle(self, x):
            return None

    clock, sleep = _clock()
    client = NimClient(["mat-key-111111", "mat-key-222222"], provider=PROVIDERS["matilda"],
                       transport=httpx.MockTransport(handler), clock=clock, sleep=sleep, rng=_InOrder())
    reply = client.complete(["matilda"], [{"role": "user", "content": "hi"}], thinking=False, max_tokens=20)
    assert reply.text == "READY" and reply.key_slot == 2
    assert used[0] == "mat-key-111111" and used[-1] == "mat-key-222222"


def test_active_provider_persists_across_limit_updates(tmp_path):
    settings = Settings(data_root=tmp_path, dotenv=tmp_path / ".env")
    settings.update({"llm": {"provider": "grok"}})
    settings.update({"agent": {"part_steps": 300}})
    assert active_provider_name(tmp_path, environ={}) == "grok", "no usable grok key, but it is the only one switched on"
    assert settings.values()["agent"]["part_steps"] == 300
    with pytest.raises(ValueError, match="unknown provider"):
        settings.update({"llm": {"provider": "bogus"}})


def test_keys_are_written_per_provider_and_never_shown_whole(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("OTHER=1\nGEMINI_API_KEY_2=AIza-old-0000\nGOOGLE_API_KEY_2=AIza-old-1111\nOPENAI_API_KEY=sk-openai-OLD0000\n")
    settings = Settings(data_root=tmp_path, dotenv=dotenv)
    settings.set_keys(["AIza-new-1111aaaa", "AIza-new-2222bbbb"], provider="gemini")
    text = dotenv.read_text()
    assert "GOOGLE_API_KEY=AIza-new-1111aaaa,AIza-new-2222bbbb" in text
    assert "GOOGLE_API_KEY_2" not in text and "GEMINI_API_KEY_2" not in text, \
        "indexed Gemini keys under either name are replaced, not kept beside the new list"
    assert "OPENAI_API_KEY=sk-openai-OLD0000" in text, "other providers' lines are left alone"
    assert "OTHER=1" in text
    masked = settings.masked_keys("gemini")
    assert [m["masked"] for m in masked] == [mask("AIza-new-1111aaaa"), mask("AIza-new-2222bbbb")]
    assert "AIza-new-1111aaaa" not in json.dumps(masked)


def test_single_key_providers_refuse_a_list(tmp_path):
    settings = Settings(data_root=tmp_path, dotenv=tmp_path / ".env")
    with pytest.raises(ValueError, match="takes one key"):
        settings.set_keys(["sk-ant-aaaaaaaa1", "sk-ant-bbbbbbbb2"], provider="claude")


def test_planner_names_the_active_provider_when_it_cannot_run(tmp_path, monkeypatch):
    for var in ("NVIDIA_NIM_API_KEYS", "NVIDIA_NIM_API_KEY_1", "XAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(data_root=tmp_path, dotenv=tmp_path / ".env")
    settings.update({"llm": {"provider": "grok"}})
    problem = planner_problem(tmp_path, environ={})
    assert problem == "Grok keys are not configured; the planner cannot run"
    assert planner_problem(tmp_path, environ={"XAI_API_KEY": "xai-123456789"}) is None
