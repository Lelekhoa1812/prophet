"""Provider registry: key variables, endpoints, request shapes per provider, settings persistence and planner gating.

Offline only. No test calls a real provider; the HTTP layer is an httpx MockTransport.
"""
from __future__ import annotations

import json

import httpx
import pytest

from prophet_agent.nim import NimClient, NotConfigured
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
    assert load_provider_keys(PROVIDERS["claude"], env) == ["sk-ant-1234567890"], "Claude is a single key"
    assert load_provider_keys(PROVIDERS["matilda"], env) == ["mat-key-111111"]
    assert load_provider_keys(PROVIDERS["openai"], env) == ["sk-openai-0000000"]
    assert load_provider_keys(PROVIDERS["grok"], env) == []


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
    assert PROVIDERS["claude"].default_models == ("claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5")
    assert PROVIDERS["openai"].default_models == ("gpt-6.2-astra", "gpt-6.2-sol", "gpt-6-luna")
    assert PROVIDERS["grok"].default_models == ("grok-4.7", "grok-4.3")
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


def test_matilda_without_an_endpoint_is_a_configuration_error(monkeypatch):
    monkeypatch.delenv("MATILDA_BASE_URL", raising=False)
    with pytest.raises(NotConfigured, match="MATILDA_BASE_URL"):
        NimClient(["mat-key-111111"], provider=PROVIDERS["matilda"])


def test_active_provider_persists_across_limit_updates(tmp_path):
    settings = Settings(data_root=tmp_path, dotenv=tmp_path / ".env")
    settings.update({"llm": {"provider": "grok"}})
    settings.update({"agent": {"part_steps": 300}})
    assert active_provider_name(tmp_path) == "grok"
    assert settings.values()["agent"]["part_steps"] == 300
    with pytest.raises(ValueError, match="unknown provider"):
        settings.update({"llm": {"provider": "bogus"}})


def test_keys_are_written_per_provider_and_never_shown_whole(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("OTHER=1\nGOOGLE_API_KEY_2=AIza-old-0000\nOPENAI_API_KEY=sk-openai-OLD0000\n")
    settings = Settings(data_root=tmp_path, dotenv=dotenv)
    settings.set_keys(["AIza-new-1111aaaa", "AIza-new-2222bbbb"], provider="gemini")
    text = dotenv.read_text()
    assert "GOOGLE_API_KEY=AIza-new-1111aaaa,AIza-new-2222bbbb" in text
    assert "GOOGLE_API_KEY_2" not in text, "indexed Gemini keys are replaced, not kept beside the new list"
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
