"""Model settings, offline: the catalog, the token ledger's window maths, enable/disable and budget validation, key checks
that take invalid keys out of planner use, and one-call model verification through a mock transport.

No test reaches a real provider. Keys are fake values set in the environment or a temporary .env, and every data
folder is temporary, so the real data/ and .env are never read or written.
"""
from __future__ import annotations

import json
from collections import Counter

import httpx
import pytest

from prophet_agent.nim.model_checks import MODEL_CHECK_FILE, verify_models
from prophet_agent.nim.providers import CATALOG, PROVIDERS, model_info
from prophet_agent.settings import Settings, key_status
from prophet_agent.usage import DAY, WINDOWS, UsageLedger, ledger_for

FAKE_KEY = "mk-test-AAAA1111"


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    for name in ["MATILDA_API_KEY"] + [f"MATILDA_API_KEY_{i}" for i in range(1, 10)]:
        monkeypatch.delenv(name, raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("# test only\n")
    return Settings(data_root=tmp_path / "data", dotenv=dotenv)


# --- catalog -----------------------------------------------------------------------------------------------------------

def test_catalog_has_the_agreed_models_per_provider():
    counts = Counter(m.provider for m in CATALOG)
    assert counts == {"nim": 5, "claude": 4, "openai": 4, "grok": 3, "gemini": 6, "matilda": 1}
    ids = [m.id for m in CATALOG]
    assert len(ids) == len(set(ids)), "model ids must be unique across providers"
    assert model_info("claude-haiku-5-5").provider == "claude"


# --- usage ledger ------------------------------------------------------------------------------------------------------

def test_rolling_windows_count_only_events_inside_them(tmp_path):
    now = 1_800_000_000.0
    ledger = UsageLedger(tmp_path / "usage.sqlite", clock=lambda: now)
    ledger.record("claude", "m1", {"prompt_tokens": 60, "completion_tokens": 40}, at=now - 2 * DAY)   # outside 1d
    ledger.record("claude", "m1", {"prompt_tokens": 30, "completion_tokens": 20}, at=now - 3600)
    ledger.record("claude", "m1", {"prompt_tokens": 5, "completion_tokens": 5}, at=now)
    ledger.record("claude", "m2", {"prompt_tokens": 999, "completion_tokens": 1}, at=now)              # other model
    assert ledger.used("m1", "1d", now) == 60
    assert ledger.used("m1", "7d", now) == 160
    assert ledger.used("m1", "30d", now) == 160
    assert ledger.used("m2", "1d", now) == 1000


def test_an_event_exactly_one_window_old_has_left_the_window(tmp_path):
    now = 1_800_000_000.0
    ledger = UsageLedger(tmp_path / "usage.sqlite", clock=lambda: now)
    ledger.record("claude", "m1", {"prompt_tokens": 7, "completion_tokens": 0}, at=now - DAY)
    assert ledger.used("m1", "1d", now) == 0
    assert ledger.used("m1", "7d", now) == 7


def test_reset_is_when_the_oldest_counted_event_leaves_the_window(tmp_path):
    now = 1_800_000_000.0
    ledger = UsageLedger(tmp_path / "usage.sqlite", clock=lambda: now)
    ledger.record("claude", "m1", {"prompt_tokens": 1, "completion_tokens": 1}, at=now - 2 * DAY)
    ledger.record("claude", "m1", {"prompt_tokens": 1, "completion_tokens": 1}, at=now - 3600)
    assert ledger.reset_at("m1", "1d", now) == pytest.approx(now - 3600 + DAY)
    assert ledger.reset_at("m1", "7d", now) == pytest.approx(now - 2 * DAY + WINDOWS["7d"])
    assert ledger.reset_at("nothing", "7d", now) == pytest.approx(now + WINDOWS["7d"])


def test_missing_or_bogus_token_counts_are_recorded_as_zero(tmp_path):
    now = 1_800_000_000.0
    ledger = UsageLedger(tmp_path / "usage.sqlite", clock=lambda: now)
    ledger.record("claude", "m1", {"prompt_tokens": True, "completion_tokens": -4}, at=now)
    ledger.record("claude", "m1", None, at=now)
    assert ledger.used("m1", "1d", now) == 0


def test_summary_lists_each_model_with_budgets_and_resets(tmp_path):
    now = 1_800_000_000.0
    ledger = UsageLedger(tmp_path / "usage.sqlite", clock=lambda: now)
    ledger.record("claude", "claude-haiku-5-5", {"prompt_tokens": 10, "completion_tokens": 5}, at=now)
    row = ledger.summary([model_info("claude-haiku-5-5")], {"claude-haiku-5-5": {"1d": 100}}, now)[0]
    assert row["used"] == {"1d": 15, "7d": 15, "30d": 15}
    assert row["budget"] == {"1d": 100, "7d": None, "30d": None}
    assert set(row["resets"]) == {"1d", "7d", "30d"}


def test_ledger_for_returns_one_shared_ledger_per_folder(tmp_path):
    assert ledger_for(tmp_path / "a") is ledger_for(tmp_path / "a")
    assert ledger_for(tmp_path / "a") is not ledger_for(tmp_path / "b")


# --- enable, disable and budgets ---------------------------------------------------------------------------------------

def test_unknown_provider_model_or_budget_is_refused(settings):
    with pytest.raises(ValueError):
        settings.update({"llm": {"providers": {"nope": {"enabled": True}}}})
    with pytest.raises(ValueError):
        settings.update({"llm": {"models": {"gpt-9-unknown": {"enabled": False}}}})
    with pytest.raises(ValueError):
        settings.update({"llm": {"models": {"claude-haiku-5-5": {"enabled": "no"}}}})
    with pytest.raises(ValueError):
        settings.update({"llm": {"budgets": {"claude-haiku-5-5": {"1w": 5}}}})
    with pytest.raises(ValueError):
        settings.update({"llm": {"budgets": {"claude-haiku-5-5": {"1d": -1}}}})
    with pytest.raises(ValueError):
        settings.update({"llm": {"budgets": {"claude-haiku-5-5": {"1d": True}}}})


def test_budgets_are_saved_and_null_means_no_limit(settings):
    settings.update({"llm": {"budgets": {"claude-haiku-5-5": {"1d": 1000, "7d": None}}}})
    again = Settings(data_root=settings.data_root, dotenv=settings.dotenv)
    budget = again.budgets()["claude-haiku-5-5"]
    assert budget["1d"] == 1000
    assert budget["7d"] is None
    assert budget["30d"] is None


def test_a_disabled_model_or_provider_cannot_be_chosen(settings):
    env = {"ANTHROPIC_API_KEY": FAKE_KEY}                             # with a key, Claude is on by default
    settings.check_model_choice("claude-haiku-5-5", environ=env)     # no error
    settings.update({"llm": {"models": {"claude-haiku-5-5": {"enabled": False}}}})
    with pytest.raises(ValueError):
        settings.check_model_choice("claude-haiku-5-5", environ=env)
    settings.update({"llm": {"providers": {"claude": {"enabled": False}}}})
    with pytest.raises(ValueError):
        settings.check_model_choice("claude-sonnet-5-5", environ=env)
    with pytest.raises(ValueError):
        settings.check_model_choice("not-a-model", environ=env)


def test_a_legacy_provider_choice_means_only_that_provider_is_on(settings):
    settings.path.parent.mkdir(parents=True, exist_ok=True)
    settings.path.write_text(json.dumps({"llm": {"provider": "matilda"}}))
    fresh = Settings(data_root=settings.data_root, dotenv=settings.dotenv)
    assert fresh.provider_enabled("matilda", environ={}) is True
    assert fresh.provider_enabled("claude", environ={}) is False


# --- key checks ---------------------------------------------------------------------------------------------------------

def test_http_status_maps_to_valid_invalid_or_untested():
    assert key_status(200) == "valid"
    assert key_status(401) == "invalid"
    assert key_status(403) == "invalid"
    assert key_status(429) == "untested"
    assert key_status(503) == "untested"
    assert key_status(None) == "untested"


def test_an_invalid_key_is_taken_out_of_planner_use_and_a_valid_one_is_restored(settings, monkeypatch):
    monkeypatch.setenv("MATILDA_API_KEY", FAKE_KEY)
    env = {"MATILDA_API_KEY": FAKE_KEY}
    assert settings.usable_keys("matilda", environ=env) == [FAKE_KEY]

    out = settings.save_key_checks("matilda", [{"slot": 1, "status": 401, "ms": 3}])
    assert out[0]["status"] == "invalid" and out[0]["http_status"] == 401 and out[0]["ok"] is False
    assert FAKE_KEY not in json.dumps(out), "the stored check must never carry the key value"
    assert settings.usable_keys("matilda", environ=env) == []

    settings.save_key_checks("matilda", [{"slot": 1, "status": 200, "ms": 3}])
    assert settings.usable_keys("matilda", environ=env) == [FAKE_KEY]


def test_a_rate_limited_key_stays_in_use(settings, monkeypatch):
    monkeypatch.setenv("MATILDA_API_KEY", FAKE_KEY)
    out = settings.save_key_checks("matilda", [{"slot": 1, "status": 429, "ms": 3}])
    assert out[0]["status"] == "untested" and out[0]["http_status"] == 429
    assert settings.usable_keys("matilda", environ={"MATILDA_API_KEY": FAKE_KEY}) == [FAKE_KEY]


# --- verification through a mock transport (one call per model, no real provider) --------------------------------------

def _sse_ok(text: str, prompt: int, completion: int) -> str:
    chunk = {"model": "matilda", "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
    final = {"model": "matilda", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}
    return f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(final)}\n\ndata: [DONE]\n\n"


def test_verify_models_makes_one_call_per_model_and_records_the_usage(tmp_path):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, text=_sse_ok("ok", 12, 2), headers={"content-type": "text/event-stream"})

    data = tmp_path / "data"
    doc = verify_models(providers=["matilda"], data_root=data, dotenv=tmp_path / ".env",
                        environ={"MATILDA_API_KEY": FAKE_KEY}, path=data / MODEL_CHECK_FILE,
                        transport=httpx.MockTransport(handler))

    assert len(calls) == 1, "exactly one tiny completion for the one matilda model"
    assert doc["models"]["matilda"]["status"] == "verified"
    assert doc["models"]["matilda"]["http_status"] == 200
    assert doc["models"]["claude-haiku-5-5"]["status"] == "untested"
    assert doc["models"]["claude-haiku-5-5"]["error"] == "no key configured"
    assert ledger_for(data).used("matilda", "1d") == 14
    assert FAKE_KEY not in (data / MODEL_CHECK_FILE).read_text()


def test_a_refused_model_is_failed_with_its_status_and_no_key_in_the_error(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": f"bad key {FAKE_KEY}"})

    data = tmp_path / "data"
    doc = verify_models(providers=["matilda"], data_root=data, dotenv=tmp_path / ".env",
                        environ={"MATILDA_API_KEY": FAKE_KEY}, path=data / MODEL_CHECK_FILE,
                        transport=httpx.MockTransport(handler))
    row = doc["models"]["matilda"]
    assert row["status"] == "failed" and row["http_status"] == 401
    assert FAKE_KEY not in json.dumps(doc)
    assert FAKE_KEY not in (data / MODEL_CHECK_FILE).read_text()


def test_a_provider_with_a_key_that_is_not_requested_is_not_called(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no call expected")

    data = tmp_path / "data"
    doc = verify_models(providers=[], data_root=data, dotenv=tmp_path / ".env",
                        environ={"MATILDA_API_KEY": FAKE_KEY}, path=data / MODEL_CHECK_FILE,
                        transport=httpx.MockTransport(handler))
    assert doc["models"]["matilda"]["status"] == "untested"
    assert doc["models"]["matilda"]["error"] == "not called in this run"


def test_every_catalog_model_gets_a_row(tmp_path):
    data = tmp_path / "data"
    doc = verify_models(providers=[], data_root=data, dotenv=tmp_path / ".env", environ={},
                        path=data / MODEL_CHECK_FILE, transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert set(doc["models"]) == {m.id for m in CATALOG}
    assert all(row["status"] == "untested" for row in doc["models"].values())
    assert set(PROVIDERS) >= {row["provider"] for row in doc["models"].values()}
