"""A token budget must be a positive whole number or null (no limit). Before the fix, 0 passed validation, and the gate in
nim/transport.py reads a falsy budget as "no limit", so a model set to 0 was never blocked. The UI already refuses 0; the
validation layer must refuse it too, with a message that names the field, and nothing is written when it is refused.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.settings import Settings
from prophet_agent.tools import default_tools

MODEL = "claude-haiku-5-5"


@pytest.fixture()
def settings(tmp_path):
    return Settings(data_root=tmp_path / "data", dotenv=tmp_path / ".env")


def _patch(value):
    return {"llm": {"budgets": {MODEL: {"1d": value}}}}


@pytest.mark.parametrize("value", [0, -1, -5000])
def test_a_zero_or_negative_budget_is_refused_and_nothing_is_written(settings, value):
    with pytest.raises(ValueError, match=rf"budget {MODEL} 1d"):
        settings.update(_patch(value))
    assert not settings.path.exists(), "a refused budget must not create or change settings.json"
    assert settings.budgets()[MODEL]["1d"] is None


def test_a_refused_zero_does_not_erase_a_budget_already_saved(settings):
    settings.update(_patch(5000))
    with pytest.raises(ValueError):
        settings.update(_patch(0))
    assert settings.budgets()[MODEL]["1d"] == 5000
    assert json.loads(settings.path.read_text())["llm"]["budgets"][MODEL]["1d"] == 5000


def test_a_positive_budget_still_saves(settings):
    settings.update(_patch(5000))
    assert settings.budgets()[MODEL]["1d"] == 5000
    assert settings.budgets()[MODEL]["7d"] is None


def test_null_still_clears_a_budget_to_no_limit(settings):
    settings.update(_patch(5000))
    settings.update(_patch(None))
    assert settings.budgets()[MODEL]["1d"] is None


def test_put_settings_refuses_a_zero_budget_with_422_and_writes_nothing(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    client = TestClient(create_app(service), base_url="http://127.0.0.1:8000")
    r = client.put("/api/settings", json=_patch(0))
    assert r.status_code == 422
    assert MODEL in r.json()["detail"] and "1d" in r.json()["detail"]
    assert service.settings.budgets()[MODEL]["1d"] is None
    assert not service.settings.path.exists()


def test_put_settings_still_saves_a_positive_budget(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    client = TestClient(create_app(service), base_url="http://127.0.0.1:8000")
    assert client.put("/api/settings", json=_patch(5000)).status_code == 200
    assert service.settings.budgets()[MODEL]["1d"] == 5000
